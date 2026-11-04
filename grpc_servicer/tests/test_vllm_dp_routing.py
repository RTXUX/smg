"""Endpoint ownership and global-to-local pins, without importing vLLM/torch."""

import ast
import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import grpc
import pytest
from smg_grpc_proto import vllm_engine_pb2
from smg_grpc_servicer.vllm.dp import local_dp_rank, served_dp_ranks


def _engine(ranks):
    return SimpleNamespace(
        engine_core=SimpleNamespace(engine_ranks_managed=ranks),
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=16)),
    )


@pytest.mark.parametrize("ranks", [[8], [8, 10, 15], [15, 8], range(8, 16)])
def test_pins_follow_the_managed_engine_order(ranks):
    engine = _engine(ranks)
    assert served_dp_ranks(engine) == list(ranks)
    for index, rank in enumerate(ranks):
        assert local_dp_rank(engine, rank) == index
    assert local_dp_rank(engine, None) is None
    with pytest.raises(ValueError, match="not served"):
        local_dp_rank(engine, 0)


def test_unknown_and_empty_topology_are_distinct():
    assert served_dp_ranks(_engine(None)) is None
    assert local_dp_rank(_engine(None), 5) == 5  # legacy full-group pin
    assert served_dp_ranks(_engine([])) == []
    with pytest.raises(ValueError, match="not served"):
        local_dp_rank(_engine([]), 0)


@pytest.mark.parametrize("ranks", [[-1], [16], [8, 8], [True], [1.5], "8,9"])
def test_invalid_engine_topology_is_not_advertised(ranks):
    with pytest.raises(ValueError):
        served_dp_ranks(_engine(ranks))


def _generate_method():
    """Run the actual RPC method with a fake engine at the AsyncLLM boundary."""
    source = Path(__file__).parents[1] / "smg_grpc_servicer/vllm/servicer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VllmEngineServicer"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "Generate"
    )
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "asyncio": asyncio,
        "time": time,
        "grpc": grpc,
        "logger": logging.getLogger(__name__),
        "mm_batches": lambda request: [],
        "params_from_request": lambda request: None,
        "_prompt_length": lambda prompt: len(prompt),
        "trace_headers": lambda metadata: {},
        "local_dp_rank": local_dp_rank,
        "VLLMClientError": type("VLLMClientError", (Exception,), {}),
        "GracefulHTTPError": type("GracefulHTTPError", (Exception,), {}),
        "grpc_code_for": lambda error: grpc.StatusCode.INVALID_ARGUMENT,
        "MmProcessorUnavailable": type("MmProcessorUnavailable", (Exception,), {}),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Generate"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ranks,pin,expected", [([8], 8, 0), ([8, 10], 10, 1), ([8, 10], None, None), (None, 3, 3)]
)
async def test_generate_translates_the_global_pin(ranks, pin, expected):
    seen = []

    async def generate(**kwargs):
        seen.append(kwargs["data_parallel_rank"])
        if False:
            yield  # An engine stream with no output still exercises submission.

    engine = _engine(ranks)
    engine.generate = generate
    servicer = SimpleNamespace(
        engine=engine,
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda params: None,
        _loads=SimpleNamespace(submitted=Mock(), finished=Mock()),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    request = vllm_engine_pb2.GenerateRequest(
        request_id="dp-pin", text="hello", data_parallel_rank=pin
    )
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    assert [response async for response in _generate_method()(servicer, request, context)] == []
    context.abort.assert_not_awaited()
    assert seen == [expected]


@pytest.mark.asyncio
async def test_generate_rejects_an_unowned_rank_before_submission():
    engine = _engine([8, 10])
    engine.generate = Mock()
    servicer = SimpleNamespace(
        engine=engine,
        _sampling_params_from_proto=lambda *args, **kwargs: SimpleNamespace(
            logprobs=None, prompt_logprobs=None
        ),
        _tokenization_kwargs_from_proto=lambda params: None,
        _loads=SimpleNamespace(submitted=Mock(), finished=Mock()),
        _notify_kv_transfer_rejected=AsyncMock(),
    )
    request = vllm_engine_pb2.GenerateRequest(
        request_id="invalid-dp", text="hello", data_parallel_rank=0
    )
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    assert [response async for response in _generate_method()(servicer, request, context)] == []
    engine.generate.assert_not_called()
    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT
