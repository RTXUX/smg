"""Numeric priority reaches AsyncLLM through the real Python servicer."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("vllm")
from smg_grpc_proto import vllm_engine_pb2  # noqa: E402
from smg_grpc_servicer.vllm.servicer import VllmEngineServicer  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [None, 0, 17, -3, -(2**31), 2**31 - 1])
@pytest.mark.parametrize("stream", [False, True])
async def test_generate_priority_is_negated_once(priority, stream):
    seen = []

    async def generate(*, priority, request_id, **kwargs):
        seen.append(priority)
        yield RequestOutput(
            request_id=request_id,
            prompt=None,
            prompt_token_ids=[1, 2],
            prompt_logprobs=None,
            outputs=[CompletionOutput(0, "", [42], None, None, finish_reason="length")],
            finished=True,
        )

    engine = SimpleNamespace(
        generate=generate,
        renderer=SimpleNamespace(process_for_engine=lambda prompt, **kwargs: prompt),
    )
    servicer = VllmEngineServicer(engine, start_time=0.0)
    request = vllm_engine_pb2.GenerateRequest(
        request_id="priority",
        tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 2]),
        sampling_params=vllm_engine_pb2.SamplingParams(max_tokens=1),
        stream=stream,
    )
    if priority is not None:
        request.priority = priority
    context = SimpleNamespace(abort=AsyncMock(), invocation_metadata=lambda: ())
    responses = [response async for response in servicer.Generate(request, context)]
    context.abort.assert_not_awaited()
    assert responses
    assert seen == [-(priority or 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("priority", [None, 0, 17, -3, -(2**31), 2**31 - 1])
async def test_embed_priority_is_negated_once(priority):
    seen = []

    async def encode(*, priority, **kwargs):
        seen.append(priority)
        yield SimpleNamespace(
            finished=True,
            prompt_token_ids=[1, 2],
            outputs=SimpleNamespace(data=SimpleNamespace(tolist=lambda: [0.25])),
        )

    servicer = VllmEngineServicer(SimpleNamespace(encode=encode), start_time=0.0)
    request = vllm_engine_pb2.EmbedRequest(
        request_id="priority", tokenized=vllm_engine_pb2.TokenizedInput(input_ids=[1, 2])
    )
    if priority is not None:
        request.priority = priority
    context = SimpleNamespace(abort=AsyncMock())
    response = await servicer.Embed(request, context)
    context.abort.assert_not_awaited()
    assert response.embedding == [0.25]
    assert seen == [-(priority or 0)]
