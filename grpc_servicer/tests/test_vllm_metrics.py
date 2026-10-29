"""The native registry survives the generated metrics RPC unchanged."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import grpc
import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from smg_grpc_proto import vllm_engine_pb2_grpc
from smg_grpc_proto.generated import common_pb2

_path = Path(__file__).parents[1] / "smg_grpc_servicer" / "vllm" / "metrics.py"
_spec = importlib.util.spec_from_file_location("vllm_metrics_under_test", _path)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)


@pytest.mark.asyncio
async def test_native_registry_round_trip_does_not_reset_or_relabel(monkeypatch):
    registry = CollectorRegistry()
    counter = Counter(
        "vllm:generation_tokens", "Tokens", ["model_name", "engine"], registry=registry
    )
    histogram = Histogram(
        "vllm:time_to_first_token_seconds",
        "TTFT",
        ["model_name", "engine"],
        buckets=[0.1, 1.0],
        registry=registry,
    )
    counter.labels('model"\\\n', "1").inc(7)
    histogram.labels('model"\\\n', "1").observe(0.2)
    # vLLM owns registry selection, including multiprocess mode. Do not fall
    # back to prometheus_client.REGISTRY or create another metrics registry.
    module = ModuleType("vllm.v1.metrics.prometheus")
    module.get_prometheus_registry = lambda: registry
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class Servicer(vllm_engine_pb2_grpc.VllmEngineServicer):
        async def GetMetrics(self, request, context):
            return await metrics.get_metrics()

    server = grpc.aio.server()
    vllm_engine_pb2_grpc.add_VllmEngineServicer_to_server(Servicer(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)
            for increment in (0, 3):
                counter.labels('model"\\\n', "1").inc(increment)
                expected = generate_latest(registry).decode()
                for _ in range(2):
                    response = await stub.GetMetrics(common_pb2.GetMetricsRequest(), timeout=2)
                    assert response.prometheus_text == expected
    finally:
        await server.stop(None)


@pytest.mark.asyncio
async def test_empty_registry_is_a_successful_empty_snapshot(monkeypatch):
    monkeypatch.setattr(metrics, "snapshot", lambda: "")
    assert (await metrics.get_metrics()).prometheus_text == ""
