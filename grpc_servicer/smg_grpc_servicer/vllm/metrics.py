"""Export the same registry as vLLM's HTTP /metrics, without an HTTP server."""

import asyncio

from smg_grpc_proto.generated import common_pb2


def snapshot() -> str:
    # Lazy imports keep the protocol handler usable without importing an engine
    # at module load time. vLLM selects its multiprocess registry when needed.
    from prometheus_client import generate_latest
    from vllm.v1.metrics.prometheus import get_prometheus_registry

    return generate_latest(get_prometheus_registry()).decode("utf-8")


async def get_metrics() -> common_pb2.GetMetricsResponse:
    # Registry serialization (including multiprocess file reads) must not block
    # the asyncio loop that forwards generation streams.
    return common_pb2.GetMetricsResponse(prometheus_text=await asyncio.to_thread(snapshot))
