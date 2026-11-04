"""The Python servicer's HTTP endpoint for vLLM's native Prometheus registry."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from smg_grpc_servicer.rust_lifecycle import ServicerConfigError, metrics_bind_address
from smg_grpc_servicer.vllm import METRICS_HOST_ENV, METRICS_PORT_ENV

logger = logging.getLogger(__name__)


def snapshot() -> bytes:
    # Resolve on every scrape: the engine can configure multiprocessing after
    # this listener binds. Keep its native labels, counters and histograms.
    from prometheus_client import generate_latest
    from vllm.v1.metrics.prometheus import get_prometheus_registry

    return generate_latest(get_prometheus_registry())


@asynccontextmanager
async def metrics_server(args):
    """Bind before engine startup, and close on return, failure or cancellation."""
    address = metrics_bind_address(args, port_env=METRICS_PORT_ENV, host_env=METRICS_HOST_ENV)
    if address is None:
        yield None
        return

    from aiohttp import web
    from prometheus_client import CONTENT_TYPE_LATEST

    async def scrape(request):
        # Registry serialization and multiprocess file reads stay off the
        # event loop that forwards inference streams.
        return web.Response(
            body=await asyncio.to_thread(snapshot),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    app = web.Application()
    app.router.add_get("/metrics", scrape)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    bind = urlsplit(f"//{address}")
    try:
        try:
            await web.TCPSite(runner, bind.hostname, bind.port).start()
        except OSError as error:
            raise ServicerConfigError(
                f"Cannot bind metrics listener at {address}: {error}"
            ) from error
        logger.info("vLLM Prometheus metrics listening at %s", runner.addresses)
        yield runner.addresses
    finally:
        await runner.cleanup()
