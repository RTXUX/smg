"""Native metrics served over HTTP for the lifetime of the Python launcher."""

import asyncio
import socket
import sys
import threading
import types
from types import SimpleNamespace

import aiohttp
import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram
from smg_grpc_servicer.rust_lifecycle import ServicerConfigError
from smg_grpc_servicer.vllm import metrics


@pytest.fixture(autouse=True)
def clean_metrics_env(monkeypatch):
    monkeypatch.delenv("SMG_VLLM_SERVICER_METRICS_PORT", raising=False)
    monkeypatch.delenv("SMG_VLLM_SERVICER_METRICS_HOST", raising=False)


def test_http_scrapes_native_registry_and_stops(monkeypatch):
    registry = CollectorRegistry()
    calls = Counter(
        "vllm:request_success", "Completed", ["model_name", "engine"], registry=registry
    )
    latency = Histogram("vllm:e2e_request_latency_seconds", "Latency", registry=registry)
    calls.labels("test/model", "8").inc(2)
    latency.observe(0.25)
    threads = []

    def get_registry():
        threads.append(threading.get_ident())
        return registry

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.metrics.prometheus",
        types.SimpleNamespace(get_prometheus_registry=get_registry),
    )

    async def run():
        args = SimpleNamespace(host="127.0.0.1", port=50051, metrics_port=0)
        async with metrics.metrics_server(args) as addresses:
            host, port = addresses[0][:2]
            async with aiohttp.ClientSession() as client:
                for expected in [2, 3]:
                    async with client.get(f"http://{host}:{port}/metrics") as response:
                        assert response.status == 200
                        assert response.headers["Content-Type"].startswith("text/plain")
                        text = await response.text()
                        assert (
                            f'vllm:request_success_total{{engine="8",model_name="test/model"}} {expected}.0'
                            in text
                        )
                        assert "vllm:e2e_request_latency_seconds_count 1.0" in text
                    calls.labels("test/model", "8").inc()
                async with client.get(f"http://{host}:{port}/missing") as response:
                    assert response.status == 404
        assert threads and all(thread != threading.get_ident() for thread in threads)
        with socket.socket() as sock:
            assert sock.connect_ex((host, port)) != 0

    asyncio.run(run())


def test_metrics_disabled_and_busy_port(monkeypatch):
    async def run():
        async with metrics.metrics_server(SimpleNamespace()) as addresses:
            assert addresses is None
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            args = SimpleNamespace(host="127.0.0.1", port=50051, metrics_port=busy.getsockname()[1])
            with pytest.raises(ServicerConfigError, match="Cannot bind metrics listener"):
                async with metrics.metrics_server(args):
                    pytest.fail("engine must not start")

    asyncio.run(run())


def test_launcher_cleanup_on_failure_and_cancellation(monkeypatch):
    from smg_grpc_servicer.vllm import launcher_switch, rust

    monkeypatch.setattr(rust, "default_kv_events_config", lambda args: None)
    monkeypatch.setattr(rust, "require_python_media_mode", lambda args: None)
    monkeypatch.setattr(metrics, "snapshot", lambda: b"native_metric 1\n")

    async def run():
        for fail in [True, False]:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            started = asyncio.Event()

            async def launch(args):
                async with aiohttp.ClientSession() as client:
                    async with client.get(f"http://127.0.0.1:{port}/metrics") as response:
                        assert await response.text() == "native_metric 1\n"
                started.set()
                if fail:
                    raise RuntimeError("engine failed")
                await asyncio.Event().wait()

            args = SimpleNamespace(
                servicer_impl="python", host="127.0.0.1", port=50051, metrics_port=port
            )
            task = asyncio.create_task(launcher_switch.with_servicer_switch(launch)(args))
            await asyncio.wait_for(started.wait(), 5)
            if fail:
                with pytest.raises(RuntimeError, match="engine failed"):
                    await task
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            with socket.socket() as probe:
                assert probe.connect_ex(("127.0.0.1", port)) != 0

    asyncio.run(run())


def test_python_plugin_accepts_metrics_and_validates_port():
    import argparse

    from smg_grpc_servicer.vllm import plugin

    parser = argparse.ArgumentParser()
    parser.add_argument("--grpc", action="store_true")
    args = SimpleNamespace(
        grpc=True, servicer_impl="python", host="127.0.0.1", port=50051, metrics_port=0
    )
    assert plugin.check_servicer_impl(parser, args) == "python"
    args.metrics_port = 50051
    with pytest.raises(SystemExit) as error:
        plugin.check_servicer_impl(parser, args)
    assert error.value.code == 2
