"""The ``--metrics-port`` and ``--metrics-host`` flags the engine launchers
register for the servicers' Prometheus endpoints. The Rust launchers share
one registration; vLLM also enables it for its Python implementation.

Stdlib only on purpose, like the servicer-implementation flag: the engine
plugins that add the flags are loaded in every engine process and import
nothing heavier than the package init. The lifecycle module that resolves the
flags into a bind address (``rust_lifecycle.RustLauncher.metrics_bind_address``)
is imported by the launchers, later and lazily.

Each flag's store action also writes its value into the launcher's
environment variable (``SMG_<ENGINE>_SERVICER_METRICS_PORT`` and
``..._HOST``): SGLang's and TokenSpeed's ``ServerArgs`` keep only their own
fields, so a flag given on their command lines travels to the launcher as the
variable, and the precedence the three launchers share (the flag, else the
variable, else no endpoint) holds by construction.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

METRICS_PORT_FLAG = "--metrics-port"
METRICS_HOST_FLAG = "--metrics-host"


class _EnvStoreAction(argparse._StoreAction):  # noqa: SLF001 — SGLang's --config merger admits store actions only
    """A store action that also exports the value under the launcher's
    variable, so the launcher reads a flag through the same name as the
    deployment's export. It derives from argparse's store action because
    SGLang's ``--config`` merger takes only store and store_true options."""

    def __init__(self, *args: Any, env_name: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.env_name = env_name

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        super().__call__(parser, namespace, values, option_string)
        if values is not None:
            os.environ[self.env_name] = str(values)


def add_metrics_arguments(
    parser: argparse.ArgumentParser,
    *,
    port_env: str,
    host_env: str,
    server_class: str,
    python_supported: bool = False,
) -> bool:
    """Add ``--metrics-port`` and ``--metrics-host`` to a launcher's parser
    that lacks them; returns whether they were added. ``server_class`` names
    the Rust server the help text points at. ``python_supported`` also permits
    the Python implementation in that help text."""
    actions = parser._option_string_actions  # noqa: SLF001 — argparse's registry of option strings
    if METRICS_PORT_FLAG in actions:
        return False
    parser.add_argument(
        METRICS_PORT_FLAG,
        dest="metrics_port",
        type=int,
        default=None,
        action=_EnvStoreAction,
        env_name=port_env,
        help=(
            "Serve Prometheus metrics (GET /metrics) "
            + ("of either Python or Rust servicers " if python_supported else f"of {server_class} ")
            + "on this port, on a listener of its own; 0 binds an "
            f"ephemeral one. Unset falls back to ${port_env}; unset there too, no "
            "endpoint. Must differ from the gRPC port."
            + ("" if python_supported else " Needs the Rust implementation.")
        ),
    )
    parser.add_argument(
        METRICS_HOST_FLAG,
        dest="metrics_host",
        default=None,
        action=_EnvStoreAction,
        env_name=host_env,
        help=(
            "The host the metrics listener binds; defaults to the gRPC listener's "
            f"host. Unset falls back to ${host_env}."
        ),
    )
    return True
