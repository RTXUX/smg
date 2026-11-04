"""Global DP ownership on the wire, frontend-local rank indices in AsyncLLM."""

from typing import Any

from smg_grpc_servicer.vllm.config_view import EngineConfigView


def served_dp_ranks(engine: Any) -> list[int] | None:
    """Read the client's actual engine identities, preserving its index order.

    Older clients without this information leave topology unspecified. Do not
    infer ownership from global DP size: hybrid/external clients serve subsets.
    An empty managed list is authoritative (no engines), not legacy metadata.
    """
    managed = getattr(getattr(engine, "engine_core", None), "engine_ranks_managed", None)
    if managed is None:
        return None
    if not isinstance(managed, (list, tuple, range)):
        raise ValueError("engine_ranks_managed must be a sequence of global DP ranks")
    ranks = list(managed)
    size = EngineConfigView(getattr(engine, "vllm_config", None)).parallel.data_parallel_size
    if any(
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or (size is not None and rank >= size)
        for rank in ranks
    ) or len(set(ranks)) != len(ranks):
        raise ValueError(f"invalid managed DP ranks: {ranks}")
    return ranks


def local_dp_rank(engine: Any, global_rank: int | None) -> int | None:
    """Translate a gRPC global rank pin to AsyncLLM's managed-engine index."""
    if global_rank is None:
        return None
    ranks = served_dp_ranks(engine)
    if ranks is None:
        # Preserve the historical contract for clients without rank metadata.
        return global_rank
    try:
        return ranks.index(global_rank)
    except ValueError:
        raise ValueError(
            f"DP rank {global_rank} is not served by this endpoint ({ranks})"
        ) from None
