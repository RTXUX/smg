"""Regenerate the parity oracle from a vLLM checkout, without loading torch/GPU.

Usage: python generate_vllm_metrics.py /path/to/vllm > vllm_metrics.json
Executes vLLM's actual PrometheusStatLogger, IterationStats and bucket helpers.
Only optional connector/performance integrations are stubbed (not compared).
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import json
import runpy
import sys
import time
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import prometheus_client as prom

root = Path(sys.argv[1]) / "vllm"


class Noop:
    def __init__(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass


class FinishReason(enum.StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"
    ERROR = "error"
    REPETITION = "repetition"


class CacheHitSource(enum.StrEnum):
    DEVICE = "device"
    HOST = "host"
    DISK = "disk"
    P2P = "p2p"
    EXTERNAL_UNSPECIFIED = "external_unspecified"


globals_ = {
    "__name__": __name__,
    "dataclass": dataclasses.dataclass,
    "field": dataclasses.field,
    "time": time,
    "prometheus_client": prom,
    "Counter": prom.Counter,
    "Gauge": prom.Gauge,
    "Histogram": prom.Histogram,
    "AggregateStatLoggerBase": object,
    "unregister_vllm_metrics": lambda: None,
    "envs": NS(VLLM_COMPUTE_NANS_IN_LOGITS=False),
    "KVConnectorProm": Noop,
    "ECConnectorProm": Noop,
    "PerfMetricsProm": Noop,
    "FinishReason": FinishReason,
    "CacheHitSource": CacheHitSource,
    "WAITING_REASON_CAPACITY": "capacity",
    "WAITING_REASON_DEFERRED": "deferred",
    "KV_FETCH_STAGES": (),
    "ADMISSION_REASONS": ("max_num_queued_reqs", "max_num_queued_tokens"),
    **runpy.run_path(str(root / "v1/metrics/buckets.py")),
}

globals_["__name__"] = __name__


def load(relative, names):
    tree = ast.parse((root / relative).read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    assert len(nodes) == len(names)
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(root / relative), "exec"), globals_)


load("v1/metrics/utils.py", ["create_metric_per_engine"])
load("v1/spec_decode/metrics.py", ["SpecDecodingProm"])
load(
    "v1/metrics/stats.py",
    [
        "RequestStateStats",
        "FinishedRequestStats",
        "PrefillStats",
        "PromptTokenStats",
        "IterationStats",
    ],
)
load("v1/metrics/loggers.py", ["PrometheusStatLogger"])
engine = ModuleType("vllm.v1.engine")
engine.EngineCoreEventType = NS(QUEUED=1, SCHEDULED=2, PREEMPTED=3)
sys.modules[engine.__name__] = engine
config = NS(
    model_config=NS(served_model_name="fixture", max_model_len=4096, is_diffusion=False),
    observability_config=NS(
        show_hidden_metrics=False, kv_cache_metrics=True, custom_histogram_buckets={}
    ),
    speculative_config=NS(num_speculative_tokens=3),
    kv_transfer_config=None,
    watermark_config=None,
    lora_config=None,
)
logger = globals_["PrometheusStatLogger"](config, [0])
request = globals_["RequestStateStats"](arrival_time=100.0)
lora = NS(request_waiting=lambda *a: None, request_running=lambda *a: None)
prefill = globals_["PrefillStats"](
    num_prompt_tokens=100,
    num_computed_tokens=60,
    num_cached_tokens=40,
    num_local_cached_tokens=30,
    num_external_cached_tokens=10,
)
first = NS(
    request_id="a",
    new_token_ids=[1],
    prefill_stats=prefill,
    events=[NS(type=1, timestamp=10.0), NS(type=2, timestamp=10.5)],
    num_nans_in_logits=0,
)
second = NS(
    request_id="a",
    new_token_ids=[2, 3],
    prefill_stats=None,
    events=[NS(type=3, timestamp=11.25), NS(type=2, timestamp=11.5)],
    num_nans_in_logits=0,
)
cache = NS(queries=100, hits=30)
stats = NS(
    num_running_reqs=2,
    num_waiting_reqs=3,
    num_skipped_waiting_reqs=1,
    kv_cache_usage=0.5,
    queue_stats=None,
    prefix_cache_stats=cache,
    connector_prefix_cache_stats=NS(queries=20, hits=10, hits_by_source={}),
    spec_decoding_stats=NS(
        num_drafts=1,
        num_draft_tokens=3,
        num_accepted_tokens=2,
        num_accepted_tokens_per_pos=[1, 1, 0],
    ),
    kv_connector_stats=None,
    ec_connector_stats=None,
    perf_stats=None,
    kv_cache_eviction_events=[
        NS(lifetime_seconds=2.0, idle_seconds=0.2, reuse_gaps_seconds=[0.1, 0.3])
    ],
)
for output, monotonic, wall, is_prefill in [
    (first, 11.0, 101.0, True),
    (second, 12.0, 102.0, False),
]:
    iteration = globals_["IterationStats"]()
    iteration.iteration_timestamp = wall
    iteration.update_from_output(output, monotonic, is_prefill, request, lora, None)
    if not is_prefill:
        iteration.update_from_finished_request(FinishReason.LENGTH, "a", 100, 16, request, 40)
    logger.record(stats if is_prefill else None, iteration, engine_idx=0)

result = {}
for family in prom.REGISTRY.collect():
    if not family.name.startswith("vllm:"):
        continue
    for sample in family.samples:
        if sample.name.endswith("_created"):
            continue
        labels = dict(sample.labels)
        if "le" in labels and labels["le"] != "+Inf":
            labels["le"] = format(float(labels["le"]), "g")
        key = sample.name + json.dumps(labels, sort_keys=True, separators=(",", ":"))
        result[key] = sample.value
print(json.dumps(result, indent=2, sort_keys=True))
