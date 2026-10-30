# vLLM metric parity fixture

`vllm_metrics.json` was generated from the adjacent vLLM checkout at
`c62d291b537d9de264508a6965ca0d7270f4a83a` using:

```sh
python generate_vllm_metrics.py /path/to/vllm > vllm_metrics.json
```

The script executes the checkout's actual `PrometheusStatLogger`,
`IterationStats`, speculative-decoding collector and histogram bucket helper.
It loads their definitions without vLLM's GPU imports; optional KV connector
and performance plugins are stubbed and are not part of the comparison.
The input has a cached prefill, a multi-token decode step, a preemption,
external prefix-cache hits, speculative acceptance and cache eviction samples.
Wall-clock and engine monotonic timestamps are deliberately distinct.

The Rust test compares every sample it exports against this independent
Python result, including zero series and cumulative histogram buckets. Python
`*_created` timestamps are excluded; histogram bound strings are normalized
so `1` and `1.0` compare as the same bound. Python families not implemented by
Rust remain in the fixture, but are not claimed as covered. Separate tests
cover stats-only batches, nonzero DP ranks, cancellation, custom buckets,
registry isolation, disabled statistics and the real gRPC/ZMQ path.
