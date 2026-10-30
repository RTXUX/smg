// Metric definitions from Apache-2.0 vLLM v1/metrics/loggers.py and
// v1/spec_decode/metrics.py. Keep names, HELP, labels and bucket families aligned.
use super::{Definition, Kind};

pub(super) const DEFINITIONS: &[Definition] = &[
    Definition { name: "num_requests_running", help: "Number of requests in model execution batches.", kind: Kind::Gauge, labels: &[], buckets: "" },
    Definition { name: "num_requests_waiting", help: "Number of requests waiting to be processed.", kind: Kind::Gauge, labels: &[], buckets: "" },
    Definition { name: "num_requests_waiting_by_reason", help: "Number of waiting requests by reason. Reason labels: 'capacity' = waiting for scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons equals vllm:num_requests_waiting.", kind: Kind::Gauge, labels: &["reason"], buckets: "" },
    Definition { name: "kv_cache_usage_perc", help: "KV-cache usage. 1 means 100 percent usage.", kind: Kind::Gauge, labels: &[], buckets: "" },
    Definition { name: "prefix_cache_queries", help: "Prefix cache queries, in terms of number of queried tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "prefix_cache_hits", help: "Prefix cache hits, in terms of number of cached tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "external_prefix_cache_queries", help: "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "external_prefix_cache_hits", help: "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "num_preemptions", help: "Cumulative number of preemption from the engine.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "prompt_tokens", help: "Number of prefill tokens processed.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "prompt_tokens_by_source", help: "Number of prompt tokens by source.", kind: Kind::Counter, labels: &["source"], buckets: "" },
    Definition { name: "prompt_tokens_cached", help: "Number of cached prompt tokens (local + external).", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "generation_tokens", help: "Number of generation tokens processed.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "request_success", help: "Count of successfully processed requests.", kind: Kind::Counter, labels: &["finished_reason"], buckets: "" },
    Definition { name: "request_prompt_tokens", help: "Number of prefill tokens processed.", kind: Kind::Histogram, labels: &[], buckets: "request_tokens" },
    Definition { name: "request_generation_tokens", help: "Number of generation tokens processed.", kind: Kind::Histogram, labels: &[], buckets: "request_tokens" },
    Definition { name: "iteration_tokens_total", help: "Histogram of number of tokens per engine_step.", kind: Kind::Histogram, labels: &[], buckets: "iteration_tokens" },
    Definition { name: "request_params_max_tokens", help: "Histogram of the max_tokens request parameter.", kind: Kind::Histogram, labels: &[], buckets: "request_tokens" },
    Definition { name: "time_to_first_token_seconds", help: "Histogram of time to first token in seconds.", kind: Kind::Histogram, labels: &[], buckets: "time_to_first_token" },
    Definition { name: "inter_token_latency_seconds", help: "Histogram of inter-token latency in seconds.", kind: Kind::Histogram, labels: &[], buckets: "inter_token_latency" },
    Definition { name: "request_time_per_output_token_seconds", help: "Histogram of time_per_output_token_seconds per request.", kind: Kind::Histogram, labels: &[], buckets: "inter_token_latency" },
    Definition { name: "e2e_request_latency_seconds", help: "Histogram of e2e request latency in seconds.", kind: Kind::Histogram, labels: &[], buckets: "request_latency" },
    Definition { name: "request_queue_time_seconds", help: "Histogram of time spent in WAITING phase for request.", kind: Kind::Histogram, labels: &[], buckets: "request_latency" },
    Definition { name: "request_inference_time_seconds", help: "Histogram of time spent in RUNNING phase for request.", kind: Kind::Histogram, labels: &[], buckets: "request_latency" },
    Definition { name: "request_prefill_time_seconds", help: "Histogram of time spent in PREFILL phase for request.", kind: Kind::Histogram, labels: &[], buckets: "request_latency" },
    Definition { name: "request_decode_time_seconds", help: "Histogram of time spent in DECODE phase for request.", kind: Kind::Histogram, labels: &[], buckets: "request_latency" },
    Definition { name: "request_num_preemptions", help: "Histogram of the number of times a request was preempted.", kind: Kind::Histogram, labels: &[], buckets: "request_num_preemptions" },
    Definition { name: "request_prefill_kv_computed_tokens", help: "Histogram of new KV tokens computed during prefill (excluding cached tokens).", kind: Kind::Histogram, labels: &[], buckets: "request_tokens" },
    Definition { name: "corrupted_requests", help: "Corrupted requests, in terms of total number of requests with NaNs in logits.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "kv_block_lifetime_seconds", help: "Histogram of KV cache block lifetime from allocation to eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).", kind: Kind::Histogram, labels: &[], buckets: "kv_cache_residency" },
    Definition { name: "kv_block_idle_before_evict_seconds", help: "Histogram of idle time before KV cache block eviction. Sampled metrics (controlled by --kv-cache-metrics-sample).", kind: Kind::Histogram, labels: &[], buckets: "kv_cache_residency" },
    Definition { name: "kv_block_reuse_gap_seconds", help: "Histogram of time gaps between consecutive KV cache block accesses. Only the most recent accesses are recorded (ring buffer). Sampled metrics (controlled by --kv-cache-metrics-sample).", kind: Kind::Histogram, labels: &[], buckets: "kv_cache_residency" },
    Definition { name: "spec_decode_num_drafts", help: "Number of spec decoding drafts.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "spec_decode_num_draft_tokens", help: "Number of draft tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "spec_decode_num_accepted_tokens", help: "Number of accepted tokens.", kind: Kind::Counter, labels: &[], buckets: "" },
    Definition { name: "spec_decode_num_accepted_tokens_per_pos", help: "Accepted tokens per draft position.", kind: Kind::Counter, labels: &["position"], buckets: "" },
];

pub(super) fn default_buckets(family: &str, max_model_len: u32) -> Vec<f64> {
    match family {
        "request_latency" => vec![
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
            120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0,
        ],
        "time_to_first_token" => vec![
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0,
        ],
        "inter_token_latency" => vec![
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5,
            10.0, 20.0, 40.0, 80.0,
        ],
        "iteration_tokens" => vec![
            1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0,
            16384.0,
        ],
        "request_params_n" => vec![1.0, 2.0, 5.0, 10.0, 20.0],
        "request_num_preemptions" => vec![1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
        "kv_cache_residency" => vec![
            0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0,
            60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0,
        ],
        "request_tokens" => {
            let mut buckets = Vec::new();
            let mut scale = 1_u64;
            while scale <= u64::from(max_model_len) {
                for mantissa in [1, 2, 5] {
                    let value = mantissa * scale;
                    if value <= u64::from(max_model_len) {
                        buckets.push(value as f64);
                    }
                }
                scale *= 10;
            }
            buckets
        }
        _ => Vec::new(),
    }
}
