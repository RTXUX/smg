//! Worker-owned vLLM metrics. Observe the engine wire once, before load
//! projection and per-request stream fan-out; scraping only reads the registry.
//! Accounting follows vLLM's IterationStats and PrometheusStatLogger.
use std::{
    collections::{BTreeMap, HashMap},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

use engine_zmq_client::{
    connector::EngineObserver,
    protocol::{
        vllm::{
            output::{EngineCoreEventType, EngineCoreFinishReason, EngineCoreOutput},
            request::EngineCoreRequest,
            stats::SchedulerStats,
            VllmProtocol,
        },
        EngineBatch,
    },
};
use prometheus::{CounterVec, GaugeVec, HistogramOpts, HistogramVec, Opts, Registry, TextEncoder};
use serde::{Deserialize, Serialize};

mod definitions;
#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Resolved vLLM bucket families, including user overrides.
    pub histogram_buckets: BTreeMap<String, Vec<f64>>,
    pub kv_cache_metrics: bool,
    pub compute_nans: bool,
    pub num_speculative_tokens: u32,
    pub is_diffusion: bool,
}
#[derive(Clone, Copy)]
enum Kind {
    Counter,
    Gauge,
    Histogram,
}
struct Definition {
    name: &'static str,
    help: &'static str,
    kind: Kind,
    labels: &'static [&'static str],
    buckets: &'static str,
}

#[derive(Default)]
struct RequestStats {
    engine: u32,
    arrival: f64,
    prompt: u64,
    cached: u64,
    prefill_recorded: bool,
    remote_cached: Option<u64>,
    max_tokens: Option<u32>,
    generated: u64,
    preemptions: u64,
    queued: Option<f64>,
    scheduled: Option<f64>,
    first: Option<f64>,
    last: f64,
    corrupted: bool,
}

pub(super) struct VllmMetrics {
    registry: Registry,
    model: String,
    counters: HashMap<&'static str, CounterVec>,
    gauges: HashMap<&'static str, GaugeVec>,
    histograms: HashMap<&'static str, HistogramVec>,
    requests: Mutex<HashMap<String, RequestStats>>,
    config: MetricsConfig,
}

impl VllmMetrics {
    pub(super) fn new(
        model: String,
        max_model_len: u32,
        config: MetricsConfig,
    ) -> Result<Self, prometheus::Error> {
        let mut this = Self {
            registry: Registry::new(),
            model,
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            requests: Mutex::new(HashMap::new()),
            config,
        };
        for def in definitions::DEFINITIONS {
            if def.name.starts_with("kv_block_") && !this.config.kv_cache_metrics {
                continue;
            }
            if def.name == "corrupted_requests" && !this.config.compute_nans {
                continue;
            }
            if def.name.starts_with("spec_decode_")
                && (this.config.num_speculative_tokens == 0 || this.config.is_diffusion)
            {
                continue;
            }
            let mut labels = vec!["model_name", "engine"];
            labels.extend(def.labels);
            let name = format!("vllm:{}", def.name);
            match def.kind {
                Kind::Counter => {
                    // rust-prometheus does not add Python's _total suffix.
                    let metric =
                        CounterVec::new(Opts::new(format!("{name}_total"), def.help), &labels)?;
                    this.registry.register(Box::new(metric.clone()))?;
                    this.counters.insert(def.name, metric);
                }
                Kind::Gauge => {
                    let metric = GaugeVec::new(Opts::new(name, def.help), &labels)?;
                    this.registry.register(Box::new(metric.clone()))?;
                    this.gauges.insert(def.name, metric);
                }
                Kind::Histogram => {
                    let buckets = this
                        .config
                        .histogram_buckets
                        .get(def.buckets)
                        .cloned()
                        .unwrap_or_else(|| {
                            definitions::default_buckets(def.buckets, max_model_len)
                        });
                    let metric = HistogramVec::new(
                        HistogramOpts::new(name, def.help).buckets(buckets),
                        &labels,
                    )?;
                    this.registry.register(Box::new(metric.clone()))?;
                    this.histograms.insert(def.name, metric);
                }
            }
        }
        Ok(this)
    }

    pub(super) fn render(&self) -> Result<String, prometheus::Error> {
        TextEncoder::new().encode_to_string(&self.registry.gather())
    }

    fn inc(&self, name: &str, rank: u32, value: u64, extra: &[&str]) {
        if let Some(metric) = self.counters.get(name) {
            let rank = rank.to_string();
            let mut labels = vec![self.model.as_str(), rank.as_str()];
            labels.extend_from_slice(extra);
            metric.with_label_values(&labels).inc_by(value as f64);
        }
    }
    fn gauge(&self, name: &str, rank: u32, value: f64, extra: &[&str]) {
        if let Some(metric) = self.gauges.get(name) {
            let rank = rank.to_string();
            let mut labels = vec![self.model.as_str(), rank.as_str()];
            labels.extend_from_slice(extra);
            metric.with_label_values(&labels).set(value);
        }
    }
    fn observe(&self, name: &str, rank: u32, value: f64) {
        if let Some(metric) = self.histograms.get(name) {
            metric
                .with_label_values(&[&self.model, &rank.to_string()])
                .observe(value);
        }
    }

    /// Initialize bounded per-engine label sets before the first request.
    pub(super) fn initialize_rank(&self, rank: u32) {
        for def in definitions::DEFINITIONS {
            if def.labels.is_empty() {
                match def.kind {
                    Kind::Counter => self.inc(def.name, rank, 0, &[]),
                    Kind::Gauge => self.gauge(def.name, rank, 0.0, &[]),
                    Kind::Histogram => {
                        if let Some(h) = self.histograms.get(def.name) {
                            h.with_label_values(&[&self.model, &rank.to_string()]);
                        }
                    }
                }
            }
        }
        for source in ["local_compute", "local_cache_hit", "external_kv_transfer"] {
            self.inc("prompt_tokens_by_source", rank, 0, &[source]);
        }
        for reason in ["stop", "length", "abort", "error", "repetition"] {
            self.inc("request_success", rank, 0, &[reason]);
        }
        for reason in ["capacity", "deferred"] {
            self.gauge("num_requests_waiting_by_reason", rank, 0.0, &[reason]);
        }
        for position in 0..self.config.num_speculative_tokens {
            self.inc(
                "spec_decode_num_accepted_tokens_per_pos",
                rank,
                0,
                &[&position.to_string()],
            );
        }
    }

    fn scheduler(&self, rank: u32, stats: &SchedulerStats) {
        self.gauge(
            "num_requests_running",
            rank,
            stats.num_running_reqs as f64,
            &[],
        );
        self.gauge(
            "num_requests_waiting",
            rank,
            (stats.num_waiting_reqs + stats.num_skipped_waiting_reqs) as f64,
            &[],
        );
        self.gauge(
            "num_requests_waiting_by_reason",
            rank,
            stats.num_waiting_reqs as f64,
            &["capacity"],
        );
        self.gauge(
            "num_requests_waiting_by_reason",
            rank,
            stats.num_skipped_waiting_reqs as f64,
            &["deferred"],
        );
        self.gauge("kv_cache_usage_perc", rank, stats.kv_cache_usage, &[]);
        self.inc(
            "prefix_cache_queries",
            rank,
            stats.prefix_cache_stats.base.queries,
            &[],
        );
        self.inc(
            "prefix_cache_hits",
            rank,
            stats.prefix_cache_stats.base.hits,
            &[],
        );
        if let Some(cache) = &stats.connector_prefix_cache_stats {
            self.inc(
                "external_prefix_cache_queries",
                rank,
                cache.base.queries,
                &[],
            );
            self.inc("external_prefix_cache_hits", rank, cache.base.hits, &[]);
        }
        if let Some(spec) = &stats.spec_decoding_stats {
            self.inc("spec_decode_num_drafts", rank, spec.num_drafts, &[]);
            self.inc(
                "spec_decode_num_draft_tokens",
                rank,
                spec.num_draft_tokens,
                &[],
            );
            self.inc(
                "spec_decode_num_accepted_tokens",
                rank,
                spec.num_accepted_tokens,
                &[],
            );
            for (pos, accepted) in spec
                .num_accepted_tokens_per_pos
                .iter()
                .take(self.config.num_speculative_tokens as usize)
                .enumerate()
            {
                self.inc(
                    "spec_decode_num_accepted_tokens_per_pos",
                    rank,
                    *accepted,
                    &[&pos.to_string()],
                );
            }
        }
        for eviction in &stats.kv_cache_eviction_events {
            self.observe("kv_block_lifetime_seconds", rank, eviction.lifetime_seconds);
            self.observe(
                "kv_block_idle_before_evict_seconds",
                rank,
                eviction.idle_seconds,
            );
            for gap in &eviction.reuse_gaps_seconds {
                self.observe("kv_block_reuse_gap_seconds", rank, *gap);
            }
        }
    }

    /// A stop string is recognized in the frontend after the engine's delta
    /// was counted. Complete it before the stream drop queues an engine abort.
    pub(super) fn frontend_finished(&self, id: &str, reason: &str) {
        let request = self
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
        if let Some(request) = request {
            self.finish(&request, reason, wall_time());
        }
    }

    fn finish(&self, request: &RequestStats, reason: &str, now: f64) {
        let rank = request.engine;
        self.inc("request_success", rank, 1, &[reason]);
        self.observe("e2e_request_latency_seconds", rank, now - request.arrival);
        self.observe("request_prompt_tokens", rank, request.prompt as f64);
        self.observe("request_generation_tokens", rank, request.generated as f64);
        self.observe("request_num_preemptions", rank, request.preemptions as f64);
        self.observe(
            "request_prefill_kv_computed_tokens",
            rank,
            request.prompt.saturating_sub(request.cached) as f64,
        );
        if let Some(max_tokens) = request.max_tokens.filter(|v| *v > 0) {
            self.observe("request_params_max_tokens", rank, f64::from(max_tokens));
        }
        if request.corrupted {
            self.inc("corrupted_requests", rank, 1, &[]);
        }
        // Errors/cancellations can finish before scheduling or the first
        // token. An absent timestamp is not monotonic time zero: only observe
        // phases whose endpoints actually occurred.
        if let (Some(queued), Some(scheduled)) = (request.queued, request.scheduled) {
            self.observe("request_queue_time_seconds", rank, scheduled - queued);
        }
        if let Some(first) = request.first {
            let decode = request.last - first;
            if let Some(scheduled) = request.scheduled {
                self.observe("request_prefill_time_seconds", rank, first - scheduled);
                self.observe(
                    "request_inference_time_seconds",
                    rank,
                    request.last - scheduled,
                );
            }
            self.observe("request_decode_time_seconds", rank, decode);
            self.observe(
                "request_time_per_output_token_seconds",
                rank,
                if request.generated > 1 {
                    decode / (request.generated - 1) as f64
                } else {
                    0.0
                },
            );
        }
    }

    fn output(
        &self,
        request: &mut RequestStats,
        output: &EngineCoreOutput,
        timestamp: f64,
        now: f64,
    ) -> u64 {
        let rank = request.engine;
        for event in output.events.as_deref().unwrap_or_default() {
            match event.r#type {
                EngineCoreEventType::Queued => {
                    request.queued.get_or_insert(event.timestamp);
                }
                EngineCoreEventType::Scheduled => {
                    request.scheduled.get_or_insert(event.timestamp);
                }
                EngineCoreEventType::Preempted => {
                    request.preemptions += 1;
                    self.inc("num_preemptions", rank, 1, &[]);
                }
            }
        }
        request.corrupted |= output.num_nans_in_logits > 0;
        let mut computed = 0;
        // Prefill accounting is independent of token timing (pooling outputs
        // can carry prefill statistics without any generated tokens).
        if !request.prefill_recorded {
            if let Some(prefill) = &output.prefill_stats {
                request.prefill_recorded = true;
                request.cached = u64::from(prefill.num_cached_tokens);
                computed = u64::from(prefill.num_computed_tokens);
                self.inc(
                    "prompt_tokens",
                    rank,
                    u64::from(prefill.num_prompt_tokens),
                    &[],
                );
                self.inc("prompt_tokens_cached", rank, request.cached, &[]);
                for (source, value) in [
                    ("local_compute", prefill.num_computed_tokens),
                    ("local_cache_hit", prefill.num_local_cached_tokens),
                    ("external_kv_transfer", prefill.num_external_cached_tokens),
                ] {
                    self.inc("prompt_tokens_by_source", rank, u64::from(value), &[source]);
                }
            }
        }
        // vLLM's decode frontend reports the remote prefill worker's
        // cache hits in finished-request statistics. Per-iteration token
        // counters above still use this engine's actual prefill stats.
        if let Some(cached) = request.remote_cached {
            request.cached = cached;
        }
        // Terminal errors and event-only outputs do not produce a token.
        if output.new_token_ids.is_empty() {
            return computed;
        }
        if request.first.is_none() {
            self.observe("time_to_first_token_seconds", rank, now - request.arrival);
            request.first = Some(timestamp);
        } else {
            // One observation per engine output, including multi-token spec
            // decode steps, exactly as IterationStats (not divided by tokens).
            self.observe(
                "inter_token_latency_seconds",
                rank,
                timestamp - request.last,
            );
        }
        request.last = timestamp;
        request.generated += output.new_token_ids.len() as u64;
        self.inc(
            "generation_tokens",
            rank,
            output.new_token_ids.len() as u64,
            &[],
        );
        computed + output.new_token_ids.len() as u64
    }

    fn batch_at(&self, batch: &EngineBatch<EngineCoreOutput>, now: f64) {
        let Some(telemetry) = &batch.vllm_telemetry else {
            return;
        };
        if let Some(stats) = &telemetry.scheduler_stats {
            self.scheduler(batch.engine_index, stats);
        }
        let mut requests = self
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut tokens = 0;
        for output in &batch.outputs {
            let Some(request) = requests.get_mut(&output.request_id) else {
                continue;
            };
            tokens += self.output(request, output, telemetry.timestamp, now);
            if let Some(reason) = output.finish_reason {
                self.finish(
                    request,
                    match reason {
                        EngineCoreFinishReason::Stop => "stop",
                        EngineCoreFinishReason::Length => "length",
                        EngineCoreFinishReason::Abort => "abort",
                        EngineCoreFinishReason::Error => "error",
                        EngineCoreFinishReason::Repetition => "repetition",
                    },
                    now,
                );
                requests.remove(&output.request_id);
            }
        }
        for id in &batch.finished_request_ids {
            requests.remove(id);
        }
        self.observe("iteration_tokens_total", batch.engine_index, tokens as f64);
    }
}

fn wall_time() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
impl EngineObserver<VllmProtocol> for VllmMetrics {
    fn connected(&self, ranks: &[u32]) {
        for rank in ranks {
            self.initialize_rank(*rank);
        }
    }
    fn submitted(&self, engine: u32, request: &EngineCoreRequest) {
        let remote_cached = request
            .sampling_params
            .as_ref()
            .and_then(|params| params.extra_args.as_ref())
            .and_then(|args| args.get("kv_transfer_params"))
            .filter(|params| {
                params
                    .get("do_remote_prefill")
                    .and_then(serde_json::Value::as_bool)
                    == Some(true)
            })
            .and_then(|params| params.get("remote_prefill_cached_tokens"))
            .and_then(serde_json::Value::as_u64);
        let stats = RequestStats {
            engine,
            remote_cached,
            arrival: request.arrival_time,
            prompt: request
                .prompt_token_ids
                .as_ref()
                .map_or(0, |ids| ids.len() as u64),
            max_tokens: request.sampling_params.as_ref().map(|p| p.max_tokens),
            ..Default::default()
        };
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(request.request_id.clone(), stats);
    }
    fn batch(&self, batch: &EngineBatch<EngineCoreOutput>) {
        self.batch_at(batch, wall_time());
    }
    fn removed(&self, id: &str, aborted: bool) {
        let request = self
            .requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
        if let Some(request) = request.filter(|_| aborted) {
            self.finish(&request, "abort", wall_time());
        }
    }
    fn closed(&self) {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
    }
}
