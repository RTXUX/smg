use engine_zmq_client::protocol::vllm::{
    output::EngineCoreEvent,
    sampling::EngineCoreSamplingParams,
    stats::{KvCacheEvictionEvent, PrefillStats, SpecDecodingStats},
    VllmTelemetry,
};

use super::*;

fn collector() -> VllmMetrics {
    let metrics = VllmMetrics::new(
        "fixture".into(),
        4096,
        MetricsConfig {
            kv_cache_metrics: true,
            num_speculative_tokens: 3,
            ..Default::default()
        },
    )
    .unwrap();
    metrics.initialize_rank(0);
    metrics
}
fn request(id: &str) -> EngineCoreRequest {
    EngineCoreRequest {
        request_id: id.into(),
        arrival_time: 100.0,
        prompt_token_ids: Some(vec![1; 100]),
        sampling_params: Some(EngineCoreSamplingParams {
            max_tokens: 16,
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn first() -> EngineBatch<EngineCoreOutput> {
    let mut stats = SchedulerStats {
        num_running_reqs: 2,
        num_waiting_reqs: 3,
        num_skipped_waiting_reqs: 1,
        kv_cache_usage: 0.5,
        ..Default::default()
    };
    stats.prefix_cache_stats.base.queries = 100;
    stats.prefix_cache_stats.base.hits = 30;
    let mut external = stats.prefix_cache_stats.clone();
    external.base.queries = 20;
    external.base.hits = 10;
    stats.connector_prefix_cache_stats = Some(external);
    stats.spec_decoding_stats = Some(SpecDecodingStats {
        num_spec_tokens: 3,
        num_drafts: 1,
        num_draft_tokens: 3,
        num_accepted_tokens: 2,
        num_accepted_tokens_per_pos: vec![1, 1, 0],
    });
    stats.kv_cache_eviction_events = vec![KvCacheEvictionEvent {
        lifetime_seconds: 2.0,
        idle_seconds: 0.2,
        reuse_gaps_seconds: vec![0.1, 0.3],
    }];
    EngineBatch {
        vllm_telemetry: Some(VllmTelemetry {
            timestamp: 11.0,
            scheduler_stats: Some(Box::new(stats)),
        }),
        outputs: vec![EngineCoreOutput {
            request_id: "a".into(),
            new_token_ids: vec![1],
            prefill_stats: Some(PrefillStats {
                num_prompt_tokens: 100,
                num_computed_tokens: 60,
                num_cached_tokens: 40,
                num_local_cached_tokens: 30,
                num_external_cached_tokens: 10,
                ..Default::default()
            }),
            events: Some(vec![
                EngineCoreEvent {
                    r#type: EngineCoreEventType::Queued,
                    timestamp: 10.0,
                },
                EngineCoreEvent {
                    r#type: EngineCoreEventType::Scheduled,
                    timestamp: 10.5,
                },
            ]),
            ..Default::default()
        }],
        ..Default::default()
    }
}
fn second() -> EngineBatch<EngineCoreOutput> {
    EngineBatch {
        vllm_telemetry: Some(VllmTelemetry {
            timestamp: 12.0,
            scheduler_stats: None,
        }),
        outputs: vec![EngineCoreOutput {
            request_id: "a".into(),
            new_token_ids: vec![2, 3],
            finish_reason: Some(EngineCoreFinishReason::Length),
            events: Some(vec![
                EngineCoreEvent {
                    r#type: EngineCoreEventType::Preempted,
                    timestamp: 11.25,
                },
                EngineCoreEvent {
                    r#type: EngineCoreEventType::Scheduled,
                    timestamp: 11.5,
                },
            ]),
            ..Default::default()
        }],
        finished_request_ids: vec!["a".into()],
        ..Default::default()
    }
}

// Compare the actual Rust registry to a fixture produced by Python vLLM's
// PrometheusStatLogger + IterationStats, including every histogram bucket.
#[test]
fn matches_python_vllm_metrics_for_prefill_decode_and_preemption() {
    let metrics = collector();
    metrics.submitted(0, &request("a"));
    metrics.batch_at(&first(), 101.0);
    metrics.batch_at(&second(), 102.0);
    let expected: BTreeMap<String, f64> =
        serde_json::from_str(include_str!("../../../tests/fixtures/vllm_metrics.json")).unwrap();
    let mut compared = 0;
    let mut check = |name: &str, labels: &BTreeMap<String, String>, value: f64| {
        let key = format!("{name}{}", serde_json::to_string(labels).unwrap());
        let reference = expected
            .get(&key)
            .unwrap_or_else(|| panic!("missing Python sample: {key}"));
        assert!(
            (reference - value).abs() < 1e-9,
            "{key}: Python={reference}, Rust={value}"
        );
        compared += 1;
    };
    for family in metrics.registry.gather() {
        for metric in family.get_metric() {
            let labels: BTreeMap<_, _> = metric
                .get_label()
                .iter()
                .map(|p| (p.name().to_owned(), p.value().to_owned()))
                .collect();
            match family.get_field_type() {
                prometheus::proto::MetricType::COUNTER => {
                    check(family.name(), &labels, metric.get_counter().get_value())
                }
                prometheus::proto::MetricType::GAUGE => {
                    check(family.name(), &labels, metric.get_gauge().get_value())
                }
                prometheus::proto::MetricType::HISTOGRAM => {
                    let hist = metric.get_histogram();
                    check(
                        &format!("{}_count", family.name()),
                        &labels,
                        hist.get_sample_count() as f64,
                    );
                    check(
                        &format!("{}_sum", family.name()),
                        &labels,
                        hist.get_sample_sum(),
                    );
                    let mut labels = labels;
                    for bucket in hist.get_bucket() {
                        labels.insert("le".into(), bucket.upper_bound().to_string());
                        check(
                            &format!("{}_bucket", family.name()),
                            &labels,
                            bucket.cumulative_count() as f64,
                        );
                    }
                    labels.insert("le".into(), "+Inf".into());
                    check(
                        &format!("{}_bucket", family.name()),
                        &labels,
                        hist.get_sample_count() as f64,
                    );
                }
                kind => panic!("unexpected metric type: {kind:?}"),
            }
        }
    }
    assert!(compared > 300, "only {compared} samples compared");
    assert!(metrics.requests.lock().unwrap().is_empty());
    assert_eq!(
        metrics.render().unwrap(),
        metrics.render().unwrap(),
        "scraping resets no values"
    );
}

#[test]
fn stats_only_batches_and_dp_ranks_are_recorded_once() {
    let metrics = collector();
    metrics.initialize_rank(7);
    let mut batch = first();
    batch.outputs.clear();
    batch.engine_index = 7;
    metrics.batch_at(&batch, 101.0);
    let text = metrics.render().unwrap();
    assert!(text
        .lines()
        .any(|l| l.starts_with("vllm:prefix_cache_hits_total{")
            && l.contains("engine=\"7\"")
            && l.ends_with(" 30")));
    assert!(text
        .lines()
        .any(|l| l.starts_with("vllm:prefix_cache_hits_total{")
            && l.contains("engine=\"0\"")
            && l.ends_with(" 0")));
    assert_eq!(text, metrics.render().unwrap());
}

#[test]
fn aborted_and_refused_requests_release_tracking_and_ignore_late_outputs() {
    let metrics = collector();
    metrics.submitted(0, &request("a"));
    metrics.removed("a", false);
    metrics.batch_at(&first(), 101.0);
    assert!(metrics.requests.lock().unwrap().is_empty());
    let text = metrics.render().unwrap();
    assert!(text
        .lines()
        .any(|l| l.starts_with("vllm:generation_tokens_total{") && l.ends_with(" 0")));
    metrics.submitted(0, &request("a"));
    metrics.removed("a", true);
    metrics.removed("a", true);
    assert!(metrics
        .render()
        .unwrap()
        .lines()
        .any(|l| l.starts_with("vllm:request_success_total{")
            && l.contains("finished_reason=\"abort\"")
            && l.ends_with(" 1")));
    metrics.submitted(0, &request("b"));
    metrics.closed();
    assert!(metrics.requests.lock().unwrap().is_empty());
}

#[test]
fn registries_are_isolated_and_custom_buckets_are_used() {
    let mut config = MetricsConfig::default();
    config
        .histogram_buckets
        .insert("time_to_first_token".into(), vec![0.25, 2.0]);
    let a = VllmMetrics::new("a".into(), 4096, config).unwrap();
    a.initialize_rank(0);
    let b = collector();
    a.submitted(0, &request("a"));
    a.batch_at(&first(), 101.0);
    let text = a.render().unwrap();
    assert!(text.lines().any(
        |l| l.starts_with("vllm:time_to_first_token_seconds_bucket{")
            && l.contains("le=\"2\"")
            && l.ends_with(" 1")
    ));
    assert!(!text.contains("spec_decode_"));
    assert!(!b.render().unwrap().contains("model_name=\"a\""));
}

#[test]
fn remote_prefill_cache_hits_only_override_finished_request_statistics() {
    let metrics = collector();
    let mut req = request("a");
    req.sampling_params.as_mut().unwrap().extra_args = Some(HashMap::from([(
        "kv_transfer_params".into(),
        serde_json::json!({
            "do_remote_prefill": true,
            "remote_prefill_cached_tokens": 7,
        }),
    )]));
    metrics.submitted(0, &req);
    metrics.batch_at(&first(), 101.0);
    metrics.batch_at(&second(), 102.0);
    let text = metrics.render().unwrap();
    assert!(text
        .lines()
        .any(|s| s.starts_with("vllm:prompt_tokens_cached_total{") && s.ends_with(" 40")));
    assert!(text.lines().any(
        |s| s.starts_with("vllm:request_prefill_kv_computed_tokens_sum{") && s.ends_with(" 93")
    ));
}
