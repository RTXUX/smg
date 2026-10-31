//! Per-request body-path decision (`--max-buffered-request-bytes`).
//!
//! An eligible request larger than the retry-buffer cap must flow to the
//! worker as a chunked stream, byte-for-byte; with retries disabled it must
//! stream at any size. Under the cap with retries enabled, without a
//! Content-Length, under a text-needing policy, or in a registry serving
//! more than one model, the buffered typed path (which re-serializes and
//! sends a Content-Length) must be used instead.

use std::{process::Command, sync::Arc};

use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE, TRANSFER_ENCODING},
        HeaderMap, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::stream;
use serde_json::{json, Value};
use smg::{
    config::{PolicyConfig, RouterConfig},
    worker::{BasicWorkerBuilder, ModelCard, Worker, WorkerType},
};
use tokio::sync::Mutex;
use tower::ServiceExt;

use crate::common::AppTestContext;

type Captured = Arc<Mutex<Vec<(HeaderMap, Bytes)>>>;

/// Loopback engine stub: captures every forwarded generate/chat request and
/// answers with canned JSON.
#[expect(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
async fn spawn_capture_worker() -> (String, Captured) {
    let captured: Captured = Arc::new(Mutex::new(Vec::new()));

    async fn capture(State(sink): State<Captured>, headers: HeaderMap, body: Bytes) -> Response {
        let payload: Value = serde_json::from_slice(&body).unwrap();
        sink.lock().await.push((headers, body));
        if payload["stream"] == true {
            (
                [(CONTENT_TYPE, "text/event-stream")],
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
            )
                .into_response()
        } else {
            Json(json!({"text": "ok"})).into_response()
        }
    }

    let app = axum::Router::new()
        .route("/generate", axum::routing::post(capture))
        .route("/v1/chat/completions", axum::routing::post(capture))
        .route("/v1/messages", axum::routing::post(capture))
        .route("/v1/messages/count_tokens", axum::routing::post(capture))
        // Outsize axum's 2MB default so multi-MiB streamed forwards land.
        .layer(axum::extract::DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(Arc::clone(&captured));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (format!("http://{addr}"), captured)
}

fn least_load_policy() -> PolicyConfig {
    PolicyConfig::LeastLoad {
        kv_pressure_weight: 0.15,
        mean_prefill_tokens: 1024,
        default_throughput: 2000.0,
        max_waiting_requests: 0,
    }
}

fn cache_aware_policy() -> PolicyConfig {
    PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)
}

fn base_config(max_payload_size: usize) -> RouterConfig {
    RouterConfig::builder()
        .regular_mode(vec![])
        .host("127.0.0.1")
        .port(3002)
        .max_payload_size(max_payload_size)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked()
}

/// App wired through the real `build_app` stack with the capture stub as the
/// only worker.
async fn streaming_app(
    policy: PolicyConfig,
    max_buffered_request_bytes: u64,
    max_payload_size: usize,
) -> (axum::Router, Captured, AppTestContext) {
    let mut config = base_config(max_payload_size);
    config.max_buffered_request_bytes = max_buffered_request_bytes;
    streaming_app_with_config(policy, config).await
}

async fn streaming_app_with_config(
    policy: PolicyConfig,
    mut config: RouterConfig,
) -> (axum::Router, Captured, AppTestContext) {
    let (worker_url, captured) = spawn_capture_worker().await;

    config.policy = policy;
    config.health_check.disable_health_check = true;

    let ctx = AppTestContext::new_with_config(config, vec![]).await;

    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new(&worker_url)
            .worker_type(WorkerType::Regular)
            .models(vec![ModelCard::new("mock-model")])
            .health_config(openai_protocol::worker::HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build(),
    );
    ctx.app_context.worker_registry.register(worker);

    let app = ctx.create_app();
    (app, captured, ctx)
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
fn generate_payload(text_len: usize) -> Vec<u8> {
    serde_json::to_vec(&json!({"text": "x".repeat(text_len), "stream": false})).unwrap()
}

#[expect(
    clippy::unwrap_used,
    reason = "test infrastructure - panicking on failure is intentional"
)]
fn chunked_json_request(uri: &str, payload: Vec<u8>, content_length: Option<u64>) -> Request<Body> {
    let chunks: Vec<Result<Bytes, std::io::Error>> = payload
        .chunks(1024)
        .map(|c| Ok(Bytes::copy_from_slice(c)))
        .collect();
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json");
    if let Some(len) = content_length {
        builder = builder.header(CONTENT_LENGTH, len);
    }
    builder
        .body(Body::from_stream(stream::iter(chunks)))
        .unwrap()
}

/// Each environment value gets a fresh process: the production setting is
/// cached, and changing the parent environment would race unrelated tests.
#[tokio::test]
async fn anthropic_preamble_env_controls_both_endpoints() {
    const CHILD: &str = "SMG_PREAMBLE_TEST_EXPECT_ENABLED";
    const ENV: &str = "SMG_STRIP_ANTHROPIC_PREAMBLE";
    const TEST: &str =
        "routing::stream_request_body_test::anthropic_preamble_env_controls_both_endpoints";
    let Ok(expected) = std::env::var(CHILD) else {
        for (value, enabled) in [
            (None, false),
            (Some(""), false),
            (Some("0"), false),
            (Some("false"), false),
            (Some("off"), false),
            (Some("no"), false),
            (Some("invalid"), false),
            (Some("1"), true),
            (Some("true"), true),
            (Some("on"), true),
            (Some("yes"), true),
            (Some("  TrUe \t"), true),
        ] {
            let mut child = Command::new(std::env::current_exe().unwrap());
            child
                .args([TEST, "--exact", "--nocapture"])
                .env(CHILD, enabled.to_string())
                .env_remove(ENV);
            if let Some(value) = value {
                child.env(ENV, value);
            }
            let output = child.output().unwrap();
            assert!(
                output.status.success(),
                "{ENV}={value:?}:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let enabled = expected == "true";

    // Exercise both the normal buffered path and requests eligible to stream
    // their raw bodies (past the buffer cap or with retries disabled).
    for (cap, disable_retries) in [(1024 * 1024, false), (0, false), (1024 * 1024, true)] {
        let mut config = base_config(1024 * 1024);
        config.max_buffered_request_bytes = cap;
        config.disable_retries = disable_retries;
        let (app, captured, ctx) = streaming_app_with_config(least_load_policy(), config).await;
        let raw_body_eligible = cap == 0 || disable_retries;

        for (system, stripped) in [
            (
                json!(" \nx-anthropic-billing-header: session\r\n  You are helpful."),
                json!("  You are helpful."),
            ),
            (
                json!([
                    {"type": "text", "text": "x-anthropic-billing-header: session\nPrompt",
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "x-anthropic-billing-header: later\nKeep this."}
                ]),
                json!([
                    {"type": "text", "text": "Prompt", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "x-anthropic-billing-header: later\nKeep this."}
                ]),
            ),
            (Value::Null, Value::Null),
        ] {
            for (route, stream) in [
                ("/v1/messages", false),
                ("/v1/messages", true),
                ("/v1/messages/count_tokens", false),
            ] {
                let mut payload = json!({
                    "model": "mock-model",
                    "system": system,
                    "messages": [{"role": "user", "content": "x-anthropic-billing-header: user\nKeep me."}]
                });
                if route == "/v1/messages" {
                    payload["max_tokens"] = json!(32);
                    payload["stream"] = json!(stream);
                }
                let bytes = serde_json::to_vec(&payload).unwrap();
                let response = app
                    .clone()
                    .oneshot(chunked_json_request(
                        route,
                        bytes.clone(),
                        Some(bytes.len() as u64),
                    ))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                if stream {
                    assert_eq!(response.headers()[CONTENT_TYPE], "text/event-stream");
                }
                let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                if stream {
                    assert!(std::str::from_utf8(&response_body)
                        .unwrap()
                        .contains("event: message_stop"));
                }
                let (headers, body) = captured.lock().await.pop().expect("forwarded request");
                let forwarded: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    &forwarded["system"],
                    if enabled { &stripped } else { &system }
                );
                assert_eq!(forwarded["messages"], payload["messages"]);
                let buffered = enabled || !raw_body_eligible || route.ends_with("/count_tokens");
                assert_eq!(
                    headers.contains_key(CONTENT_LENGTH),
                    buffered,
                    "{route}, enabled={enabled}, cap={cap}, disable_retries={disable_retries}"
                );
                if !buffered {
                    assert_eq!(body.as_ref(), bytes.as_slice());
                }
            }
        }

        // The switch must not force other APIs to buffer or alter their prompts.
        let payload = serde_json::to_vec(&json!({
            "model": "mock-model",
            "messages": [{"role": "system", "content": "x-anthropic-billing-header: chat\nKeep me."}]
        })).unwrap();
        let response = app
            .oneshot(chunked_json_request(
                "/v1/chat/completions",
                payload.clone(),
                Some(payload.len() as u64),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let (headers, body) = captured.lock().await.pop().unwrap();
        assert_eq!(headers.contains_key(CONTENT_LENGTH), !raw_body_eligible);
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["messages"],
            serde_json::from_slice::<Value>(&payload).unwrap()["messages"]
        );
        ctx.shutdown().await;
    }
}

#[tokio::test]
async fn large_generate_body_streams_to_worker_chunked() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let parsed: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(parsed["text"], "ok");

    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert_eq!(
        body.as_ref(),
        payload.as_slice(),
        "streamed body must reach the worker byte-for-byte"
    );
    assert!(
        headers.get(CONTENT_LENGTH).is_none(),
        "streamed forward must not carry a Content-Length"
    );
    assert_eq!(
        headers.get(TRANSFER_ENCODING).and_then(|v| v.to_str().ok()),
        Some("chunked")
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn large_chat_body_streams_to_worker_chunked() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = serde_json::to_vec(&json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "y".repeat(16 * 1024)}]
    }))
    .unwrap();

    let req = chunked_json_request(
        "/v1/chat/completions",
        payload.clone(),
        Some(payload.len() as u64),
    );
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert_eq!(body.as_ref(), payload.as_slice());
    assert!(headers.get(CONTENT_LENGTH).is_none());
    drop(captured);
    ctx.shutdown().await;
}

/// The buffered typed path re-serializes a chat request: one with tools and
/// `tool_choice: "none"` reaches the worker with both as sent, so an upstream
/// that renders the prompt itself declares the tools, as the engine's own
/// server does for such a request.
#[tokio::test]
async fn tool_choice_none_forwards_the_tools_to_the_worker() {
    let (app, captured, ctx) =
        streaming_app(least_load_policy(), 64 * 1024, 64 * 1024 * 1024).await;
    let payload = serde_json::to_vec(&json!({
        "model": "mock-model",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": "none"
    }))
    .unwrap();

    let req = chunked_json_request(
        "/v1/chat/completions",
        payload.clone(),
        Some(payload.len() as u64),
    );
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "the buffered typed path re-serializes the request"
    );
    let forwarded: Value = serde_json::from_slice(body).unwrap();
    assert_eq!(forwarded["tool_choice"], json!("none"), "{forwarded}");
    assert_eq!(
        forwarded["tools"].as_array().map(Vec::len),
        Some(1),
        "the tools reach the upstream under none: {forwarded}"
    );
    assert_eq!(
        forwarded["tools"][0]["function"]["name"],
        json!("get_weather")
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn body_under_retry_buffer_cap_stays_buffered() {
    let (app, captured, ctx) =
        streaming_app(least_load_policy(), 64 * 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(64);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "the buffered path sends a fixed-length body"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn retries_disabled_streams_a_small_body() {
    let mut config = base_config(64 * 1024 * 1024);
    config.disable_retries = true;
    let (app, captured, ctx) = streaming_app_with_config(least_load_policy(), config).await;
    let payload = generate_payload(64);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, body) = captured.first().unwrap();
    assert_eq!(body.as_ref(), payload.as_slice());
    assert!(
        headers.get(CONTENT_LENGTH).is_none(),
        "a pure-forward request must stream at any size"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn routing_key_override_buffers_and_prefers_header_over_fresh_rids() {
    let (url_a, captured_a) = spawn_capture_worker().await;
    let (url_b, captured_b) = spawn_capture_worker().await;

    let mut config = base_config(64 * 1024 * 1024);
    config.disable_retries = true;
    config.policy = PolicyConfig::RoundRobin;
    config.routing_key_override.enabled = true;
    config.health_check.disable_health_check = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    for url in [&url_a, &url_b] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new("mock-model")])
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        ctx.app_context.worker_registry.register(worker);
    }
    let app = ctx.create_app();

    let rids = [
        "request:group:59bc514bcf6d43a7b523563a1e9769a0",
        "request:group:9045ab50ac21477aa6b4f6bd29c475b0",
    ];
    for rid in rids {
        let payload = serde_json::to_vec(&json!({
            "input_ids": [1, 2, 3],
            "stream": false,
            "rid": rid,
        }))
        .unwrap();
        let mut req =
            chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
        req.headers_mut()
            .insert("x-smg-routing-key", "conversation-42".parse().unwrap());
        req.headers_mut()
            .insert("x-smg-routing-tokens", "1,2,3".parse().unwrap());
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    let captured_a = captured_a.lock().await;
    let captured_b = captured_b.lock().await;
    assert!(
        matches!((captured_a.len(), captured_b.len()), (2, 0) | (0, 2)),
        "the conversation header must override fresh per-request IDs"
    );
    for ((headers, body), rid) in captured_a.iter().chain(captured_b.iter()).zip(rids) {
        assert!(
            headers.get(CONTENT_LENGTH).is_some(),
            "routing-key override must use the buffered typed path"
        );
        let body: Value = serde_json::from_slice(body).unwrap();
        assert_eq!(body["rid"].as_str(), Some(rid));
        assert_eq!(body["input_ids"], json!([1, 2, 3]));
    }
    drop(captured_a);
    drop(captured_b);
    ctx.shutdown().await;
}

#[tokio::test]
async fn default_config_buffers_small_and_streams_past_one_mib() {
    let (app, captured, ctx) =
        streaming_app_with_config(least_load_policy(), base_config(64 * 1024 * 1024)).await;

    let small = generate_payload(64);
    let req = chunked_json_request("/generate", small.clone(), Some(small.len() as u64));
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    {
        let captured = captured.lock().await;
        let (headers, _) = captured.first().unwrap();
        assert!(
            headers.get(CONTENT_LENGTH).is_some(),
            "a retryable body under the 1MiB default must buffer"
        );
    }

    let app = ctx.create_app();
    let large = generate_payload(2 * 1024 * 1024);
    let req = chunked_json_request("/generate", large.clone(), Some(large.len() as u64));
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, body) = captured.last().unwrap();
    assert_eq!(body.as_ref(), large.as_slice());
    assert!(
        headers.get(CONTENT_LENGTH).is_none(),
        "a body past the 1MiB default must stream and forfeit retries"
    );
    drop(captured);
    ctx.shutdown().await;
}

/// Streamed selection is content-blind, so a second model in the registry
/// must force the buffered typed path — which reads the model and lands the
/// request on the right worker even past the retry-buffer cap.
#[tokio::test]
async fn multi_model_registry_buffers_and_routes_to_the_right_model() {
    let (url_a, captured_a) = spawn_capture_worker().await;
    let (url_b, captured_b) = spawn_capture_worker().await;

    let mut config = base_config(64 * 1024 * 1024);
    config.policy = least_load_policy();
    config.health_check.disable_health_check = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    for (url, model) in [(&url_a, "model-a"), (&url_b, "model-b")] {
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .models(vec![ModelCard::new(model)])
                .health_config(openai_protocol::worker::HealthCheckConfig {
                    disable_health_check: true,
                    ..Default::default()
                })
                .build(),
        );
        ctx.app_context.worker_registry.register(worker);
    }
    let app = ctx.create_app();

    let payload = serde_json::to_vec(&json!({
        "model": "model-b",
        "messages": [{"role": "user", "content": "y".repeat(2 * 1024 * 1024)}]
    }))
    .unwrap();
    let req = chunked_json_request(
        "/v1/chat/completions",
        payload.clone(),
        Some(payload.len() as u64),
    );
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    assert!(
        captured_a.lock().await.is_empty(),
        "model-a's worker must never see a model-b request"
    );
    let captured = captured_b.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "a multi-model registry must take the buffered typed path"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn missing_content_length_stays_buffered() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload, None);
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(headers.get(CONTENT_LENGTH).is_some());
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn text_needing_policy_stays_buffered_past_the_cap() {
    let (app, captured, ctx) = streaming_app(cache_aware_policy(), 1024, 64 * 1024 * 1024).await;
    let payload = generate_payload(16 * 1024);

    let req = chunked_json_request("/generate", payload.clone(), Some(payload.len() as u64));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let captured = captured.lock().await;
    let (headers, _) = captured.first().unwrap();
    assert!(
        headers.get(CONTENT_LENGTH).is_some(),
        "a text-needing policy must keep the buffered path"
    );
    drop(captured);
    ctx.shutdown().await;
}

#[tokio::test]
async fn oversized_streamed_body_is_rejected_with_413() {
    let (app, captured, ctx) = streaming_app(least_load_policy(), 1024, 4096).await;
    // The declared length passes the ingress check; the actual stream is far
    // larger, so the cap must trip while forwarding.
    let payload = generate_payload(64 * 1024);

    let req = chunked_json_request("/generate", payload, Some(2048));
    let resp = app.oneshot(req).await.unwrap();

    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(
        resp.headers()
            .get("X-SMG-Error-Code")
            .and_then(|v| v.to_str().ok()),
        Some("request_body_too_large")
    );
    assert!(captured.lock().await.is_empty());
    ctx.shutdown().await;
}
