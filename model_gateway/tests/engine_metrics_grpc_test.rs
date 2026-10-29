//! Mixed-transport scrapes preserve worker identity and do not affect health.
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{response::IntoResponse, routing::get, Router};
use smg::worker::{
    manager::EngineMetricsResult, BasicWorkerBuilder, ConnectionMode, RuntimeType, WorkerManager,
    WorkerRegistry,
};
use smg_grpc_client::{common_proto as common, vllm_proto as proto};
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tonic::{Request, Response, Status};

#[derive(Clone)]
struct Engine {
    hits: Arc<AtomicUsize>,
    unsupported: bool,
}
#[tonic::async_trait]
impl proto::vllm_engine_server::VllmEngine for Engine {
    type GenerateStream = ReceiverStream<Result<proto::GenerateResponse, Status>>;
    type GetTokenizerStream = ReceiverStream<Result<common::GetTokenizerChunk, Status>>;
    type SubscribeKvEventsStream = ReceiverStream<Result<common::KvEventBatch, Status>>;
    async fn get_metrics(
        &self,
        _: Request<common::GetMetricsRequest>,
    ) -> Result<Response<common::GetMetricsResponse>, Status> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        if self.unsupported {
            return Err(Status::unimplemented("older worker"));
        }
        Ok(Response::new(common::GetMetricsResponse { prometheus_text: "# HELP vllm:tokens_total Tokens\n# TYPE vllm:tokens_total counter\nvllm:tokens_total{engine=\"0\"} 17\n".into() }))
    }
    async fn generate(
        &self,
        _: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn embed(
        &self,
        _: Request<proto::EmbedRequest>,
    ) -> Result<Response<proto::EmbedResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn abort(
        &self,
        _: Request<proto::AbortRequest>,
    ) -> Result<Response<proto::AbortResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn health_check(
        &self,
        _: Request<proto::HealthCheckRequest>,
    ) -> Result<Response<proto::HealthCheckResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_loads(
        &self,
        _: Request<proto::GetLoadsRequest>,
    ) -> Result<Response<proto::GetLoadsResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_model_info(
        &self,
        _: Request<proto::GetModelInfoRequest>,
    ) -> Result<Response<proto::GetModelInfoResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_server_info(
        &self,
        _: Request<proto::GetServerInfoRequest>,
    ) -> Result<Response<proto::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn flush_cache(
        &self,
        _: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn get_tokenizer(
        &self,
        _: Request<common::GetTokenizerRequest>,
    ) -> Result<Response<Self::GetTokenizerStream>, Status> {
        Err(Status::unimplemented("unused"))
    }
    async fn subscribe_kv_events(
        &self,
        _: Request<common::SubscribeKvEventsRequest>,
    ) -> Result<Response<Self::SubscribeKvEventsStream>, Status> {
        Err(Status::unimplemented("unused"))
    }
}
async fn worker(unsupported: bool) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    let engine = Engine {
        hits: hits.clone(),
        unsupported,
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "test owns and aborts server task"
    )]
    let task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(proto::vllm_engine_server::VllmEngineServer::new(engine))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    (url, hits, task)
}

#[tokio::test]
async fn mixed_scrape_deduplicates_dp_and_isolates_old_workers() {
    let (url, hits, task) = worker(false).await;
    let (old, old_hits, old_task) = worker(true).await;
    let registry = Arc::new(WorkerRegistry::new());
    for rank in 0..2 {
        let spec = serde_json::from_value(serde_json::json!({
            "url": format!("{url}@{rank}"), "dp_base_url": url, "dp_rank": rank, "dp_size": 2,
        }))
        .unwrap();
        registry
            .register(Arc::new(
                BasicWorkerBuilder::from_spec(spec)
                    .connection_mode(ConnectionMode::Grpc)
                    .runtime_type(RuntimeType::Vllm)
                    .build(),
            ))
            .unwrap();
    }
    let old_worker = Arc::new(
        BasicWorkerBuilder::new(&old)
            .connection_mode(ConnectionMode::Grpc)
            .runtime_type(RuntimeType::Vllm)
            .build(),
    );
    registry.register(old_worker.clone()).unwrap();
    use smg::worker::Worker;
    let old_status = old_worker.status();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_url = format!("http://{}", listener.local_addr().unwrap());
    registry
        .register(Arc::new(BasicWorkerBuilder::new(&http_url).build()))
        .unwrap();
    let app = Router::new()
        .route(
            "/metrics",
            get(|| async {
                "# HELP http_tokens HTTP tokens\n# TYPE http_tokens gauge\nhttp_tokens 9\n"
            }),
        )
        .route(
            "/engine_metrics",
            get(move || {
                let registry = registry.clone();
                async move {
                    WorkerManager::get_engine_metrics(&registry)
                        .await
                        .into_response()
                }
            }),
        );
    #[expect(
        clippy::disallowed_methods,
        reason = "test owns and aborts server task"
    )]
    let http_task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    for _ in 0..2 {
        let response = smg::http_client::client()
            .get(format!("{http_url}/engine_metrics"))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        assert!(response.headers()["content-type"]
            .to_str()
            .unwrap()
            .contains("version=0.0.4"));
        let body = response.text().await.unwrap();
        let sample = body
            .lines()
            .find(|s| s.starts_with("vllm:tokens_total{"))
            .unwrap();
        assert!(
            sample.contains(&format!("worker_addr=\"{url}\"")) && sample.ends_with(" 17"),
            "{sample}"
        );
        assert!(body
            .lines()
            .any(|s| s.starts_with("http_tokens{") && s.ends_with(" 9")));
        assert!(body
            .lines()
            .any(|s| s.starts_with("smg_engine_metrics_scrape_success{")
                && s.contains(&old)
                && s.ends_with(" 0")));
    }
    assert_eq!(hits.load(Ordering::SeqCst), 2);
    assert_eq!(old_hits.load(Ordering::SeqCst), 2);
    assert_eq!(old_worker.status(), old_status);
    task.abort();
    old_task.abort();
    http_task.abort();
}

#[tokio::test]
async fn all_unsupported_preserves_scrape_failure() {
    let (url, _, task) = worker(true).await;
    let registry = WorkerRegistry::new();
    registry
        .register(Arc::new(
            BasicWorkerBuilder::new(&url)
                .connection_mode(ConnectionMode::Grpc)
                .runtime_type(RuntimeType::Vllm)
                .build(),
        ))
        .unwrap();
    assert!(matches!(
        WorkerManager::get_engine_metrics(&registry).await,
        EngineMetricsResult::Err(_)
    ));
    task.abort();
}
