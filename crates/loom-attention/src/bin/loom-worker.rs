use axum::{routing::get, Json, Router};
use clap::Parser;
use loom_attention::types::{
    AttentionKind, ComputeCapabilities, DType, DeviceKind, MemoryDomain, WorkerId,
};
use serde::Serialize;
use std::net::SocketAddr;

#[derive(Debug, Parser)]
#[command(name = "loom-worker")]
#[command(about = "Node-local worker for distributed partial attention")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8090")]
    bind: SocketAddr,
    #[arg(long, default_value = "attention-0")]
    worker_id: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    data_path: &'static str,
    ready_for_attention: bool,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerState {
    executor_status: &'static str,
    capabilities: ComputeCapabilities,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "loom_attention=info".to_owned()),
        )
        .init();
    let args = Args::parse();
    let device_kind = DeviceKind::Cpu;
    let capabilities = ComputeCapabilities {
        worker_id: WorkerId(args.worker_id),
        device_kind,
        memory_domains: vec![MemoryDomain::HostDram],
        attention_kinds: vec![AttentionKind::Mha, AttentionKind::Gqa],
        dtypes: vec![DType::Fp32, DType::Fp16, DType::Bf16],
        head_sizes: vec![64, 80, 96, 128],
        page_sizes: vec![1, 16, 32],
        supports_partial_softmax: true,
        supports_graph_capture: false,
    };
    let app = Router::new().route("/healthz", get(health)).route(
        "/v1/capabilities",
        get({
            let state = WorkerState {
                executor_status: "reference_contract_only",
                capabilities: capabilities.clone(),
            };
            move || async move { Json(state) }
        }),
    );
    tracing::info!(bind = %args.bind, worker = %capabilities.worker_id, "starting attention worker control endpoint");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "loom-worker",
        data_path: "not configured; HTTP is control-only",
        ready_for_attention: false,
    })
}
