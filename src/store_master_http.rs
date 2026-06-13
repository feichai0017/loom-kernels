//! HTTP front for the store's [`MasterService`] (Mooncake's `MasterService` is a
//! coro_rpc network service; here it is axum/HTTP). Out-of-process clients — a
//! real engine's KV connector (`bridge/`) — drive the **two-phase Put**, the
//! identity-guarded **Get**, **Mount**, and **Remove** over the network. No
//! object bytes flow through here: the master returns `(segment, offset)`
//! locations and the client moves the bytes via the transfer engine.
//!
//! Endpoints:
//! - `POST /v1/mount`            `{name, capacity}`
//! - `POST /v1/put_start`        `{key, identity, size, replica_num?}` → `{buffers}`
//! - `POST /v1/put_end`          `{key}`
//! - `POST /v1/put_revoke`       `{key}`
//! - `POST /v1/get_replica_list` `{key, identity}` → `{replicas}` (identity-guarded)
//! - `POST /v1/remove`           `{key, force?}`
//! - `GET  /v1/state`

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use quillcache_core::IdentityScope;
use quillcache_store::{AllocatedBuffer, ErrorCode, MasterService, Replica, ReplicateConfig};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpListener;

type Shared = Arc<Mutex<MasterService>>;

/// Map a store error to an HTTP status + message.
fn http_err(error: ErrorCode) -> (StatusCode, String) {
    let code = match error {
        ErrorCode::ObjectNotFound | ErrorCode::SegmentNotFound => StatusCode::NOT_FOUND,
        ErrorCode::ObjectAlreadyExists | ErrorCode::ObjectNotReady => StatusCode::CONFLICT,
        // The identity guard: a cross-identity Get is a forbidden reuse.
        ErrorCode::UnsafeReuse(_) => StatusCode::FORBIDDEN,
        _ => StatusCode::BAD_REQUEST,
    };
    (code, error.to_string())
}

#[derive(Deserialize)]
struct MountReq {
    name: String,
    capacity: u64,
}

#[derive(Deserialize)]
struct PutStartReq {
    key: String,
    identity: IdentityScope,
    size: u64,
    #[serde(default)]
    replica_num: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct PutStartResp {
    buffers: Vec<AllocatedBuffer>,
}

#[derive(Deserialize)]
struct KeyReq {
    key: String,
}

#[derive(Deserialize)]
struct RemoveReq {
    key: String,
    #[serde(default)]
    force: bool,
}

#[derive(Deserialize)]
struct GetReq {
    key: String,
    identity: IdentityScope,
}

#[derive(Serialize, Deserialize)]
struct GetResp {
    replicas: Vec<Replica>,
}

#[derive(Deserialize)]
struct SegmentReq {
    segment: String,
}

#[derive(Serialize)]
struct StateResp {
    objects: usize,
    segments: usize,
    capacity: u64,
    allocated: u64,
    /// Segments whose node has missed heartbeats past the TTL (failure detection).
    dead_segments: Vec<String>,
}

// ---- batch APIs (Mooncake's BatchPut / BatchGet) ----

#[derive(Deserialize)]
struct BatchPutItem {
    key: String,
    identity: IdentityScope,
    size: u64,
}

#[derive(Deserialize)]
struct BatchPutStartReq {
    items: Vec<BatchPutItem>,
    #[serde(default)]
    replica_num: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct BatchPutStartResp {
    buffers: Vec<Vec<AllocatedBuffer>>,
}

#[derive(Deserialize)]
struct BatchKeysReq {
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct BatchGetReq {
    keys: Vec<String>,
    identity: IdentityScope,
}

#[derive(Serialize, Deserialize)]
struct BatchGetResp {
    replicas: Vec<Vec<Replica>>,
}

async fn mount(State(master): State<Shared>, Json(req): Json<MountReq>) -> Json<bool> {
    master.lock().unwrap().mount_segment(req.name, req.capacity);
    Json(true)
}

async fn put_start(
    State(master): State<Shared>,
    Json(req): Json<PutStartReq>,
) -> Result<Json<PutStartResp>, (StatusCode, String)> {
    let config = req
        .replica_num
        .map(ReplicateConfig::replicas)
        .unwrap_or_default();
    let buffers = master
        .lock()
        .unwrap()
        .put_start(req.key, req.identity, req.size, &config)
        .map_err(http_err)?;
    Ok(Json(PutStartResp { buffers }))
}

async fn put_end(
    State(master): State<Shared>,
    Json(req): Json<KeyReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    master.lock().unwrap().put_end(&req.key).map_err(http_err)?;
    Ok(Json(true))
}

async fn put_revoke(
    State(master): State<Shared>,
    Json(req): Json<KeyReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    master
        .lock()
        .unwrap()
        .put_revoke(&req.key)
        .map_err(http_err)?;
    Ok(Json(true))
}

async fn get_replica_list(
    State(master): State<Shared>,
    Json(req): Json<GetReq>,
) -> Result<Json<GetResp>, (StatusCode, String)> {
    let replicas = master
        .lock()
        .unwrap()
        .get_replica_list(&req.key, &req.identity)
        .map_err(http_err)?;
    Ok(Json(GetResp { replicas }))
}

async fn remove(
    State(master): State<Shared>,
    Json(req): Json<RemoveReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    master
        .lock()
        .unwrap()
        .remove(&req.key, req.force)
        .map_err(http_err)?;
    Ok(Json(true))
}

/// Record a liveness heartbeat from a segment's node (Mooncake's client heartbeats).
async fn heartbeat(
    State(master): State<Shared>,
    Json(req): Json<SegmentReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    master
        .lock()
        .unwrap()
        .heartbeat(&req.segment)
        .map_err(http_err)?;
    Ok(Json(true))
}

async fn batch_put_start(
    State(master): State<Shared>,
    Json(req): Json<BatchPutStartReq>,
) -> Result<Json<BatchPutStartResp>, (StatusCode, String)> {
    let config = req
        .replica_num
        .map(ReplicateConfig::replicas)
        .unwrap_or_default();
    let items = req
        .items
        .into_iter()
        .map(|i| (i.key, i.identity, i.size))
        .collect();
    let buffers = master
        .lock()
        .unwrap()
        .batch_put_start(items, &config)
        .map_err(http_err)?;
    Ok(Json(BatchPutStartResp { buffers }))
}

async fn batch_put_end(
    State(master): State<Shared>,
    Json(req): Json<BatchKeysReq>,
) -> Result<Json<bool>, (StatusCode, String)> {
    master
        .lock()
        .unwrap()
        .batch_put_end(&req.keys)
        .map_err(http_err)?;
    Ok(Json(true))
}

async fn batch_get_replica_list(
    State(master): State<Shared>,
    Json(req): Json<BatchGetReq>,
) -> Result<Json<BatchGetResp>, (StatusCode, String)> {
    let replicas = master
        .lock()
        .unwrap()
        .batch_get_replica_list(&req.keys, &req.identity)
        .map_err(http_err)?;
    Ok(Json(BatchGetResp { replicas }))
}

async fn state(State(master): State<Shared>) -> Json<StateResp> {
    let master = master.lock().unwrap();
    Json(StateResp {
        objects: master.object_count(),
        segments: master.segment_count(),
        capacity: master.capacity(),
        allocated: master.allocated(),
        dead_segments: master.dead_segments(),
    })
}

fn router(shared: Shared) -> Router {
    Router::new()
        .route("/v1/mount", post(mount))
        .route("/v1/put_start", post(put_start))
        .route("/v1/put_end", post(put_end))
        .route("/v1/put_revoke", post(put_revoke))
        .route("/v1/get_replica_list", post(get_replica_list))
        .route("/v1/batch_put_start", post(batch_put_start))
        .route("/v1/batch_put_end", post(batch_put_end))
        .route("/v1/batch_get_replica_list", post(batch_get_replica_list))
        .route("/v1/remove", post(remove))
        .route("/v1/heartbeat", post(heartbeat))
        .route("/v1/state", get(state))
        .with_state(shared)
}

/// Options for the HA-capable store master.
pub struct StoreMasterOpts {
    pub addr: String,
    pub strategy: String,
    /// Snapshot file: recovered on startup if present, saved periodically.
    pub snapshot: Option<String>,
    /// Seconds between periodic snapshots (0 = only on demand / never).
    pub snapshot_interval_secs: u64,
    /// Seconds a segment may miss heartbeats before it's dead (0 = health off).
    pub segment_ttl: u64,
    /// Comma-separated etcd endpoints for leader election (HA mode). Requires the
    /// `etcd` build feature; without it a given value is ignored with a warning.
    pub etcd: Option<String>,
    /// This master's id (the leader value in the election).
    pub node_id: String,
}

pub async fn run_store_master(opts: StoreMasterOpts) -> Result<(), Box<dyn std::error::Error>> {
    // 1) HA: if etcd endpoints are configured, campaign for leadership and only
    //    the winner serves (standbys block here until the leader's lease lapses).
    #[cfg(feature = "etcd")]
    let _leadership = match &opts.etcd {
        Some(endpoints) => {
            let eps: Vec<String> = endpoints
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            println!(
                "store-master '{}' joining leader election on etcd {eps:?} …",
                opts.node_id
            );
            let mut election = quillcache_store::MasterElection::join(
                eps,
                "quillcache/store-master/leader",
                opts.node_id.clone(),
                10,
            )
            .await?;
            let leadership = election.campaign().await?; // blocks until we are leader
            println!(
                "store-master '{}' is now the LEADER — serving",
                opts.node_id
            );
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(Duration::from_secs(3)).await;
                    if election.keep_alive().await.is_err() {
                        break;
                    }
                }
            });
            Some(leadership)
        }
        None => None,
    };
    #[cfg(not(feature = "etcd"))]
    if opts.etcd.is_some() {
        eprintln!("--etcd given but built without `--features etcd`; running as a single master");
    }

    // 2) Recover from the snapshot if present, else start fresh.
    let mut master = match &opts.snapshot {
        Some(path) if std::path::Path::new(path).exists() => {
            println!("recovering master state from snapshot {path}");
            MasterService::load_snapshot(path)?
        }
        _ => MasterService::new(&opts.strategy),
    };
    if opts.segment_ttl > 0 {
        master.set_segment_ttl(opts.segment_ttl);
    }
    let shared: Shared = Arc::new(Mutex::new(master));

    // 3) Drive the logical clock on wall time so leases + segment health advance.
    {
        let s = shared.clone();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tick.tick().await;
                s.lock().unwrap().tick();
            }
        });
    }

    // 4) Periodic crash-safe snapshot of the in-memory metadata.
    if let (Some(path), true) = (opts.snapshot.clone(), opts.snapshot_interval_secs > 0) {
        let s = shared.clone();
        let interval = opts.snapshot_interval_secs;
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(interval));
            loop {
                tick.tick().await;
                if let Err(e) = s.lock().unwrap().save_snapshot(&path) {
                    eprintln!("snapshot save failed: {e}");
                }
            }
        });
    }

    let socket: SocketAddr = opts.addr.parse()?;
    let listener = TcpListener::bind(socket).await?;
    println!(
        "QuillCache store MasterService '{}' on http://{socket} (strategy: {}, snapshot: {:?}, segment_ttl: {}s)",
        opts.node_id, opts.strategy, opts.snapshot, opts.segment_ttl
    );
    println!("  POST /v1/{{mount,put_start,put_end,put_revoke,get_replica_list,remove,heartbeat}} · GET /v1/state");
    axum::serve(listener, router(shared)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn two_phase_put_then_identity_guarded_get_over_http() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let shared: Shared = Arc::new(Mutex::new(MasterService::new("random")));
        tokio::spawn(async move { axum::serve(listener, router(shared)).await.unwrap() });
        let base = format!("http://{addr}");
        let http = reqwest::Client::new();
        let id_a = serde_json::json!({"model_id":"m","tokenizer_id":"t","adapter_id":null,"tenant_id":"ten-a"});

        // Mount a segment.
        http.post(format!("{base}/v1/mount"))
            .json(&serde_json::json!({"name":"seg-0","capacity":65536}))
            .send()
            .await
            .unwrap();

        // Two-phase Put: put_start allocates a replica, put_end commits it.
        let started = http
            .post(format!("{base}/v1/put_start"))
            .json(&serde_json::json!({"key":"k","identity":id_a,"size":16,"replica_num":1}))
            .send()
            .await
            .unwrap();
        assert!(started.status().is_success());
        let body: PutStartResp = started.json().await.unwrap();
        assert_eq!(body.buffers.len(), 1);
        http.post(format!("{base}/v1/put_end"))
            .json(&serde_json::json!({"key":"k"}))
            .send()
            .await
            .unwrap();

        // Get with the writer's identity → 200.
        let get = http
            .post(format!("{base}/v1/get_replica_list"))
            .json(&serde_json::json!({"key":"k","identity":id_a}))
            .send()
            .await
            .unwrap();
        assert!(get.status().is_success());
        let got: GetResp = get.json().await.unwrap();
        assert_eq!(got.replicas.len(), 1);

        // Get with a different tenant → 403 (the identity guard, over HTTP).
        let id_b = serde_json::json!({"model_id":"m","tokenizer_id":"t","adapter_id":null,"tenant_id":"ten-b"});
        let refused = http
            .post(format!("{base}/v1/get_replica_list"))
            .json(&serde_json::json!({"key":"k","identity":id_b}))
            .send()
            .await
            .unwrap();
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn batch_put_then_batch_get_over_http() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let shared: Shared = Arc::new(Mutex::new(MasterService::new("random")));
        tokio::spawn(async move { axum::serve(listener, router(shared)).await.unwrap() });
        let base = format!("http://{addr}");
        let http = reqwest::Client::new();
        let id = serde_json::json!({"model_id":"m","tokenizer_id":"t","adapter_id":null,"tenant_id":"ten-a"});

        http.post(format!("{base}/v1/mount"))
            .json(&serde_json::json!({"name":"seg-0","capacity":65536}))
            .send()
            .await
            .unwrap();

        // Batch-allocate three keys in one call.
        let started = http
            .post(format!("{base}/v1/batch_put_start"))
            .json(&serde_json::json!({
                "items": [
                    {"key":"a","identity":id,"size":64},
                    {"key":"b","identity":id,"size":64},
                    {"key":"c","identity":id,"size":64},
                ],
                "replica_num": 1
            }))
            .send()
            .await
            .unwrap();
        assert!(started.status().is_success());
        let body: BatchPutStartResp = started.json().await.unwrap();
        assert_eq!(body.buffers.len(), 3);

        http.post(format!("{base}/v1/batch_put_end"))
            .json(&serde_json::json!({"keys":["a","b","c"]}))
            .send()
            .await
            .unwrap();

        let got = http
            .post(format!("{base}/v1/batch_get_replica_list"))
            .json(&serde_json::json!({"keys":["a","b","c"],"identity":id}))
            .send()
            .await
            .unwrap();
        assert!(got.status().is_success());
        let resp: BatchGetResp = got.json().await.unwrap();
        assert_eq!(resp.replicas.len(), 3);
        assert!(resp.replicas.iter().all(|r| r.len() == 1));
    }
}
