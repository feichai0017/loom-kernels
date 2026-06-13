//! P/D proxy / router — the running service that joins the pieces: it puts the
//! QuillCache store on the request hot path by orchestrating a true mid-request
//! disaggregated prefill→decode flow across two engines that share one store.
//!
//! This is the **router** in vLLM's disaggregation handshake. For each
//! `POST /v1/chat/completions` it mints a unique `transfer_id` and threads it
//! through vLLM's request-level `kv_transfer_params`:
//!   1. send the prompt to the **prefill** engine (the `kv_producer`) tagged
//!      `do_remote_decode` + `transfer_id`. Its QuillCache connector offloads the
//!      request's KV to the shared store under `qc-pd/{transfer_id}` and does no
//!      real decoding (`max_tokens=1` as a belt-and-suspenders cap);
//!   2. send the original request to the **decode** engine (the `kv_consumer`)
//!      tagged `do_remote_prefill` + the same `transfer_id`. Its connector pulls
//!      the KV named by that id, skips prefill, and generates; that response is
//!      returned to the caller.
//!
//! The `transfer_id` — not the prompt content — names the KV, so this works for
//! unique/uncacheable prompts (true P/D), and degrades gracefully: against plain
//! `kv_both` engines the `kv_transfer_params` are simply ignored and the
//! connectors fall back to content-addressed prefix reuse.
//!
//! One-shot `qc-pd/{transfer_id}` entries are reclaimed by the store's eviction
//! (they're unpinned); explicit post-decode removal is a follow-up.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::TcpListener;

#[derive(Clone)]
struct ProxyState {
    client: reqwest::Client,
    prefill_url: String,
    decode_url: String,
    // Per-process base (startup nanos) + monotonic counter → unique transfer_ids
    // that don't collide across requests or across proxy restarts.
    id_base: u64,
    next_id: Arc<AtomicU64>,
}

impl ProxyState {
    /// A unique handshake id naming this request's KV in the shared store.
    fn mint_transfer_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("pd-{:x}-{}", self.id_base, n)
    }
}

/// Set `kv_transfer_params` on a JSON request body (creating it if absent).
fn set_kv_transfer_params(body: &mut Value, params: Value) {
    if let Some(obj) = body.as_object_mut() {
        obj.insert("kv_transfer_params".into(), params);
    }
}

/// Orchestrate prefill (producer offloads KV) → decode (consumer pulls + generates).
async fn chat(
    State(st): State<Arc<ProxyState>>,
    Json(body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let transfer_id = st.mint_transfer_id();

    // 1) Prefill (kv_producer): tag the request `do_remote_decode` so the engine
    //    only prefills and its connector offloads the KV under this transfer_id.
    //    max_tokens=1 caps any decode the engine would otherwise do.
    let mut prefill = body.clone();
    if let Some(obj) = prefill.as_object_mut() {
        obj.insert("max_tokens".into(), Value::from(1));
        obj.insert("stream".into(), Value::from(false));
    }
    set_kv_transfer_params(
        &mut prefill,
        serde_json::json!({ "do_remote_decode": true, "transfer_id": transfer_id }),
    );
    st.client
        .post(format!("{}/v1/chat/completions", st.prefill_url))
        .json(&prefill)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("prefill engine: {e}")))?
        .error_for_status()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("prefill engine: {e}")))?;

    // 2) Decode (kv_consumer): tag the original request `do_remote_prefill` with
    //    the SAME transfer_id so the engine pulls the producer's KV and skips
    //    prefill, then generates. Awaiting (1) first guarantees the KV is committed.
    let mut decode = body.clone();
    set_kv_transfer_params(
        &mut decode,
        serde_json::json!({ "do_remote_prefill": true, "transfer_id": transfer_id }),
    );
    let resp = st
        .client
        .post(format!("{}/v1/chat/completions", st.decode_url))
        .json(&decode)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("decode engine: {e}")))?;

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::OK);
    let text = resp
        .text()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    Ok((
        status,
        [
            (header::CONTENT_TYPE, "application/json"),
            (
                header::HeaderName::from_static("x-quillcache-pd"),
                "prefill→store→decode",
            ),
        ],
        text,
    ))
}

async fn state(State(st): State<Arc<ProxyState>>) -> Json<Value> {
    Json(serde_json::json!({
        "mode": "pd-proxy",
        "flow": "prefill(do_remote_decode) → store[qc-pd/{transfer_id}] → decode(do_remote_prefill)",
        "prefill": st.prefill_url,
        "decode": st.decode_url,
    }))
}

fn router(st: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/state", get(state))
        .with_state(st)
}

pub async fn run_pd_proxy(
    bind: String,
    prefill_url: String,
    decode_url: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let id_base = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let st = Arc::new(ProxyState {
        client: reqwest::Client::new(),
        prefill_url: prefill_url.trim_end_matches('/').to_string(),
        decode_url: decode_url.trim_end_matches('/').to_string(),
        id_base,
        next_id: Arc::new(AtomicU64::new(0)),
    });
    let socket: SocketAddr = bind.parse()?;
    let listener = TcpListener::bind(socket).await?;
    println!("QuillCache P/D proxy on http://{socket}  (prefill → store → decode)");
    println!("  prefill: {}   decode: {}", st.prefill_url, st.decode_url);
    axum::serve(listener, router(st)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::State as AxState;
    use std::sync::Mutex;

    type Seen = Arc<Mutex<Vec<Value>>>;

    // A mock engine that records each request body it received and echoes its name.
    async fn mock_engine(
        AxState((name, seen)): AxState<(String, Seen)>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        seen.lock().unwrap().push(body.clone());
        Json(serde_json::json!({"engine": name}))
    }

    async fn spawn_mock(name: &str) -> (String, Seen) {
        let seen: Seen = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route("/v1/chat/completions", post(mock_engine))
            .with_state((name.to_string(), seen.clone()));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), seen)
    }

    #[tokio::test]
    async fn proxy_warms_prefill_then_returns_decode() {
        let (prefill_url, prefill_seen) = spawn_mock("prefill").await;
        let (decode_url, decode_seen) = spawn_mock("decode").await;

        let st = Arc::new(ProxyState {
            client: reqwest::Client::new(),
            prefill_url: prefill_url.clone(),
            decode_url: decode_url.clone(),
            id_base: 0xabc,
            next_id: Arc::new(AtomicU64::new(0)),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router(st)).await.unwrap() });

        let http = reqwest::Client::new();
        let out: Value = http
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({"model":"m","messages":[],"max_tokens":16}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();

        // The caller gets the DECODE engine's response.
        assert_eq!(out["engine"], "decode");

        let prefill = prefill_seen.lock().unwrap();
        let decode = decode_seen.lock().unwrap();
        assert_eq!(prefill.len(), 1);
        assert_eq!(decode.len(), 1);
        let p = &prefill[0];
        let d = &decode[0];

        // Prefill (producer): do_remote_decode + max_tokens capped to 1.
        assert_eq!(
            p["kv_transfer_params"]["do_remote_decode"],
            Value::from(true)
        );
        assert_eq!(p["max_tokens"], Value::from(1));
        // Decode (consumer): do_remote_prefill + the caller's original max_tokens.
        assert_eq!(
            d["kv_transfer_params"]["do_remote_prefill"],
            Value::from(true)
        );
        assert_eq!(d["max_tokens"], Value::from(16));
        // Both carry the SAME router-minted transfer_id (the handshake correlation).
        let pid = p["kv_transfer_params"]["transfer_id"].as_str().unwrap();
        let did = d["kv_transfer_params"]["transfer_id"].as_str().unwrap();
        assert_eq!(pid, did);
        assert!(
            pid.starts_with("pd-"),
            "transfer_id should be router-minted: {pid}"
        );
    }
}
