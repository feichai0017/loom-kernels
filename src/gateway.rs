use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::StreamExt;
use quillcache_core::{
    AdmissionDecision, CacheObservation, CacheTier, CoScheduler, CoSchedulerAction,
    CoSchedulerActionKind, CoSchedulerPlan, CoSchedulerSnapshot, CoSchedulerTelemetry,
    ControlPlane, IngestSummary, PlanAction, PlanActionKind, RequestPlan, ServingMode,
    SloObservation, TransferObservation,
};
use quillcache_core::{
    DataPlane, DataPlaneAction, EngineEndpoint, ExternalKvBlockKey, IndexBackend, KvBlockKey,
    KvEventBatch, MemoryIndex, NoDataPlane, RequestKvHints, RequestShape, SloTarget,
};
use quillcache_core::{
    DynamoCostRouter, GreedyStatePlaneRouter, LeastLoadedRouter, PrefixAffinityRouter,
    RoundRobinRouter, RoutingPolicy, SessionAffinityRouter, SloAwareRouter,
};
use quillcache_store::{StoreDataPlane, StoreTierConfig};
use quillcache_transfer_engine::LayerTransferTelemetry;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("failed to read config {path}: {source}")]
    ReadConfig {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse config {path}: {source}")]
    ParseConfig {
        path: String,
        source: serde_yaml::Error,
    },
    #[error("gateway config must include at least one engine")]
    NoEngines,
    #[error("failed to bind gateway on {addr}: {source}")]
    Bind {
        addr: SocketAddr,
        source: std::io::Error,
    },
    #[error("action_sink kind=http requires url")]
    ActionSinkMissingUrl,
    #[error("gateway server failed: {0}")]
    Serve(std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub bind: SocketAddr,
    pub engines: Vec<EngineEndpoint>,
    /// Routing policy: "prefix-affinity" (cache-affine across the fleet),
    /// "round-robin" (spread baseline), "least-loaded", "slo-aware" (SLO as a
    /// near-hard constraint), "session-affinity", "dynamo-cost" (mirrors NVIDIA
    /// Dynamo's KV-router cost function), or "greedy" (default).
    #[serde(default)]
    pub policy: Option<String>,
    /// Residency index backend: "memory" (default, ephemeral), "holt"
    /// (persistent ART), or "rocksdb" (persistent LSM). The persistent backends
    /// need the matching build feature; otherwise the gateway warns and uses
    /// memory. A persistent index keeps fleet residency across restarts.
    #[serde(default)]
    pub index: Option<String>,
    /// On-disk path for a persistent index backend (default
    /// `quillcache-residency`).
    #[serde(default)]
    pub index_path: Option<String>,
    /// Runtime KV tensor data-plane adapter. `none` keeps the previous inferred
    /// placement behavior; `tiered` enables an in-process HBM/DRAM/SSD control
    /// plane that performs real admission, promotion, demotion, and eviction.
    #[serde(default)]
    pub data_plane: Option<DataPlaneConfig>,
    /// Optional synchronous action sink. `http` posts planner/data-plane actions
    /// to an external adapter, for example vLLM kv_transfer, SGLang/LMCache, or
    /// a Dynamo KVBM bridge.
    #[serde(default)]
    pub action_sink: Option<ActionSinkConfig>,
    /// Route via the Mooncake Conductor (the prefix-cache table + the Dynamo cost
    /// function), fed by the same KV events + inferred placement, instead of the
    /// residency-snapshot router. Off by default.
    #[serde(default)]
    pub conductor: Option<bool>,
    /// Cluster co-scheduler actuator settings. The observation plane is always
    /// exposed in `/v1/state`; `apply=true` lets gateway-local actions mutate
    /// runtime state.
    #[serde(default)]
    pub co_scheduler: Option<CoSchedulerConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPlaneConfig {
    #[serde(default = "default_data_plane_kind")]
    pub kind: String,
    #[serde(default)]
    pub hbm_capacity_bytes: Option<u64>,
    #[serde(default)]
    pub cpu_dram_capacity_bytes: Option<u64>,
    #[serde(default)]
    pub local_ssd_capacity_bytes: Option<u64>,
}

fn default_data_plane_kind() -> String {
    "none".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSinkConfig {
    #[serde(default = "default_action_sink_kind")]
    pub kind: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_action_sink_fail_open")]
    pub fail_open: bool,
    #[serde(default = "default_action_sink_timeout_ms")]
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSchedulerConfig {
    #[serde(default = "default_co_scheduler_apply")]
    pub apply: bool,
    /// Violation budget used when an `AdmissionReject` action enables gateway
    /// admission control. `0` means reject any request predicted to miss SLO.
    #[serde(default = "default_admission_slo_violation_limit_us")]
    pub admission_slo_violation_limit_us: u64,
    #[serde(default = "default_transfer_depth_initial")]
    pub transfer_depth_initial: usize,
    #[serde(default = "default_transfer_depth_min")]
    pub transfer_depth_min: usize,
    #[serde(default = "default_transfer_depth_max")]
    pub transfer_depth_max: usize,
    #[serde(default = "default_transfer_first_layer_budget_ms")]
    pub transfer_first_layer_budget_ms: f64,
    #[serde(default = "default_transfer_min_overlap_efficiency_pct")]
    pub transfer_min_overlap_efficiency_pct: f64,
    #[serde(default = "default_transfer_high_queue_depth")]
    pub transfer_high_queue_depth: u64,
}

impl Default for CoSchedulerConfig {
    fn default() -> Self {
        Self {
            apply: default_co_scheduler_apply(),
            admission_slo_violation_limit_us: default_admission_slo_violation_limit_us(),
            transfer_depth_initial: default_transfer_depth_initial(),
            transfer_depth_min: default_transfer_depth_min(),
            transfer_depth_max: default_transfer_depth_max(),
            transfer_first_layer_budget_ms: default_transfer_first_layer_budget_ms(),
            transfer_min_overlap_efficiency_pct: default_transfer_min_overlap_efficiency_pct(),
            transfer_high_queue_depth: default_transfer_high_queue_depth(),
        }
    }
}

fn default_co_scheduler_apply() -> bool {
    true
}

fn default_admission_slo_violation_limit_us() -> u64 {
    0
}

fn default_transfer_depth_initial() -> usize {
    1
}

fn default_transfer_depth_min() -> usize {
    1
}

fn default_transfer_depth_max() -> usize {
    8
}

fn default_transfer_first_layer_budget_ms() -> f64 {
    20.0
}

fn default_transfer_min_overlap_efficiency_pct() -> f64 {
    50.0
}

fn default_transfer_high_queue_depth() -> u64 {
    16
}

fn default_action_sink_kind() -> String {
    "none".to_string()
}

fn default_action_sink_fail_open() -> bool {
    true
}

fn default_action_sink_timeout_ms() -> u64 {
    250
}

impl GatewayConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, GatewayError> {
        let path_ref = path.as_ref();
        let raw = fs::read_to_string(path_ref).map_err(|source| GatewayError::ReadConfig {
            path: path_ref.display().to_string(),
            source,
        })?;
        let config: Self =
            serde_yaml::from_str(&raw).map_err(|source| GatewayError::ParseConfig {
                path: path_ref.display().to_string(),
                source,
            })?;
        if config.engines.is_empty() {
            return Err(GatewayError::NoEngines);
        }
        Ok(config)
    }
}

/// Live SLO goodput: of the requests served, how many produced their first token
/// within the request's TTFT SLO budget. Measured from the gateway's own clock
/// (arrival → first streamed chunk), not the cost model — a real online metric.
#[derive(Debug, Default)]
struct SloGoodput {
    served: AtomicU64,
    met: AtomicU64,
    ttft_ms_sum: AtomicU64,
    ttft_budget_ms_sum: AtomicU64,
}

impl SloGoodput {
    fn record(&self, ttft_ms: u64, budget_ms: u64) {
        self.served.fetch_add(1, Ordering::Relaxed);
        self.ttft_ms_sum.fetch_add(ttft_ms, Ordering::Relaxed);
        self.ttft_budget_ms_sum
            .fetch_add(budget_ms, Ordering::Relaxed);
        if ttft_ms <= budget_ms {
            self.met.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn observation(&self) -> SloObservation {
        let served = self.served.load(Ordering::Relaxed);
        if served == 0 {
            return SloObservation::default();
        }
        let met = self.met.load(Ordering::Relaxed);
        let ttft_sum = self.ttft_ms_sum.load(Ordering::Relaxed);
        let budget_sum = self.ttft_budget_ms_sum.load(Ordering::Relaxed);
        SloObservation {
            ttft_p99_ms: None,
            ttft_mean_ms: Some(ttft_sum as f64 / served as f64),
            tpot_p99_ms: None,
            ttft_budget_ms: Some(budget_sum / served),
            tpot_budget_ms: None,
            slo_miss_pct: pct(served.saturating_sub(met), served),
            goodput_pct: pct(met, served),
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        let served = self.served.load(Ordering::Relaxed);
        let met = self.met.load(Ordering::Relaxed);
        let sum = self.ttft_ms_sum.load(Ordering::Relaxed);
        let observation = self.observation();
        json!({
            "served": served,
            "met_slo": met,
            "goodput_pct": observation.goodput_pct.unwrap_or(0.0),
            "mean_ttft_ms": if served > 0 { sum as f64 / served as f64 } else { 0.0 },
            "mean_ttft_budget_ms": observation.ttft_budget_ms.unwrap_or(0),
        })
    }
}

/// Cumulative routing counters for the Prometheus `/metrics` endpoint. Raw
/// counters (rates are computed at query time, the Prometheus convention); fed
/// from each request's [`GatewayRouteTrace`] so the cache effectiveness + the
/// identity guard are observable as fleet-wide totals, not just per-request headers.
#[derive(Debug, Default)]
struct GatewayMetrics {
    requests_total: AtomicU64,
    reusable_blocks_total: AtomicU64,
    local_hits_total: AtomicU64,
    transfer_blocks_total: AtomicU64,
    recompute_blocks_total: AtomicU64,
    reuse_refused_total: AtomicU64,
    transfer_requests_total: AtomicU64,
    transfer_estimated_us_sum: AtomicU64,
    transfer_first_estimated_us_sum: AtomicU64,
    transfer_measured_events_total: AtomicU64,
    transfer_measured_first_layer_events_total: AtomicU64,
    transfer_measured_bytes_total: AtomicU64,
    transfer_measured_layers_total: AtomicU64,
    transfer_measured_first_layer_us_sum: AtomicU64,
    transfer_measured_full_us_sum: AtomicU64,
    transfer_measured_overlap_us_sum: AtomicU64,
    transfer_measured_queue_depth: AtomicU64,
}

impl GatewayMetrics {
    fn record(&self, t: &GatewayRouteTrace) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.reusable_blocks_total
            .fetch_add(t.reusable_blocks as u64, Ordering::Relaxed);
        self.local_hits_total
            .fetch_add(t.local_hits as u64, Ordering::Relaxed);
        self.transfer_blocks_total
            .fetch_add(t.transfer_blocks as u64, Ordering::Relaxed);
        self.recompute_blocks_total
            .fetch_add(t.recompute_blocks as u64, Ordering::Relaxed);
        self.reuse_refused_total
            .fetch_add(t.reuse_refused as u64, Ordering::Relaxed);
        if t.transfer_blocks > 0 {
            self.transfer_requests_total.fetch_add(1, Ordering::Relaxed);
            self.transfer_estimated_us_sum
                .fetch_add(t.estimated_transfer_us, Ordering::Relaxed);
            self.transfer_first_estimated_us_sum
                .fetch_add(t.estimated_first_transfer_us, Ordering::Relaxed);
        }
    }

    fn observation(&self) -> CacheObservation {
        let requests = self.requests_total.load(Ordering::Relaxed);
        let reusable_blocks = self.reusable_blocks_total.load(Ordering::Relaxed);
        let local_hits = self.local_hits_total.load(Ordering::Relaxed);
        let remote_hits = self.transfer_blocks_total.load(Ordering::Relaxed);
        let recompute_blocks = self.recompute_blocks_total.load(Ordering::Relaxed);
        let reuse_refused = self.reuse_refused_total.load(Ordering::Relaxed);
        let routed_blocks = local_hits + remote_hits + recompute_blocks;
        CacheObservation {
            requests,
            reusable_blocks,
            local_hits,
            remote_hits,
            recompute_blocks,
            reuse_refused,
            local_hit_rate_pct: pct(local_hits, routed_blocks),
            remote_hit_rate_pct: pct(remote_hits, routed_blocks),
        }
    }

    fn record_transfer_telemetry(&self, event: &TransferTelemetryEvent) {
        self.transfer_measured_events_total
            .fetch_add(1, Ordering::Relaxed);
        self.transfer_measured_bytes_total
            .fetch_add(event.telemetry.bytes, Ordering::Relaxed);
        self.transfer_measured_layers_total
            .fetch_add(event.telemetry.layers as u64, Ordering::Relaxed);
        if let Some(first_us) = event.telemetry.time_to_first_layer_us {
            self.transfer_measured_first_layer_events_total
                .fetch_add(1, Ordering::Relaxed);
            self.transfer_measured_first_layer_us_sum
                .fetch_add(first_us, Ordering::Relaxed);
        }
        self.transfer_measured_full_us_sum
            .fetch_add(event.telemetry.full_transfer_us, Ordering::Relaxed);
        self.transfer_measured_overlap_us_sum
            .fetch_add(event.telemetry.overlap_window_us, Ordering::Relaxed);
        if let Some(queue_depth) = event.queue_depth {
            self.transfer_measured_queue_depth
                .store(queue_depth, Ordering::Relaxed);
        }
    }

    fn transfer_observation(&self) -> TransferObservation {
        let measured_events = self.transfer_measured_events_total.load(Ordering::Relaxed);
        if measured_events > 0 {
            let first_events = self
                .transfer_measured_first_layer_events_total
                .load(Ordering::Relaxed);
            let first_us = self
                .transfer_measured_first_layer_us_sum
                .load(Ordering::Relaxed);
            let full_us = self.transfer_measured_full_us_sum.load(Ordering::Relaxed);
            let overlap_us = self
                .transfer_measured_overlap_us_sum
                .load(Ordering::Relaxed);
            let bytes = self.transfer_measured_bytes_total.load(Ordering::Relaxed);
            return TransferObservation {
                time_to_first_layer_ms: avg_us_to_ms(first_us, first_events),
                full_transfer_ms: avg_us_to_ms(full_us, measured_events),
                overlap_saved_ms: avg_us_to_ms(overlap_us, measured_events),
                exposed_transfer_ms: avg_us_to_ms(
                    full_us.saturating_sub(overlap_us),
                    measured_events,
                ),
                overlap_efficiency_pct: overlap_efficiency_pct(overlap_us, full_us),
                queue_depth: self.transfer_measured_queue_depth.load(Ordering::Relaxed),
                bandwidth_mbps: bandwidth_mbps(bytes, full_us),
            };
        }

        let transfer_requests = self.transfer_requests_total.load(Ordering::Relaxed);
        if transfer_requests == 0 {
            return TransferObservation::default();
        }
        let full_us = self.transfer_estimated_us_sum.load(Ordering::Relaxed);
        let first_us = self.transfer_first_estimated_us_sum.load(Ordering::Relaxed);
        TransferObservation {
            time_to_first_layer_ms: Some(first_us as f64 / transfer_requests as f64 / 1_000.0),
            full_transfer_ms: Some(full_us as f64 / transfer_requests as f64 / 1_000.0),
            overlap_saved_ms: avg_us_to_ms(full_us.saturating_sub(first_us), transfer_requests),
            exposed_transfer_ms: avg_us_to_ms(first_us.min(full_us), transfer_requests),
            overlap_efficiency_pct: overlap_efficiency_pct(
                full_us.saturating_sub(first_us),
                full_us,
            ),
            queue_depth: 0,
            bandwidth_mbps: None,
        }
    }

    fn transfer_summary(&self) -> TransferTelemetrySummary {
        let measured_events = self.transfer_measured_events_total.load(Ordering::Relaxed);
        let first_events = self
            .transfer_measured_first_layer_events_total
            .load(Ordering::Relaxed);
        let bytes = self.transfer_measured_bytes_total.load(Ordering::Relaxed);
        let layers = self.transfer_measured_layers_total.load(Ordering::Relaxed);
        let first_us = self
            .transfer_measured_first_layer_us_sum
            .load(Ordering::Relaxed);
        let full_us = self.transfer_measured_full_us_sum.load(Ordering::Relaxed);
        let overlap_us = self
            .transfer_measured_overlap_us_sum
            .load(Ordering::Relaxed);
        TransferTelemetrySummary {
            measured_events,
            measured_first_layer_events: first_events,
            measured_bytes: bytes,
            measured_layers: layers,
            avg_time_to_first_layer_ms: avg_us_to_ms(first_us, first_events),
            avg_full_transfer_ms: avg_us_to_ms(full_us, measured_events),
            avg_overlap_window_ms: avg_us_to_ms(overlap_us, measured_events),
            avg_exposed_transfer_ms: avg_us_to_ms(
                full_us.saturating_sub(overlap_us),
                measured_events,
            ),
            overlap_efficiency_pct: overlap_efficiency_pct(overlap_us, full_us),
            latest_queue_depth: self.transfer_measured_queue_depth.load(Ordering::Relaxed),
            bandwidth_mbps: bandwidth_mbps(bytes, full_us),
        }
    }
}

fn pct(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(100.0 * numerator as f64 / denominator as f64)
    }
}

fn avg_us_to_ms(total_us: u64, count: u64) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total_us as f64 / count as f64 / 1_000.0)
    }
}

fn bandwidth_mbps(bytes: u64, total_us: u64) -> Option<f64> {
    if total_us == 0 {
        None
    } else {
        Some(bytes as f64 * 8.0 / total_us as f64)
    }
}

fn overlap_efficiency_pct(overlap_us: u64, full_transfer_us: u64) -> Option<f64> {
    if full_transfer_us == 0 {
        None
    } else {
        Some(100.0 * overlap_us.min(full_transfer_us) as f64 / full_transfer_us as f64)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferTelemetryEvent {
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub source_engine_id: Option<String>,
    #[serde(default)]
    pub target_engine_id: Option<String>,
    #[serde(default)]
    pub backend: Option<String>,
    #[serde(default)]
    pub queue_depth: Option<u64>,
    pub telemetry: LayerTransferTelemetry,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TransferTelemetrySummary {
    pub measured_events: u64,
    pub measured_first_layer_events: u64,
    pub measured_bytes: u64,
    pub measured_layers: u64,
    pub avg_time_to_first_layer_ms: Option<f64>,
    pub avg_full_transfer_ms: Option<f64>,
    pub avg_overlap_window_ms: Option<f64>,
    pub avg_exposed_transfer_ms: Option<f64>,
    pub overlap_efficiency_pct: Option<f64>,
    pub latest_queue_depth: u64,
    pub bandwidth_mbps: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedCoSchedulerAction {
    pub epoch: u64,
    pub kind: CoSchedulerActionKind,
    pub applied: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSchedulerActuatorSnapshot {
    pub apply: bool,
    pub last_applied_epoch: u64,
    pub admission_reject_enabled: bool,
    pub admission_slo_violation_limit_us: Option<u64>,
    pub transfer_max_inflight: usize,
    pub last_applied_actions: Vec<AppliedCoSchedulerAction>,
}

#[derive(Debug)]
struct CoSchedulerActuator {
    config: CoSchedulerConfig,
    last_applied_epoch: u64,
    admission_reject_enabled: bool,
    transfer_max_inflight: usize,
    last_applied_actions: Vec<AppliedCoSchedulerAction>,
}

impl CoSchedulerActuator {
    fn new(config: Option<CoSchedulerConfig>) -> Self {
        let config = config.unwrap_or_default();
        let transfer_max_inflight = clamp_transfer_depth(
            config.transfer_depth_initial,
            config.transfer_depth_min,
            config.transfer_depth_max,
        );
        Self {
            config,
            last_applied_epoch: 0,
            admission_reject_enabled: false,
            transfer_max_inflight,
            last_applied_actions: Vec::new(),
        }
    }

    fn plan_dry_run(&self) -> bool {
        !self.config.apply
    }

    fn apply(
        &mut self,
        plan: &CoSchedulerPlan,
        snapshot: &CoSchedulerSnapshot,
        control: &mut ControlPlane,
    ) -> Vec<AppliedCoSchedulerAction> {
        if !self.config.apply || plan.epoch <= self.last_applied_epoch {
            return Vec::new();
        }

        let mut applied = Vec::new();
        let mut saw_admission_reject = false;
        for action in &plan.actions {
            match action.kind {
                CoSchedulerActionKind::AdmissionReject => {
                    saw_admission_reject = true;
                    control.set_admission_slo_limit(self.config.admission_slo_violation_limit_us);
                    self.admission_reject_enabled = true;
                    applied.push(applied_action(
                        action,
                        true,
                        "enabled gateway admission rejection",
                    ));
                }
                CoSchedulerActionKind::ReplicateHotPrefix => {
                    let Some(source_worker_id) = action.source_worker_id.as_deref() else {
                        applied.push(applied_action(action, false, "missing source worker"));
                        continue;
                    };
                    let Some(target_worker_id) = action.target_worker_id.as_deref() else {
                        applied.push(applied_action(action, false, "missing target worker"));
                        continue;
                    };
                    let Some(prefix_hash) = action.prefix_hash.as_deref() else {
                        applied.push(applied_action(action, false, "missing prefix hash"));
                        continue;
                    };
                    let tier = action.tier.unwrap_or(CacheTier::Hbm);
                    let cache_actions = control.replicate_prefix(
                        source_worker_id,
                        target_worker_id,
                        prefix_hash,
                        tier,
                    );
                    applied.push(applied_action(
                        action,
                        true,
                        format!(
                            "replicated hot prefix to {target_worker_id} ({}) cache actions",
                            cache_actions.len()
                        ),
                    ));
                }
                CoSchedulerActionKind::TuneTransferDepth => {
                    let old = self.transfer_max_inflight;
                    self.transfer_max_inflight = self.next_transfer_depth(snapshot);
                    applied.push(applied_action(
                        action,
                        true,
                        format!(
                            "set transfer max_inflight {old}->{}",
                            self.transfer_max_inflight
                        ),
                    ));
                }
                _ => {}
            }
        }

        if !saw_admission_reject && self.admission_reject_enabled {
            control.clear_admission_slo_limit();
            self.admission_reject_enabled = false;
            applied.push(AppliedCoSchedulerAction {
                epoch: plan.epoch,
                kind: CoSchedulerActionKind::AdmissionReject,
                applied: true,
                reason: "disabled gateway admission rejection; pressure cleared".to_string(),
            });
        }

        self.last_applied_epoch = plan.epoch;
        self.last_applied_actions = applied.clone();
        applied
    }

    fn snapshot(&self, control: &ControlPlane) -> CoSchedulerActuatorSnapshot {
        CoSchedulerActuatorSnapshot {
            apply: self.config.apply,
            last_applied_epoch: self.last_applied_epoch,
            admission_reject_enabled: self.admission_reject_enabled,
            admission_slo_violation_limit_us: control.admission_slo_limit(),
            transfer_max_inflight: self.transfer_max_inflight,
            last_applied_actions: self.last_applied_actions.clone(),
        }
    }

    fn next_transfer_depth(&self, snapshot: &CoSchedulerSnapshot) -> usize {
        let min = self.config.transfer_depth_min.max(1);
        let max = self.config.transfer_depth_max.max(min);
        let current = self.transfer_max_inflight.clamp(min, max);
        let first_layer_slow = snapshot
            .transfer
            .time_to_first_layer_ms
            .map(|ms| ms > self.config.transfer_first_layer_budget_ms)
            .unwrap_or(false);
        let queue_high = snapshot.transfer.queue_depth >= self.config.transfer_high_queue_depth;
        if first_layer_slow || queue_high {
            return current.saturating_sub(1).max(min);
        }
        let overlap_low = snapshot
            .transfer
            .overlap_efficiency_pct
            .map(|pct| pct < self.config.transfer_min_overlap_efficiency_pct)
            .unwrap_or(false);
        if overlap_low {
            return current.saturating_add(1).min(max);
        }
        current.saturating_add(1).min(max)
    }
}

fn clamp_transfer_depth(value: usize, min: usize, max: usize) -> usize {
    let min = min.max(1);
    let max = max.max(min);
    value.clamp(min, max)
}

fn applied_action(
    action: &CoSchedulerAction,
    applied: bool,
    reason: impl Into<String>,
) -> AppliedCoSchedulerAction {
    AppliedCoSchedulerAction {
        epoch: action.epoch,
        kind: action.kind,
        applied,
        reason: reason.into(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoSchedulerRuntimeState {
    pub snapshot: CoSchedulerSnapshot,
    pub plan: CoSchedulerPlan,
    pub applied_actions: Vec<AppliedCoSchedulerAction>,
    pub actuator: CoSchedulerActuatorSnapshot,
}

#[derive(Debug, Clone)]
struct GatewayState {
    control: Arc<RwLock<ControlPlane>>,
    client: Client,
    action_sink: Option<ActionSink>,
    slo: Arc<SloGoodput>,
    metrics: Arc<GatewayMetrics>,
    co_scheduler: Arc<CoScheduler>,
    co_scheduler_epoch: Arc<AtomicU64>,
    co_scheduler_actuator: Arc<RwLock<CoSchedulerActuator>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayRouteTrace {
    pub request_id: String,
    pub mode: ServingMode,
    pub engine_id: String,
    pub prefill_engine_id: Option<String>,
    pub decode_engine_id: String,
    pub planner_actions: usize,
    pub reusable_blocks: usize,
    pub local_hits: usize,
    pub transfer_blocks: usize,
    pub recompute_blocks: usize,
    /// Content-matching blocks the identity guard refused (resident only under a
    /// different identity — a naive content cache would have served them).
    pub reuse_refused: usize,
    pub estimated_ttft_us: u64,
    pub estimated_tpot_us: u64,
    pub estimated_transfer_us: u64,
    pub estimated_first_transfer_us: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TransferCostSummary {
    estimated_transfer_us: u64,
    estimated_first_transfer_us: u64,
}

fn transfer_cost_summary(plan: &RequestPlan) -> TransferCostSummary {
    let mut summary = TransferCostSummary::default();
    for action in &plan.actions {
        if action.kind != PlanActionKind::Fetch {
            continue;
        }
        summary.estimated_transfer_us = summary
            .estimated_transfer_us
            .saturating_add(action.estimated_us);
        summary.estimated_first_transfer_us = if summary.estimated_first_transfer_us == 0 {
            action.estimated_us
        } else {
            summary.estimated_first_transfer_us.min(action.estimated_us)
        };
    }
    summary
}

#[derive(Debug, Clone)]
struct ActionSink {
    url: String,
    fail_open: bool,
    timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ActionSinkSnapshot {
    kind: String,
    url: Option<String>,
    fail_open: bool,
    timeout_ms: u64,
}

impl ActionSinkSnapshot {
    fn disabled() -> Self {
        Self {
            kind: "none".to_string(),
            url: None,
            fail_open: true,
            timeout_ms: 0,
        }
    }
}

impl ActionSink {
    fn snapshot(&self) -> ActionSinkSnapshot {
        ActionSinkSnapshot {
            kind: "http".to_string(),
            url: Some(self.url.clone()),
            fail_open: self.fail_open,
            timeout_ms: self.timeout.as_millis() as u64,
        }
    }

    async fn publish(
        &self,
        client: &Client,
        event: &ActionSinkEvent,
    ) -> Result<(), reqwest::Error> {
        client
            .post(&self.url)
            .timeout(self.timeout)
            .json(event)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSinkPhase {
    Planned,
    Committed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSinkPlan {
    pub mode: ServingMode,
    pub execution_worker_id: String,
    pub prefill_worker_id: Option<String>,
    pub decode_worker_id: String,
    pub actions: Vec<PlanAction>,
}

impl From<&RequestPlan> for ActionSinkPlan {
    fn from(plan: &RequestPlan) -> Self {
        Self {
            mode: plan.mode,
            execution_worker_id: plan.execution_worker_id.clone(),
            prefill_worker_id: plan.prefill_worker_id.clone(),
            decode_worker_id: plan.decode_worker_id.clone(),
            actions: plan.actions.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionSinkEvent {
    pub schema_version: u32,
    pub phase: ActionSinkPhase,
    pub openai_path: String,
    pub request: RequestShape,
    pub route: GatewayRouteTrace,
    pub plan: ActionSinkPlan,
    pub cache_actions: Vec<DataPlaneAction>,
}

impl ActionSinkEvent {
    pub fn new(
        phase: ActionSinkPhase,
        openai_path: &str,
        request: &RequestShape,
        route: &GatewayRouteTrace,
        plan: ActionSinkPlan,
        cache_actions: Vec<DataPlaneAction>,
    ) -> Self {
        Self {
            schema_version: 1,
            phase,
            openai_path: openai_path.to_string(),
            request: request.clone(),
            route: route.clone(),
            plan,
            cache_actions,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionSinkDelivery {
    Disabled,
    Sent,
    Failed,
}

impl ActionSinkDelivery {
    fn as_header(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Sent => "sent",
            Self::Failed => "failed",
        }
    }

    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Failed, _) | (_, Self::Failed) => Self::Failed,
            (Self::Sent, _) | (_, Self::Sent) => Self::Sent,
            _ => Self::Disabled,
        }
    }
}

pub async fn run_from_config_path(path: impl AsRef<Path>) -> Result<(), GatewayError> {
    let config = GatewayConfig::from_path(path)?;
    run(config).await
}

pub async fn run(config: GatewayConfig) -> Result<(), GatewayError> {
    let policy = build_policy(config.policy.as_deref());
    let policy_name = policy.name().to_string();
    let index = build_index(&config);
    let index_name = index.name().to_string();
    let data_plane = build_data_plane(config.data_plane.as_ref());
    let data_plane_name = data_plane.name().to_string();
    let action_sink = build_action_sink(config.action_sink.as_ref())?;
    let action_sink_name = action_sink
        .as_ref()
        .map(|sink| sink.snapshot().kind)
        .unwrap_or_else(|| "none".to_string());
    let use_conductor = config.conductor.unwrap_or(false);
    let control = ControlPlane::with_index_and_policy(config.engines, index, policy)
        .with_data_plane(data_plane)
        .with_conductor_routing(use_conductor);
    tracing::info!(
        policy = %policy_name,
        index = %index_name,
        data_plane = %data_plane_name,
        action_sink = %action_sink_name,
        "control plane configured"
    );
    let state = GatewayState {
        control: Arc::new(RwLock::new(control)),
        client: Client::new(),
        action_sink,
        slo: Arc::new(SloGoodput::default()),
        metrics: Arc::new(GatewayMetrics::default()),
        co_scheduler: Arc::new(CoScheduler::default()),
        co_scheduler_epoch: Arc::new(AtomicU64::new(0)),
        co_scheduler_actuator: Arc::new(RwLock::new(CoSchedulerActuator::new(
            config.co_scheduler.clone(),
        ))),
    };
    let control = state.control.clone();
    let app = router(state);
    let listener = TcpListener::bind(config.bind)
        .await
        .map_err(|source| GatewayError::Bind {
            addr: config.bind,
            source,
        })?;

    tracing::info!(addr = %config.bind, "starting QuillCache gateway");
    // Persist the residency index on shutdown so a persistent backend survives a
    // restart (in-memory no-ops).
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            control.read().await.flush();
            tracing::info!("flushed residency index on shutdown");
        })
        .await
        .map_err(GatewayError::Serve)
}

/// Resolve when the process receives Ctrl-C or (on Unix) SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Build the residency index backend from config. Persistent backends are
/// feature-gated; if the requested one is not compiled in (or fails to open),
/// the gateway warns and falls back to the in-memory reference index, so a
/// misconfigured backend degrades instead of failing to start.
fn build_index(config: &GatewayConfig) -> Box<dyn IndexBackend> {
    match config.index.as_deref().unwrap_or("memory") {
        "memory" => Box::new(MemoryIndex::new()),
        #[cfg(feature = "holt")]
        "holt" => {
            let path = config
                .index_path
                .clone()
                .unwrap_or_else(|| "quillcache-residency".to_string());
            match quillcache_core::HoltIndex::open(&path) {
                Ok(index) => {
                    tracing::info!(path = %path, "persistent ART (Holt) residency index");
                    Box::new(index)
                }
                Err(error) => {
                    tracing::error!(?error, "failed to open Holt index; using in-memory");
                    Box::new(MemoryIndex::new())
                }
            }
        }
        #[cfg(feature = "rocksdb")]
        "rocksdb" => {
            let path = config
                .index_path
                .clone()
                .unwrap_or_else(|| "quillcache-residency".to_string());
            match quillcache_core::RocksIndex::open(&path) {
                Ok(index) => {
                    tracing::info!(path = %path, "persistent LSM (RocksDB) residency index");
                    Box::new(index)
                }
                Err(error) => {
                    tracing::error!(?error, "failed to open RocksDB index; using in-memory");
                    Box::new(MemoryIndex::new())
                }
            }
        }
        other => {
            tracing::warn!(
                backend = other,
                "index backend unavailable (needs a build feature); using in-memory"
            );
            Box::new(MemoryIndex::new())
        }
    }
}

fn build_data_plane(config: Option<&DataPlaneConfig>) -> Box<dyn DataPlane> {
    let Some(config) = config else {
        return Box::new(NoDataPlane);
    };
    match config.kind.as_str() {
        "none" => Box::new(NoDataPlane),
        "tiered" | "store" => {
            let defaults = StoreTierConfig::default();
            Box::new(StoreDataPlane::new(StoreTierConfig {
                hbm_capacity_bytes: config
                    .hbm_capacity_bytes
                    .unwrap_or(defaults.hbm_capacity_bytes),
                cpu_dram_capacity_bytes: config
                    .cpu_dram_capacity_bytes
                    .unwrap_or(defaults.cpu_dram_capacity_bytes),
                local_ssd_capacity_bytes: config
                    .local_ssd_capacity_bytes
                    .unwrap_or(defaults.local_ssd_capacity_bytes),
            }))
        }
        other => {
            tracing::warn!(data_plane = other, "unknown data plane; using none");
            Box::new(NoDataPlane)
        }
    }
}

fn build_action_sink(
    config: Option<&ActionSinkConfig>,
) -> Result<Option<ActionSink>, GatewayError> {
    let Some(config) = config else {
        return Ok(None);
    };
    match config.kind.as_str() {
        "none" => Ok(None),
        "http" => {
            let url = config
                .url
                .clone()
                .ok_or(GatewayError::ActionSinkMissingUrl)?;
            Ok(Some(ActionSink {
                url,
                fail_open: config.fail_open,
                timeout: Duration::from_millis(config.timeout_ms),
            }))
        }
        other => {
            tracing::warn!(action_sink = other, "unknown action sink; disabling");
            Ok(None)
        }
    }
}

/// Build a routing policy from its config name (default: cache-aware greedy).
fn build_policy(name: Option<&str>) -> Box<dyn RoutingPolicy> {
    match name.unwrap_or("greedy") {
        "prefix-affinity" | "affinity" => Box::new(PrefixAffinityRouter::default()),
        "round-robin" | "roundrobin" => Box::new(RoundRobinRouter::default()),
        "least-loaded" | "load" => Box::new(LeastLoadedRouter::default()),
        "slo-aware" | "slo" => Box::new(SloAwareRouter::default()),
        "session-affinity" | "session" => Box::new(SessionAffinityRouter::default()),
        "dynamo-cost" | "dynamo" | "kv-router" => Box::new(DynamoCostRouter::default()),
        _ => Box::new(GreedyStatePlaneRouter::default()),
    }
}

fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/state", get(state_snapshot))
        .route("/metrics", get(metrics_endpoint))
        .route("/v1/kv-events", post(ingest_kv_events))
        .route("/v1/transfer-telemetry", post(ingest_transfer_telemetry))
        .route("/v1/chat/completions", post(proxy_chat_completions))
        .route("/v1/completions", post(proxy_completions))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

/// Prometheus-text metrics: cache effectiveness (local hits / transfers /
/// recomputes / reusable), the identity guard (refused reuse), SLO goodput, and
/// resident-block occupancy — the observability the production gap analysis flagged.
async fn metrics_endpoint(State(state): State<GatewayState>) -> impl IntoResponse {
    let m = &state.metrics;
    let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
    let resident = { state.control.read().await.residency().len() };
    let body = format!(
        concat!(
            "# HELP quillcache_requests_total Requests routed by the gateway.\n",
            "# TYPE quillcache_requests_total counter\n",
            "quillcache_requests_total {req}\n",
            "# HELP quillcache_local_hits_total KV blocks served from a worker's local HBM.\n",
            "# TYPE quillcache_local_hits_total counter\n",
            "quillcache_local_hits_total {lh}\n",
            "# HELP quillcache_transfer_blocks_total KV blocks fetched from the store / another worker.\n",
            "# TYPE quillcache_transfer_blocks_total counter\n",
            "quillcache_transfer_blocks_total {tb}\n",
            "# HELP quillcache_transfer_requests_total Requests whose plan includes remote KV fetches.\n",
            "# TYPE quillcache_transfer_requests_total counter\n",
            "quillcache_transfer_requests_total {trq}\n",
            "# HELP quillcache_transfer_estimated_us_sum Sum of planner-estimated full KV fetch time.\n",
            "# TYPE quillcache_transfer_estimated_us_sum counter\n",
            "quillcache_transfer_estimated_us_sum {tes}\n",
            "# HELP quillcache_transfer_first_estimated_us_sum Sum of planner-estimated first KV fetch time.\n",
            "# TYPE quillcache_transfer_first_estimated_us_sum counter\n",
            "quillcache_transfer_first_estimated_us_sum {tfs}\n",
            "# HELP quillcache_transfer_measured_events_total Layer-wise transfer telemetry events reported by the transfer engine.\n",
            "# TYPE quillcache_transfer_measured_events_total counter\n",
            "quillcache_transfer_measured_events_total {transfer_measured_events}\n",
            "# HELP quillcache_transfer_measured_first_layer_events_total Measured transfer events that made layer 0 consumable.\n",
            "# TYPE quillcache_transfer_measured_first_layer_events_total counter\n",
            "quillcache_transfer_measured_first_layer_events_total {tmfe}\n",
            "# HELP quillcache_transfer_measured_bytes_total Bytes covered by measured transfer telemetry.\n",
            "# TYPE quillcache_transfer_measured_bytes_total counter\n",
            "quillcache_transfer_measured_bytes_total {tmb}\n",
            "# HELP quillcache_transfer_measured_layers_total Layers covered by measured transfer telemetry.\n",
            "# TYPE quillcache_transfer_measured_layers_total counter\n",
            "quillcache_transfer_measured_layers_total {tml}\n",
            "# HELP quillcache_transfer_measured_first_layer_us_sum Sum of measured consumer-visible first-layer latency.\n",
            "# TYPE quillcache_transfer_measured_first_layer_us_sum counter\n",
            "quillcache_transfer_measured_first_layer_us_sum {tmf}\n",
            "# HELP quillcache_transfer_measured_full_us_sum Sum of measured full-transfer latency.\n",
            "# TYPE quillcache_transfer_measured_full_us_sum counter\n",
            "quillcache_transfer_measured_full_us_sum {tmfull}\n",
            "# HELP quillcache_transfer_measured_overlap_us_sum Sum of measured transfer overlap windows.\n",
            "# TYPE quillcache_transfer_measured_overlap_us_sum counter\n",
            "quillcache_transfer_measured_overlap_us_sum {tmo}\n",
            "# HELP quillcache_transfer_measured_queue_depth Latest reported transfer queue depth.\n",
            "# TYPE quillcache_transfer_measured_queue_depth gauge\n",
            "quillcache_transfer_measured_queue_depth {tmq}\n",
            "# HELP quillcache_recompute_blocks_total KV blocks recomputed on a cache miss.\n",
            "# TYPE quillcache_recompute_blocks_total counter\n",
            "quillcache_recompute_blocks_total {rb}\n",
            "# HELP quillcache_reusable_blocks_total Reusable prefix blocks the planner found.\n",
            "# TYPE quillcache_reusable_blocks_total counter\n",
            "quillcache_reusable_blocks_total {rub}\n",
            "# HELP quillcache_reuse_refused_total Content-matching blocks the identity guard refused.\n",
            "# TYPE quillcache_reuse_refused_total counter\n",
            "quillcache_reuse_refused_total {rr}\n",
            "# HELP quillcache_slo_served_total Streamed responses measured for SLO goodput.\n",
            "# TYPE quillcache_slo_served_total counter\n",
            "quillcache_slo_served_total {srv}\n",
            "# HELP quillcache_slo_met_total Responses whose first token met the TTFT budget.\n",
            "# TYPE quillcache_slo_met_total counter\n",
            "quillcache_slo_met_total {met}\n",
            "# HELP quillcache_slo_ttft_ms_sum Sum of measured TTFT in milliseconds.\n",
            "# TYPE quillcache_slo_ttft_ms_sum counter\n",
            "quillcache_slo_ttft_ms_sum {ttft}\n",
            "# HELP quillcache_resident_blocks Current resident KV blocks in the index.\n",
            "# TYPE quillcache_resident_blocks gauge\n",
            "quillcache_resident_blocks {res}\n",
        ),
        req = g(&m.requests_total),
        lh = g(&m.local_hits_total),
        tb = g(&m.transfer_blocks_total),
        trq = g(&m.transfer_requests_total),
        tes = g(&m.transfer_estimated_us_sum),
        tfs = g(&m.transfer_first_estimated_us_sum),
        transfer_measured_events = g(&m.transfer_measured_events_total),
        tmfe = g(&m.transfer_measured_first_layer_events_total),
        tmb = g(&m.transfer_measured_bytes_total),
        tml = g(&m.transfer_measured_layers_total),
        tmf = g(&m.transfer_measured_first_layer_us_sum),
        tmfull = g(&m.transfer_measured_full_us_sum),
        tmo = g(&m.transfer_measured_overlap_us_sum),
        tmq = g(&m.transfer_measured_queue_depth),
        rb = g(&m.recompute_blocks_total),
        rub = g(&m.reusable_blocks_total),
        rr = g(&m.reuse_refused_total),
        srv = g(&state.slo.served),
        met = g(&state.slo.met),
        ttft = g(&state.slo.ttft_ms_sum),
        res = resident,
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

async fn state_snapshot(State(state): State<GatewayState>) -> impl IntoResponse {
    let co_scheduler_runtime = run_co_scheduler_cycle(&state).await;
    let control = state.control.read().await;
    Json(json!({
        "engines": control.engines(),
        "workers": control.workers(),
        "index": control.residency().metrics(),
        "index_backend": control.residency().name(),
        "data_plane": control.data_plane().name(),
        "data_plane_metrics": control.data_plane().metrics(),
        "data_plane_residency": control.data_plane().snapshot(),
        "action_sink": state.action_sink.as_ref().map(ActionSink::snapshot).unwrap_or_else(ActionSinkSnapshot::disabled),
        "slo_goodput": state.slo.snapshot(),
        "transfer_telemetry": state.metrics.transfer_summary(),
        "co_scheduler": {
            "policy": state.co_scheduler.policy,
            "snapshot": co_scheduler_runtime.snapshot,
            "active_plan": co_scheduler_runtime.plan,
            "applied_actions": co_scheduler_runtime.applied_actions,
            "actuator": co_scheduler_runtime.actuator,
        },
        "resident_blocks": control.residency().len(),
        "residency": control.residency().snapshot(),
    }))
}

async fn run_co_scheduler_cycle(state: &GatewayState) -> CoSchedulerRuntimeState {
    let mut control = state.control.write().await;
    let epoch = state
        .co_scheduler_epoch
        .fetch_add(1, Ordering::Relaxed)
        .saturating_add(1);
    let hotness_threshold = state
        .co_scheduler
        .policy
        .min_hot_prefix_hits
        .min(u64::from(u32::MAX)) as u32;
    let snapshot = control.co_scheduler_snapshot(
        epoch,
        CoSchedulerTelemetry {
            slo: state.slo.observation(),
            cache: state.metrics.observation(),
            transfer: state.metrics.transfer_observation(),
        },
        hotness_threshold,
    );
    let dry_run = state.co_scheduler_actuator.read().await.plan_dry_run();
    let plan = state.co_scheduler.plan(&snapshot, dry_run);
    let applied_actions = {
        let mut actuator = state.co_scheduler_actuator.write().await;
        actuator.apply(&plan, &snapshot, &mut control)
    };
    let actuator = state.co_scheduler_actuator.read().await.snapshot(&control);
    CoSchedulerRuntimeState {
        snapshot,
        plan,
        applied_actions,
        actuator,
    }
}

async fn ingest_kv_events(
    State(state): State<GatewayState>,
    Json(batch): Json<KvEventBatch>,
) -> Result<Json<IngestSummary>, GatewayHttpError> {
    let mut control = state.control.write().await;
    let summary = control.ingest(batch)?;
    Ok(Json(summary))
}

async fn ingest_transfer_telemetry(
    State(state): State<GatewayState>,
    Json(event): Json<TransferTelemetryEvent>,
) -> Json<TransferTelemetrySummary> {
    state.metrics.record_transfer_telemetry(&event);
    Json(state.metrics.transfer_summary())
}

async fn proxy_chat_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayHttpError> {
    proxy_openai_path(state, headers, body, "/v1/chat/completions").await
}

async fn proxy_completions(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayHttpError> {
    proxy_openai_path(state, headers, body, "/v1/completions").await
}

async fn proxy_openai_path(
    state: GatewayState,
    headers: HeaderMap,
    body: Bytes,
    path: &str,
) -> Result<Response, GatewayHttpError> {
    // Arrival clock for the SLO-goodput measurement (arrival → first token).
    let request_start = Instant::now();
    let mut payload: Value = serde_json::from_slice(&body)?;
    let request_shape = request_shape_from_payload(&mut payload);
    let ttft_budget_ms = request_shape.slo.ttft_ms;
    let co_scheduler_runtime = run_co_scheduler_cycle(&state).await;
    let transfer_max_inflight = co_scheduler_runtime.actuator.transfer_max_inflight;
    inject_transfer_max_inflight(&mut payload, transfer_max_inflight);
    let clean_body = serde_json::to_vec(&payload)?;

    let (engine, trace, action_plan) = {
        let control = state.control.read().await;
        let plan = match control.admit(&request_shape)? {
            AdmissionDecision::Admit(plan) => *plan,
            AdmissionDecision::Reject {
                reason,
                best_slo_violation_us,
            } => {
                return Err(GatewayHttpError::AdmissionRejected {
                    reason,
                    best_slo_violation_us,
                });
            }
        };
        let decision = &plan.route;
        let transfer_costs = transfer_cost_summary(&plan);
        // Identity guard: how many content-matching blocks we refused to reuse
        // because they belong to another identity (the safety property, made
        // observable on the live path).
        let audit = control.audit_reuse(&request_shape);
        let engine = control
            .engine(&plan.execution_worker_id)
            .cloned()
            .ok_or_else(|| GatewayHttpError::MissingEngine(plan.execution_worker_id.clone()))?;
        let trace = GatewayRouteTrace {
            request_id: decision.request_id.clone(),
            mode: plan.mode,
            engine_id: plan.execution_worker_id.clone(),
            prefill_engine_id: plan.prefill_worker_id.clone(),
            decode_engine_id: plan.decode_worker_id.clone(),
            planner_actions: plan.actions.len(),
            reusable_blocks: decision.reusable_blocks(),
            local_hits: decision.local_hits.len(),
            transfer_blocks: decision.transfers.len(),
            recompute_blocks: decision.recomputes.len(),
            reuse_refused: audit.refused_unsafe,
            estimated_ttft_us: decision.estimated_ttft_us,
            estimated_tpot_us: decision.estimated_tpot_us,
            estimated_transfer_us: transfer_costs.estimated_transfer_us,
            estimated_first_transfer_us: transfer_costs.estimated_first_transfer_us,
        };
        let action_plan = ActionSinkPlan::from(&plan);
        (engine, trace, action_plan)
    };

    state.metrics.record(&trace);

    if trace.reuse_refused > 0 {
        tracing::warn!(
            request_id = %trace.request_id,
            reuse_refused = trace.reuse_refused,
            "identity guard refused unsafe cross-identity reuse"
        );
    }

    let planned_delivery = dispatch_action_sink_event(
        &state,
        ActionSinkEvent::new(
            ActionSinkPhase::Planned,
            path,
            &request_shape,
            &trace,
            action_plan.clone(),
            Vec::new(),
        ),
    )
    .await?;

    let target_url = format!("{}{}", engine.base_url.trim_end_matches('/'), path);
    tracing::info!(
        engine_id = %trace.engine_id,
        request_id = %trace.request_id,
        reusable_blocks = trace.reusable_blocks,
        recompute_blocks = trace.recompute_blocks,
        "proxying request"
    );

    let mut request = state.client.post(target_url).body(clean_body);
    for (name, value) in headers.iter() {
        if should_forward_header(name) {
            request = request.header(name, value);
        }
    }
    request = request.header("x-quillcache-engine-id", trace.engine_id.as_str());
    request = request.header("x-quillcache-request-id", trace.request_id.as_str());
    request = request.header("x-quillcache-mode", serving_mode_header(trace.mode));
    request = request.header(
        "x-quillcache-decode-engine-id",
        trace.decode_engine_id.as_str(),
    );
    if let Some(prefill_engine_id) = trace.prefill_engine_id.as_deref() {
        request = request.header("x-quillcache-prefill-engine-id", prefill_engine_id);
    }
    request = request.header(
        "x-quillcache-reusable-blocks",
        trace.reusable_blocks.to_string(),
    );
    request = request.header(
        "x-quillcache-transfer-max-inflight",
        transfer_max_inflight.to_string(),
    );

    // Count this request as in flight on the chosen engine for the prefill
    // window (dispatch → response headers), so concurrent requests see the load
    // and a cache-aware policy spreads instead of dog-piling the one cache-hot
    // engine. This is the gateway's own load signal feeding back into routing.
    state.control.write().await.begin_request(&trace.engine_id);
    let send_result = request.send().await;
    state.control.write().await.end_request(&trace.engine_id);
    let upstream = send_result?;
    let status = StatusCode::from_u16(upstream.status().as_u16())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

    // Close the residency loop: record where we placed this request's prefix
    // blocks, so the next request for the same prefix sees them resident on this
    // engine — cache-aware routing now works end-to-end without a KV-events
    // bridge. (Tier 2 events later correct this inference on eviction.)
    let mut cache_action_list = Vec::new();
    if status.is_success() {
        let mut control = state.control.write().await;
        cache_action_list =
            control.observe_placement(&trace.engine_id, &request_shape, DEFAULT_BLOCK_BYTES);
    }
    let committed_delivery = if status.is_success() {
        dispatch_action_sink_event(
            &state,
            ActionSinkEvent::new(
                ActionSinkPhase::Committed,
                path,
                &request_shape,
                &trace,
                action_plan,
                cache_action_list.clone(),
            ),
        )
        .await?
    } else {
        ActionSinkDelivery::Disabled
    };
    let action_sink_delivery = planned_delivery.merge(committed_delivery);

    let mut response = Response::builder().status(status);
    for (name, value) in upstream.headers() {
        if should_return_header(name) {
            response = response.header(name, value);
        }
    }
    response = response
        .header("x-quillcache-engine-id", trace.engine_id)
        .header("x-quillcache-mode", serving_mode_header(trace.mode))
        .header("x-quillcache-decode-engine-id", trace.decode_engine_id)
        .header(
            "x-quillcache-prefill-engine-id",
            trace.prefill_engine_id.unwrap_or_default(),
        )
        .header(
            "x-quillcache-planner-actions",
            trace.planner_actions.to_string(),
        )
        .header(
            "x-quillcache-cache-actions",
            cache_action_list.len().to_string(),
        )
        .header("x-quillcache-action-sink", action_sink_delivery.as_header())
        .header("x-quillcache-request-id", trace.request_id)
        .header("x-quillcache-local-hits", trace.local_hits.to_string())
        .header(
            "x-quillcache-transfer-blocks",
            trace.transfer_blocks.to_string(),
        )
        .header(
            "x-quillcache-recompute-blocks",
            trace.recompute_blocks.to_string(),
        )
        .header(
            "x-quillcache-reuse-refused",
            trace.reuse_refused.to_string(),
        )
        .header(
            "x-quillcache-estimated-ttft-us",
            trace.estimated_ttft_us.to_string(),
        )
        .header(
            "x-quillcache-estimated-transfer-us",
            trace.estimated_transfer_us.to_string(),
        )
        .header(
            "x-quillcache-estimated-first-transfer-us",
            trace.estimated_first_transfer_us.to_string(),
        )
        .header(
            "x-quillcache-transfer-max-inflight",
            transfer_max_inflight.to_string(),
        );
    // Stream the upstream body straight through (SSE chunks forwarded as they
    // arrive) instead of buffering it, so the client's time-to-first-token
    // reflects the real engine — QuillCache's decision headers are already set
    // above and flush with the response head, before the first token. On the
    // first chunk we record real TTFT against the SLO budget for live goodput.
    let slo = state.slo.clone();
    let mut recorded = false;
    let stream = upstream.bytes_stream().inspect(move |_chunk| {
        if !recorded {
            recorded = true;
            let ttft_ms = request_start.elapsed().as_millis() as u64;
            slo.record(ttft_ms, ttft_budget_ms);
        }
    });
    response
        .body(axum::body::Body::from_stream(stream))
        .map_err(GatewayHttpError::BuildResponse)
}

async fn dispatch_action_sink_event(
    state: &GatewayState,
    event: ActionSinkEvent,
) -> Result<ActionSinkDelivery, GatewayHttpError> {
    let Some(sink) = state.action_sink.as_ref() else {
        return Ok(ActionSinkDelivery::Disabled);
    };
    match sink.publish(&state.client, &event).await {
        Ok(()) => Ok(ActionSinkDelivery::Sent),
        Err(error) if sink.fail_open => {
            tracing::warn!(
                ?error,
                phase = ?event.phase,
                request_id = %event.route.request_id,
                "action sink delivery failed; continuing because fail_open=true"
            );
            Ok(ActionSinkDelivery::Failed)
        }
        Err(error) => Err(GatewayHttpError::ActionSink(error.to_string())),
    }
}

fn serving_mode_header(mode: ServingMode) -> &'static str {
    match mode {
        ServingMode::Aggregated => "aggregated",
        ServingMode::Disaggregated => "disaggregated",
    }
}

fn inject_transfer_max_inflight(payload: &mut Value, transfer_max_inflight: usize) {
    let Some(object) = payload.as_object_mut() else {
        return;
    };
    let params = object
        .entry("kv_transfer_params")
        .or_insert_with(|| json!({}));
    if !params.is_object() {
        *params = json!({});
    }
    if let Some(params) = params.as_object_mut() {
        params.insert(
            "quillcache_transfer_max_inflight".to_string(),
            json!(transfer_max_inflight.max(1)),
        );
    }
}

fn request_shape_from_payload(payload: &mut Value) -> RequestShape {
    let hints = payload
        .as_object_mut()
        .and_then(|object| object.remove("quillcache"))
        .and_then(|value| serde_json::from_value::<RequestKvHints>(value).ok());

    let model_id = payload
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown-model")
        .to_string();
    let tokenizer_id = hints
        .as_ref()
        .and_then(|hints| hints.tokenizer_id.clone())
        .unwrap_or_else(|| model_id.clone());
    let tenant_id = hints
        .as_ref()
        .and_then(|hints| hints.tenant_id.clone())
        .unwrap_or_else(|| "default".to_string());
    let estimated_decode_tokens = hints
        .as_ref()
        .and_then(|hints| hints.estimated_decode_tokens)
        .or_else(|| {
            payload
                .get("max_tokens")
                .and_then(Value::as_u64)
                .map(|v| v as u32)
        })
        .unwrap_or(128);
    let id = hints
        .as_ref()
        .and_then(|hints| hints.request_id.clone())
        .unwrap_or_else(fallback_request_id);
    let session_id = hints.as_ref().and_then(|hints| hints.session_id.clone());

    let blocks = hints
        .as_ref()
        .filter(|hints| !hints.block_hashes.is_empty())
        .map(|hints| hints.to_blocks(&model_id, &tokenizer_id, &tenant_id))
        .unwrap_or_else(|| fallback_blocks(payload, &model_id, &tokenizer_id, &tenant_id));

    RequestShape {
        id,
        model_id,
        tokenizer_id,
        adapter_id: hints.and_then(|hints| hints.adapter_id),
        tenant_id,
        session_id,
        blocks,
        estimated_decode_tokens,
        slo: SloTarget::default(),
    }
}

/// Inferred bytes per KV block when recording placement (no engine event yet to
/// give the real size). 4 MiB ≈ a 64-token block for a mid-size model.
const DEFAULT_BLOCK_BYTES: u64 = 4 * 1024 * 1024;

/// Approx. characters per fallback block (no tokenizer in the gateway, so we
/// chunk prompt text). ~4 chars/token ⇒ ~64 tokens/block.
const FALLBACK_BLOCK_CHARS: usize = 256;
/// Cap fallback blocks per request so a huge prompt can't explode the index.
const FALLBACK_MAX_BLOCKS: usize = 64;

/// Derive prefix blocks from the request itself when the client sends no
/// `quillcache` hints. Each block hash is **prefix-inclusive** (a hash of all
/// prompt text up to and including the block), so two requests that share a
/// leading prefix — e.g. the same system prompt or RAG context — produce the
/// same leading block hashes and route cache-affinely. The diverging suffix (the
/// user's question) only changes the trailing blocks. This is a tokenizer-free
/// approximation of how engines hash KV blocks; precise hashes arrive via
/// `quillcache` hints or `/v1/kv-events`.
fn fallback_blocks(
    payload: &Value,
    model_id: &str,
    tokenizer_id: &str,
    tenant_id: &str,
) -> Vec<KvBlockKey> {
    let text = prompt_text(payload);
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        let mut hasher = DefaultHasher::new();
        payload.to_string().hash(&mut hasher);
        return vec![KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: model_id.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            adapter_id: None,
            tenant_id: tenant_id.to_string(),
            prefix_hash: "root".to_string(),
            block_hash: format!("pfx-{:016x}", hasher.finish()),
            block_index: 0,
            token_count: 64,
        })];
    }

    let mut blocks = Vec::new();
    let mut parent = "root".to_string();
    let mut start = 0usize;
    let mut idx = 0u32;
    while start < chars.len() && blocks.len() < FALLBACK_MAX_BLOCKS {
        let end = (start + FALLBACK_BLOCK_CHARS).min(chars.len());
        // Prefix-inclusive content hash: bind the whole chain up to `end`.
        let prefix_text: String = chars[..end].iter().collect();
        let mut hasher = DefaultHasher::new();
        prefix_text.hash(&mut hasher);
        let block_hash = format!("pfx-{:016x}", hasher.finish());
        blocks.push(KvBlockKey::external_hash(ExternalKvBlockKey {
            model_id: model_id.to_string(),
            tokenizer_id: tokenizer_id.to_string(),
            adapter_id: None,
            tenant_id: tenant_id.to_string(),
            prefix_hash: parent.clone(),
            block_hash: block_hash.clone(),
            block_index: idx,
            token_count: ((end - start) as u32).div_ceil(4).max(1),
        }));
        parent = block_hash;
        start = end;
        idx += 1;
    }
    blocks
}

/// Flatten the request's prompt to text for fallback block hashing: chat
/// `messages` become `role:content` lines; a completion `prompt` is used as-is.
fn prompt_text(payload: &Value) -> String {
    if let Some(messages) = payload.get("messages").and_then(Value::as_array) {
        let mut text = String::new();
        for message in messages {
            if let Some(role) = message.get("role").and_then(Value::as_str) {
                text.push_str(role);
                text.push(':');
            }
            if let Some(content) = message.get("content").and_then(Value::as_str) {
                text.push_str(content);
                text.push('\n');
            }
        }
        text
    } else if let Some(prompt) = payload.get("prompt").and_then(Value::as_str) {
        prompt.to_string()
    } else {
        payload.to_string()
    }
}

fn fallback_request_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!("req-{nanos}")
}

fn should_forward_header(name: &HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "host"
            | "content-length"
            | "connection"
            | "upgrade"
            | "proxy-authorization"
            | "proxy-authenticate"
            | "te"
            | "trailer"
            | "transfer-encoding"
    )
}

fn should_return_header(name: &HeaderName) -> bool {
    should_forward_header(name) && name != HeaderName::from_static("content-length")
}

#[derive(Debug, Error)]
enum GatewayHttpError {
    #[error("invalid JSON request body: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Control(#[from] quillcache_core::ControlError),
    #[error("routed to unknown engine: {0}")]
    MissingEngine(String),
    #[error(
        "request rejected by QuillCache admission control: {reason} (best_slo_violation_us={best_slo_violation_us})"
    )]
    AdmissionRejected {
        reason: String,
        best_slo_violation_us: u64,
    },
    #[error("upstream request failed: {0}")]
    Upstream(#[from] reqwest::Error),
    #[error("action sink delivery failed: {0}")]
    ActionSink(String),
    #[error("failed to build response: {0}")]
    BuildResponse(axum::http::Error),
}

impl IntoResponse for GatewayHttpError {
    fn into_response(self) -> Response {
        let message = self.to_string();
        let status = match self {
            Self::Json(_) => StatusCode::BAD_REQUEST,
            Self::AdmissionRejected { .. } => StatusCode::TOO_MANY_REQUESTS,
            Self::MissingEngine(_) => StatusCode::BAD_GATEWAY,
            Self::Control(_) | Self::Upstream(_) | Self::ActionSink(_) | Self::BuildResponse(_) => {
                StatusCode::BAD_GATEWAY
            }
        };
        let body = Json(json!({
            "error": {
                "message": message,
                "type": "quillcache_gateway_error"
            }
        }));
        (status, body).into_response()
    }
}

fn _assert_header_value_send_sync(_: HeaderValue) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_quillcache_hints_before_forwarding() {
        let mut value = json!({
            "model": "Qwen/Qwen3-0.6B",
            "messages": [{"role": "user", "content": "hello"}],
            "quillcache": {
                "request_id": "req-a",
                "block_hashes": ["h0"],
                "block_tokens": 16
            }
        });
        let shape = request_shape_from_payload(&mut value);

        assert!(value.get("quillcache").is_none());
        assert_eq!(shape.id, "req-a");
        assert_eq!(shape.blocks[0].block_hash, "h0");
    }

    #[test]
    fn build_index_defaults_to_memory_and_degrades_gracefully() {
        let base = GatewayConfig {
            bind: "127.0.0.1:0".parse().unwrap(),
            engines: vec![],
            policy: None,
            index: None,
            index_path: None,
            data_plane: None,
            action_sink: None,
            conductor: None,
            co_scheduler: None,
        };
        // No backend configured -> in-memory reference.
        assert_eq!(build_index(&base).name(), "memory");
        // An unavailable / uncompiled backend falls back to memory, not a panic.
        let unknown = GatewayConfig {
            index: Some("not-a-backend".to_string()),
            ..base.clone()
        };
        assert_eq!(build_index(&unknown).name(), "memory");
    }

    #[test]
    fn build_data_plane_supports_tiered_runtime_backend() {
        assert_eq!(build_data_plane(None).name(), "none");
        let config = DataPlaneConfig {
            kind: "tiered".to_string(),
            hbm_capacity_bytes: Some(1024),
            cpu_dram_capacity_bytes: Some(2048),
            local_ssd_capacity_bytes: Some(4096),
        };
        let data_plane = build_data_plane(Some(&config));
        // The "tiered" config kind now builds the real store data plane.
        assert_eq!(data_plane.name(), "store");
        assert_eq!(data_plane.metrics().resident_blocks, 0);
    }

    #[test]
    fn build_action_sink_defaults_to_none_and_supports_http() {
        assert!(build_action_sink(None).unwrap().is_none());
        let sink = build_action_sink(Some(&ActionSinkConfig {
            kind: "http".to_string(),
            url: Some("http://127.0.0.1:9090/v1/quillcache/actions".to_string()),
            fail_open: false,
            timeout_ms: 500,
        }))
        .unwrap()
        .unwrap();

        let snapshot = sink.snapshot();
        assert_eq!(snapshot.kind, "http");
        assert!(!snapshot.fail_open);
        assert_eq!(snapshot.timeout_ms, 500);
    }

    #[test]
    fn co_scheduler_admission_action_controls_gateway_admission() {
        let engine = EngineEndpoint {
            id: "decode-a".to_string(),
            kind: quillcache_core::EngineKind::Vllm,
            role: quillcache_core::EngineRole::Aggregated,
            base_url: "http://127.0.0.1:8000".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            tenant_id: "tenant".to_string(),
            locality_domain: "local".to_string(),
        };
        let mut control = ControlPlane::new(vec![engine]);
        let mut actuator = CoSchedulerActuator::new(Some(CoSchedulerConfig {
            apply: true,
            admission_slo_violation_limit_us: 0,
            ..CoSchedulerConfig::default()
        }));
        let action = CoSchedulerAction {
            epoch: 1,
            kind: CoSchedulerActionKind::AdmissionReject,
            target_worker_id: None,
            source_worker_id: None,
            prefix_hash: None,
            tier: None,
            value: Some("enable_overload_rejection".to_string()),
            reason: "test overload".to_string(),
        };
        let plan = CoSchedulerPlan {
            epoch: 1,
            dry_run: false,
            actions: vec![action],
        };

        let applied = actuator.apply(&plan, &CoSchedulerSnapshot::default(), &mut control);
        assert_eq!(applied.len(), 1);
        assert_eq!(control.admission_slo_limit(), Some(0));

        let cold = RequestShape {
            id: "r".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            adapter_id: None,
            tenant_id: "tenant".to_string(),
            session_id: None,
            blocks: vec![KvBlockKey::new(
                "model", "tok", "tenant", "root", "cold", 0, 64,
            )],
            estimated_decode_tokens: 8,
            slo: SloTarget {
                ttft_ms: 0,
                tpot_ms: 0,
            },
        };
        assert!(matches!(
            control.admit(&cold).unwrap(),
            AdmissionDecision::Reject { .. }
        ));

        let clear = CoSchedulerPlan {
            epoch: 2,
            dry_run: false,
            actions: vec![],
        };
        actuator.apply(&clear, &CoSchedulerSnapshot::default(), &mut control);
        assert_eq!(control.admission_slo_limit(), None);
        assert!(matches!(
            control.admit(&cold).unwrap(),
            AdmissionDecision::Admit(_)
        ));
    }

    #[test]
    fn co_scheduler_replicate_action_updates_residency() {
        let engine_a = EngineEndpoint {
            id: "decode-a".to_string(),
            kind: quillcache_core::EngineKind::Vllm,
            role: quillcache_core::EngineRole::Aggregated,
            base_url: "http://127.0.0.1:8000".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            tenant_id: "tenant".to_string(),
            locality_domain: "local".to_string(),
        };
        let engine_b = EngineEndpoint {
            id: "decode-b".to_string(),
            base_url: "http://127.0.0.1:8001".to_string(),
            ..engine_a.clone()
        };
        let mut control = ControlPlane::new(vec![engine_a, engine_b]);
        let block = KvBlockKey::new("model", "tok", "tenant", "root", "hot", 0, 64);
        let request = RequestShape {
            id: "r".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            adapter_id: None,
            tenant_id: "tenant".to_string(),
            session_id: None,
            blocks: vec![block.clone()],
            estimated_decode_tokens: 8,
            slo: SloTarget::default(),
        };
        control.observe_placement("decode-a", &request, 1024);
        let mut actuator = CoSchedulerActuator::new(Some(CoSchedulerConfig::default()));
        let action = CoSchedulerAction {
            epoch: 1,
            kind: CoSchedulerActionKind::ReplicateHotPrefix,
            target_worker_id: Some("decode-b".to_string()),
            source_worker_id: Some("decode-a".to_string()),
            prefix_hash: Some("root".to_string()),
            tier: Some(CacheTier::Hbm),
            value: None,
            reason: "hot prefix".to_string(),
        };

        let applied = actuator.apply(
            &CoSchedulerPlan {
                epoch: 1,
                dry_run: false,
                actions: vec![action],
            },
            &CoSchedulerSnapshot::default(),
            &mut control,
        );

        assert_eq!(applied.len(), 1);
        assert!(applied[0].applied);
        let holders = control
            .residency()
            .locate(&block)
            .into_iter()
            .map(|r| r.worker_id)
            .collect::<Vec<_>>();
        assert!(holders.contains(&"decode-a".to_string()));
        assert!(holders.contains(&"decode-b".to_string()));
    }

    #[test]
    fn co_scheduler_tune_transfer_depth_updates_actuator() {
        let engine = EngineEndpoint {
            id: "decode-a".to_string(),
            kind: quillcache_core::EngineKind::Vllm,
            role: quillcache_core::EngineRole::Aggregated,
            base_url: "http://127.0.0.1:8000".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            tenant_id: "tenant".to_string(),
            locality_domain: "local".to_string(),
        };
        let mut control = ControlPlane::new(vec![engine]);
        let mut actuator = CoSchedulerActuator::new(Some(CoSchedulerConfig {
            apply: true,
            transfer_depth_initial: 1,
            transfer_depth_max: 4,
            ..CoSchedulerConfig::default()
        }));
        let action = |epoch| CoSchedulerAction {
            epoch,
            kind: CoSchedulerActionKind::TuneTransferDepth,
            target_worker_id: None,
            source_worker_id: None,
            prefix_hash: None,
            tier: None,
            value: None,
            reason: "transfer pressure".to_string(),
        };
        let mut snapshot = CoSchedulerSnapshot::default();
        snapshot.transfer.time_to_first_layer_ms = Some(18.0);
        snapshot.transfer.full_transfer_ms = Some(100.0);
        snapshot.transfer.overlap_efficiency_pct = Some(20.0);

        actuator.apply(
            &CoSchedulerPlan {
                epoch: 1,
                dry_run: false,
                actions: vec![action(1)],
            },
            &snapshot,
            &mut control,
        );
        assert_eq!(actuator.snapshot(&control).transfer_max_inflight, 2);

        snapshot.transfer.queue_depth = 32;
        actuator.apply(
            &CoSchedulerPlan {
                epoch: 2,
                dry_run: false,
                actions: vec![action(2)],
            },
            &snapshot,
            &mut control,
        );
        assert_eq!(actuator.snapshot(&control).transfer_max_inflight, 1);
    }

    #[test]
    fn inject_transfer_max_inflight_preserves_existing_kv_transfer_params() {
        let mut payload = json!({
            "model": "m",
            "prompt": "hello",
            "kv_transfer_params": {
                "do_remote_prefill": true,
                "transfer_id": "pd-1"
            }
        });
        inject_transfer_max_inflight(&mut payload, 4);
        let params = payload.get("kv_transfer_params").unwrap();
        assert_eq!(
            params.get("do_remote_prefill").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            params.get("transfer_id").and_then(Value::as_str),
            Some("pd-1")
        );
        assert_eq!(
            params
                .get("quillcache_transfer_max_inflight")
                .and_then(Value::as_u64),
            Some(4)
        );
    }

    #[test]
    fn action_sink_event_carries_request_plan_and_cache_actions() {
        let request = RequestShape {
            id: "req-a".to_string(),
            model_id: "model".to_string(),
            tokenizer_id: "tok".to_string(),
            adapter_id: None,
            tenant_id: "tenant".to_string(),
            session_id: None,
            blocks: vec![],
            estimated_decode_tokens: 1,
            slo: SloTarget::default(),
        };
        let route = GatewayRouteTrace {
            request_id: "req-a".to_string(),
            mode: ServingMode::Aggregated,
            engine_id: "vllm-a".to_string(),
            prefill_engine_id: None,
            decode_engine_id: "vllm-a".to_string(),
            planner_actions: 1,
            reusable_blocks: 0,
            local_hits: 0,
            transfer_blocks: 0,
            recompute_blocks: 0,
            reuse_refused: 0,
            estimated_ttft_us: 10,
            estimated_tpot_us: 20,
            estimated_transfer_us: 0,
            estimated_first_transfer_us: 0,
        };
        let plan = ActionSinkPlan {
            mode: ServingMode::Aggregated,
            execution_worker_id: "vllm-a".to_string(),
            prefill_worker_id: None,
            decode_worker_id: "vllm-a".to_string(),
            actions: vec![],
        };
        let event = ActionSinkEvent::new(
            ActionSinkPhase::Planned,
            "/v1/chat/completions",
            &request,
            &route,
            plan,
            vec![],
        );

        assert_eq!(event.schema_version, 1);
        assert_eq!(event.phase, ActionSinkPhase::Planned);
        assert_eq!(event.request.id, "req-a");
        assert_eq!(event.route.engine_id, "vllm-a");
    }

    #[test]
    fn metrics_aggregate_route_traces_into_totals() {
        let mk = |local, transfer, recompute, refused, reusable, transfer_us, first_us| {
            GatewayRouteTrace {
                request_id: "r".to_string(),
                mode: ServingMode::Aggregated,
                engine_id: "e".to_string(),
                prefill_engine_id: None,
                decode_engine_id: "e".to_string(),
                planner_actions: 0,
                reusable_blocks: reusable,
                local_hits: local,
                transfer_blocks: transfer,
                recompute_blocks: recompute,
                reuse_refused: refused,
                estimated_ttft_us: 0,
                estimated_tpot_us: 0,
                estimated_transfer_us: transfer_us,
                estimated_first_transfer_us: first_us,
            }
        };
        let m = GatewayMetrics::default();
        m.record(&mk(3, 1, 2, 1, 4, 1_200, 400));
        m.record(&mk(2, 0, 1, 0, 2, 0, 0));
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        assert_eq!(g(&m.requests_total), 2);
        assert_eq!(g(&m.local_hits_total), 5);
        assert_eq!(g(&m.transfer_blocks_total), 1);
        assert_eq!(g(&m.recompute_blocks_total), 3);
        assert_eq!(g(&m.reuse_refused_total), 1);
        assert_eq!(g(&m.reusable_blocks_total), 6);
        assert_eq!(g(&m.transfer_requests_total), 1);
        assert_eq!(g(&m.transfer_estimated_us_sum), 1_200);
        assert_eq!(g(&m.transfer_first_estimated_us_sum), 400);

        let transfer = m.transfer_observation();
        assert_eq!(transfer.full_transfer_ms, Some(1.2));
        assert_eq!(transfer.time_to_first_layer_ms, Some(0.4));
        assert_eq!(transfer.overlap_saved_ms, Some(0.8));
        assert_eq!(transfer.exposed_transfer_ms, Some(0.4));
        assert_eq!(transfer.overlap_efficiency_pct, Some(800.0 / 12.0));
    }

    #[test]
    fn measured_transfer_telemetry_overrides_planner_estimates() {
        let trace = GatewayRouteTrace {
            request_id: "r".to_string(),
            mode: ServingMode::Aggregated,
            engine_id: "e".to_string(),
            prefill_engine_id: None,
            decode_engine_id: "e".to_string(),
            planner_actions: 0,
            reusable_blocks: 1,
            local_hits: 0,
            transfer_blocks: 1,
            recompute_blocks: 0,
            reuse_refused: 0,
            estimated_ttft_us: 0,
            estimated_tpot_us: 0,
            estimated_transfer_us: 99_000,
            estimated_first_transfer_us: 88_000,
        };
        let event = TransferTelemetryEvent {
            request_id: Some("r".to_string()),
            source_engine_id: Some("src".to_string()),
            target_engine_id: Some("e".to_string()),
            backend: Some("tcp".to_string()),
            queue_depth: Some(7),
            telemetry: LayerTransferTelemetry {
                layers: 2,
                bytes: 1_000,
                max_inflight: 2,
                time_to_first_layer_us: Some(1_000),
                full_transfer_us: 4_000,
                overlap_window_us: 3_000,
            },
        };

        let m = GatewayMetrics::default();
        m.record(&trace);
        m.record_transfer_telemetry(&event);

        let observation = m.transfer_observation();
        assert_eq!(observation.time_to_first_layer_ms, Some(1.0));
        assert_eq!(observation.full_transfer_ms, Some(4.0));
        assert_eq!(observation.overlap_saved_ms, Some(3.0));
        assert_eq!(observation.exposed_transfer_ms, Some(1.0));
        assert_eq!(observation.overlap_efficiency_pct, Some(75.0));
        assert_eq!(observation.queue_depth, 7);
        assert_eq!(observation.bandwidth_mbps, Some(2.0));

        let summary = m.transfer_summary();
        assert_eq!(summary.measured_events, 1);
        assert_eq!(summary.measured_first_layer_events, 1);
        assert_eq!(summary.measured_bytes, 1_000);
        assert_eq!(summary.measured_layers, 2);
        assert_eq!(summary.avg_time_to_first_layer_ms, Some(1.0));
        assert_eq!(summary.avg_full_transfer_ms, Some(4.0));
        assert_eq!(summary.avg_overlap_window_ms, Some(3.0));
        assert_eq!(summary.avg_exposed_transfer_ms, Some(1.0));
        assert_eq!(summary.overlap_efficiency_pct, Some(75.0));
        assert_eq!(summary.latest_queue_depth, 7);
        assert_eq!(summary.bandwidth_mbps, Some(2.0));
    }

    #[test]
    fn transfer_cost_summary_reads_fetch_actions() {
        let block = KvBlockKey::new("m", "t", "tenant", "root", "b", 0, 64);
        let plan = RequestPlan {
            mode: ServingMode::Aggregated,
            execution_worker_id: "e".to_string(),
            prefill_worker_id: None,
            decode_worker_id: "e".to_string(),
            route: quillcache_core::RouteDecision {
                request_id: "r".to_string(),
                worker_id: "e".to_string(),
                local_hits: vec![],
                transfers: vec![],
                recomputes: vec![],
                estimated_ttft_us: 0,
                estimated_tpot_us: 0,
                slo_violation_us: 0,
            },
            actions: vec![
                PlanAction {
                    kind: PlanActionKind::Fetch,
                    worker_id: "e".to_string(),
                    source_worker_id: Some("src-a".to_string()),
                    key: Some(block.clone()),
                    tier: None,
                    estimated_us: 900,
                },
                PlanAction {
                    kind: PlanActionKind::Fetch,
                    worker_id: "e".to_string(),
                    source_worker_id: Some("src-b".to_string()),
                    key: Some(block),
                    tier: None,
                    estimated_us: 300,
                },
                PlanAction {
                    kind: PlanActionKind::Decode,
                    worker_id: "e".to_string(),
                    source_worker_id: None,
                    key: None,
                    tier: None,
                    estimated_us: 50,
                },
            ],
        };

        let summary = transfer_cost_summary(&plan);
        assert_eq!(summary.estimated_transfer_us, 1_200);
        assert_eq!(summary.estimated_first_transfer_us, 300);
    }

    #[test]
    fn builds_fallback_block_when_no_hints_exist() {
        let mut value = json!({
            "model": "Qwen/Qwen3-0.6B",
            "prompt": "hello",
            "max_tokens": 8
        });
        let shape = request_shape_from_payload(&mut value);

        assert_eq!(shape.model_id, "Qwen/Qwen3-0.6B");
        assert_eq!(shape.estimated_decode_tokens, 8);
        // "hello" is one short block.
        assert_eq!(shape.blocks.len(), 1);
        assert!(shape.blocks[0].block_hash.starts_with("pfx-"));
    }

    #[test]
    fn shared_system_prompt_yields_shared_prefix_blocks() {
        // A long shared system prompt (spans several fallback blocks) followed by
        // a per-request user turn — the multi-tenant shared-prompt case.
        let system = "You are a careful assistant. ".repeat(40);
        let make = |question: &str| {
            json!({
                "model": "m",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": question}
                ]
            })
        };
        let mut a = make("What is 2 + 2?");
        let mut b = make("Name a primary color.");
        let sa = request_shape_from_payload(&mut a);
        let sb = request_shape_from_payload(&mut b);

        // The shared system prefix yields identical leading block hashes (the
        // cache-affinity signal)...
        assert!(sa.blocks.len() >= 2 && sb.blocks.len() >= 2);
        assert_eq!(sa.blocks[0].block_hash, sb.blocks[0].block_hash);
        // ...while the diverging user turn changes the trailing block.
        assert_ne!(
            sa.blocks.last().unwrap().block_hash,
            sb.blocks.last().unwrap().block_hash
        );
    }
}
