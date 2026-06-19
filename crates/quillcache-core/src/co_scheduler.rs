//! Cluster-level KV-cache co-scheduler.
//!
//! This sits above the per-request router. The router answers "where should this
//! request go now"; the co-scheduler answers "what should the fleet change next"
//! from live SLO, cache, transfer, and tier-pressure observations. P0 is dry-run
//! by design: it emits actions an actuator can later bind to vLLM/SGLang,
//! LMCache/KVBM, or the transfer engine.

use serde::{Deserialize, Serialize};

use crate::{CacheTier, EngineRole, WorkerState};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoSchedulerSnapshot {
    pub epoch: u64,
    #[serde(default)]
    pub workers: Vec<WorkerState>,
    #[serde(default)]
    pub slo: SloObservation,
    #[serde(default)]
    pub cache: CacheObservation,
    #[serde(default)]
    pub transfer: TransferObservation,
    #[serde(default)]
    pub tier: TierObservation,
    #[serde(default)]
    pub hot_prefixes: Vec<HotPrefixObservation>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CoSchedulerTelemetry {
    #[serde(default)]
    pub slo: SloObservation,
    #[serde(default)]
    pub cache: CacheObservation,
    #[serde(default)]
    pub transfer: TransferObservation,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SloObservation {
    pub ttft_p99_ms: Option<f64>,
    pub ttft_mean_ms: Option<f64>,
    pub tpot_p99_ms: Option<f64>,
    pub ttft_budget_ms: Option<u64>,
    pub tpot_budget_ms: Option<u64>,
    pub slo_miss_pct: Option<f64>,
    pub goodput_pct: Option<f64>,
}

impl SloObservation {
    fn observed_ttft_ms(&self) -> Option<f64> {
        self.ttft_p99_ms.or(self.ttft_mean_ms)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheObservation {
    pub requests: u64,
    pub reusable_blocks: u64,
    pub local_hits: u64,
    pub remote_hits: u64,
    pub recompute_blocks: u64,
    pub reuse_refused: u64,
    pub local_hit_rate_pct: Option<f64>,
    pub remote_hit_rate_pct: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TransferObservation {
    pub time_to_first_layer_ms: Option<f64>,
    pub full_transfer_ms: Option<f64>,
    pub overlap_saved_ms: Option<f64>,
    pub exposed_transfer_ms: Option<f64>,
    pub overlap_efficiency_pct: Option<f64>,
    pub queue_depth: u64,
    pub bandwidth_mbps: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierObservation {
    pub hbm_used_bytes: u64,
    pub hbm_capacity_bytes: u64,
    pub cpu_dram_used_bytes: u64,
    pub cpu_dram_capacity_bytes: u64,
    pub local_ssd_used_bytes: u64,
    pub local_ssd_capacity_bytes: u64,
    pub evictions: u64,
}

impl TierObservation {
    pub fn hbm_pressure_pct(&self) -> Option<f64> {
        pct(self.hbm_used_bytes, self.hbm_capacity_bytes)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HotPrefixObservation {
    pub prefix_hash: String,
    pub holders: Vec<String>,
    pub estimated_hits_per_interval: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoSchedulerActionKind {
    AdjustPdRatio,
    AdjustHbmSplit,
    ReplicateHotPrefix,
    PromotePrefix,
    DemotePrefix,
    TuneTransferDepth,
    SelectTransferBackend,
    AdmissionReject,
    PinPrefix,
    UnpinPrefix,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoSchedulerAction {
    pub epoch: u64,
    pub kind: CoSchedulerActionKind,
    pub target_worker_id: Option<String>,
    pub source_worker_id: Option<String>,
    pub prefix_hash: Option<String>,
    pub tier: Option<CacheTier>,
    pub value: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoSchedulerPlan {
    pub epoch: u64,
    pub dry_run: bool,
    pub actions: Vec<CoSchedulerAction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoSchedulerPolicy {
    pub max_actions: usize,
    pub min_goodput_pct: f64,
    pub max_slo_miss_pct: f64,
    pub remote_hit_rate_threshold_pct: f64,
    pub hbm_pressure_threshold_pct: f64,
    pub high_transfer_queue_depth: u64,
    pub time_to_first_layer_budget_ms: f64,
    pub min_overlap_efficiency_pct: f64,
    pub target_hot_prefix_replicas: usize,
    pub min_hot_prefix_hits: u64,
}

impl Default for CoSchedulerPolicy {
    fn default() -> Self {
        Self {
            max_actions: 32,
            min_goodput_pct: 95.0,
            max_slo_miss_pct: 10.0,
            remote_hit_rate_threshold_pct: 20.0,
            hbm_pressure_threshold_pct: 90.0,
            high_transfer_queue_depth: 16,
            time_to_first_layer_budget_ms: 20.0,
            min_overlap_efficiency_pct: 50.0,
            target_hot_prefix_replicas: 2,
            min_hot_prefix_hits: 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoScheduler {
    pub policy: CoSchedulerPolicy,
}

impl Default for CoScheduler {
    fn default() -> Self {
        Self::new(CoSchedulerPolicy::default())
    }
}

impl CoScheduler {
    pub fn new(policy: CoSchedulerPolicy) -> Self {
        Self { policy }
    }

    pub fn dry_run(&self, snapshot: &CoSchedulerSnapshot) -> CoSchedulerPlan {
        self.plan(snapshot, true)
    }

    pub fn plan(&self, snapshot: &CoSchedulerSnapshot, dry_run: bool) -> CoSchedulerPlan {
        let mut actions = Vec::new();
        self.plan_pd_pressure(snapshot, &mut actions);
        self.plan_hot_prefix_replication(snapshot, &mut actions);
        self.plan_hbm_pressure(snapshot, &mut actions);
        self.plan_transfer_pressure(snapshot, &mut actions);
        self.plan_admission_pressure(snapshot, &mut actions);
        CoSchedulerPlan {
            epoch: snapshot.epoch,
            dry_run,
            actions,
        }
    }

    fn plan_pd_pressure(
        &self,
        snapshot: &CoSchedulerSnapshot,
        actions: &mut Vec<CoSchedulerAction>,
    ) {
        if !self.has_room(actions) || !self.slo_under_pressure(&snapshot.slo) {
            return;
        }
        let can_prefill = snapshot
            .workers
            .iter()
            .any(|worker| worker.role.can_prefill());
        let can_decode = snapshot
            .workers
            .iter()
            .any(|worker| worker.role.can_decode());
        if !can_prefill || !can_decode {
            return;
        }
        let dedicated_prefill = snapshot
            .workers
            .iter()
            .filter(|worker| worker.role == EngineRole::Prefill)
            .count();
        let dedicated_decode = snapshot
            .workers
            .iter()
            .filter(|worker| worker.role == EngineRole::Decode)
            .count();
        let total_prefill_queue: u64 = snapshot
            .workers
            .iter()
            .map(|worker| u64::from(worker.queued_prefill_tokens))
            .sum();
        let total_decode_load: u64 = snapshot
            .workers
            .iter()
            .map(|worker| u64::from(worker.running_decodes))
            .sum();
        let value = if dedicated_prefill == 0 || dedicated_decode == 0 {
            "split_aggregated_worker_for_pd".to_string()
        } else if total_prefill_queue > total_decode_load.saturating_mul(512) {
            "increase_prefill_share".to_string()
        } else {
            "rebalance_prefill_decode_share".to_string()
        };

        actions.push(CoSchedulerAction {
            epoch: snapshot.epoch,
            kind: CoSchedulerActionKind::AdjustPdRatio,
            target_worker_id: None,
            source_worker_id: None,
            prefix_hash: None,
            tier: None,
            value: Some(value),
            reason: format!(
                "ttft/goodput pressure: ttft={:?}ms budget={:?}ms goodput={:?}%",
                snapshot.slo.observed_ttft_ms(),
                snapshot.slo.ttft_budget_ms,
                snapshot.slo.goodput_pct
            ),
        });
    }

    fn plan_hot_prefix_replication(
        &self,
        snapshot: &CoSchedulerSnapshot,
        actions: &mut Vec<CoSchedulerAction>,
    ) {
        if !self.has_room(actions) || !self.remote_reuse_under_pressure(snapshot) {
            return;
        }
        let mut hot: Vec<&HotPrefixObservation> = snapshot
            .hot_prefixes
            .iter()
            .filter(|prefix| {
                prefix.estimated_hits_per_interval >= self.policy.min_hot_prefix_hits
                    && !prefix.holders.is_empty()
                    && prefix.holders.len() < self.policy.target_hot_prefix_replicas
            })
            .collect();
        hot.sort_by(|a, b| {
            b.estimated_hits_per_interval
                .cmp(&a.estimated_hits_per_interval)
                .then_with(|| a.prefix_hash.cmp(&b.prefix_hash))
        });

        for prefix in hot {
            if !self.has_room(actions) {
                break;
            }
            let Some(target) = least_loaded_non_holder(&snapshot.workers, &prefix.holders) else {
                continue;
            };
            actions.push(CoSchedulerAction {
                epoch: snapshot.epoch,
                kind: CoSchedulerActionKind::ReplicateHotPrefix,
                target_worker_id: Some(target.id.clone()),
                source_worker_id: prefix.holders.first().cloned(),
                prefix_hash: Some(prefix.prefix_hash.clone()),
                tier: Some(CacheTier::Hbm),
                value: Some(format!(
                    "target_replicas={}",
                    self.policy.target_hot_prefix_replicas
                )),
                reason: format!(
                    "hot remote prefix: hits={} holders={} remote_hit_rate={:?}%",
                    prefix.estimated_hits_per_interval,
                    prefix.holders.len(),
                    snapshot.cache.remote_hit_rate_pct
                ),
            });
        }
    }

    fn plan_hbm_pressure(
        &self,
        snapshot: &CoSchedulerSnapshot,
        actions: &mut Vec<CoSchedulerAction>,
    ) {
        if !self.has_room(actions) {
            return;
        }
        let Some(pressure) = snapshot.tier.hbm_pressure_pct() else {
            return;
        };
        if pressure < self.policy.hbm_pressure_threshold_pct {
            return;
        }
        let target_worker_id = most_hbm_pressured_worker(&snapshot.workers).map(|w| w.id.clone());
        actions.push(CoSchedulerAction {
            epoch: snapshot.epoch,
            kind: CoSchedulerActionKind::DemotePrefix,
            target_worker_id,
            source_worker_id: None,
            prefix_hash: None,
            tier: Some(CacheTier::CpuDram),
            value: Some("demote_cold_hbm_blocks".to_string()),
            reason: format!(
                "hbm pressure {:.1}% exceeds {:.1}%",
                pressure, self.policy.hbm_pressure_threshold_pct
            ),
        });
    }

    fn plan_transfer_pressure(
        &self,
        snapshot: &CoSchedulerSnapshot,
        actions: &mut Vec<CoSchedulerAction>,
    ) {
        if !self.has_room(actions) {
            return;
        }
        let slow_first_layer = snapshot
            .transfer
            .time_to_first_layer_ms
            .map(|ms| ms > self.policy.time_to_first_layer_budget_ms)
            .unwrap_or(false);
        let queued = snapshot.transfer.queue_depth >= self.policy.high_transfer_queue_depth;
        let low_overlap_efficiency = match (
            snapshot.transfer.overlap_efficiency_pct,
            snapshot.transfer.full_transfer_ms,
        ) {
            (Some(efficiency), Some(full_transfer_ms)) => {
                full_transfer_ms > self.policy.time_to_first_layer_budget_ms
                    && efficiency < self.policy.min_overlap_efficiency_pct
            }
            _ => false,
        };
        if !slow_first_layer && !queued && !low_overlap_efficiency {
            return;
        }
        actions.push(CoSchedulerAction {
            epoch: snapshot.epoch,
            kind: CoSchedulerActionKind::TuneTransferDepth,
            target_worker_id: None,
            source_worker_id: None,
            prefix_hash: None,
            tier: None,
            value: Some("prioritize_first_layer_or_switch_backend".to_string()),
            reason: format!(
                "transfer pressure: first_layer={:?}ms queue_depth={} overlap_efficiency={:?}%",
                snapshot.transfer.time_to_first_layer_ms,
                snapshot.transfer.queue_depth,
                snapshot.transfer.overlap_efficiency_pct
            ),
        });
    }

    fn plan_admission_pressure(
        &self,
        snapshot: &CoSchedulerSnapshot,
        actions: &mut Vec<CoSchedulerAction>,
    ) {
        if !self.has_room(actions) {
            return;
        }
        let slo_miss_bad = snapshot
            .slo
            .slo_miss_pct
            .map(|pct| pct > self.policy.max_slo_miss_pct)
            .unwrap_or(false);
        let transfer_queued =
            snapshot.transfer.queue_depth >= self.policy.high_transfer_queue_depth;
        let hbm_full = snapshot
            .tier
            .hbm_pressure_pct()
            .map(|pct| pct >= self.policy.hbm_pressure_threshold_pct)
            .unwrap_or(false);
        if !(slo_miss_bad && transfer_queued && hbm_full) {
            return;
        }
        actions.push(CoSchedulerAction {
            epoch: snapshot.epoch,
            kind: CoSchedulerActionKind::AdmissionReject,
            target_worker_id: None,
            source_worker_id: None,
            prefix_hash: None,
            tier: None,
            value: Some("enable_overload_rejection".to_string()),
            reason: format!(
                "slo miss {:?}% with transfer queue {} and hbm pressure {:?}%",
                snapshot.slo.slo_miss_pct,
                snapshot.transfer.queue_depth,
                snapshot.tier.hbm_pressure_pct()
            ),
        });
    }

    fn slo_under_pressure(&self, slo: &SloObservation) -> bool {
        let ttft_bad = match (slo.observed_ttft_ms(), slo.ttft_budget_ms) {
            (Some(observed), Some(budget)) => observed > budget as f64,
            _ => false,
        };
        let goodput_bad = slo
            .goodput_pct
            .map(|pct| pct < self.policy.min_goodput_pct)
            .unwrap_or(false);
        let slo_miss_bad = slo
            .slo_miss_pct
            .map(|pct| pct > self.policy.max_slo_miss_pct)
            .unwrap_or(false);
        ttft_bad || goodput_bad || slo_miss_bad
    }

    fn remote_reuse_under_pressure(&self, snapshot: &CoSchedulerSnapshot) -> bool {
        let remote_hit_bad = snapshot
            .cache
            .remote_hit_rate_pct
            .map(|pct| pct >= self.policy.remote_hit_rate_threshold_pct)
            .unwrap_or(false);
        let full_transfer_bad = match (
            snapshot.transfer.full_transfer_ms,
            snapshot.transfer.time_to_first_layer_ms,
        ) {
            (Some(full), Some(first)) => {
                full > first && full > self.policy.time_to_first_layer_budget_ms
            }
            (Some(full), None) => full > self.policy.time_to_first_layer_budget_ms,
            _ => false,
        };
        remote_hit_bad || full_transfer_bad
    }

    fn has_room(&self, actions: &[CoSchedulerAction]) -> bool {
        actions.len() < self.policy.max_actions
    }
}

fn pct(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(100.0 * numerator as f64 / denominator as f64)
    }
}

fn least_loaded_non_holder<'a>(
    workers: &'a [WorkerState],
    holders: &[String],
) -> Option<&'a WorkerState> {
    workers
        .iter()
        .filter(|worker| !holders.iter().any(|holder| holder == &worker.id))
        .min_by_key(|worker| {
            (
                worker.running_decodes,
                worker.queued_prefill_tokens,
                worker.id.clone(),
            )
        })
}

fn most_hbm_pressured_worker(workers: &[WorkerState]) -> Option<&WorkerState> {
    workers
        .iter()
        .filter(|worker| worker.hbm_capacity_bytes > 0)
        .max_by(|a, b| {
            let left = u128::from(a.hbm_used_bytes) * u128::from(b.hbm_capacity_bytes);
            let right = u128::from(b.hbm_used_bytes) * u128::from(a.hbm_capacity_bytes);
            left.cmp(&right).then_with(|| a.id.cmp(&b.id))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: &str, role: EngineRole) -> WorkerState {
        WorkerState::new(id, "rack-a").with_role(role)
    }

    fn healthy_snapshot() -> CoSchedulerSnapshot {
        CoSchedulerSnapshot {
            epoch: 7,
            workers: vec![worker("w0", EngineRole::Aggregated)],
            slo: SloObservation {
                ttft_p99_ms: Some(300.0),
                ttft_mean_ms: Some(120.0),
                ttft_budget_ms: Some(800),
                goodput_pct: Some(99.0),
                ..SloObservation::default()
            },
            cache: CacheObservation {
                requests: 10,
                local_hits: 90,
                remote_hits: 2,
                recompute_blocks: 1,
                local_hit_rate_pct: Some(96.8),
                remote_hit_rate_pct: Some(2.1),
                ..CacheObservation::default()
            },
            tier: TierObservation {
                hbm_used_bytes: 40,
                hbm_capacity_bytes: 100,
                ..TierObservation::default()
            },
            ..CoSchedulerSnapshot::default()
        }
    }

    #[test]
    fn healthy_snapshot_has_no_actions() {
        let scheduler = CoScheduler::default();
        let plan = scheduler.dry_run(&healthy_snapshot());
        assert!(plan.dry_run);
        assert!(plan.actions.is_empty());
    }

    #[test]
    fn ttft_pressure_emits_pd_ratio_adjustment() {
        let scheduler = CoScheduler::default();
        let mut snapshot = healthy_snapshot();
        snapshot.workers = vec![
            worker("prefill-a", EngineRole::Prefill),
            worker("decode-a", EngineRole::Decode),
        ];
        snapshot.slo.ttft_p99_ms = Some(2_000.0);
        snapshot.slo.ttft_budget_ms = Some(800);
        snapshot.slo.goodput_pct = Some(70.0);

        let plan = scheduler.dry_run(&snapshot);
        assert!(plan
            .actions
            .iter()
            .any(|action| action.kind == CoSchedulerActionKind::AdjustPdRatio));
    }

    #[test]
    fn remote_hot_prefix_emits_replication() {
        let scheduler = CoScheduler::default();
        let mut snapshot = healthy_snapshot();
        snapshot.workers = vec![
            worker("decode-a", EngineRole::Decode).with_load(0, 4),
            worker("decode-b", EngineRole::Decode).with_load(0, 0),
        ];
        snapshot.cache.remote_hit_rate_pct = Some(45.0);
        snapshot.transfer.full_transfer_ms = Some(80.0);
        snapshot.transfer.time_to_first_layer_ms = Some(10.0);
        snapshot.hot_prefixes = vec![HotPrefixObservation {
            prefix_hash: "sys-prompt".to_string(),
            holders: vec!["decode-a".to_string()],
            estimated_hits_per_interval: 64,
            bytes: 256 << 20,
        }];

        let plan = scheduler.dry_run(&snapshot);
        let action = plan
            .actions
            .iter()
            .find(|action| action.kind == CoSchedulerActionKind::ReplicateHotPrefix)
            .expect("replicate hot prefix");
        assert_eq!(action.prefix_hash.as_deref(), Some("sys-prompt"));
        assert_eq!(action.target_worker_id.as_deref(), Some("decode-b"));
    }

    #[test]
    fn hbm_pressure_emits_demote_action() {
        let scheduler = CoScheduler::default();
        let mut snapshot = healthy_snapshot();
        snapshot.workers = vec![WorkerState {
            hbm_used_bytes: 95,
            hbm_capacity_bytes: 100,
            ..worker("w0", EngineRole::Aggregated)
        }];
        snapshot.tier.hbm_used_bytes = 95;
        snapshot.tier.hbm_capacity_bytes = 100;

        let plan = scheduler.dry_run(&snapshot);
        assert!(plan
            .actions
            .iter()
            .any(|action| action.kind == CoSchedulerActionKind::DemotePrefix));
    }

    #[test]
    fn transfer_queue_pressure_tunes_depth() {
        let scheduler = CoScheduler::default();
        let mut snapshot = healthy_snapshot();
        snapshot.transfer.queue_depth = 32;

        let plan = scheduler.dry_run(&snapshot);
        assert!(plan
            .actions
            .iter()
            .any(|action| action.kind == CoSchedulerActionKind::TuneTransferDepth));
    }

    #[test]
    fn low_overlap_efficiency_tunes_depth() {
        let scheduler = CoScheduler::default();
        let mut snapshot = healthy_snapshot();
        snapshot.transfer.time_to_first_layer_ms = Some(18.0);
        snapshot.transfer.full_transfer_ms = Some(100.0);
        snapshot.transfer.overlap_efficiency_pct = Some(20.0);

        let plan = scheduler.dry_run(&snapshot);
        let action = plan
            .actions
            .iter()
            .find(|action| action.kind == CoSchedulerActionKind::TuneTransferDepth)
            .expect("tune transfer depth");
        assert!(action.reason.contains("overlap_efficiency=Some(20.0)%"));
    }
}
