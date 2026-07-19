//! SLO-aware planning for distributed attention.
//!
//! The planner compares local execution, query routing, KV staging, and
//! sharded execution using one explicit latency model. Storage and transport
//! implementations remain outside this module.

use crate::attention::AttentionExecutionMode;
use crate::types::{MemoryDomain, WorkerId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    #[error("no feasible attention execution candidate")]
    NoFeasibleCandidate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionCandidate {
    pub worker_id: WorkerId,
    pub mode: AttentionExecutionMode,
    pub source_domain: MemoryDomain,
    pub queue_us: u64,
    pub query_bytes: u64,
    pub output_bytes: u64,
    pub kv_stage_bytes: u64,
    pub effective_bandwidth_bytes_per_us: f64,
    pub fixed_transport_us: u64,
    pub kernel_us: u64,
    pub merge_us: u64,
    pub feasible: bool,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPlan {
    pub worker_id: WorkerId,
    pub mode: AttentionExecutionMode,
    pub estimated_us: u64,
    pub meets_deadline: bool,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionCostModel {
    /// Headroom for model error and runtime variance.
    pub risk_multiplier: f64,
}

impl Default for AttentionCostModel {
    fn default() -> Self {
        Self {
            risk_multiplier: 1.10,
        }
    }
}

impl AttentionCostModel {
    pub fn estimate_us(&self, candidate: &AttentionCandidate) -> u64 {
        let transport_bytes = candidate
            .query_bytes
            .saturating_add(candidate.output_bytes)
            .saturating_add(candidate.kv_stage_bytes);
        let transfer_us = if transport_bytes == 0 {
            0.0
        } else if candidate.effective_bandwidth_bytes_per_us > 0.0 {
            transport_bytes as f64 / candidate.effective_bandwidth_bytes_per_us
        } else {
            f64::INFINITY
        };
        let base = candidate.queue_us as f64
            + candidate.fixed_transport_us as f64
            + transfer_us
            + candidate.kernel_us as f64
            + candidate.merge_us as f64;
        let estimate = (base * self.risk_multiplier).ceil();
        if estimate.is_finite() && estimate <= u64::MAX as f64 {
            estimate as u64
        } else {
            u64::MAX
        }
    }

    pub fn choose(
        &self,
        candidates: &[AttentionCandidate],
        budget_us: u64,
    ) -> Result<AttentionPlan, SchedulerError> {
        let (candidate, estimate) = candidates
            .iter()
            .filter(|candidate| candidate.feasible)
            .map(|candidate| (candidate, self.estimate_us(candidate)))
            .min_by_key(|(_, estimate)| *estimate)
            .ok_or(SchedulerError::NoFeasibleCandidate)?;

        Ok(AttentionPlan {
            worker_id: candidate.worker_id.clone(),
            mode: candidate.mode,
            estimated_us: estimate,
            meets_deadline: estimate <= budget_us,
            explanation: candidate.explanation.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        worker: &str,
        mode: AttentionExecutionMode,
        query: u64,
        output: u64,
        kv: u64,
    ) -> AttentionCandidate {
        AttentionCandidate {
            worker_id: WorkerId(worker.into()),
            mode,
            source_domain: MemoryDomain::RemoteHbm,
            queue_us: 10,
            query_bytes: query,
            output_bytes: output,
            kv_stage_bytes: kv,
            effective_bandwidth_bytes_per_us: 1_000.0,
            fixed_transport_us: 5,
            kernel_us: 20,
            merge_us: 2,
            feasible: true,
            explanation: worker.into(),
        }
    }

    #[test]
    fn routes_query_when_history_is_much_larger_than_query_and_output() {
        let fetch = candidate(
            "local",
            AttentionExecutionMode::StageKv,
            0,
            0,
            128 * 1024 * 1024,
        );
        let route = candidate(
            "remote",
            AttentionExecutionMode::RouteQuery,
            16 * 1024,
            16 * 1024,
            0,
        );
        let plan = AttentionCostModel::default()
            .choose(&[fetch, route], 1_000)
            .unwrap();
        assert_eq!(plan.mode, AttentionExecutionMode::RouteQuery);
        assert!(plan.meets_deadline);
    }

    #[test]
    fn returns_fastest_plan_even_when_slo_cannot_be_met() {
        let slow = candidate(
            "remote",
            AttentionExecutionMode::RouteQuery,
            1_000_000,
            1_000_000,
            0,
        );
        let plan = AttentionCostModel::default().choose(&[slow], 1).unwrap();
        assert!(!plan.meets_deadline);
    }
}
