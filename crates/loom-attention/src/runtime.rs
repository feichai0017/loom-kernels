//! Node-local runtime state used on the model-forward critical path.
//!
//! The global controller never participates in a per-layer lookup. A step pins
//! a page-table generation and lease set; commit is rejected if either changed.

use crate::attention::KvView;
use crate::pool::ReadLease;
use crate::scheduler::AttentionPlan;
use crate::types::{PoolObjectRef, SequenceBlockRef, SequenceId, WorkerId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    #[error("unknown sequence: {0}")]
    UnknownSequence(String),
    #[error("sequence already has an active step: {0}")]
    StepAlreadyActive(String),
    #[error("step handle is stale")]
    StalePlan,
    #[error("a required KV lease has expired: {0}")]
    LeaseExpired(String),
    #[error("sequence layout is invalid: {0}")]
    InvalidSequence(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageBinding {
    pub sequence_ref: SequenceBlockRef,
    pub object: PoolObjectRef,
    pub lease_id: String,
    pub lease_expires_at_unix_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveTail {
    pub block_index: u32,
    pub tokens: u32,
    pub capacity_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailCommit {
    Mutable { tokens: u32 },
    Sealed { block_index: u32 },
}

impl ActiveTail {
    fn append(&mut self, tokens: u32) -> Result<TailCommit, RuntimeError> {
        let next = self
            .tokens
            .checked_add(tokens)
            .ok_or_else(|| RuntimeError::InvalidSequence("tail token count overflow".into()))?;
        if next > self.capacity_tokens {
            return Err(RuntimeError::InvalidSequence(format!(
                "tail append exceeds block capacity: {next} > {}",
                self.capacity_tokens
            )));
        }
        self.tokens = next;
        if self.tokens == self.capacity_tokens {
            let sealed = self.block_index;
            self.block_index += 1;
            self.tokens = 0;
            Ok(TailCommit::Sealed {
                block_index: sealed,
            })
        } else {
            Ok(TailCommit::Mutable {
                tokens: self.tokens,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerStepPlan {
    pub layer_id: u32,
    pub attention: AttentionPlan,
    pub kv: KvView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanHandle {
    pub plan_id: u64,
    pub sequence_id: SequenceId,
    pub sequence_generation: u64,
    pub page_table_generation: u64,
    pub layers: Vec<LayerStepPlan>,
}

#[derive(Debug)]
struct SequenceState {
    generation: u64,
    page_table_generation: u64,
    page_table: BTreeMap<(u32, u32), PageBinding>,
    leases: HashMap<String, ReadLease>,
    active_tail: ActiveTail,
    active_plan_id: Option<u64>,
    execution_worker: WorkerId,
}

#[derive(Debug, Default)]
pub struct SequenceRuntime {
    sequences: HashMap<SequenceId, SequenceState>,
    next_plan_id: u64,
}

impl SequenceRuntime {
    pub fn open_sequence(
        &mut self,
        sequence_id: SequenceId,
        execution_worker: WorkerId,
        block_tokens: u32,
    ) -> Result<(), RuntimeError> {
        if block_tokens == 0 {
            return Err(RuntimeError::InvalidSequence(
                "block_tokens must be non-zero".into(),
            ));
        }
        if self.sequences.contains_key(&sequence_id) {
            return Err(RuntimeError::InvalidSequence(format!(
                "sequence {} is already open",
                sequence_id.0
            )));
        }
        self.sequences.insert(
            sequence_id,
            SequenceState {
                generation: 1,
                page_table_generation: 0,
                page_table: BTreeMap::new(),
                leases: HashMap::new(),
                active_tail: ActiveTail {
                    block_index: 0,
                    tokens: 0,
                    capacity_tokens: block_tokens,
                },
                active_plan_id: None,
                execution_worker,
            },
        );
        Ok(())
    }

    pub fn install_pages(
        &mut self,
        sequence_id: &SequenceId,
        lease: ReadLease,
        pages: Vec<(SequenceBlockRef, PoolObjectRef)>,
    ) -> Result<(), RuntimeError> {
        let state = self.state_mut(sequence_id)?;
        if pages
            .iter()
            .any(|(_, object)| !lease.objects.contains(object))
        {
            return Err(RuntimeError::InvalidSequence(
                "page object is not covered by the supplied lease".into(),
            ));
        }
        for (sequence_ref, object) in pages {
            if sequence_ref.sequence_id != *sequence_id {
                return Err(RuntimeError::InvalidSequence(
                    "page belongs to a different sequence".into(),
                ));
            }
            state.page_table.insert(
                (sequence_ref.layer_id, sequence_ref.logical_block),
                PageBinding {
                    sequence_ref,
                    object,
                    lease_id: lease.lease_id.clone(),
                    lease_expires_at_unix_us: lease.expires_at_unix_us,
                },
            );
        }
        state.leases.insert(lease.lease_id.clone(), lease);
        state.page_table_generation += 1;
        Ok(())
    }

    pub fn begin_step(
        &mut self,
        sequence_id: &SequenceId,
        now_unix_us: u64,
        layers: Vec<LayerStepPlan>,
    ) -> Result<PlanHandle, RuntimeError> {
        self.next_plan_id += 1;
        let plan_id = self.next_plan_id;
        let state = self.state_mut(sequence_id)?;
        if state.active_plan_id.is_some() {
            return Err(RuntimeError::StepAlreadyActive(sequence_id.0.clone()));
        }
        if let Some(binding) = state
            .page_table
            .values()
            .find(|binding| now_unix_us >= binding.lease_expires_at_unix_us)
        {
            return Err(RuntimeError::LeaseExpired(binding.lease_id.clone()));
        }
        for layer in &layers {
            if layer.kv.page_table_generation != state.page_table_generation {
                return Err(RuntimeError::StalePlan);
            }
            if layer
                .kv
                .lease_ids
                .iter()
                .any(|lease_id| !state.leases.contains_key(lease_id))
            {
                return Err(RuntimeError::InvalidSequence(
                    "attention plan references an unknown KV lease".into(),
                ));
            }
            for block in &layer.kv.blocks {
                let installed = state.page_table.values().any(|binding| {
                    binding.sequence_ref.layer_id == layer.layer_id
                        && binding.sequence_ref.block_id == *block
                        && layer.kv.lease_ids.contains(&binding.lease_id)
                });
                if !installed {
                    return Err(RuntimeError::InvalidSequence(
                        "attention plan references an unbound KV block".into(),
                    ));
                }
            }
        }
        state.active_plan_id = Some(plan_id);
        Ok(PlanHandle {
            plan_id,
            sequence_id: sequence_id.clone(),
            sequence_generation: state.generation,
            page_table_generation: state.page_table_generation,
            layers,
        })
    }

    pub fn commit_step(
        &mut self,
        handle: PlanHandle,
        appended_tokens: u32,
    ) -> Result<TailCommit, RuntimeError> {
        let state = self.state_mut(&handle.sequence_id)?;
        if state.active_plan_id != Some(handle.plan_id)
            || state.generation != handle.sequence_generation
            || state.page_table_generation != handle.page_table_generation
        {
            return Err(RuntimeError::StalePlan);
        }
        let result = state.active_tail.append(appended_tokens)?;
        state.active_plan_id = None;
        Ok(result)
    }

    pub fn abort_step(&mut self, handle: &PlanHandle) -> Result<(), RuntimeError> {
        let state = self.state_mut(&handle.sequence_id)?;
        if state.active_plan_id != Some(handle.plan_id) {
            return Err(RuntimeError::StalePlan);
        }
        state.active_plan_id = None;
        Ok(())
    }

    pub fn execution_worker(&self, sequence_id: &SequenceId) -> Result<&WorkerId, RuntimeError> {
        Ok(&self.state(sequence_id)?.execution_worker)
    }

    pub fn kv_view(&self, sequence_id: &SequenceId, layer_id: u32) -> Result<KvView, RuntimeError> {
        let state = self.state(sequence_id)?;
        let mut blocks = Vec::new();
        let mut lease_ids = BTreeSet::new();
        for ((binding_layer, _), binding) in &state.page_table {
            if *binding_layer == layer_id {
                blocks.push(binding.sequence_ref.block_id.clone());
                lease_ids.insert(binding.lease_id.clone());
            }
        }
        Ok(KvView {
            blocks,
            page_table_generation: state.page_table_generation,
            lease_ids: lease_ids.into_iter().collect(),
        })
    }

    pub fn close_sequence(&mut self, sequence_id: &SequenceId) -> Result<(), RuntimeError> {
        self.sequences
            .remove(sequence_id)
            .map(|_| ())
            .ok_or_else(|| RuntimeError::UnknownSequence(sequence_id.0.clone()))
    }

    fn state(&self, sequence_id: &SequenceId) -> Result<&SequenceState, RuntimeError> {
        self.sequences
            .get(sequence_id)
            .ok_or_else(|| RuntimeError::UnknownSequence(sequence_id.0.clone()))
    }

    fn state_mut(&mut self, sequence_id: &SequenceId) -> Result<&mut SequenceState, RuntimeError> {
        self.sequences
            .get_mut(sequence_id)
            .ok_or_else(|| RuntimeError::UnknownSequence(sequence_id.0.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::ReadLease;
    use crate::types::{IdentityScope, KvBlockId};

    fn block() -> KvBlockId {
        KvBlockId {
            scope: IdentityScope {
                tenant_id: "tenant".into(),
                model_id: "model".into(),
                tokenizer_id: "tokenizer".into(),
                adapter_id: None,
            },
            prefix_hash: "prefix".into(),
            block_hash: "block".into(),
            layer_id: 0,
            block_index: 0,
            token_count: 4,
        }
    }

    fn install(runtime: &mut SequenceRuntime, sequence: &SequenceId, expires: u64) {
        let object = PoolObjectRef {
            pool_id: "pool".into(),
            object_key: "object".into(),
            generation: 1,
            layout_digest: "layout".into(),
            checksum: None,
        };
        runtime
            .install_pages(
                sequence,
                ReadLease {
                    lease_id: "lease".into(),
                    pool_id: "pool".into(),
                    expires_at_unix_us: expires,
                    objects: vec![object.clone()],
                },
                vec![(
                    SequenceBlockRef {
                        sequence_id: sequence.clone(),
                        layer_id: 0,
                        logical_block: 0,
                        block_id: block(),
                        version: 1,
                    },
                    object,
                )],
            )
            .unwrap();
    }

    #[test]
    fn expired_lease_blocks_step_before_attention_runs() {
        let mut runtime = SequenceRuntime::default();
        let sequence = SequenceId("sequence".into());
        runtime
            .open_sequence(sequence.clone(), WorkerId("engine".into()), 4)
            .unwrap();
        install(&mut runtime, &sequence, 100);
        assert_eq!(
            runtime.begin_step(&sequence, 100, vec![]).unwrap_err(),
            RuntimeError::LeaseExpired("lease".into())
        );
    }

    #[test]
    fn tail_seals_exactly_at_block_boundary() {
        let mut runtime = SequenceRuntime::default();
        let sequence = SequenceId("sequence".into());
        runtime
            .open_sequence(sequence.clone(), WorkerId("engine".into()), 4)
            .unwrap();
        let plan = runtime.begin_step(&sequence, 0, vec![]).unwrap();
        assert_eq!(
            runtime.commit_step(plan, 4).unwrap(),
            TailCommit::Sealed { block_index: 0 }
        );
    }

    #[test]
    fn only_one_step_may_mutate_a_sequence() {
        let mut runtime = SequenceRuntime::default();
        let sequence = SequenceId("sequence".into());
        runtime
            .open_sequence(sequence.clone(), WorkerId("engine".into()), 4)
            .unwrap();
        let plan = runtime.begin_step(&sequence, 0, vec![]).unwrap();
        assert_eq!(
            runtime.begin_step(&sequence, 0, vec![]).unwrap_err(),
            RuntimeError::StepAlreadyActive("sequence".into())
        );
        runtime.abort_step(&plan).unwrap();
    }

    #[test]
    fn builds_generation_pinned_kv_view_from_installed_pages() {
        let mut runtime = SequenceRuntime::default();
        let sequence = SequenceId("sequence".into());
        runtime
            .open_sequence(sequence.clone(), WorkerId("engine".into()), 4)
            .unwrap();
        install(&mut runtime, &sequence, 100);

        let view = runtime.kv_view(&sequence, 0).unwrap();
        assert_eq!(view.blocks, vec![block()]);
        assert_eq!(view.page_table_generation, 1);
        assert_eq!(view.lease_ids, vec!["lease"]);
    }
}
