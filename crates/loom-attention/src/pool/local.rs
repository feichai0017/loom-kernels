//! Deterministic metadata-only pool used by CI and runtime unit tests.
//!
//! It models object generations, read leases, and ordered events. Tensor bytes
//! stay behind opaque handles, matching the external-pool ownership boundary.

use super::{
    KvPool, PoolCapabilities, PoolError, PoolEvent, PoolEventKind, ReadLease, ResolvedBlock,
    SealedBlock, StageCompletion, StageRequest,
};
use crate::types::{
    KvBlockId, MemoryDomain, PhysicalReplica, PoolObjectRef, ReplicaState, WorkerId,
};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
struct LocalState {
    blocks: HashMap<KvBlockId, ResolvedBlock>,
    events: Vec<PoolEvent>,
    live_leases: HashSet<String>,
}

#[derive(Debug)]
pub struct LocalKvPool {
    pool_id: String,
    worker_id: WorkerId,
    next_id: AtomicU64,
    state: Mutex<LocalState>,
}

impl LocalKvPool {
    pub fn new(pool_id: impl Into<String>, worker_id: WorkerId) -> Self {
        Self {
            pool_id: pool_id.into(),
            worker_id,
            next_id: AtomicU64::new(1),
            state: Mutex::new(LocalState {
                blocks: HashMap::new(),
                events: Vec::new(),
                live_leases: HashSet::new(),
            }),
        }
    }

    fn allocate_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    fn push_event(&self, state: &mut LocalState, mut event: PoolEvent) {
        event.sequence = state.events.len() as u64 + 1;
        state.events.push(event);
    }
}

#[async_trait]
impl KvPool for LocalKvPool {
    fn capabilities(&self) -> PoolCapabilities {
        PoolCapabilities {
            pool_id: self.pool_id.clone(),
            memory_domains: vec![MemoryDomain::HostDram],
            supports_events: true,
            supports_leases: true,
            supports_direct_device_stage: false,
        }
    }

    async fn resolve(&self, blocks: &[KvBlockId]) -> Result<Vec<ResolvedBlock>, PoolError> {
        let state = self.state.lock().unwrap();
        blocks
            .iter()
            .map(|block| {
                state
                    .blocks
                    .get(block)
                    .cloned()
                    .ok_or_else(|| PoolError::NotFound(block.block_hash.clone()))
            })
            .collect()
    }

    async fn acquire_read_lease(
        &self,
        objects: &[PoolObjectRef],
        expires_at_unix_us: u64,
    ) -> Result<ReadLease, PoolError> {
        let mut state = self.state.lock().unwrap();
        for object in objects {
            let found = state.blocks.values().any(|block| {
                block.replicas.iter().any(|replica| {
                    replica.object == *object && replica.state == ReplicaState::Ready
                })
            });
            if !found {
                return Err(PoolError::NotFound(object.object_key.clone()));
            }
        }
        let lease_id = self.allocate_id("lease");
        state.live_leases.insert(lease_id.clone());
        Ok(ReadLease {
            lease_id,
            pool_id: self.pool_id.clone(),
            expires_at_unix_us,
            objects: objects.to_vec(),
        })
    }

    async fn stage_into(&self, request: StageRequest) -> Result<StageCompletion, PoolError> {
        let state = self.state.lock().unwrap();
        if !state.live_leases.contains(&request.lease.lease_id) {
            return Err(PoolError::LeaseExpired(request.lease.lease_id));
        }
        let replica = state
            .blocks
            .values()
            .flat_map(|block| &block.replicas)
            .find(|replica| replica.object == request.object)
            .ok_or_else(|| PoolError::NotFound(request.object.object_key.clone()))?;
        Ok(StageCompletion {
            completion_id: self.allocate_id("stage"),
            bytes: replica.bytes.min(request.destination.bytes),
            source_domain: replica.memory_domain,
        })
    }

    async fn publish_sealed(&self, block: SealedBlock) -> Result<PoolObjectRef, PoolError> {
        let object = PoolObjectRef {
            pool_id: self.pool_id.clone(),
            object_key: format!(
                "{}/{}/layer-{}/block-{}",
                block.block_id.scope.model_id,
                block.block_id.block_hash,
                block.block_id.layer_id,
                block.block_id.block_index
            ),
            generation: self.next_id.fetch_add(1, Ordering::Relaxed),
            layout_digest: block.layout.layout_digest.clone(),
            checksum: block.checksum.clone(),
        };
        let replica = PhysicalReplica {
            object: object.clone(),
            worker_id: self.worker_id.clone(),
            memory_domain: MemoryDomain::HostDram,
            bytes: block.layout.block_bytes(),
            worker_epoch: 1,
            state: ReplicaState::Ready,
            opaque_handle: Some(format!("local:0x{:x}", block.source.address)),
        };
        let resolved = ResolvedBlock {
            block_id: block.block_id.clone(),
            layout: block.layout,
            replicas: vec![replica.clone()],
        };

        let mut state = self.state.lock().unwrap();
        state.blocks.insert(block.block_id.clone(), resolved);
        self.push_event(
            &mut state,
            PoolEvent {
                pool_id: self.pool_id.clone(),
                sequence: 0,
                kind: PoolEventKind::ObjectReady,
                block_id: Some(block.block_id),
                object: Some(object.clone()),
                replica: Some(replica),
                worker_id: None,
                worker_epoch: None,
            },
        );
        Ok(object)
    }

    async fn release_lease(&self, lease: ReadLease) -> Result<(), PoolError> {
        self.state
            .lock()
            .unwrap()
            .live_leases
            .remove(&lease.lease_id);
        Ok(())
    }

    async fn events_since(&self, after_sequence: u64) -> Result<Vec<PoolEvent>, PoolError> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .events
            .iter()
            .filter(|event| event.sequence > after_sequence)
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::SealedBlock;
    use super::*;
    use crate::types::{
        AttentionKind, DType, DeviceKind, IdentityScope, KvBlockId, KvLayout, TensorHandle,
    };

    fn sealed() -> SealedBlock {
        SealedBlock {
            block_id: KvBlockId {
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
                token_count: 16,
            },
            layout: KvLayout {
                attention_kind: AttentionKind::Gqa,
                dtype: DType::Bf16,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                block_tokens: 16,
                tensor_parallel_rank: 0,
                tensor_parallel_size: 1,
                layout_digest: "layout".into(),
            },
            source: TensorHandle {
                owner: WorkerId("engine-0".into()),
                device_kind: DeviceKind::Cuda,
                device_index: 0,
                address: 0x1000,
                bytes: 65_536,
                registration_key: None,
                generation: 1,
            },
            checksum: None,
        }
    }

    #[tokio::test]
    async fn publish_resolve_and_lease_round_trip() {
        let pool = LocalKvPool::new("local", WorkerId("pool-0".into()));
        let block = sealed();
        let object = pool.publish_sealed(block.clone()).await.unwrap();
        let resolved = pool.resolve(&[block.block_id]).await.unwrap();
        assert_eq!(resolved[0].replicas[0].object, object);

        let lease = pool.acquire_read_lease(&[object], 10_000).await.unwrap();
        assert!(lease.valid_at(9_999));
        pool.release_lease(lease).await.unwrap();
        assert_eq!(pool.events_since(0).await.unwrap().len(), 1);
    }
}
