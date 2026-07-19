//! Contract between QuillCache and external KV pools such as Mooncake.
//!
//! Pools own sealed object allocation, placement, eviction, replication, and
//! durability. QuillCache consumes object references and short-lived leases.

use async_trait::async_trait;
use quillcache_types::{
    KvBlockId, KvLayout, MemoryDomain, PhysicalReplica, PoolObjectRef, TensorHandle, WorkerId,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PoolError {
    #[error("KV object was not found: {0}")]
    NotFound(String),
    #[error("KV object layout mismatch: {0}")]
    LayoutMismatch(String),
    #[error("KV read lease expired: {0}")]
    LeaseExpired(String),
    #[error("KV pool is unavailable: {0}")]
    Unavailable(String),
    #[error("KV pool rejected the operation: {0}")]
    Rejected(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolCapabilities {
    pub pool_id: String,
    pub memory_domains: Vec<MemoryDomain>,
    pub supports_events: bool,
    pub supports_leases: bool,
    pub supports_direct_device_stage: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedBlock {
    pub block_id: KvBlockId,
    pub layout: KvLayout,
    pub replicas: Vec<PhysicalReplica>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadLease {
    pub lease_id: String,
    pub pool_id: String,
    pub expires_at_unix_us: u64,
    pub objects: Vec<PoolObjectRef>,
}

impl ReadLease {
    pub fn valid_at(&self, now_unix_us: u64) -> bool {
        now_unix_us < self.expires_at_unix_us
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageRequest {
    pub lease: ReadLease,
    pub object: PoolObjectRef,
    pub destination: TensorHandle,
    pub deadline_unix_us: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageCompletion {
    pub completion_id: String,
    pub bytes: u64,
    pub source_domain: MemoryDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedBlock {
    pub block_id: KvBlockId,
    pub layout: KvLayout,
    pub source: TensorHandle,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolEventKind {
    ObjectReady,
    ReplicaAdded,
    ReplicaRemoved,
    ObjectDeleted,
    WorkerEpochChanged,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolEvent {
    pub pool_id: String,
    pub sequence: u64,
    pub kind: PoolEventKind,
    pub block_id: Option<KvBlockId>,
    pub object: Option<PoolObjectRef>,
    pub replica: Option<PhysicalReplica>,
    pub worker_id: Option<WorkerId>,
    pub worker_epoch: Option<u64>,
}

#[async_trait]
pub trait KvPool: std::fmt::Debug + Send + Sync {
    fn capabilities(&self) -> PoolCapabilities;

    async fn resolve(&self, blocks: &[KvBlockId]) -> Result<Vec<ResolvedBlock>, PoolError>;

    async fn acquire_read_lease(
        &self,
        objects: &[PoolObjectRef],
        expires_at_unix_us: u64,
    ) -> Result<ReadLease, PoolError>;

    async fn stage_into(&self, request: StageRequest) -> Result<StageCompletion, PoolError>;

    async fn publish_sealed(&self, block: SealedBlock) -> Result<PoolObjectRef, PoolError>;

    async fn release_lease(&self, lease: ReadLease) -> Result<(), PoolError>;

    /// Return events strictly after `after_sequence`, in storage-pool order.
    async fn events_since(&self, after_sequence: u64) -> Result<Vec<PoolEvent>, PoolError>;
}
