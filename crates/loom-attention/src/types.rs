//! Stable, dependency-light types shared by Loom services and adapters.
//!
//! This module deliberately knows nothing about Mooncake, vLLM, CUDA, Holt, or
//! any wire protocol. Physical implementations live behind outer-layer traits.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SequenceId(pub String);

impl From<&str> for SequenceId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WorkerId(pub String);

impl From<&str> for WorkerId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IdentityScope {
    pub tenant_id: String,
    pub model_id: String,
    pub tokenizer_id: String,
    pub adapter_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DType {
    Fp32,
    Fp16,
    Bf16,
    Fp8E4M3,
    Int8,
}

impl DType {
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Fp8E4M3 | Self::Int8 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    Mha,
    Gqa,
    Mla,
    Sparse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Rocm,
    Npu,
    NearStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryDomain {
    LocalHbm,
    RemoteHbm,
    HostDram,
    LocalSsd,
    ObjectStore,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvLayout {
    pub attention_kind: AttentionKind,
    pub dtype: DType,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_tokens: u32,
    pub tensor_parallel_rank: u32,
    pub tensor_parallel_size: u32,
    /// Engine-defined digest over physical stride/order and RoPE conventions.
    pub layout_digest: String,
}

impl KvLayout {
    pub fn block_bytes(&self) -> u64 {
        2_u64
            * u64::from(self.num_kv_heads)
            * u64::from(self.head_dim)
            * u64::from(self.block_tokens)
            * self.dtype.bytes_per_element() as u64
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct KvBlockId {
    pub scope: IdentityScope,
    pub prefix_hash: String,
    pub block_hash: String,
    pub layer_id: u32,
    pub block_index: u32,
    pub token_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceBlockRef {
    pub sequence_id: SequenceId,
    pub layer_id: u32,
    pub logical_block: u32,
    pub block_id: KvBlockId,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolObjectRef {
    pub pool_id: String,
    pub object_key: String,
    pub generation: u64,
    pub layout_digest: String,
    pub checksum: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaState {
    Allocating,
    Writing,
    Ready,
    Evicting,
    Deleted,
}

impl ReplicaState {
    pub const fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Allocating, Self::Writing)
                | (Self::Writing, Self::Ready)
                | (Self::Ready, Self::Evicting)
                | (Self::Evicting, Self::Deleted)
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalReplica {
    pub object: PoolObjectRef,
    pub worker_id: WorkerId,
    pub memory_domain: MemoryDomain,
    pub bytes: u64,
    pub worker_epoch: u64,
    pub state: ReplicaState,
    /// Ephemeral pool-defined handle. It is never written to the persistent catalog.
    pub opaque_handle: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorHandle {
    pub owner: WorkerId,
    pub device_kind: DeviceKind,
    pub device_index: u32,
    pub address: u64,
    pub bytes: u64,
    pub registration_key: Option<String>,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeCapabilities {
    pub worker_id: WorkerId,
    pub device_kind: DeviceKind,
    pub memory_domains: Vec<MemoryDomain>,
    pub attention_kinds: Vec<AttentionKind>,
    pub dtypes: Vec<DType>,
    pub head_sizes: Vec<u32>,
    pub page_sizes: Vec<u32>,
    pub supports_partial_softmax: bool,
    pub supports_graph_capture: bool,
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_layout_reports_k_and_v_bytes() {
        let layout = KvLayout {
            attention_kind: AttentionKind::Gqa,
            dtype: DType::Bf16,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_tokens: 16,
            tensor_parallel_rank: 0,
            tensor_parallel_size: 1,
            layout_digest: "gqa-bf16".into(),
        };
        assert_eq!(layout.block_bytes(), 65_536);
    }

    #[test]
    fn replica_state_machine_rejects_ready_to_deleted_jump() {
        assert!(ReplicaState::Writing.can_transition_to(ReplicaState::Ready));
        assert!(!ReplicaState::Ready.can_transition_to(ReplicaState::Deleted));
    }
}
