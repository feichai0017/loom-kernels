//! Store data model (Mooncake's `mooncake-store/include/types.h`).
//!
//! Object keys are **opaque strings** whose value is independent of the key
//! (Mooncake does not content-address — unlike Redis); the prefix-hash chaining
//! that maps KV-cache blocks to keys lives in the integration / connector layer.

use serde::{Deserialize, Serialize};

/// An opaque object key (Mooncake's `ObjectKey`).
pub type ObjectKey = String;
/// A mounted RAM segment's name (Mooncake's segment name).
pub type SegmentName = String;
/// A replica's id within an object's [`crate::replica::ReplicaList`].
pub type ReplicaId = u32;

/// A contiguous byte range to read into / write from (Mooncake's `Slice`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Slice {
    pub offset: u64,
    pub size: u64,
}

/// How an object should be replicated (Mooncake's `ReplicateConfig`): how many
/// copies, and whether to soft-/hard-pin them against eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicateConfig {
    pub replica_num: usize,
    pub with_soft_pin: bool,
    pub with_hard_pin: bool,
    pub preferred_segment: Option<SegmentName>,
}

impl Default for ReplicateConfig {
    fn default() -> Self {
        Self {
            replica_num: 1,
            with_soft_pin: false,
            with_hard_pin: false,
            preferred_segment: None,
        }
    }
}

impl ReplicateConfig {
    /// `n` replicas, no pinning.
    pub fn replicas(n: usize) -> Self {
        Self {
            replica_num: n.max(1),
            ..Default::default()
        }
    }
}

/// Store error codes (a representative subset of Mooncake's `ErrorCode`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ErrorCode {
    #[error("object not found")]
    ObjectNotFound,
    #[error("object already exists")]
    ObjectAlreadyExists,
    #[error("object is being written (not yet readable)")]
    ObjectNotReady,
    #[error("segment not found")]
    SegmentNotFound,
    #[error("no segment has enough free space")]
    NoAvailableSegment,
    #[error("buffer overflow")]
    BufferOverflow,
    #[error("invalid replica")]
    InvalidReplica,
    /// QuillCache's identity guard: the object is resident but was written by a
    /// different identity, so serving it would be a cross-tenant leak or a
    /// cross-adapter/model correctness error. Mooncake keys are identity-agnostic;
    /// this variant is our addition.
    #[error("unsafe cross-identity reuse refused ({0:?})")]
    UnsafeReuse(quillcache_core::ReuseViolation),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replicate_config_defaults_to_one_unpinned_replica() {
        let cfg = ReplicateConfig::default();
        assert_eq!(cfg.replica_num, 1);
        assert!(!cfg.with_soft_pin && !cfg.with_hard_pin);
        assert_eq!(ReplicateConfig::replicas(3).replica_num, 3);
        // `replicas(0)` is clamped to a usable single replica.
        assert_eq!(ReplicateConfig::replicas(0).replica_num, 1);
    }
}
