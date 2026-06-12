//! Replica model (Mooncake's `mooncake-store/include/replica.h`). An object's
//! bytes can live in several places at once; each is a [`Replica`]. A replica is
//! either in a mounted RAM segment ([`ReplicaData::Memory`]) or on a node's
//! durable disk tier ([`ReplicaData::Disk`]) — the latter is QuillCache's
//! crash-consistent persistent tier (Mooncake's pool is volatile DRAM).

use crate::allocator::AllocatedBuffer;
use crate::types::ReplicaId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A replica's lifecycle (Mooncake's `ReplicaStatus`). The two-phase `Put`
/// drives `Initialized` → (client writes) → `Complete`; `Processing` is the
/// in-flight window, `Failed`/`Removed` the terminal aborts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaStatus {
    Undefined,
    Initialized,
    Processing,
    Complete,
    Removed,
    Failed,
}

/// Where a replica's bytes live (Mooncake's `MemoryReplicaData` /
/// `DiskReplicaData` variant).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaData {
    /// In a mounted RAM segment, at an allocated buffer.
    Memory(AllocatedBuffer),
    /// On a node's durable SSD tier — file-backed, crash-consistent (the
    /// QuillCache persistent tier, addressed by the owning node + path).
    Disk { file_path: String, object_size: u64 },
}

impl ReplicaData {
    pub fn segment_name(&self) -> Option<&str> {
        match self {
            ReplicaData::Memory(buffer) => Some(buffer.segment_name()),
            ReplicaData::Disk { .. } => None,
        }
    }

    pub fn size(&self) -> u64 {
        match self {
            ReplicaData::Memory(buffer) => buffer.size,
            ReplicaData::Disk { object_size, .. } => *object_size,
        }
    }

    pub fn is_disk(&self) -> bool {
        matches!(self, ReplicaData::Disk { .. })
    }
}

/// One copy of an object (Mooncake's `Replica`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replica {
    pub id: ReplicaId,
    pub status: ReplicaStatus,
    pub ref_count: u32,
    pub data: ReplicaData,
}

impl Replica {
    /// A freshly-allocated replica, before the client has written its bytes.
    pub fn new(id: ReplicaId, data: ReplicaData) -> Self {
        Self {
            id,
            status: ReplicaStatus::Initialized,
            ref_count: 0,
            data,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.status == ReplicaStatus::Complete
    }

    pub fn segment_name(&self) -> Option<&str> {
        self.data.segment_name()
    }

    pub fn size(&self) -> u64 {
        self.data.size()
    }
}

/// All replicas of one object (Mooncake's `ReplicaList`).
pub type ReplicaList = HashMap<ReplicaId, Replica>;

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(id: ReplicaId, off: u64) -> Replica {
        Replica::new(
            id,
            ReplicaData::Memory(AllocatedBuffer {
                segment_name: "seg-0".into(),
                offset: off,
                size: 64,
            }),
        )
    }

    #[test]
    fn replica_starts_initialized_and_completes() {
        let mut r = mem(0, 0);
        assert_eq!(r.status, ReplicaStatus::Initialized);
        assert!(!r.is_complete());
        assert_eq!(r.segment_name(), Some("seg-0"));
        assert_eq!(r.size(), 64);
        r.status = ReplicaStatus::Complete;
        assert!(r.is_complete());
    }

    #[test]
    fn replica_round_trips_through_serde() {
        let r = mem(7, 128);
        let json = serde_json::to_string(&r).unwrap();
        let back: Replica = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, 7);
        assert_eq!(back.segment_name(), Some("seg-0"));
        // A disk replica (the crash-consistent tier) also round-trips.
        let disk = Replica::new(
            1,
            ReplicaData::Disk {
                file_path: "/pool/obj-1".into(),
                object_size: 4096,
            },
        );
        let back: Replica = serde_json::from_str(&serde_json::to_string(&disk).unwrap()).unwrap();
        assert!(back.data.is_disk());
        assert_eq!(back.size(), 4096);
    }
}
