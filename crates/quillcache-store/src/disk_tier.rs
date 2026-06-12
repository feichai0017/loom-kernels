//! Crash-consistent durable tier for the store's `Disk` replicas — QuillCache's
//! persistent, immediately-reusable tier. **Mooncake's pool is volatile DRAM**,
//! so a durable replica surviving a process restart is our 2nd differentiator
//! (the seam from the novelty positioning: the safety invariant held across
//! tiering / eviction / crash-recovery).
//!
//! It reuses [`LocalKvStore`]'s SSD tier — object-first atomic publish (file
//! fsynced, then a single WAL commit fsynced) + recover-on-reopen + the identity
//! guard. An `(ObjectKey, IdentityScope)` is mapped to the store's
//! content-hash-addressed [`KvBlockKey`] so the *same* guard rejects a
//! cross-identity read here too — even after recovery.

use crate::replica::ReplicaData;
use crate::types::{ErrorCode, ObjectKey};
use crate::{LocalKvStore, StoreError};
use bytes::Bytes;
use quillcache_core::{IdentityScope, KvBlockKey};
use std::path::Path;

/// Map an object key + the writing identity to the content-hash-addressed key
/// the durable store guards on (block_hash = the object key; identity carried in
/// the model/tokenizer/adapter/tenant fields).
fn durable_key(key: &ObjectKey, scope: &IdentityScope) -> KvBlockKey {
    KvBlockKey {
        model_id: scope.model_id.clone(),
        tokenizer_id: scope.tokenizer_id.clone(),
        adapter_id: scope.adapter_id.clone(),
        tenant_id: scope.tenant_id.clone(),
        prefix_hash: String::new(),
        block_hash: key.clone(),
        block_index: 0,
        token_count: 0,
    }
}

/// A node's durable disk tier, backing [`ReplicaData::Disk`].
#[derive(Debug)]
pub struct DiskTier {
    store: LocalKvStore,
}

impl DiskTier {
    /// Open a fresh durable tier backed by `dir` (a clean WAL).
    pub fn open(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        // dram_capacity 0 so every put is demoted to the SSD tier immediately
        // (durable); ssd_capacity unbounded (the tier policy lives in the master).
        Ok(Self {
            store: LocalKvStore::new(dir.as_ref(), 0, u64::MAX)?,
        })
    }

    /// Reopen a durable tier, recovering committed objects from the WAL — the
    /// crash-recovery path (half-written / corrupted objects are never served).
    pub fn recover(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self {
            store: LocalKvStore::recover(dir.as_ref(), 0, u64::MAX)?,
        })
    }

    /// Durably store an object's bytes and return a [`ReplicaData::Disk`]. The
    /// write is object-first: the block file is fsynced, then a single WAL commit
    /// is fsynced (the atomic publish point) — a crash before the commit leaves an
    /// orphan file that is never served.
    pub fn put(
        &mut self,
        key: &ObjectKey,
        scope: &IdentityScope,
        data: Bytes,
    ) -> Result<ReplicaData, ErrorCode> {
        let object_size = data.len() as u64;
        let durable = durable_key(key, scope);
        self.store.put_dram(durable.clone(), data);
        self.store
            .demote_to_ssd(&durable)
            .map_err(|e| ErrorCode::Io(e.to_string()))?;
        Ok(ReplicaData::Disk {
            file_path: format!("durable:{key}"),
            object_size,
        })
    }

    /// Read a durable object's bytes, identity-guarded (a cross-identity read is
    /// refused with [`ErrorCode::UnsafeReuse`], the same as in memory).
    pub fn get(&mut self, key: &ObjectKey, scope: &IdentityScope) -> Result<Bytes, ErrorCode> {
        match self.store.get(&durable_key(key, scope)) {
            Ok(bytes) => Ok(bytes),
            Err(StoreError::Unsafe(violation)) => Err(ErrorCode::UnsafeReuse(violation)),
            Err(_) => Err(ErrorCode::ObjectNotFound),
        }
    }

    /// Number of durable objects.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut dir = std::env::temp_dir();
        dir.push(format!("qc-disk-tier-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn scope(tenant: &str) -> IdentityScope {
        IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: tenant.into(),
        }
    }

    #[test]
    fn durable_object_survives_restart_and_stays_identity_guarded() {
        let dir = tmp("survive");
        let owner = scope("ten-a");
        let data = Bytes::from_static(b"durable-kv-bytes-on-the-ssd-tier");
        let key = "obj-1".to_string();

        // Write durably, then drop the tier — simulating process death (the block
        // file + the WAL commit remain on disk).
        {
            let mut tier = DiskTier::open(&dir).unwrap();
            let replica = tier.put(&key, &owner, data.clone()).unwrap();
            assert!(matches!(replica, ReplicaData::Disk { .. }));
            assert_eq!(&tier.get(&key, &owner).unwrap()[..], &data[..]);
        }

        // Reopen + recover from the WAL: the object SURVIVES the restart.
        let mut tier = DiskTier::recover(&dir).unwrap();
        assert_eq!(tier.len(), 1);
        assert_eq!(&tier.get(&key, &owner).unwrap()[..], &data[..]);

        // The identity guard holds after recovery: tenant-b is refused the same
        // content key (a cross-tenant durable read is a privacy leak).
        assert!(matches!(
            tier.get(&key, &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
