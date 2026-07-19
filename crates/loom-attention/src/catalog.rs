//! Derived KV catalog for routing and recovery.
//!
//! External pools remain authoritative for object lifetime. The hot directory
//! tracks live replicas from ordered events; Holt persists only stable object
//! references that can be revalidated with the pool after restart.

use crate::pool::{PoolEvent, PoolEventKind, ResolvedBlock};
use crate::types::{
    IdentityScope, KvBlockId, KvLayout, MemoryDomain, PhysicalReplica, PoolObjectRef, ReplicaState,
    WorkerId,
};
use holt::{Durability, RangeEntry, Tree, TreeBuilder};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use thiserror::Error;

const SEP: u8 = 0;

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("catalog storage failed: {0}")]
    Storage(String),
    #[error("catalog record could not be decoded: {0}")]
    Decode(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogReplica {
    pub worker_id: WorkerId,
    pub memory_domain: MemoryDomain,
    pub bytes: u64,
    pub worker_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogRecord {
    pub block_id: KvBlockId,
    pub object: PoolObjectRef,
    pub layout: KvLayout,
    pub replicas: Vec<CatalogReplica>,
    pub committed_at_unix_us: u64,
}

impl CatalogRecord {
    pub fn from_resolved(block: &ResolvedBlock, committed_at_unix_us: u64) -> Option<Self> {
        let object = block.replicas.first()?.object.clone();
        Some(Self {
            block_id: block.block_id.clone(),
            object,
            layout: block.layout.clone(),
            replicas: block
                .replicas
                .iter()
                .filter(|replica| replica.state == ReplicaState::Ready)
                .map(|replica| CatalogReplica {
                    worker_id: replica.worker_id.clone(),
                    memory_domain: replica.memory_domain,
                    bytes: replica.bytes,
                    worker_epoch: replica.worker_epoch,
                })
                .collect(),
            committed_at_unix_us,
        })
    }
}

pub trait PersistentCatalog: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn put(&mut self, record: CatalogRecord) -> Result<(), CatalogError>;
    fn get(&self, block: &KvBlockId) -> Result<Option<CatalogRecord>, CatalogError>;
    fn scan_prefix(
        &self,
        scope: &IdentityScope,
        prefix_hash: &str,
    ) -> Result<Vec<CatalogRecord>, CatalogError>;
    fn remove(&mut self, block: &KvBlockId) -> Result<bool, CatalogError>;
    fn flush(&self) -> Result<(), CatalogError>;
}

#[derive(Debug, Default)]
pub struct MemoryCatalog {
    records: BTreeMap<KvBlockId, CatalogRecord>,
}

impl PersistentCatalog for MemoryCatalog {
    fn name(&self) -> &str {
        "memory"
    }

    fn put(&mut self, record: CatalogRecord) -> Result<(), CatalogError> {
        self.records.insert(record.block_id.clone(), record);
        Ok(())
    }

    fn get(&self, block: &KvBlockId) -> Result<Option<CatalogRecord>, CatalogError> {
        Ok(self.records.get(block).cloned())
    }

    fn scan_prefix(
        &self,
        scope: &IdentityScope,
        prefix_hash: &str,
    ) -> Result<Vec<CatalogRecord>, CatalogError> {
        Ok(self
            .records
            .values()
            .filter(|record| {
                record.block_id.scope == *scope && record.block_id.prefix_hash == prefix_hash
            })
            .cloned()
            .collect())
    }

    fn remove(&mut self, block: &KvBlockId) -> Result<bool, CatalogError> {
        Ok(self.records.remove(block).is_some())
    }

    fn flush(&self) -> Result<(), CatalogError> {
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct HotResidencyDirectory {
    records: HashMap<KvBlockId, Vec<PhysicalReplica>>,
    worker_epochs: HashMap<WorkerId, u64>,
    last_event_sequences: HashMap<String, u64>,
}

impl HotResidencyDirectory {
    pub fn apply(&mut self, event: PoolEvent) {
        let last_sequence = self
            .last_event_sequences
            .entry(event.pool_id.clone())
            .or_default();
        if event.sequence <= *last_sequence {
            return;
        }
        *last_sequence = event.sequence;
        match event.kind {
            PoolEventKind::ObjectReady | PoolEventKind::ReplicaAdded => {
                if let (Some(block), Some(replica)) = (event.block_id, event.replica) {
                    if replica.state == ReplicaState::Ready {
                        let replicas = self.records.entry(block).or_default();
                        replicas.retain(|current| {
                            current.worker_id != replica.worker_id
                                || current.memory_domain != replica.memory_domain
                        });
                        replicas.push(replica);
                    }
                }
            }
            PoolEventKind::ReplicaRemoved => {
                if let (Some(block), Some(replica)) = (event.block_id, event.replica) {
                    if let Some(replicas) = self.records.get_mut(&block) {
                        replicas.retain(|current| {
                            current.worker_id != replica.worker_id
                                || current.memory_domain != replica.memory_domain
                        });
                        if replicas.is_empty() {
                            self.records.remove(&block);
                        }
                    }
                }
            }
            PoolEventKind::ObjectDeleted => {
                if let Some(block) = event.block_id {
                    self.records.remove(&block);
                }
            }
            PoolEventKind::WorkerEpochChanged => {
                if let (Some(worker), Some(epoch)) = (event.worker_id, event.worker_epoch) {
                    self.worker_epochs.insert(worker.clone(), epoch);
                    for replicas in self.records.values_mut() {
                        replicas.retain(|replica| {
                            replica.worker_id != worker || replica.worker_epoch == epoch
                        });
                    }
                    self.records.retain(|_, replicas| !replicas.is_empty());
                }
            }
        }
    }

    pub fn locate(&self, block: &KvBlockId) -> &[PhysicalReplica] {
        self.records.get(block).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn last_event_sequence(&self, pool_id: &str) -> u64 {
        self.last_event_sequences
            .get(pool_id)
            .copied()
            .unwrap_or_default()
    }
}

pub struct HoltCatalog {
    tree: Tree,
    dir: PathBuf,
}

impl std::fmt::Debug for HoltCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HoltCatalog")
            .field("dir", &self.dir)
            .finish()
    }
}

impl HoltCatalog {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, CatalogError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|error| CatalogError::Storage(error.to_string()))?;
        let tree = TreeBuilder::new(dir.join("catalog.holt"))
            .durability(Durability::Wal { sync: false })
            .open()
            .map_err(|error| CatalogError::Storage(error.to_string()))?;
        Ok(Self { tree, dir })
    }

    fn encode_scope(buffer: &mut Vec<u8>, scope: &IdentityScope) {
        for part in [
            scope.tenant_id.as_str(),
            scope.model_id.as_str(),
            scope.tokenizer_id.as_str(),
            scope.adapter_id.as_deref().unwrap_or(""),
        ] {
            buffer.extend_from_slice(part.as_bytes());
            buffer.push(SEP);
        }
    }

    fn prefix_key(scope: &IdentityScope, prefix_hash: &str) -> Vec<u8> {
        let mut key = vec![1];
        Self::encode_scope(&mut key, scope);
        key.extend_from_slice(prefix_hash.as_bytes());
        key.push(SEP);
        key
    }

    fn block_key(block: &KvBlockId) -> Vec<u8> {
        let mut key = Self::prefix_key(&block.scope, &block.prefix_hash);
        key.extend_from_slice(&block.layer_id.to_be_bytes());
        key.extend_from_slice(&block.block_index.to_be_bytes());
        key.push(SEP);
        key.extend_from_slice(block.block_hash.as_bytes());
        key.push(SEP);
        key
    }

    fn scan(&self, prefix: &[u8]) -> Result<Vec<CatalogRecord>, CatalogError> {
        let mut records = Vec::new();
        for entry in self.tree.scan(prefix) {
            let entry = entry.map_err(|error| CatalogError::Storage(error.to_string()))?;
            if let RangeEntry::Key { value, .. } = entry {
                records.push(
                    serde_json::from_slice(&value)
                        .map_err(|error| CatalogError::Decode(error.to_string()))?,
                );
            }
        }
        Ok(records)
    }
}

impl PersistentCatalog for HoltCatalog {
    fn name(&self) -> &str {
        "holt"
    }

    fn put(&mut self, record: CatalogRecord) -> Result<(), CatalogError> {
        let key = Self::block_key(&record.block_id);
        let value =
            serde_json::to_vec(&record).map_err(|error| CatalogError::Decode(error.to_string()))?;
        self.tree
            .put(&key, &value)
            .map_err(|error| CatalogError::Storage(error.to_string()))?;
        Ok(())
    }

    fn get(&self, block: &KvBlockId) -> Result<Option<CatalogRecord>, CatalogError> {
        Ok(self.scan(&Self::block_key(block))?.into_iter().next())
    }

    fn scan_prefix(
        &self,
        scope: &IdentityScope,
        prefix_hash: &str,
    ) -> Result<Vec<CatalogRecord>, CatalogError> {
        self.scan(&Self::prefix_key(scope, prefix_hash))
    }

    fn remove(&mut self, block: &KvBlockId) -> Result<bool, CatalogError> {
        let key = Self::block_key(block);
        self.tree
            .delete(&key)
            .map_err(|error| CatalogError::Storage(error.to_string()))
    }

    fn flush(&self) -> Result<(), CatalogError> {
        self.tree
            .checkpoint()
            .map_err(|error| CatalogError::Storage(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AttentionKind, DType};

    fn record(prefix: &str, block_hash: &str) -> CatalogRecord {
        let scope = IdentityScope {
            tenant_id: "tenant".into(),
            model_id: "model".into(),
            tokenizer_id: "tokenizer".into(),
            adapter_id: None,
        };
        CatalogRecord {
            block_id: KvBlockId {
                scope,
                prefix_hash: prefix.into(),
                block_hash: block_hash.into(),
                layer_id: 0,
                block_index: 0,
                token_count: 16,
            },
            object: PoolObjectRef {
                pool_id: "pool".into(),
                object_key: block_hash.into(),
                generation: 1,
                layout_digest: "layout".into(),
                checksum: None,
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
            replicas: vec![],
            committed_at_unix_us: 1,
        }
    }

    #[test]
    fn memory_catalog_scans_identity_scoped_prefix() {
        let mut catalog = MemoryCatalog::default();
        let first = record("prefix", "a");
        let second = record("other", "b");
        catalog.put(first.clone()).unwrap();
        catalog.put(second).unwrap();
        let found = catalog
            .scan_prefix(&first.block_id.scope, "prefix")
            .unwrap();
        assert_eq!(found, vec![first]);
    }

    #[test]
    fn holt_catalog_recovers_prefix_records() {
        let path = std::env::temp_dir().join(format!("loom-catalog-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        let first = record("prefix", "a");
        {
            let mut catalog = HoltCatalog::open(&path).unwrap();
            catalog.put(first.clone()).unwrap();
            catalog.flush().unwrap();
        }
        let catalog = HoltCatalog::open(&path).unwrap();
        assert_eq!(catalog.get(&first.block_id).unwrap(), Some(first));
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn hot_directory_orders_events_independently_per_pool() {
        let first = record("prefix", "a");
        let second = record("prefix", "b");
        let replica = |record: &CatalogRecord| PhysicalReplica {
            object: record.object.clone(),
            worker_id: WorkerId("worker".into()),
            memory_domain: MemoryDomain::RemoteHbm,
            bytes: record.layout.block_bytes(),
            worker_epoch: 1,
            state: ReplicaState::Ready,
            opaque_handle: Some("ephemeral".into()),
        };
        let event = |pool_id: &str, sequence: u64, record: &CatalogRecord| PoolEvent {
            pool_id: pool_id.into(),
            sequence,
            kind: PoolEventKind::ObjectReady,
            block_id: Some(record.block_id.clone()),
            object: Some(record.object.clone()),
            replica: Some(replica(record)),
            worker_id: None,
            worker_epoch: None,
        };

        let mut directory = HotResidencyDirectory::default();
        directory.apply(event("pool-a", 2, &first));
        directory.apply(event("pool-a", 1, &second));
        directory.apply(event("pool-b", 1, &second));

        assert_eq!(directory.locate(&first.block_id).len(), 1);
        assert_eq!(directory.locate(&second.block_id).len(), 1);
        assert_eq!(directory.last_event_sequence("pool-a"), 2);
        assert_eq!(directory.last_event_sequence("pool-b"), 1);
    }
}
