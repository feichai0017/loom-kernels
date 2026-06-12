//! Store clients (Mooncake's `mooncake-store` `Client` — `DummyClient` /
//! `RealClient`). A client drives the [`MasterService`] two-phase Put/Get and
//! moves the object bytes to/from the master-allocated `(segment, offset)`.
//!
//! - [`DummyClient`] — in-process: the segments are local byte arenas the client
//!   reads/writes directly (no transfer engine). Mooncake's first target.
//! - [`RealClient`] — distributed: the segments are
//!   [`quillcache_transfer_engine::TransferEngine`] RAM segments on (possibly
//!   remote) storage nodes; bytes move over the transfer engine. This is the
//!   real end-to-end Put→Get — `put_start` → transfer WRITE → `put_end`, and
//!   `get_replica_list` → transfer READ — with the identity guard enforced at
//!   the metadata layer *before* any byte moves.

use crate::allocator::AllocatedBuffer;
use crate::master_service::MasterService;
use crate::replica::{Replica, ReplicaData};
use crate::types::{ErrorCode, ReplicateConfig};
use bytes::Bytes;
use quillcache_core::IdentityScope;
use quillcache_transfer_engine::{OpCode, TransferEngine, TransferRequest};
use std::collections::HashMap;
use std::sync::Arc;

/// The `(segment, offset, size)` of the first complete in-memory replica.
fn first_memory_buffer(replicas: &[Replica]) -> Result<AllocatedBuffer, ErrorCode> {
    replicas
        .iter()
        .find_map(|r| match &r.data {
            ReplicaData::Memory(buffer) => Some(buffer.clone()),
            _ => None,
        })
        .ok_or(ErrorCode::InvalidReplica)
}

/// In-process client: segments are local byte arenas (Mooncake's `DummyClient`).
#[derive(Debug)]
pub struct DummyClient {
    master: MasterService,
    arenas: HashMap<String, Vec<u8>>,
}

impl DummyClient {
    pub fn new(strategy: &str) -> Self {
        Self {
            master: MasterService::new(strategy),
            arenas: HashMap::new(),
        }
    }

    pub fn mount(&mut self, name: &str, capacity: u64) {
        self.master.mount_segment(name, capacity);
        self.arenas
            .insert(name.to_string(), vec![0u8; capacity as usize]);
    }

    /// Two-phase Put: allocate replicas, write the bytes into each, commit.
    pub fn put(
        &mut self,
        key: &str,
        identity: IdentityScope,
        data: &[u8],
        config: &ReplicateConfig,
    ) -> Result<(), ErrorCode> {
        let buffers =
            self.master
                .put_start(key.to_string(), identity, data.len() as u64, config)?;
        for buffer in &buffers {
            let arena = self
                .arenas
                .get_mut(&buffer.segment_name)
                .ok_or(ErrorCode::SegmentNotFound)?;
            let (start, end) = (
                buffer.offset as usize,
                (buffer.offset + buffer.size) as usize,
            );
            arena[start..end].copy_from_slice(data);
        }
        self.master.put_end(key)
    }

    /// Identity-guarded Get: read the bytes from a complete replica's arena.
    pub fn get(&mut self, key: &str, identity: &IdentityScope) -> Result<Vec<u8>, ErrorCode> {
        let replicas = self.master.get_replica_list(key, identity)?;
        let buffer = first_memory_buffer(&replicas)?;
        let arena = self
            .arenas
            .get(&buffer.segment_name)
            .ok_or(ErrorCode::SegmentNotFound)?;
        Ok(arena[buffer.offset as usize..(buffer.offset + buffer.size) as usize].to_vec())
    }

    pub fn remove(&mut self, key: &str, force: bool) -> Result<(), ErrorCode> {
        self.master.remove(key, force)
    }

    pub fn exist(&self, key: &str) -> bool {
        self.master.exist_key(key)
    }

    pub fn master(&self) -> &MasterService {
        &self.master
    }
}

/// Distributed client: object bytes move over the transfer engine to/from the
/// master-allocated `(segment, offset)` on storage nodes (Mooncake's `RealClient`).
#[derive(Debug)]
pub struct RealClient {
    master: MasterService,
    engine: Arc<TransferEngine>,
}

impl RealClient {
    /// `engine` is this client's transfer engine (its RAM segment is scratch for
    /// the staged read/write buffers); the master decides which storage segment
    /// each replica lands on.
    pub fn new(strategy: &str, engine: Arc<TransferEngine>) -> Self {
        Self {
            master: MasterService::new(strategy),
            engine,
        }
    }

    /// Mount a storage segment by name + capacity. The name must match a storage
    /// node's published transfer-engine segment so the client can open it.
    pub fn mount(&mut self, name: &str, capacity: u64) {
        self.master.mount_segment(name, capacity);
    }

    /// Two-phase Put: allocate replicas, then for each, stage the bytes locally
    /// and WRITE them to the replica's `(segment, offset)` over the transfer
    /// engine; commit with `put_end`.
    pub async fn put(
        &mut self,
        key: &str,
        identity: IdentityScope,
        data: &[u8],
        config: &ReplicateConfig,
    ) -> Result<(), ErrorCode> {
        let buffers =
            self.master
                .put_start(key.to_string(), identity, data.len() as u64, config)?;
        for buffer in &buffers {
            let segment = self
                .engine
                .open_segment(&buffer.segment_name)
                .map_err(|_| ErrorCode::SegmentNotFound)?;
            let source = self.engine.register_local_memory(data);
            let batch = self.engine.allocate_batch_id(1);
            self.engine
                .submit_transfer(
                    batch,
                    vec![TransferRequest {
                        opcode: OpCode::Write,
                        source_offset: source,
                        target_id: segment,
                        target_offset: buffer.offset,
                        length: buffer.size,
                    }],
                )
                .await
                .map_err(|_| ErrorCode::BufferOverflow)?;
            self.engine.free_batch_id(batch);
        }
        self.master.put_end(key)
    }

    /// Identity-guarded Get: locate a replica (refused before any byte moves if
    /// the identity mismatches), then READ its bytes over the transfer engine.
    pub async fn get(&mut self, key: &str, identity: &IdentityScope) -> Result<Bytes, ErrorCode> {
        let replicas = self.master.get_replica_list(key, identity)?;
        let buffer = first_memory_buffer(&replicas)?;
        let segment = self
            .engine
            .open_segment(&buffer.segment_name)
            .map_err(|_| ErrorCode::SegmentNotFound)?;
        let dest = self.engine.register_zeroed(buffer.size as usize);
        let batch = self.engine.allocate_batch_id(1);
        self.engine
            .submit_transfer(
                batch,
                vec![TransferRequest {
                    opcode: OpCode::Read,
                    source_offset: dest,
                    target_id: segment,
                    target_offset: buffer.offset,
                    length: buffer.size,
                }],
            )
            .await
            .map_err(|_| ErrorCode::BufferOverflow)?;
        self.engine.free_batch_id(batch);
        Ok(self.engine.read_local(dest, buffer.size))
    }

    pub fn master(&self) -> &MasterService {
        &self.master
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_transfer_engine::{InMemoryMetadata, MetadataBackend};

    fn scope(tenant: &str) -> IdentityScope {
        IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: tenant.into(),
        }
    }

    #[test]
    fn dummy_client_put_get_is_identity_guarded() {
        let mut c = DummyClient::new("random");
        c.mount("seg-0", 1 << 16);
        c.mount("seg-1", 1 << 16);
        let id = scope("ten-a");
        let data = b"system-prompt-kv-bytes";

        // Two replicas, on distinct segments — both arenas hold the bytes.
        c.put("k", id.clone(), data, &ReplicateConfig::replicas(2))
            .unwrap();
        assert_eq!(&c.get("k", &id).unwrap()[..], data);

        // Same content key, different tenant → refused before any read.
        assert!(matches!(
            c.get("k", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));

        c.remove("k", true).unwrap();
        assert!(!c.exist("k"));
    }

    #[tokio::test]
    async fn real_client_put_get_moves_bytes_over_the_transfer_engine() {
        let md: Arc<dyn MetadataBackend> = Arc::new(InMemoryMetadata::new());
        // Storage node "B": its transfer-engine RAM segment is the pool.
        let _node_b = TransferEngine::init("B", md.clone(), "127.0.0.1:0")
            .await
            .unwrap();
        // Client "A": its own engine stages bytes + drives the transfers.
        let engine_a = TransferEngine::init("A", md.clone(), "127.0.0.1:0")
            .await
            .unwrap();
        let mut client = RealClient::new("random", engine_a);
        // The master places objects on segment "B" (node B's pool).
        client.mount("B", 1 << 16);

        let id = scope("ten-a");
        let data = b"real-bytes-over-the-transfer-engine";
        client
            .put("k", id.clone(), data, &ReplicateConfig::replicas(1))
            .await
            .unwrap();
        // Get reads the bytes back from node B over TCP.
        let got = client.get("k", &id).await.unwrap();
        assert_eq!(&got[..], data);

        // Cross-identity get is refused by the master — no bytes move.
        assert!(matches!(
            client.get("k", &scope("ten-b")).await,
            Err(ErrorCode::UnsafeReuse(_))
        ));
    }
}
