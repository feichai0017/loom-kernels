//! MasterService (Mooncake's `mooncake-store/include/master_service.h`) — the
//! store's control plane: object metadata, replica allocation, the **two-phase
//! Put**, and lease-based eviction. **No object bytes flow through it** — clients
//! move bytes via the transfer engine directly to/from the allocated buffers;
//! the master only decides *where* and tracks *what is readable*.
//!
//! Two-phase Put (Mooncake's `PutStart` / `PutEnd` / `PutRevoke`):
//! 1. [`MasterService::put_start`] allocates `replica_num` replicas (distinct
//!    segments, via the [`AllocationStrategy`]) and returns their buffers; the
//!    object is `Initialized` (not yet readable).
//! 2. the client writes the bytes into those buffers (transfer engine);
//! 3. [`MasterService::put_end`] flips the replicas to `Complete` (readable), or
//!    [`MasterService::put_revoke`] aborts and frees them.
//!
//! **QuillCache's identity guard** is woven into [`MasterService::get_replica_list`]:
//! each object records the [`IdentityScope`] that wrote it, and a get from a
//! mismatched identity is refused with [`ErrorCode::UnsafeReuse`] — Mooncake
//! isolates by `tenant_id` but not by model / tokenizer / adapter, so extending
//! the guard to the full identity is our addition.

use crate::allocation_strategy::{create_allocation_strategy, AllocationStrategy};
use crate::allocator::{AllocatedBuffer, BufferAllocator};
use crate::count_min_sketch::CountMinSketch;
use crate::offset_allocator::OffsetBufferAllocator;
use crate::op_log::{OpLog, OpLogEntry};
use crate::replica::{Replica, ReplicaData, ReplicaList, ReplicaStatus};
use crate::sharded_map::ShardedMap;
use crate::types::{ErrorCode, ObjectKey, ReplicaId, ReplicateConfig, SegmentName};
use quillcache_core::IdentityScope;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Per-object control-plane metadata (server-side; never sent to clients).
#[derive(Debug)]
struct ObjectMetadata {
    replicas: ReplicaList,
    /// The identity that wrote this object — QuillCache's guard.
    identity: IdentityScope,
    /// Logical time the read lease expires (blocks remove / eviction until then).
    lease_until: u64,
    /// Logical time of last access (approximate-LRU key).
    last_access: u64,
    soft_pinned: bool,
    hard_pinned: bool,
}

impl ObjectMetadata {
    fn has_complete_replica(&self) -> bool {
        self.replicas.values().any(Replica::is_complete)
    }
}

/// A serializable, consistent copy of the master's in-memory metadata — mounted
/// segments, object replicas, leases/pins, allocation strategy, and the clock.
/// Mooncake's periodic metadata snapshot, taken so a restarted master (or a
/// newly-elected leader, under etcd HA) can rebuild state; changes after the last
/// snapshot are lost (the same bound Mooncake documents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MasterSnapshot {
    pub version: u32,
    pub strategy: String,
    pub clock: u64,
    pub lease_ttl: u64,
    pub high_watermark: f64,
    pub eviction_ratio: f64,
    pub segment_ttl: u64,
    pub next_replica_id: ReplicaId,
    pub segments: Vec<SegmentSnapshot>,
    pub objects: Vec<ObjectSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentSnapshot {
    pub name: SegmentName,
    pub capacity: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSnapshot {
    pub key: ObjectKey,
    pub replicas: Vec<Replica>,
    pub identity: IdentityScope,
    pub lease_until: u64,
    pub last_access: u64,
    pub soft_pinned: bool,
    pub hard_pinned: bool,
}

/// Mutable allocator/segment state behind one mutex (the "segment lock" in the
/// lock order). Allocation, freeing, and replica-id assignment all need it.
#[derive(Debug)]
struct SegmentState {
    /// Mounted segments → their buffer allocators (one per segment).
    allocators: Vec<Box<dyn BufferAllocator>>,
    strategy: Box<dyn AllocationStrategy>,
    next_replica_id: ReplicaId,
}

impl SegmentState {
    fn capacity(&self) -> u64 {
        self.allocators.iter().map(|a| a.capacity()).sum()
    }

    fn allocated(&self) -> u64 {
        self.allocators.iter().map(|a| a.allocated()).sum()
    }

    fn free_replicas(&mut self, replicas: &ReplicaList) {
        for replica in replicas.values() {
            if let ReplicaData::Memory(buffer) = &replica.data {
                if let Some(allocator) = self
                    .allocators
                    .iter_mut()
                    .find(|a| a.segment_name() == buffer.segment_name)
                {
                    allocator.deallocate(buffer);
                }
            }
        }
    }
}

/// The store's metadata / allocation / eviction authority. Concurrent by design
/// (Mooncake's sharded master): object metadata lives in a per-shard-locked
/// [`ShardedMap`], and the allocators sit behind a separate [`Mutex`].
///
/// **Lock order (deadlock-free):** a read or single-object write locks only its
/// object shard; an allocator-touching op locks `segments` (outer) then a shard
/// or all shards (inner) — never the reverse. `segment_heartbeat` and `hotness`
/// are held only briefly and never nested with the others.
#[derive(Debug)]
pub struct MasterService {
    objects: ShardedMap<ObjectMetadata>,
    segments: Mutex<SegmentState>,
    /// The allocation-strategy name, kept so a snapshot can rebuild the same one.
    strategy_name: String,
    clock: AtomicU64,
    lease_ttl: u64,
    high_watermark: f64,
    eviction_ratio: f64,
    /// Last logical-clock tick each segment's node heartbeated — Mooncake's
    /// periodic client heartbeats, the basis for failure detection.
    segment_heartbeat: Mutex<HashMap<SegmentName, u64>>,
    /// Ticks a segment may miss before it is treated as dead. `0` disables the
    /// health check (every mounted segment is considered alive).
    segment_ttl: AtomicU64,
    /// Approximate per-key access frequency (Mooncake's CountMinSketch), bumped on
    /// every guarded read so eviction / promotion can favour hot keys.
    hotness: Mutex<CountMinSketch>,
    /// Optional durable op-log (Mooncake's HA OpLog). When set, committed
    /// mutations are appended + fsynced so the master can recover to the last op
    /// (not just the last snapshot). `None` = no logging (the default).
    oplog: Option<Mutex<OpLog>>,
}

impl MasterService {
    pub fn new(strategy: &str) -> Self {
        Self {
            objects: ShardedMap::default(),
            segments: Mutex::new(SegmentState {
                allocators: Vec::new(),
                strategy: create_allocation_strategy(strategy),
                next_replica_id: 0,
            }),
            strategy_name: strategy.to_string(),
            clock: AtomicU64::new(0),
            lease_ttl: 5,
            high_watermark: 0.95,
            eviction_ratio: 0.1,
            segment_heartbeat: Mutex::new(HashMap::new()),
            segment_ttl: AtomicU64::new(0),
            hotness: Mutex::new(CountMinSketch::default()),
            oplog: None,
        }
    }

    fn segments(&self) -> std::sync::MutexGuard<'_, SegmentState> {
        self.segments.lock().expect("segments mutex poisoned")
    }

    /// Turn on durable op-logging to `path` (Mooncake's HA OpLog). Call during
    /// construction, before the master is shared. Existing entries are kept; pair
    /// with periodic snapshots for compaction.
    pub fn enable_oplog(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.oplog = Some(Mutex::new(OpLog::open(path)?));
        Ok(())
    }

    /// Append a committed mutation to the op-log if one is enabled (best effort —
    /// the snapshot is the backstop). Locks only the op-log mutex (a leaf).
    fn log_op(&self, entry: OpLogEntry) {
        if let Some(oplog) = &self.oplog {
            let _ = oplog.lock().expect("oplog mutex poisoned").append(&entry);
        }
    }

    /// Advance the logical clock (drives leases + LRU without wall-clock, so
    /// tests are deterministic).
    pub fn tick(&self) -> u64 {
        self.clock.fetch_add(1, Ordering::Relaxed) + 1
    }

    // ---- segment lifecycle (Mooncake's MountSegment / UnmountSegment) ----

    pub fn mount_segment(&self, name: impl Into<SegmentName>, capacity: u64) {
        let name = name.into();
        // A freshly-mounted segment is alive as of now (heartbeat lock released
        // before the segments lock — never nested).
        self.segment_heartbeat
            .lock()
            .expect("heartbeat mutex poisoned")
            .insert(name.clone(), self.clock.load(Ordering::Relaxed));
        self.segments()
            .allocators
            .push(Box::new(OffsetBufferAllocator::new(name.clone(), capacity)));
        self.log_op(OpLogEntry::SegmentMounted { name, capacity });
    }

    /// Unmount a segment: drop any replicas living on it (they become
    /// unreadable), then remove its allocator.
    pub fn unmount_segment(&self, name: &str) -> Result<(), ErrorCode> {
        let mut segs = self.segments();
        if !segs.allocators.iter().any(|a| a.segment_name() == name) {
            return Err(ErrorCode::SegmentNotFound);
        }
        // Drop replicas on this segment, then any now-empty objects (segments
        // held → all shards, the fixed outer→inner order).
        self.objects.with_all(|shards| {
            for shard in shards.iter_mut() {
                for obj in shard.values_mut() {
                    obj.replicas.retain(|_, r| r.segment_name() != Some(name));
                }
                shard.retain(|_, o| !o.replicas.is_empty());
            }
        });
        segs.allocators.retain(|a| a.segment_name() != name);
        drop(segs);
        self.segment_heartbeat
            .lock()
            .expect("heartbeat mutex poisoned")
            .remove(name);
        self.log_op(OpLogEntry::SegmentUnmounted {
            name: name.to_string(),
        });
        Ok(())
    }

    // ---- HA: heartbeat-based segment health (Mooncake's client heartbeats) ----

    /// Enable failure detection: a segment that misses a heartbeat for more than
    /// `ttl` logical ticks is treated as dead and its replicas are not handed
    /// out. `0` disables the check (the default — every mounted segment is alive).
    pub fn set_segment_ttl(&self, ttl: u64) {
        self.segment_ttl.store(ttl, Ordering::Relaxed);
    }

    /// Record a liveness heartbeat from a segment's node. Unknown segment → error.
    pub fn heartbeat(&self, segment: &str) -> Result<(), ErrorCode> {
        let now = self.clock.load(Ordering::Relaxed);
        let mut hb = self
            .segment_heartbeat
            .lock()
            .expect("heartbeat mutex poisoned");
        match hb.get_mut(segment) {
            Some(last) => {
                *last = now;
                Ok(())
            }
            None => Err(ErrorCode::SegmentNotFound),
        }
    }

    /// Whether a mounted segment is alive (heartbeated within `segment_ttl`).
    /// With the check disabled (`segment_ttl == 0`), any mounted segment is alive.
    pub fn segment_alive(&self, segment: &str) -> bool {
        let ttl = self.segment_ttl.load(Ordering::Relaxed);
        let now = self.clock.load(Ordering::Relaxed);
        match self
            .segment_heartbeat
            .lock()
            .expect("heartbeat mutex poisoned")
            .get(segment)
        {
            Some(&last) => ttl == 0 || now.saturating_sub(last) <= ttl,
            None => false,
        }
    }

    /// Mounted segments that have missed heartbeats past the TTL — failure
    /// detection surfaces them so the control plane can re-replicate / route away.
    pub fn dead_segments(&self) -> Vec<String> {
        // Collect names under the segments lock, then check liveness separately
        // (heartbeat lock never nested under segments).
        let names: Vec<String> = self
            .segments()
            .allocators
            .iter()
            .map(|a| a.segment_name().to_string())
            .collect();
        names
            .into_iter()
            .filter(|name| !self.segment_alive(name))
            .collect()
    }

    // ---- two-phase Put ----

    /// Phase 1: allocate `config.replica_num` replicas for `key` and return their
    /// buffers; the object is recorded `Initialized` (not yet readable).
    pub fn put_start(
        &self,
        key: ObjectKey,
        identity: IdentityScope,
        size: u64,
        config: &ReplicateConfig,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        let mut segs = self.segments();
        // Dup check + reclaim any in-flight leftover (segments held → this key's
        // shard, the fixed outer→inner order).
        let leftover = self.objects.with_shard(&key, |s| {
            if s.get(&key)
                .is_some_and(ObjectMetadata::has_complete_replica)
            {
                return Err(ErrorCode::ObjectAlreadyExists);
            }
            Ok(s.remove(&key))
        })?;
        if let Some(old) = leftover {
            segs.free_replicas(&old.replicas);
        }

        let preferred = config.preferred_segment.as_deref();
        let buffers = self.allocate_replicas(&mut segs, size, config.replica_num, preferred)?;

        let mut replicas = ReplicaList::new();
        for buffer in &buffers {
            let id = segs.next_replica_id;
            segs.next_replica_id += 1;
            replicas.insert(id, Replica::new(id, ReplicaData::Memory(buffer.clone())));
        }
        let now = self.clock.load(Ordering::Relaxed);
        let metadata = ObjectMetadata {
            replicas,
            identity,
            lease_until: 0,
            last_access: now,
            soft_pinned: config.with_soft_pin,
            hard_pinned: config.with_hard_pin,
        };
        self.objects.with_shard(&key, |s| {
            s.insert(key.clone(), metadata);
        });
        Ok(buffers)
    }

    /// Phase 2: flip the object's replicas to `Complete` (readable). Single-object
    /// write — locks only this key's shard.
    pub fn put_end(&self, key: &str) -> Result<(), ErrorCode> {
        let logging = self.oplog.is_some();
        let entry = self.objects.with_shard(key, |s| {
            let object = s.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
            for replica in object.replicas.values_mut() {
                replica.status = ReplicaStatus::Complete;
            }
            // Build the durable entry while we hold the shard (cheap; only if the
            // op-log is on). It is appended after the shard lock is released.
            Ok::<_, ErrorCode>(logging.then(|| OpLogEntry::PutCommitted {
                key: key.to_string(),
                identity: object.identity.clone(),
                replicas: object.replicas.values().cloned().collect(),
                soft_pinned: object.soft_pinned,
                hard_pinned: object.hard_pinned,
            }))
        })?;
        if let Some(entry) = entry {
            self.log_op(entry);
        }
        Ok(())
    }

    /// Abort an in-flight Put: free the allocation and drop the object.
    pub fn put_revoke(&self, key: &str) -> Result<(), ErrorCode> {
        let mut segs = self.segments();
        let object = self
            .objects
            .with_shard(key, |s| s.remove(key))
            .ok_or(ErrorCode::ObjectNotFound)?;
        segs.free_replicas(&object.replicas);
        Ok(())
    }

    // ---- read path (identity-guarded) ----

    /// Return the object's complete replicas, **identity-guarded**, and grant a
    /// read lease. Refuses a requester whose identity doesn't match the writer's.
    pub fn get_replica_list(
        &self,
        key: &str,
        requester: &IdentityScope,
    ) -> Result<Vec<Replica>, ErrorCode> {
        let now = self.clock.load(Ordering::Relaxed);
        let ttl = self.segment_ttl.load(Ordering::Relaxed);
        // Failure detection: a Memory replica on a segment whose node stopped
        // heartbeating is treated as lost; a Disk replica is durable, so kept. The
        // alive set is read once from the heartbeat map (its keys are the mounted
        // segments) — no segments lock, so reads stay concurrent.
        let alive: HashSet<String> = {
            let hb = self
                .segment_heartbeat
                .lock()
                .expect("heartbeat mutex poisoned");
            hb.iter()
                .filter(|(_, &last)| ttl == 0 || now.saturating_sub(last) <= ttl)
                .map(|(name, _)| name.clone())
                .collect()
        };
        // Record this access in the frequency sketch (hot-key tracking).
        self.hotness
            .lock()
            .expect("hotness mutex poisoned")
            .increment(key);
        // Only this key's shard is locked — concurrent with reads of other keys.
        self.objects.with_shard(key, |s| {
            let object = s.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
            // QuillCache identity guard: a content-hash key can be requested under
            // a different identity — refuse cross-tenant / cross-adapter / model.
            if let Some(violation) = object.identity.reuse_violation_against(requester) {
                return Err(ErrorCode::UnsafeReuse(violation));
            }
            let complete: Vec<Replica> = object
                .replicas
                .values()
                .filter(|r| r.is_complete())
                .filter(|r| match r.segment_name() {
                    Some(seg) => alive.contains(seg),
                    None => true,
                })
                .cloned()
                .collect();
            if complete.is_empty() {
                return Err(ErrorCode::ObjectNotReady);
            }
            object.last_access = now;
            object.lease_until = now + self.lease_ttl;
            Ok(complete)
        })
    }

    // ---- batch APIs (Mooncake's BatchPut / BatchGet — one round-trip for many
    // keys; our connector offloads/loads a prefix's layers as one batch) ----

    /// Allocate replicas for many objects in one call. Transactional: if any key
    /// can't be allocated, the ones already started in this batch are revoked and
    /// the error is returned (no partial batch is left behind).
    pub fn batch_put_start(
        &self,
        items: Vec<(ObjectKey, IdentityScope, u64)>,
        config: &ReplicateConfig,
    ) -> Result<Vec<Vec<AllocatedBuffer>>, ErrorCode> {
        let mut out = Vec::with_capacity(items.len());
        let mut started: Vec<ObjectKey> = Vec::new();
        for (key, identity, size) in items {
            match self.put_start(key.clone(), identity, size, config) {
                Ok(buffers) => {
                    out.push(buffers);
                    started.push(key);
                }
                Err(e) => {
                    for k in &started {
                        let _ = self.put_revoke(k);
                    }
                    return Err(e);
                }
            }
        }
        Ok(out)
    }

    /// Commit many objects' replicas (flip to readable). Errors on the first key
    /// that isn't in flight.
    pub fn batch_put_end(&self, keys: &[String]) -> Result<(), ErrorCode> {
        for key in keys {
            self.put_end(key)?;
        }
        Ok(())
    }

    /// Revoke many in-flight Puts (free their allocations). Errors on the first
    /// key not in flight.
    pub fn batch_put_revoke(&self, keys: &[String]) -> Result<(), ErrorCode> {
        for key in keys {
            self.put_revoke(key)?;
        }
        Ok(())
    }

    // ---- upsert (Mooncake's UpsertStart/End/Revoke) ----

    /// Mooncake's `UpsertStart`. If the key is absent (or only in-flight) this is
    /// [`Self::put_start`]. If it exists complete at the **same** size, reuse its
    /// buffers in place — return them and flip the replicas back to `Initialized`
    /// so the client rewrites them (no re-allocation). If the size **differs**,
    /// free the old replicas and allocate new ones. Refused with
    /// [`ErrorCode::ObjectNotReady`] while a read lease is active (the object is
    /// busy), mirroring Mooncake's `OBJECT_REPLICA_BUSY`.
    pub fn upsert_start(
        &self,
        key: ObjectKey,
        identity: IdentityScope,
        size: u64,
        config: &ReplicateConfig,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        let now = self.clock.load(Ordering::Relaxed);
        // Inspect current state under this key's shard only.
        let (exists_complete, leased, existing_size) =
            self.objects.with_shard(&key, |s| match s.get(&key) {
                None => (false, false, 0),
                Some(o) => (
                    o.has_complete_replica(),
                    o.lease_until > now,
                    o.replicas.values().next().map(|r| r.size()).unwrap_or(0),
                ),
            });
        if !exists_complete {
            // Absent or only an in-flight leftover — PutStart reclaims + allocates.
            return self.put_start(key, identity, size, config);
        }
        if leased {
            return Err(ErrorCode::ObjectNotReady);
        }
        if existing_size == size {
            // In-place: reuse the existing buffers, re-open for writing (shard-only).
            return Ok(self.objects.with_shard(&key, |s| {
                let object = s.get_mut(&key).expect("object present");
                object.identity = identity;
                object.last_access = now;
                let mut buffers = Vec::new();
                for replica in object.replicas.values_mut() {
                    replica.status = ReplicaStatus::Initialized;
                    if let ReplicaData::Memory(buffer) = &replica.data {
                        buffers.push(buffer.clone());
                    }
                }
                buffers
            }));
        }
        // Size changed: free the old replicas (segments → shard), then allocate
        // fresh via put_start — the segments lock is released first, no reentrancy.
        {
            let mut segs = self.segments();
            if let Some(old) = self.objects.with_shard(&key, |s| s.remove(&key)) {
                segs.free_replicas(&old.replicas);
            }
        }
        self.put_start(key, identity, size, config)
    }

    /// Mooncake's `UpsertEnd` — same as committing a Put.
    pub fn upsert_end(&self, key: &str) -> Result<(), ErrorCode> {
        self.put_end(key)
    }

    /// Mooncake's `UpsertRevoke` — same as aborting a Put.
    pub fn upsert_revoke(&self, key: &str) -> Result<(), ErrorCode> {
        self.put_revoke(key)
    }

    /// Identity-guarded Get for many keys. Errors on the first key that is
    /// missing / not ready / refused (the connector wants all of a prefix's
    /// layers, or it recomputes the prefix).
    pub fn batch_get_replica_list(
        &self,
        keys: &[String],
        requester: &IdentityScope,
    ) -> Result<Vec<Vec<Replica>>, ErrorCode> {
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            out.push(self.get_replica_list(key, requester)?);
        }
        Ok(out)
    }

    pub fn exist_key(&self, key: &str) -> bool {
        self.objects.with_shard(key, |s| {
            s.get(key).is_some_and(ObjectMetadata::has_complete_replica)
        })
    }

    /// Existence for many keys in one call (Mooncake's `BatchExistKey`).
    pub fn batch_exist_key(&self, keys: &[String]) -> Vec<bool> {
        keys.iter().map(|k| self.exist_key(k)).collect()
    }

    /// Mooncake's `GetReplicaListByRegex`: every object whose key matches
    /// `pattern` and whose identity the requester is allowed to read, mapped to
    /// its complete replicas. Cross-identity matches are skipped (the guard), not
    /// errored — a bulk query returns what the caller may see.
    pub fn get_replica_list_by_regex(
        &self,
        pattern: &str,
        requester: &IdentityScope,
    ) -> Result<HashMap<String, Vec<Replica>>, ErrorCode> {
        let re = Regex::new(pattern).map_err(|e| ErrorCode::Io(format!("bad regex: {e}")))?;
        // Collect matching keys first (all shards, briefly) so the per-key
        // get_replica_list calls each take just their own shard afterwards.
        let keys: Vec<String> = self.objects.with_all(|shards| {
            shards
                .iter()
                .flat_map(|s| s.keys())
                .filter(|k| re.is_match(k))
                .cloned()
                .collect()
        });
        let mut out = HashMap::new();
        for key in keys {
            if let Ok(replicas) = self.get_replica_list(&key, requester) {
                out.insert(key, replicas);
            }
        }
        Ok(out)
    }

    /// Remove an object and free its replicas. Blocked while a read lease is
    /// active unless `force`.
    pub fn remove(&self, key: &str, force: bool) -> Result<(), ErrorCode> {
        let now = self.clock.load(Ordering::Relaxed);
        let mut segs = self.segments();
        let object = self.objects.with_shard(key, |s| match s.get(key) {
            None => Err(ErrorCode::ObjectNotFound),
            Some(o) if o.lease_until > now && !force => Err(ErrorCode::ObjectNotReady),
            Some(_) => Ok(s.remove(key).expect("present")),
        })?;
        segs.free_replicas(&object.replicas);
        drop(segs);
        self.log_op(OpLogEntry::Removed {
            key: key.to_string(),
        });
        Ok(())
    }

    // ---- observability ----

    pub fn segment_count(&self) -> usize {
        self.segments().allocators.len()
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn capacity(&self) -> u64 {
        self.segments().capacity()
    }
    pub fn allocated(&self) -> u64 {
        self.segments().allocated()
    }

    /// Approximate access frequency for `key` (Mooncake's CountMinSketch
    /// estimate) — how hot the key is, for frequency-aware eviction / promotion.
    pub fn hotness(&self, key: &str) -> u8 {
        self.hotness
            .lock()
            .expect("hotness mutex poisoned")
            .count(key)
    }

    // ---- HA: snapshot + recovery (Mooncake's metadata snapshot thread) ----

    /// Take a consistent [`MasterSnapshot`] of the current in-memory metadata.
    pub fn snapshot(&self) -> MasterSnapshot {
        // segments (outer) → all shards (inner) — the fixed order.
        let segs = self.segments();
        let objects = self.objects.with_all(|shards| {
            shards
                .iter()
                .flat_map(|s| s.iter())
                .map(|(key, o)| ObjectSnapshot {
                    key: key.clone(),
                    replicas: o.replicas.values().cloned().collect(),
                    identity: o.identity.clone(),
                    lease_until: o.lease_until,
                    last_access: o.last_access,
                    soft_pinned: o.soft_pinned,
                    hard_pinned: o.hard_pinned,
                })
                .collect()
        });
        MasterSnapshot {
            version: 1,
            strategy: self.strategy_name.clone(),
            clock: self.clock.load(Ordering::Relaxed),
            lease_ttl: self.lease_ttl,
            high_watermark: self.high_watermark,
            eviction_ratio: self.eviction_ratio,
            segment_ttl: self.segment_ttl.load(Ordering::Relaxed),
            next_replica_id: segs.next_replica_id,
            segments: segs
                .allocators
                .iter()
                .map(|a| SegmentSnapshot {
                    name: a.segment_name().to_string(),
                    capacity: a.capacity(),
                })
                .collect(),
            objects,
        }
    }

    /// Rebuild a master from a snapshot: re-mount the segments and re-reserve each
    /// replica's exact `(offset, size)` so the allocator layout matches, then
    /// restore objects, leases, pins, and the clock.
    pub fn recover(snapshot: MasterSnapshot) -> Result<Self, ErrorCode> {
        let master = MasterService::new(&snapshot.strategy);
        master.clock.store(snapshot.clock, Ordering::Relaxed);
        master
            .segment_ttl
            .store(snapshot.segment_ttl, Ordering::Relaxed);
        // lease_ttl / watermarks are immutable after construction; set them on the
        // freshly-owned value before it is shared.
        let mut master = master;
        master.lease_ttl = snapshot.lease_ttl;
        master.high_watermark = snapshot.high_watermark;
        master.eviction_ratio = snapshot.eviction_ratio;
        let master = master;
        for seg in &snapshot.segments {
            master.mount_segment(seg.name.clone(), seg.capacity);
        }
        for obj in snapshot.objects {
            // Re-reserve each Memory replica's exact range so the allocator's
            // free-list reflects the recovered layout (Disk replicas are durable).
            {
                let mut segs = master.segments();
                for replica in &obj.replicas {
                    if let ReplicaData::Memory(buf) = &replica.data {
                        let allocator = segs
                            .allocators
                            .iter_mut()
                            .find(|a| a.segment_name() == buf.segment_name())
                            .ok_or(ErrorCode::SegmentNotFound)?;
                        if !allocator.reserve(buf.offset, buf.size) {
                            return Err(ErrorCode::InvalidReplica);
                        }
                    }
                }
            }
            let mut replicas = ReplicaList::new();
            for replica in obj.replicas {
                replicas.insert(replica.id, replica);
            }
            master.objects.with_shard(&obj.key, |s| {
                s.insert(
                    obj.key.clone(),
                    ObjectMetadata {
                        replicas,
                        identity: obj.identity,
                        lease_until: obj.lease_until,
                        last_access: obj.last_access,
                        soft_pinned: obj.soft_pinned,
                        hard_pinned: obj.hard_pinned,
                    },
                );
            });
        }
        master.segments().next_replica_id = snapshot.next_replica_id;
        Ok(master)
    }

    /// Persist a snapshot to `path` **atomically** — write a temp file, then
    /// rename — so a crash mid-write never leaves a torn snapshot (the QuillCache
    /// crash-consistency discipline applied to the master's metadata).
    pub fn save_snapshot(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        let bytes = serde_json::to_vec(&self.snapshot())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let tmp = path.with_extension("snapshot.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Recover a master from a snapshot file written by [`MasterService::save_snapshot`].
    pub fn load_snapshot(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: MasterSnapshot = serde_json::from_slice(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Self::recover(snapshot)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}")))
    }

    // ---- HA: op-log replay (Mooncake's OpLog recovery) ----

    /// Recover a master by replaying an op-log from an empty start — Mooncake's
    /// HA model (rebuild the state machine from the durable log). For snapshot +
    /// log, [`Self::recover`] from the snapshot first, then apply the post-snapshot
    /// log; this variant is the pure-log path.
    pub fn recover_from_oplog(strategy: &str, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let master = MasterService::new(strategy);
        for entry in OpLog::replay(path)? {
            master.apply_op(entry).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{e:?}"))
            })?;
        }
        Ok(master)
    }

    /// Apply one replayed [`OpLogEntry`] (recovery only — not itself logged).
    fn apply_op(&self, entry: OpLogEntry) -> Result<(), ErrorCode> {
        match entry {
            OpLogEntry::SegmentMounted { name, capacity } => {
                self.mount_segment(name, capacity);
                Ok(())
            }
            OpLogEntry::SegmentUnmounted { name } => {
                let _ = self.unmount_segment(&name);
                Ok(())
            }
            OpLogEntry::Removed { key } => {
                let mut segs = self.segments();
                if let Some(object) = self.objects.with_shard(&key, |s| s.remove(&key)) {
                    segs.free_replicas(&object.replicas);
                }
                Ok(())
            }
            OpLogEntry::PutCommitted {
                key,
                identity,
                replicas,
                soft_pinned,
                hard_pinned,
            } => {
                let mut segs = self.segments();
                for replica in &replicas {
                    if let ReplicaData::Memory(buf) = &replica.data {
                        let allocator = segs
                            .allocators
                            .iter_mut()
                            .find(|a| a.segment_name() == buf.segment_name())
                            .ok_or(ErrorCode::SegmentNotFound)?;
                        if !allocator.reserve(buf.offset, buf.size) {
                            return Err(ErrorCode::InvalidReplica);
                        }
                    }
                    segs.next_replica_id = segs.next_replica_id.max(replica.id + 1);
                }
                let now = self.clock.load(Ordering::Relaxed);
                let mut rlist = ReplicaList::new();
                for replica in replicas {
                    rlist.insert(replica.id, replica);
                }
                self.objects.with_shard(&key, |s| {
                    s.insert(
                        key.clone(),
                        ObjectMetadata {
                            replicas: rlist,
                            identity,
                            lease_until: 0,
                            last_access: now,
                            soft_pinned,
                            hard_pinned,
                        },
                    );
                });
                Ok(())
            }
        }
    }

    // ---- internals ----

    /// Allocate `replica_num` buffers of `size`; on segment exhaustion, evict the
    /// coldest unpinned objects to make room and retry once (Mooncake evicts at a
    /// high watermark or on put failure).
    fn allocate_replicas(
        &self,
        segs: &mut SegmentState,
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        self.evict_if_needed_locked(segs);
        match segs
            .strategy
            .allocate(&mut segs.allocators, size, replica_num, preferred, &[])
        {
            Ok(buffers) => Ok(buffers),
            Err(ErrorCode::NoAvailableSegment) => {
                self.evict_to_fit(segs, size.saturating_mul(replica_num as u64));
                segs.strategy
                    .allocate(&mut segs.allocators, size, replica_num, preferred, &[])
            }
            Err(other) => Err(other),
        }
    }

    /// Eviction candidates, coldest first; non-soft-pinned before soft-pinned;
    /// hard-pinned and currently-leased objects are never candidates. Locks all
    /// shards; the caller already holds `segments` (the segments → shards order).
    fn victims_coldest_first(&self) -> Vec<ObjectKey> {
        let now = self.clock.load(Ordering::Relaxed);
        self.objects.with_all(|shards| {
            let mut victims: Vec<(ObjectKey, u64, bool)> = shards
                .iter()
                .flat_map(|s| s.iter())
                .filter(|(_, o)| !o.hard_pinned && o.lease_until <= now)
                .map(|(k, o)| (k.clone(), o.last_access, o.soft_pinned))
                .collect();
            victims.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));
            victims.into_iter().map(|(k, _, _)| k).collect()
        })
    }

    /// Proactive watermark eviction: when usage exceeds the high watermark, evict
    /// the coldest unpinned objects down below the target.
    pub fn evict_if_needed(&self) -> usize {
        let mut segs = self.segments();
        self.evict_if_needed_locked(&mut segs)
    }

    fn evict_if_needed_locked(&self, segs: &mut SegmentState) -> usize {
        let capacity = segs.capacity();
        if capacity == 0 || (segs.allocated() as f64) < self.high_watermark * capacity as f64 {
            return 0;
        }
        let target = (self.high_watermark * (1.0 - self.eviction_ratio) * capacity as f64) as u64;
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if segs.allocated() <= target {
                break;
            }
            if let Some(object) = self.objects.with_shard(&key, |s| s.remove(&key)) {
                segs.free_replicas(&object.replicas);
                self.log_op(OpLogEntry::Removed { key });
                evicted += 1;
            }
        }
        evicted
    }

    /// On-demand eviction: evict coldest unpinned objects until at least `needed`
    /// bytes are free cluster-wide (best effort). The caller holds `segs`.
    fn evict_to_fit(&self, segs: &mut SegmentState, needed: u64) -> usize {
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if segs.capacity().saturating_sub(segs.allocated()) >= needed {
                break;
            }
            if let Some(object) = self.objects.with_shard(&key, |s| s.remove(&key)) {
                segs.free_replicas(&object.replicas);
                self.log_op(OpLogEntry::Removed { key });
                evicted += 1;
            }
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quillcache_core::ReuseViolation;

    fn scope(tenant: &str) -> IdentityScope {
        IdentityScope {
            model_id: "m".into(),
            tokenizer_id: "t".into(),
            adapter_id: None,
            tenant_id: tenant.into(),
        }
    }

    #[test]
    fn two_phase_put_then_get_with_replication() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        m.mount_segment("seg-1", 100);
        let id = scope("ten-a");

        // Phase 1: two replicas on distinct segments.
        let buffers = m
            .put_start("k".into(), id.clone(), 16, &ReplicateConfig::replicas(2))
            .unwrap();
        assert_eq!(buffers.len(), 2);
        assert_ne!(buffers[0].segment_name, buffers[1].segment_name);
        // Not readable until put_end.
        assert_eq!(m.get_replica_list("k", &id), Err(ErrorCode::ObjectNotReady));

        // Phase 2: now readable.
        m.put_end("k").unwrap();
        let replicas = m.get_replica_list("k", &id).unwrap();
        assert_eq!(replicas.len(), 2);
        assert!(replicas.iter().all(|r| r.is_complete()));
        assert!(m.exist_key("k"));
    }

    #[test]
    fn get_is_identity_guarded() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        // Same content-hash key, written by tenant-a.
        m.put_start(
            "hot".into(),
            scope("ten-a"),
            10,
            &ReplicateConfig::replicas(1),
        )
        .unwrap();
        m.put_end("hot").unwrap();
        // tenant-b asks for the same key → refused (a prefix-cache privacy leak).
        assert_eq!(
            m.get_replica_list("hot", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(ReuseViolation::Tenant))
        );
        // tenant-a (the writer) gets it.
        assert_eq!(m.get_replica_list("hot", &scope("ten-a")).unwrap().len(), 1);
    }

    #[test]
    fn put_revoke_frees_the_allocation() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        m.put_start(
            "k".into(),
            scope("ten-a"),
            40,
            &ReplicateConfig::replicas(1),
        )
        .unwrap();
        assert_eq!(m.allocated(), 40);
        m.put_revoke("k").unwrap();
        assert_eq!(m.allocated(), 0);
        assert!(!m.exist_key("k"));
    }

    #[test]
    fn eviction_makes_room_under_pressure_and_keeps_hot_and_pinned() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");

        // A at t0 (coldest), B at t1.
        m.put_start("A".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("A").unwrap();
        m.tick();
        m.put_start("B".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("B").unwrap();
        assert_eq!(m.allocated(), 80);

        // C (40) doesn't fit (20 free) → evict the coldest (A) to make room.
        m.tick();
        m.put_start("C".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("C").unwrap();
        assert!(!m.exist_key("A"), "coldest object A should be evicted");
        assert!(m.exist_key("B") && m.exist_key("C"));
    }

    #[test]
    fn hard_pinned_object_is_never_evicted() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        // A is hard-pinned even though it is the coldest.
        let pinned = ReplicateConfig {
            with_hard_pin: true,
            ..ReplicateConfig::replicas(1)
        };
        m.put_start("A".into(), id.clone(), 40, &pinned).unwrap();
        m.put_end("A").unwrap();
        m.tick();
        m.put_start("B".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("B").unwrap();
        // C needs room: B (the only unpinned victim) is evicted, A survives.
        m.tick();
        m.put_start("C".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("C").unwrap();
        assert!(m.exist_key("A"), "hard-pinned A must survive");
        assert!(!m.exist_key("B"));
        assert!(m.exist_key("C"));
    }

    #[test]
    fn watermark_eviction_frees_down_to_target() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        for i in 0..10 {
            let k = format!("k{i}");
            m.put_start(k.clone(), id.clone(), 10, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end(&k).unwrap();
            m.tick();
        }
        assert_eq!(m.allocated(), 100); // full, over the 95% watermark
        let evicted = m.evict_if_needed();
        assert!(evicted >= 2);
        assert_eq!(m.allocated(), 80); // freed down below target (85)
        assert!(!m.exist_key("k0") && !m.exist_key("k1")); // the two coldest
        assert!(m.exist_key("k2"));
    }

    #[test]
    fn read_lease_blocks_remove_until_it_expires() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        m.put_start("A".into(), id.clone(), 10, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("A").unwrap();
        // A get grants a lease (ttl 5).
        m.get_replica_list("A", &id).unwrap();
        assert_eq!(m.remove("A", false), Err(ErrorCode::ObjectNotReady));
        // Force ignores the lease.
        // (don't actually remove yet — test lease expiry path instead)
        for _ in 0..6 {
            m.tick();
        }
        assert!(m.remove("A", false).is_ok());
        assert!(!m.exist_key("A"));
    }

    #[test]
    fn snapshot_recovers_objects_segments_and_allocator_state() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        m.mount_segment("seg-1", 1000);
        let id = scope("ten-a");
        m.put_start("a".into(), id.clone(), 64, &ReplicateConfig::replicas(2))
            .unwrap();
        m.put_end("a").unwrap();
        m.put_start("b".into(), id.clone(), 128, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("b").unwrap();
        let allocated_before = m.allocated();

        // Round-trip through a snapshot (e.g. a restart / leader failover).
        let r = MasterService::recover(m.snapshot()).expect("recover");
        assert_eq!(r.object_count(), 2);
        assert_eq!(r.segment_count(), 2);
        assert_eq!(
            r.allocated(),
            allocated_before,
            "the allocator's allocated bytes are rebuilt exactly"
        );

        // Recovered objects are readable, still identity-guarded.
        assert_eq!(r.get_replica_list("a", &id).unwrap().len(), 2);
        assert!(matches!(
            r.get_replica_list("a", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));

        // The rebuilt allocator won't hand out the recovered ranges again: a fresh
        // Put succeeds without overlapping the reserved offsets.
        r.put_start("c".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        r.put_end("c").unwrap();
        assert!(r.get_replica_list("c", &id).is_ok());
    }

    #[test]
    fn snapshot_file_round_trip_is_atomic() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();

        let path = std::env::temp_dir().join(format!("qc-master-snap-{}.json", std::process::id()));
        m.save_snapshot(&path).unwrap();
        let r = MasterService::load_snapshot(&path).unwrap();
        assert_eq!(r.get_replica_list("k", &id).unwrap().len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn heartbeat_health_hides_replicas_on_a_dead_segment() {
        let m = MasterService::new("random");
        m.set_segment_ttl(5); // enable failure detection
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        assert!(m.segment_alive("seg-0"));
        assert!(m.get_replica_list("k", &id).is_ok());

        // Miss heartbeats past the TTL → the segment is dead and its only replica
        // is treated as lost, so the object becomes unservable.
        for _ in 0..6 {
            m.tick();
        }
        assert!(!m.segment_alive("seg-0"));
        assert_eq!(m.dead_segments(), vec!["seg-0".to_string()]);
        assert_eq!(m.get_replica_list("k", &id), Err(ErrorCode::ObjectNotReady));

        // A heartbeat brings the node back and its replica is served again.
        m.heartbeat("seg-0").unwrap();
        assert!(m.segment_alive("seg-0"));
        assert!(m.get_replica_list("k", &id).is_ok());
    }

    #[test]
    fn batch_put_then_batch_get_round_trips() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 4096);
        let id = scope("ten-a");
        let keys: Vec<String> = vec!["k0".into(), "k1".into(), "k2".into()];
        let items = keys.iter().map(|k| (k.clone(), id.clone(), 64)).collect();

        let buffers = m
            .batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(buffers.len(), 3);
        m.batch_put_end(&keys).unwrap();

        let got = m.batch_get_replica_list(&keys, &id).unwrap();
        assert_eq!(got.len(), 3);
        assert!(got.iter().all(|replicas| replicas.len() == 1));

        // The identity guard applies to the batch too.
        assert!(matches!(
            m.batch_get_replica_list(&["k0".into()], &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(_))
        ));
    }

    #[test]
    fn batch_put_start_rolls_back_on_failure() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 100);
        let id = scope("ten-a");
        // The second object (200B) can't fit a 100B segment → the batch fails and
        // the first object's allocation is rolled back (no partial batch).
        let items = vec![
            ("a".to_string(), id.clone(), 64),
            ("b".to_string(), id, 200),
        ];
        assert!(m
            .batch_put_start(items, &ReplicateConfig::replicas(1))
            .is_err());
        assert_eq!(m.object_count(), 0);
        assert_eq!(m.allocated(), 0);
    }

    #[test]
    fn upsert_reuses_buffers_in_place_when_size_unchanged() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let bufs = m
            .put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        let off = bufs[0].offset;
        let used = m.allocated();
        // Same size → reuse the buffer in place, no extra allocation.
        let bufs2 = m
            .upsert_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(bufs2[0].offset, off, "in-place upsert reuses the buffer");
        assert_eq!(
            m.allocated(),
            used,
            "no re-allocation for a same-size upsert"
        );
        assert!(!m.exist_key("k"), "re-opened for writing until committed");
        m.upsert_end("k").unwrap();
        assert!(m.exist_key("k"));
    }

    #[test]
    fn upsert_reallocates_when_size_changes() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 40, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        assert_eq!(m.allocated(), 40);
        m.upsert_start("k".into(), id.clone(), 100, &ReplicateConfig::replicas(1))
            .unwrap();
        m.upsert_end("k").unwrap();
        assert_eq!(m.allocated(), 100, "size change frees old + allocates new");
    }

    #[test]
    fn upsert_is_refused_while_leased() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("k").unwrap();
        m.get_replica_list("k", &id).unwrap(); // grants a read lease → busy
        assert_eq!(
            m.upsert_start("k".into(), id.clone(), 64, &ReplicateConfig::replicas(1)),
            Err(ErrorCode::ObjectNotReady)
        );
    }

    #[test]
    fn get_replica_list_by_regex_matches_allowed_keys_only() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let a = scope("ten-a");
        let b = scope("ten-b");
        for k in ["qc/a/1", "qc/a/2", "other"] {
            m.put_start(k.into(), a.clone(), 16, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end(k).unwrap();
        }
        // Same-pattern key under a different tenant must NOT leak to tenant-a.
        m.put_start("qc/a/secret".into(), b, 16, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("qc/a/secret").unwrap();

        let got = m.get_replica_list_by_regex("^qc/a/.*", &a).unwrap();
        let mut keys: Vec<&str> = got.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["qc/a/1", "qc/a/2"]);
    }

    #[test]
    fn batch_exist_key_reports_committed_only() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let items = vec![
            ("a".to_string(), id.clone(), 16u64),
            ("b".to_string(), id.clone(), 16u64),
        ];
        m.batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        assert_eq!(
            m.batch_exist_key(&["a".into(), "b".into()]),
            vec![false, false]
        );
        m.batch_put_end(&["a".into(), "b".into()]).unwrap();
        assert_eq!(
            m.batch_exist_key(&["a".into(), "b".into(), "z".into()]),
            vec![true, true, false]
        );
    }

    #[test]
    fn batch_put_revoke_frees_inflight() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        let items = vec![
            ("a".to_string(), id.clone(), 16u64),
            ("b".to_string(), id, 16u64),
        ];
        m.batch_put_start(items, &ReplicateConfig::replicas(1))
            .unwrap();
        m.batch_put_revoke(&["a".into(), "b".into()]).unwrap();
        assert_eq!(m.allocated(), 0);
        assert_eq!(m.object_count(), 0);
    }

    #[test]
    fn guarded_reads_bump_key_hotness() {
        let m = MasterService::new("random");
        m.mount_segment("seg-0", 1000);
        let id = scope("ten-a");
        m.put_start("hot".into(), id.clone(), 16, &ReplicateConfig::replicas(1))
            .unwrap();
        m.put_end("hot").unwrap();
        assert_eq!(m.hotness("hot"), 0);
        for _ in 0..5 {
            m.get_replica_list("hot", &id).unwrap();
        }
        assert_eq!(m.hotness("hot"), 5, "each guarded read bumps the sketch");
        assert_eq!(m.hotness("cold"), 0);
    }

    #[test]
    fn recovers_state_by_replaying_the_oplog() {
        let dir = std::env::temp_dir().join(format!("qc-master-oplog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("master.oplog");
        let _ = std::fs::remove_file(&path);
        let id = scope("ten-a");
        {
            let mut m = MasterService::new("random");
            m.enable_oplog(&path).unwrap();
            m.mount_segment("seg-0", 1000);
            m.put_start("keep".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end("keep").unwrap();
            m.put_start("gone".into(), id.clone(), 64, &ReplicateConfig::replicas(1))
                .unwrap();
            m.put_end("gone").unwrap();
            m.remove("gone", true).unwrap();
        }
        // Rebuild purely from the log: segment re-mounted, "keep" durable + still
        // identity-guarded, "gone" gone.
        let r = MasterService::recover_from_oplog("random", &path).unwrap();
        assert_eq!(r.segment_count(), 1);
        assert!(r.exist_key("keep"));
        assert!(!r.exist_key("gone"));
        assert_eq!(r.get_replica_list("keep", &id).unwrap().len(), 1);
        assert_eq!(
            r.get_replica_list("keep", &scope("ten-b")),
            Err(ErrorCode::UnsafeReuse(ReuseViolation::Tenant))
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concurrent_puts_and_gets_across_keys_do_not_deadlock() {
        use std::sync::Arc;
        use std::thread;
        // The whole point of the sharded master: many threads Put/Get distinct
        // keys concurrently (allocation serializes on `segments`, object shards run
        // in parallel). A lock-order slip would hang this (CI times out).
        let m = Arc::new(MasterService::new("random"));
        m.mount_segment("seg-0", 1 << 20);
        let id = scope("ten-a");

        let writers: Vec<_> = (0..4u64)
            .map(|t| {
                let m = Arc::clone(&m);
                let id = id.clone();
                thread::spawn(move || {
                    for i in 0..50u64 {
                        let k = format!("t{t}-k{i}");
                        m.put_start(k.clone(), id.clone(), 64, &ReplicateConfig::replicas(1))
                            .unwrap();
                        m.put_end(&k).unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        assert_eq!(m.object_count(), 4 * 50);

        let readers: Vec<_> = (0..4u64)
            .map(|t| {
                let m = Arc::clone(&m);
                let id = id.clone();
                thread::spawn(move || {
                    for i in 0..50u64 {
                        let k = format!("t{t}-k{i}");
                        assert_eq!(m.get_replica_list(&k, &id).unwrap().len(), 1);
                    }
                })
            })
            .collect();
        for r in readers {
            r.join().unwrap();
        }
    }
}
