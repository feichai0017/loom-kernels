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
//! mismatched identity is refused with [`ErrorCode::UnsafeReuse`] — Mooncake keys
//! are identity-agnostic, so this is our addition.

use crate::allocation_strategy::{create_allocation_strategy, AllocationStrategy};
use crate::allocator::{AllocatedBuffer, BufferAllocator, OffsetBufferAllocator};
use crate::replica::{Replica, ReplicaData, ReplicaList, ReplicaStatus};
use crate::types::{ErrorCode, ObjectKey, ReplicaId, ReplicateConfig, SegmentName};
use quillcache_core::IdentityScope;
use std::collections::HashMap;

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

/// The store's metadata / allocation / eviction authority.
#[derive(Debug)]
pub struct MasterService {
    /// Mounted segments → their buffer allocators (one per segment).
    allocators: Vec<Box<dyn BufferAllocator>>,
    objects: HashMap<ObjectKey, ObjectMetadata>,
    strategy: Box<dyn AllocationStrategy>,
    next_replica_id: ReplicaId,
    clock: u64,
    lease_ttl: u64,
    high_watermark: f64,
    eviction_ratio: f64,
}

impl MasterService {
    pub fn new(strategy: &str) -> Self {
        Self {
            allocators: Vec::new(),
            objects: HashMap::new(),
            strategy: create_allocation_strategy(strategy),
            next_replica_id: 0,
            clock: 0,
            lease_ttl: 5,
            high_watermark: 0.95,
            eviction_ratio: 0.1,
        }
    }

    /// Advance the logical clock (drives leases + LRU without wall-clock, so
    /// tests are deterministic).
    pub fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    // ---- segment lifecycle (Mooncake's MountSegment / UnmountSegment) ----

    pub fn mount_segment(&mut self, name: impl Into<SegmentName>, capacity: u64) {
        self.allocators
            .push(Box::new(OffsetBufferAllocator::new(name, capacity)));
    }

    /// Unmount a segment: drop any replicas living on it (they become
    /// unreadable), then remove its allocator.
    pub fn unmount_segment(&mut self, name: &str) -> Result<(), ErrorCode> {
        if !self.allocators.iter().any(|a| a.segment_name() == name) {
            return Err(ErrorCode::SegmentNotFound);
        }
        for obj in self.objects.values_mut() {
            obj.replicas.retain(|_, r| r.segment_name() != Some(name));
        }
        self.allocators.retain(|a| a.segment_name() != name);
        // Objects left with no replicas are gone.
        self.objects.retain(|_, o| !o.replicas.is_empty());
        Ok(())
    }

    // ---- two-phase Put ----

    /// Phase 1: allocate `config.replica_num` replicas for `key` and return their
    /// buffers; the object is recorded `Initialized` (not yet readable).
    pub fn put_start(
        &mut self,
        key: ObjectKey,
        identity: IdentityScope,
        size: u64,
        config: &ReplicateConfig,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        if let Some(existing) = self.objects.get(&key) {
            if existing.has_complete_replica() {
                return Err(ErrorCode::ObjectAlreadyExists);
            }
        }
        // Reclaim any in-flight leftover for this key before re-allocating.
        if let Some(old) = self.objects.remove(&key) {
            self.free_replicas(&old.replicas);
        }

        let preferred = config.preferred_segment.as_deref();
        let buffers = self.allocate_replicas(size, config.replica_num, preferred)?;

        let mut replicas = ReplicaList::new();
        for buffer in &buffers {
            let id = self.next_replica_id;
            self.next_replica_id += 1;
            replicas.insert(id, Replica::new(id, ReplicaData::Memory(buffer.clone())));
        }
        let now = self.clock;
        self.objects.insert(
            key,
            ObjectMetadata {
                replicas,
                identity,
                lease_until: 0,
                last_access: now,
                soft_pinned: config.with_soft_pin,
                hard_pinned: config.with_hard_pin,
            },
        );
        Ok(buffers)
    }

    /// Phase 2: flip the object's replicas to `Complete` (readable).
    pub fn put_end(&mut self, key: &str) -> Result<(), ErrorCode> {
        let object = self.objects.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
        for replica in object.replicas.values_mut() {
            replica.status = ReplicaStatus::Complete;
        }
        Ok(())
    }

    /// Abort an in-flight Put: free the allocation and drop the object.
    pub fn put_revoke(&mut self, key: &str) -> Result<(), ErrorCode> {
        let object = self.objects.remove(key).ok_or(ErrorCode::ObjectNotFound)?;
        self.free_replicas(&object.replicas);
        Ok(())
    }

    // ---- read path (identity-guarded) ----

    /// Return the object's complete replicas, **identity-guarded**, and grant a
    /// read lease. Refuses a requester whose identity doesn't match the writer's.
    pub fn get_replica_list(
        &mut self,
        key: &str,
        requester: &IdentityScope,
    ) -> Result<Vec<Replica>, ErrorCode> {
        let now = self.clock;
        let lease_ttl = self.lease_ttl;
        let object = self.objects.get_mut(key).ok_or(ErrorCode::ObjectNotFound)?;
        // QuillCache identity guard: a content-hash key can be requested under a
        // different identity — refuse cross-tenant / cross-adapter / cross-model.
        if let Some(violation) = object.identity.reuse_violation_against(requester) {
            return Err(ErrorCode::UnsafeReuse(violation));
        }
        let complete: Vec<Replica> = object
            .replicas
            .values()
            .filter(|r| r.is_complete())
            .cloned()
            .collect();
        if complete.is_empty() {
            return Err(ErrorCode::ObjectNotReady);
        }
        object.last_access = now;
        object.lease_until = now + lease_ttl;
        Ok(complete)
    }

    pub fn exist_key(&self, key: &str) -> bool {
        self.objects
            .get(key)
            .is_some_and(ObjectMetadata::has_complete_replica)
    }

    /// Remove an object and free its replicas. Blocked while a read lease is
    /// active unless `force`.
    pub fn remove(&mut self, key: &str, force: bool) -> Result<(), ErrorCode> {
        let leased = {
            let object = self.objects.get(key).ok_or(ErrorCode::ObjectNotFound)?;
            object.lease_until > self.clock
        };
        if leased && !force {
            return Err(ErrorCode::ObjectNotReady);
        }
        let object = self.objects.remove(key).unwrap();
        self.free_replicas(&object.replicas);
        Ok(())
    }

    // ---- observability ----

    pub fn segment_count(&self) -> usize {
        self.allocators.len()
    }
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }
    pub fn capacity(&self) -> u64 {
        self.allocators.iter().map(|a| a.capacity()).sum()
    }
    pub fn allocated(&self) -> u64 {
        self.allocators.iter().map(|a| a.allocated()).sum()
    }

    // ---- internals ----

    /// Allocate `replica_num` buffers of `size`; on segment exhaustion, evict the
    /// coldest unpinned objects to make room and retry once (Mooncake evicts at a
    /// high watermark or on put failure).
    fn allocate_replicas(
        &mut self,
        size: u64,
        replica_num: usize,
        preferred: Option<&str>,
    ) -> Result<Vec<AllocatedBuffer>, ErrorCode> {
        self.evict_if_needed();
        match self
            .strategy
            .allocate(&mut self.allocators, size, replica_num, preferred, &[])
        {
            Ok(buffers) => Ok(buffers),
            Err(ErrorCode::NoAvailableSegment) => {
                self.evict_to_fit(size.saturating_mul(replica_num as u64));
                self.strategy
                    .allocate(&mut self.allocators, size, replica_num, preferred, &[])
            }
            Err(other) => Err(other),
        }
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

    /// Eviction candidates, coldest first; non-soft-pinned before soft-pinned;
    /// hard-pinned and currently-leased objects are never candidates.
    fn victims_coldest_first(&self) -> Vec<ObjectKey> {
        let now = self.clock;
        let mut victims: Vec<(ObjectKey, u64, bool)> = self
            .objects
            .iter()
            .filter(|(_, o)| !o.hard_pinned && o.lease_until <= now)
            .map(|(k, o)| (k.clone(), o.last_access, o.soft_pinned))
            .collect();
        victims.sort_by(|a, b| a.2.cmp(&b.2).then(a.1.cmp(&b.1)));
        victims.into_iter().map(|(k, _, _)| k).collect()
    }

    /// Proactive watermark eviction: when usage exceeds the high watermark, evict
    /// the coldest unpinned objects down below the target.
    pub fn evict_if_needed(&mut self) -> usize {
        let capacity = self.capacity();
        if capacity == 0 || (self.allocated() as f64) < self.high_watermark * capacity as f64 {
            return 0;
        }
        let target = (self.high_watermark * (1.0 - self.eviction_ratio) * capacity as f64) as u64;
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if self.allocated() <= target {
                break;
            }
            if let Some(object) = self.objects.remove(&key) {
                self.free_replicas(&object.replicas);
                evicted += 1;
            }
        }
        evicted
    }

    /// On-demand eviction: evict coldest unpinned objects until at least `needed`
    /// bytes are free cluster-wide (best effort).
    fn evict_to_fit(&mut self, needed: u64) -> usize {
        let mut evicted = 0;
        for key in self.victims_coldest_first() {
            if self.capacity().saturating_sub(self.allocated()) >= needed {
                break;
            }
            if let Some(object) = self.objects.remove(&key) {
                self.free_replicas(&object.replicas);
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
        let mut m = MasterService::new("random");
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
}
