//! Sharded, per-shard-locked map (Mooncake's `MasterService::metadata_shards_`).
//!
//! The master's object metadata is partitioned into `N` independently-locked
//! shards keyed by `hash(key) % N`, so operations on different keys proceed
//! concurrently instead of serializing on one big lock. This is the foundation
//! of the sharded `MasterService`; it is a standalone, concurrency-tested
//! primitive so the **lock order** is proven before the master is rewired.
//!
//! ## Lock order (deadlock-free by construction)
//! - a **single-key** op locks exactly **one** shard ([`Self::with_shard`]);
//! - a **cross-shard** op locks **every** shard in **ascending index order**
//!   ([`Self::with_all`]).
//!
//! Because every multi-shard acquisition is ascending and single-shard ops take
//! just one lock, no two threads can hold-and-wait in a cycle. (A caller must not
//! lock a second shard from inside [`Self::with_shard`]; the master never does —
//! it pairs a single shard lock with the *separate* allocator lock, in that
//! fixed order.)

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

const DEFAULT_SHARDS: usize = 16;

/// A `String`-keyed map partitioned into independently-locked shards.
#[derive(Debug)]
pub struct ShardedMap<V> {
    shards: Vec<Mutex<HashMap<String, V>>>,
}

impl<V> Default for ShardedMap<V> {
    fn default() -> Self {
        Self::new(DEFAULT_SHARDS)
    }
}

impl<V> ShardedMap<V> {
    /// `num_shards` independently-locked shards (min 1).
    pub fn new(num_shards: usize) -> Self {
        let n = num_shards.max(1);
        Self {
            shards: (0..n).map(|_| Mutex::new(HashMap::new())).collect(),
        }
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_idx(&self, key: &str) -> usize {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut h);
        (h.finish() as usize) % self.shards.len()
    }

    /// Run `f` against the one shard that owns `key`, holding only that shard's
    /// lock — the concurrent hot path. Do not lock another shard inside `f`.
    pub fn with_shard<R>(&self, key: &str, f: impl FnOnce(&mut HashMap<String, V>) -> R) -> R {
        let idx = self.shard_idx(key);
        let mut guard = self.shards[idx].lock().expect("shard mutex poisoned");
        f(&mut guard)
    }

    /// Run `f` with **all** shards locked, acquired in ascending index order
    /// (the only safe way to touch the whole map — eviction scans, snapshots).
    pub fn with_all<R>(
        &self,
        f: impl FnOnce(&mut [std::sync::MutexGuard<'_, HashMap<String, V>>]) -> R,
    ) -> R {
        let mut guards: Vec<_> = self
            .shards
            .iter()
            .map(|s| s.lock().expect("shard mutex poisoned"))
            .collect();
        f(&mut guards)
    }

    /// Total entries across all shards (each shard locked briefly, in order).
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.lock().expect("shard mutex poisoned").len())
            .sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn routing_is_stable_and_roundtrips() {
        let m: ShardedMap<i32> = ShardedMap::new(8);
        m.with_shard("alpha", |s| s.insert("alpha".into(), 1));
        m.with_shard("beta", |s| s.insert("beta".into(), 2));
        // Same key always routes to the same shard → reads find the value.
        assert_eq!(m.with_shard("alpha", |s| s.get("alpha").copied()), Some(1));
        assert_eq!(m.with_shard("beta", |s| s.get("beta").copied()), Some(2));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn with_all_sees_every_shard() {
        let m: ShardedMap<i32> = ShardedMap::new(4);
        for i in 0..50 {
            let k = format!("k{i}");
            m.with_shard(&k, |s| s.insert(k.clone(), i));
        }
        let total: usize = m.with_all(|guards| guards.iter().map(|g| g.len()).sum());
        assert_eq!(total, 50, "with_all observes all shards");
        let sum: i32 = m.with_all(|guards| guards.iter().flat_map(|g| g.values()).sum());
        assert_eq!(sum, (0..50).sum::<i32>());
    }

    #[test]
    fn concurrent_distinct_key_inserts_do_not_deadlock_or_lose() {
        let m: Arc<ShardedMap<u64>> = Arc::new(ShardedMap::new(16));
        let threads: Vec<_> = (0..8u64)
            .map(|t| {
                let m = Arc::clone(&m);
                thread::spawn(move || {
                    for i in 0..200u64 {
                        let k = format!("t{t}-k{i}");
                        m.with_shard(&k, |s| s.insert(k.clone(), t * 1000 + i));
                    }
                })
            })
            .collect();
        for h in threads {
            h.join().unwrap();
        }
        assert_eq!(m.len(), 8 * 200, "no inserts lost under concurrency");
    }

    #[test]
    fn with_all_concurrent_with_with_shard_does_not_deadlock() {
        // A thread repeatedly locking ALL shards (ascending) racing threads that
        // each lock ONE shard must make progress, never deadlock. If the order
        // were inconsistent this test would hang (and CI would time out).
        let m: Arc<ShardedMap<u64>> = Arc::new(ShardedMap::new(8));
        let writers: Vec<_> = (0..4u64)
            .map(|t| {
                let m = Arc::clone(&m);
                thread::spawn(move || {
                    for i in 0..500u64 {
                        let k = format!("t{t}-{i}");
                        m.with_shard(&k, |s| {
                            s.insert(k.clone(), i);
                        });
                    }
                })
            })
            .collect();
        let scanner = {
            let m = Arc::clone(&m);
            thread::spawn(move || {
                let mut last = 0;
                for _ in 0..500 {
                    last = m.with_all(|gs| gs.iter().map(|g| g.len()).sum::<usize>());
                }
                last
            })
        };
        for w in writers {
            w.join().unwrap();
        }
        let _ = scanner.join().unwrap();
        assert_eq!(m.len(), 4 * 500);
    }
}
