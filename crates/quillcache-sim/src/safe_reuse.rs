//! Safe-reuse experiment — QuillCache's identity-governed reuse spike.
//!
//! Data-plane KV caches (FlexKV / LMCache / KVBM) reuse a block when its
//! **content hash** matches. But content hashes collide across identities — the
//! same tokens produce the same `block_hash` regardless of which tenant sent
//! them or which LoRA adapter is active — while the KV *tensors* do not. So
//! content-only reuse serves blocks it must not:
//! - across **tenants**: a privacy leak (one tenant's cached state served to
//!   another),
//! - across **adapters / models / tokenizers**: a correctness error (the cached
//!   KV is numerically wrong for the request).
//!
//! This experiment replays a workload where one popular prefix is requested by
//! many identities (the collision), and compares two reuse policies on the
//! *same* trace:
//! - **naive**: reuse on `block_hash` alone (what a data-plane cache keys on),
//! - **identity-guarded**: QuillCache's [`IdentityScope`]-checked reuse.
//!
//! It reports how many unsafe blocks the naive policy serves (split into privacy
//! leaks vs correctness errors), that the guard serves zero, and the cost of the
//! guard: the recomputes it forces — while still preserving safe, same-identity
//! reuse.

use quillcache_core::{CostModel, IdentityScope, KvBlockKey, ReuseViolation};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafeReuseConfig {
    /// Distinct popular prefixes (e.g. shared system prompts / RAG docs).
    pub distinct_prefixes: u32,
    /// Blocks per prefix.
    pub prefix_blocks: u32,
    /// Tenants that each send the shared prefix (the cross-tenant / privacy axis).
    pub tenants: u32,
    /// Extra LoRA adapters beyond the base model on tenant 0 (the cross-adapter /
    /// correctness axis).
    pub adapters: u32,
    /// Extra model variants on tenant 0 — e.g. quantizations (fp16 vs int8 vs
    /// fp8) or fine-tunes — that share token content but not KV numerics (the
    /// cross-model / correctness axis; quantization is modeled as a model id).
    pub models: u32,
    /// Extra tokenizer versions on tenant 0 — the same text tokenizes
    /// differently, so the "same" prefix is not the same token sequence (the
    /// cross-tokenizer / correctness axis).
    pub tokenizers: u32,
    /// Requests per identity. `>= 2` exercises safe same-identity reuse so the
    /// guard is shown to be *precise*, not just restrictive.
    pub repeats: u32,
    /// Tokens per block (drives the recompute cost of the guard).
    pub block_tokens: u32,
}

impl Default for SafeReuseConfig {
    fn default() -> Self {
        Self {
            distinct_prefixes: 50,
            prefix_blocks: 8,
            tenants: 8,
            adapters: 4,
            models: 2,
            tokenizers: 2,
            repeats: 2,
            block_tokens: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SafeReuseReport {
    pub identities: u64,
    pub blocks_evaluated: u64,
    /// Content-hash hits a naive (data-plane-style) cache would serve.
    pub naive_reuses: u64,
    /// Of those, how many cross an identity boundary and are unsafe.
    pub naive_unsafe: u64,
    /// Privacy leaks: same content, different tenant.
    pub unsafe_cross_tenant: u64,
    /// Correctness errors: same content, different adapter.
    pub unsafe_cross_adapter: u64,
    pub unsafe_cross_model: u64,
    pub unsafe_cross_tokenizer: u64,
    /// Token-blocks of state the naive policy serves unsafely.
    pub unsafe_tokens_served_by_naive: u64,
    /// Identity-guarded reuses (QuillCache): every one is safe.
    pub safe_reuses: u64,
    /// Recomputes the guard forces by refusing a cross-identity content hit —
    /// the cost of safety.
    pub guard_recomputes: u64,
    /// Unsafe serves the guard eliminates (it serves zero unsafe). Equals
    /// `naive_unsafe`.
    pub unsafe_blocks_avoided: u64,
    /// Prefill cost of the guard's forced recomputes, in milliseconds.
    pub guard_recompute_ms: f64,
    /// Guard overhead = forced recomputes / (forced recomputes + safe reuses).
    /// Small when same-identity reuse dominates — the realistic case — so safety
    /// is near-free; large only on adversarial all-collision workloads.
    pub safety_overhead_pct: f64,
}

fn identity(model: &str, tok: &str, adapter: Option<String>, tenant: String) -> IdentityScope {
    IdentityScope {
        model_id: model.to_string(),
        tokenizer_id: tok.to_string(),
        adapter_id: adapter,
        tenant_id: tenant,
    }
}

fn block_key(scope: &IdentityScope, content: &str, block_index: u32, tokens: u32) -> KvBlockKey {
    KvBlockKey {
        model_id: scope.model_id.clone(),
        tokenizer_id: scope.tokenizer_id.clone(),
        adapter_id: scope.adapter_id.clone(),
        tenant_id: scope.tenant_id.clone(),
        prefix_hash: content.to_string(),
        block_hash: content.to_string(),
        block_index,
        token_count: tokens,
    }
}

/// Run the safe-reuse experiment: naive content reuse vs identity-guarded reuse
/// on the same cross-identity workload.
pub fn run_safe_reuse(config: SafeReuseConfig) -> SafeReuseReport {
    let cost = CostModel::default();
    let (model, tok) = ("bench-model", "bench-tok");

    // Distinct identities that all request the same content. The base identity
    // (model, tok, no adapter, tenant-0) is listed first, so it is the cache's
    // first writer and every other identity collides against it on exactly one
    // axis — clean per-axis attribution:
    //   - one base identity per tenant   -> cross-tenant (privacy),
    //   - extra LoRA adapters on tenant 0 -> cross-adapter (correctness),
    //   - extra model variants / quants   -> cross-model (correctness),
    //   - extra tokenizer versions        -> cross-tokenizer (correctness).
    let mut identities: Vec<IdentityScope> = Vec::new();
    for t in 0..config.tenants.max(1) {
        identities.push(identity(model, tok, None, format!("tenant-{t}")));
    }
    for a in 1..=config.adapters {
        identities.push(identity(
            model,
            tok,
            Some(format!("lora-{a}")),
            "tenant-0".to_string(),
        ));
    }
    for m in 1..=config.models {
        identities.push(identity(
            &format!("{model}-q{m}"),
            tok,
            None,
            "tenant-0".to_string(),
        ));
    }
    for k in 1..=config.tokenizers {
        identities.push(identity(
            model,
            &format!("{tok}-v{k}"),
            None,
            "tenant-0".to_string(),
        ));
    }

    // Naive cache: content hash -> the (full-identity) block of its first writer.
    let mut naive_cache: HashMap<String, KvBlockKey> = HashMap::new();
    // Identity-guarded cache: (scope, content) presence.
    let mut safe_cache: HashSet<(IdentityScope, String)> = HashSet::new();

    let mut r = SafeReuseReport {
        identities: identities.len() as u64,
        blocks_evaluated: 0,
        naive_reuses: 0,
        naive_unsafe: 0,
        unsafe_cross_tenant: 0,
        unsafe_cross_adapter: 0,
        unsafe_cross_model: 0,
        unsafe_cross_tokenizer: 0,
        unsafe_tokens_served_by_naive: 0,
        safe_reuses: 0,
        guard_recomputes: 0,
        unsafe_blocks_avoided: 0,
        guard_recompute_ms: 0.0,
        safety_overhead_pct: 0.0,
    };

    let repeats = config.repeats.max(1);
    for p in 0..config.distinct_prefixes {
        for block in 0..config.prefix_blocks {
            let content = format!("pfx{p}-blk{block}");
            for scope in &identities {
                for _ in 0..repeats {
                    r.blocks_evaluated += 1;

                    // --- naive content-hash reuse ---
                    match naive_cache.get(&content) {
                        Some(owner) => {
                            r.naive_reuses += 1;
                            if let Some(violation) = scope.reuse_violation(owner) {
                                r.naive_unsafe += 1;
                                r.unsafe_tokens_served_by_naive += u64::from(config.block_tokens);
                                match violation {
                                    ReuseViolation::Tenant => r.unsafe_cross_tenant += 1,
                                    ReuseViolation::Adapter => r.unsafe_cross_adapter += 1,
                                    ReuseViolation::Model => r.unsafe_cross_model += 1,
                                    ReuseViolation::Tokenizer => r.unsafe_cross_tokenizer += 1,
                                }
                            }
                        }
                        None => {
                            naive_cache.insert(
                                content.clone(),
                                block_key(scope, &content, block, config.block_tokens),
                            );
                        }
                    }

                    // --- identity-guarded reuse (QuillCache) ---
                    let safe_key = (scope.clone(), content.clone());
                    if safe_cache.contains(&safe_key) {
                        r.safe_reuses += 1;
                    } else {
                        // First time this identity sees this content. If the naive
                        // cache holds it under a *different* identity, the guard
                        // just turned a (would-be) reuse into a recompute.
                        if let Some(owner) = naive_cache.get(&content) {
                            if scope.reuse_violation(owner).is_some() {
                                r.guard_recomputes += 1;
                            }
                        }
                        safe_cache.insert(safe_key);
                    }
                }
            }
        }
    }

    r.unsafe_blocks_avoided = r.naive_unsafe;
    r.guard_recompute_ms =
        r.guard_recomputes as f64 * cost.prefill_cost_us(config.block_tokens) as f64 / 1_000.0;
    let guard_vs_reuse = r.guard_recomputes + r.safe_reuses;
    r.safety_overhead_pct = if guard_vs_reuse > 0 {
        100.0 * r.guard_recomputes as f64 / guard_vs_reuse as f64
    } else {
        0.0
    };
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naive_reuse_serves_unsafe_blocks_the_guard_eliminates() {
        let config = SafeReuseConfig {
            distinct_prefixes: 50,
            prefix_blocks: 8,
            tenants: 8,
            adapters: 4,
            models: 0,
            tokenizers: 0,
            repeats: 2,
            block_tokens: 64,
        };
        let r = run_safe_reuse(config);

        let units = config.prefix_blocks * config.distinct_prefixes; // 8 * 50
        let units = u64::from(units);
        let repeats = u64::from(config.repeats);
        let identities = u64::from(config.tenants + config.adapters); // 8 + 4 = 12
        let cross_tenant = u64::from(config.tenants - 1); // collide against tenant 0
        let cross_adapter = u64::from(config.adapters);
        let unsafe_per = cross_tenant + cross_adapter; // 11 unsafe identities per block

        assert_eq!(r.identities, identities);
        assert_eq!(r.blocks_evaluated, identities * repeats * units);

        // Each unsafe identity serves once per repeat.
        assert_eq!(r.unsafe_cross_tenant, repeats * cross_tenant * units);
        assert_eq!(r.unsafe_cross_adapter, repeats * cross_adapter * units);
        assert_eq!(r.naive_unsafe, repeats * unsafe_per * units);
        assert_eq!(r.unsafe_cross_model, 0);

        // The guard serves zero unsafe, forcing one recompute per first
        // cross-identity occurrence, and avoids every unsafe serve.
        assert_eq!(r.guard_recomputes, unsafe_per * units);
        assert_eq!(r.unsafe_blocks_avoided, r.naive_unsafe);
        assert!(r.guard_recompute_ms > 0.0);

        // The guard is precise: same-identity repeats still reuse safely.
        assert_eq!(r.safe_reuses, identities * (repeats - 1) * units);
    }

    #[test]
    fn single_identity_has_no_unsafe_reuse() {
        let config = SafeReuseConfig {
            distinct_prefixes: 4,
            prefix_blocks: 4,
            tenants: 1,
            adapters: 0,
            models: 0,
            tokenizers: 0,
            repeats: 3,
            block_tokens: 32,
        };
        let r = run_safe_reuse(config);
        // One identity, no collisions: nothing unsafe, nothing for the guard to
        // recompute — but safe self-reuse still happens.
        assert_eq!(r.identities, 1);
        assert_eq!(r.naive_unsafe, 0);
        assert_eq!(r.guard_recomputes, 0);
        assert_eq!(r.guard_recompute_ms, 0.0);
        let units = u64::from(config.prefix_blocks * config.distinct_prefixes);
        assert_eq!(r.safe_reuses, u64::from(config.repeats - 1) * units);
    }

    #[test]
    fn safety_is_near_free_when_same_identity_reuse_dominates() {
        // Realistic: a handful of identities, mostly the same one reusing a lot.
        let realistic = run_safe_reuse(SafeReuseConfig {
            distinct_prefixes: 50,
            prefix_blocks: 8,
            tenants: 2,
            adapters: 1,
            models: 0,
            tokenizers: 0,
            repeats: 40,
            block_tokens: 64,
        });
        // Adversarial: every identity collides on every block (the default).
        let adversarial = run_safe_reuse(SafeReuseConfig::default());

        assert!(
            realistic.safety_overhead_pct < 5.0,
            "realistic overhead too high: {}%",
            realistic.safety_overhead_pct
        );
        assert!(
            adversarial.safety_overhead_pct > 25.0,
            "adversarial overhead too low: {}%",
            adversarial.safety_overhead_pct
        );
    }

    #[test]
    fn cross_model_and_tokenizer_collisions_are_classified() {
        // Only model and tokenizer variants collide (no extra tenants/adapters).
        let config = SafeReuseConfig {
            distinct_prefixes: 10,
            prefix_blocks: 4,
            tenants: 1,
            adapters: 0,
            models: 3, // e.g. fp16 vs int8 vs fp8 vs another quant
            tokenizers: 2,
            repeats: 1,
            block_tokens: 64,
        };
        let r = run_safe_reuse(config);
        let units = u64::from(config.prefix_blocks * config.distinct_prefixes);

        // 1 base + 3 model variants + 2 tokenizer versions = 6 identities.
        assert_eq!(r.identities, 6);
        // Each model/tokenizer variant collides against the base once per block.
        assert_eq!(r.unsafe_cross_model, u64::from(config.models) * units);
        assert_eq!(
            r.unsafe_cross_tokenizer,
            u64::from(config.tokenizers) * units
        );
        assert_eq!(r.unsafe_cross_tenant, 0);
        assert_eq!(r.unsafe_cross_adapter, 0);
        assert_eq!(
            r.naive_unsafe,
            u64::from(config.models + config.tokenizers) * units
        );
    }
}
