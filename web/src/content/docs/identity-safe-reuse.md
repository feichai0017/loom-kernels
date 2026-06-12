---
title: Identity-safe reuse
description: Why content-hash caches leak across tenants and adapters, and the identity guard that costs ~1.7%.
---

A KV block's **content hash** is computed from its tokens, so the same tokens
produce the same hash — *regardless of which tenant sent them, which LoRA adapter
is active, or which model/tokenizer version is loaded*. But the KV **tensors**
depend on all of those. A cache that reuses on content hash alone — which is what
the data-plane caches (Mooncake / LMCache / KVBM) key on — will serve blocks it
must not:

- across **tenants** → a **privacy leak** (one tenant's cached state served to another);
- across **adapters / models / tokenizers** → a **correctness error** (numerically wrong KV).

## The guard

QuillCache makes the reuse contract explicit. Every block carries an
`IdentityScope` (model · tokenizer · adapter · tenant), and `LocalKvStore::get`
serves a block only when the requester's identity matches:

```rust
pub fn get(&mut self, key: &KvBlockKey) -> Result<Bytes, StoreError> {
    // exact identity + content match -> serve.
    // content resident under a *different* identity -> refuse:
    //     Err(StoreError::Unsafe(ReuseViolation::Tenant | Adapter | Model | Tokenizer))
    // otherwise -> Err(StoreError::NotFound)
}
```

The same check runs inline on the live gateway: after a `tenant-a` request caches
a prefix, a `tenant-b` request for the *same content* returns
`x-quillcache-local-hits: 0` and `x-quillcache-reuse-refused: 2` — it refuses to
serve tenant A's KV to tenant B, and says so.

## Measured

On a collision-heavy workload (one popular prefix shared across many identities —
the multi-tenant shared-system-prompt / shared-RAG case):

| policy | content-hash hits | unsafe served | safe reuse kept |
| --- | --- | --- | --- |
| naive (content hash only) | 12400 | **12000 (96.8%)** | — |
| **identity guard** | — | **0** | 4800 |

The guard eliminates **all** unsafe reuse while preserving safe same-identity
reuse. And it is precise, not blunt: on a realistic mostly-same-identity mix the
overhead — forced recomputes as a fraction of all reuse work — drops to **1.7%**
(it is only 47.8% on the adversarial all-collision case). Safety is near-free
exactly where it matters.
