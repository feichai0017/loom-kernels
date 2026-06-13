---
title: Identity-safe reuse
description: Why content-hash caches leak across tenants and adapters, and the identity guard that refuses it — in memory, on disk, and at the master.
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
`IdentityScope` (model · tokenizer · adapter · tenant), and a block is served only
when the requester's identity matches. The check is the same at **every** serving
point:

- **`LocalKvStore::get`** — the in-memory byte tier;
- **`DiskTier::get`** — the durable on-disk tier (so the guard holds after a crash
  and recovery, not just in RAM);
- **`MasterService::get_replica_list`** — the metadata layer, which refuses a
  cross-identity request *before* any bytes move over the transfer engine.

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

## Precise, not blunt

The guard is keyed on identity, not content, so it is **precise**: a
same-identity request still gets its cache hit — only a genuine cross-identity
match (a different tenant / adapter / model / tokenizer for the same tokens) is
refused. On the multi-tenant shared-system-prompt / shared-RAG case, where one
popular prefix is shared across many identities, a naive content-hash cache would
serve that prefix across all of them; the guard serves **zero** unsafe reuse while
keeping every safe same-identity hit.

It is the **same guard** in memory (`LocalKvStore`), on disk (`DiskTier`), and at
the master (`MasterService`) — and it still holds after a crash and recovery. This
is QuillCache's addition; Mooncake's keys are identity-agnostic.
