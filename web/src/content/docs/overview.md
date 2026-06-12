---
title: Overview
description: What QuillCache is, and what's wired online vs a tested unit vs reserved.
---

QuillCache is a **Mooncake/Dynamo-style distributed KV cache pool and control
plane** for LLM serving, written in Rust. It sits beside real inference engines
(vLLM, SGLang) and owns the KV cache as a resource:

- a **byte pool** — DRAM + SSD tiers that hold real KV block bytes, with
  capacity-driven demotion and eviction;
- a **transfer engine** — moves blocks between nodes (TCP today, RDMA reserved);
- a **residency index** — maps each block (by identity) to where it lives
  (node + tier), persistent so it survives a restart;
- a **control plane / Conductor** — routes requests cache-aware (the Dynamo
  KV-router cost function), governs reuse, and meters SLO.

It replicates the architecture of NVIDIA Dynamo and Moonshot's Mooncake, plus
two properties the production data planes leave implicit: **identity-governed
safe reuse** and a **crash-consistent persistent tier**.

:::note[It does not run models]
No transformer kernels, no attention. The CUDA tier moves and quantizes KV
*bytes* (the data path), not inference compute.
:::

## Status — wired online vs tested unit vs reserved

Everything here is real code — there is no simulation. The honest distinction is
how far each piece is integrated:

- **✅ wired online & measured** — gateway, control plane, Dynamo-cost routing,
  persistent residency index, `StoreDataPlane` moving real bytes across
  HBM/DRAM/SSD, the identity guard, live SLO goodput, and the ART-vs-LSM storage
  study.
- **▣ tested unit (not yet on the live gateway path)** — `PooledStore`
  cross-node fetch over TCP, and `LocalKvStore::recover` crash recovery. Both are
  covered by tests; wiring them into the live gateway needs an engine
  KV-connector for the engine⟷pool byte handoff.
- **⊙ reserved / needs hardware** — `RdmaTransfer` (behind the `rdma` feature)
  and the CUDA device tier (build `quillcache-cuda` with `--features cuda` on a
  GPU box). Both are real interfaces, stubbed/fallback so the default build is
  hardware-free.

`cargo test` — 45 tests pass; `cargo fmt --check` and `cargo clippy` are clean.

## Differentiation

The reference designs (Mooncake / Dynamo / LMCache / KVBM) key reuse on a
block's **content hash** and keep the cache mostly volatile. QuillCache adds:

1. **Identity-governed safe reuse** — a block is served only when the requester's
   model · tokenizer · adapter · tenant matches, so cross-tenant leaks and
   cross-adapter/model errors are refused. See
   [Identity-safe reuse](/identity-safe-reuse/).
2. **A crash-consistent persistent tier** — the SSD tier survives a restart with
   object-first atomic publish + a WAL, so durable blocks are immediately
   reusable and corrupt/half-written ones are never served. See
   [Crash-consistent tier](/crash-consistency/).
