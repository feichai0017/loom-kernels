---
title: Crates
description: The nine crates that make up the QuillCache workspace, and what each does.
---

QuillCache is a Cargo workspace. The CUDA crate is excluded from the default
build (it needs an NVIDIA toolchain); everything else builds hardware-free.

| crate | role |
| --- | --- |
| `quillcache-gateway` | OpenAI-compatible gateway: proxy, streaming, decision headers, SLO goodput |
| `quillcache-control` | control plane: `plan()` / `observe_placement` / `audit_reuse` |
| `quillcache-router` | routing policies incl. `DynamoCostRouter` (+ greedy / SLO-aware / session / prefix-affinity / round-robin) |
| `quillcache-core` | `KvBlockKey` identity, `CostModel`, the `IndexBackend` + `DataPlane` traits, and the ART-vs-LSM `bench` |
| `quillcache-store` | `LocalKvStore` (crash-consistent byte pool), `StoreDataPlane` (tiers), `PooledStore`, `NodeRegistry` |
| `quillcache-transfer` | transfer engine: `LocalTransfer` / `TcpTransfer` / `RdmaTransfer` (reserved) |
| `quillcache-index-holt` | Holt (persistent ART) index backend |
| `quillcache-index-rocksdb` | RocksDB (LSM) index backend |
| `quillcache-cuda` | CUDA device tier: HBM↔host copies + FP8 quantize-on-offload (feature-gated, excluded from the workspace) |

## The two seams

Two traits make the system pluggable and testable:

- **`IndexBackend`** (`quillcache-core`) — the residency index. Implementations:
  `MemoryIndex` (reference), Holt (persistent ART), RocksDB (LSM). The same trait
  lets the [storage study](/storage-study/) compare engines apples-to-apples.
- **`DataPlane`** (`quillcache-core`) — the KV byte tier manager. `StoreDataPlane`
  implements it over per-worker `LocalKvStore` byte pools, so `place()` moves
  real bytes between HBM/DRAM/SSD tiers.

The **transfer engine** (`Transfer` trait) and the **node registry**
(`NodeRegistry` trait) are the seams for the distributed read path: TCP today,
RDMA reserved; an in-memory registry now, etcd pluggable behind the trait.
