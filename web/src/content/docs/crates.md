---
title: Crates
description: The five crates that make up the QuillCache workspace, and what each does.
---

QuillCache is a Cargo workspace of **five crates**. The CUDA crate is excluded
from the default build (it needs an NVIDIA toolchain); everything else builds
hardware-free.

| crate | role |
| --- | --- |
| `quillcache` (bin) | the OpenAI-compatible **gateway** (proxy · cache-aware routing · streaming · SLO), the local **cluster** demo, and `bench-index` |
| `quillcache-core` | `KvBlockKey` / `IdentityScope` identity, `CostModel`, `ReuseViolation`; the `router` (incl. `DynamoCostRouter`), `control` plane, `DataPlane` + `IndexBackend` traits, the ART-vs-LSM `bench`, and the feature-gated `index_holt` / `index_rocksdb` backends |
| `quillcache-transfer-engine` | faithful port of Mooncake's Transfer Engine: `TransferEngine` + `MultiTransport` + `Transport` (`tcp` real / `rdma` reserved) + `TransferMetadata` + `Topology` |
| `quillcache-store` | faithful port of `mooncake-store`: `Client`, `MasterService`, `OffsetBufferAllocator`, `AllocationStrategy`, `Replica`, the crash-consistent `DiskTier`, plus `LocalKvStore` (byte pool) + `StoreDataPlane` (tiers) |
| `quillcache-cuda` | CUDA device tier: HBM↔host copies + FP8 quantize-on-offload (feature-gated, excluded from the default workspace) |

The two index backends (`index_holt`, `index_rocksdb`) are **feature-gated modules
inside `quillcache-core`**, off by default — `holt` is pure Rust; `rocksdb` pulls a
C++/libclang toolchain — so the default build needs neither. They are not separate
crates.

## The seams

A few traits keep the system pluggable and testable:

- **`IndexBackend`** (`quillcache-core`) — the residency index. Implementations:
  `MemoryIndex` (reference), Holt (persistent ART), RocksDB (LSM). The same trait
  lets the [storage study](/storage-study/) compare engines apples-to-apples.
- **`DataPlane`** (`quillcache-core`) — the KV byte tier manager. `StoreDataPlane`
  implements it over per-worker `LocalKvStore` byte pools, so `place()` moves
  real bytes between HBM/DRAM/SSD tiers.
- **`Transport`** (`quillcache-transfer-engine`) — the wire under the Transfer
  Engine. `TcpTransport` moves bytes one-sidedly by `(segment, offset)` today;
  `RdmaTransport` is reserved behind the same trait.
- **`TransferMetadata`** (`quillcache-transfer-engine`) — segment / topology
  discovery: `InMemoryMetadata` now, etcd / redis / http pluggable behind the
  trait.
