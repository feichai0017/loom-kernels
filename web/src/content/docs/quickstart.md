---
title: Quick start
description: Build, test, and run QuillCache — the storage study, the gateway, the CUDA tier.
---

QuillCache is a Rust workspace. The default build needs no GPU, no RDMA, and no
C++ toolchain.

## Build & test

```bash
git clone https://github.com/feichai0017/quillcache
cd quillcache
cargo build
cargo test          # 45 tests
```

## The ART-vs-LSM storage study

The residency index is benchmarked across three backends on the same trace.
RocksDB needs a C++ toolchain; Holt (ART) is pure Rust.

```bash
cargo run --features "rocksdb holt" -- bench-index --backend holt
cargo run --features "rocksdb holt" -- bench-index --backend rocksdb
cargo run -- bench-index --backend memory
```

See [the storage study](/storage-study/) for the numbers and what they mean.

## Online mode — the gateway

Run the OpenAI-compatible gateway in front of real engines, backed by a
persistent ART (Holt) residency index that survives restarts:

```bash
cargo run --features holt -- gateway --config examples/quillcache-gateway.yaml
# set `index: holt` in the config for the persistent residency index
```

The gateway proxies `POST /v1/chat/completions` and `POST /v1/completions`,
ingests KV events at `POST /v1/kv-events`, and reports state at `GET /v1/state`.
Each response carries `x-quillcache-*` decision headers (local hits, transfers,
recomputes, refused unsafe reuse, estimated TTFT).

## The CUDA device tier

The CUDA tier (HBM↔host copies + FP16→FP8 quantize-on-offload) is a separate
crate, excluded from the default workspace so the build stays hardware-free.
Build it on a GPU box:

```bash
cd crates/quillcache-cuda
cargo build --features cuda
```
