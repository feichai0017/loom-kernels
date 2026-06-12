---
title: Mooncake / Dynamo mapping
description: Every Mooncake and NVIDIA Dynamo component, mapped to a QuillCache crate.
---

QuillCache replicates the production reference designs piece by piece, then adds
its differentiation on top. The concepts line up one-to-one — built small enough
to read end-to-end and measure.

| Mooncake / NVIDIA Dynamo | QuillCache | Status |
| --- | --- | --- |
| Mooncake Store (pooled DRAM/SSD KV) | `LocalKvStore` + `PooledStore` | ✅ real bytes |
| Mooncake Transfer Engine | `quillcache-transfer` | ✅ TCP / ⊙ RDMA reserved |
| Conductor / scheduler | `quillcache-control` + router | ✅ |
| Dynamo KV-router cost function | `DynamoCostRouter` | ✅ reproduces the worked example |
| Dynamo KVBM tiers (G1/G2/G3) | `StoreDataPlane` (HBM/DRAM/SSD) | ✅ moves real bytes |
| Dynamo KV-Cache Indexer | residency index (Holt ART) | ✅ persistent |
| Dynamo etcd / service discovery | `NodeRegistry` (`StaticRegistry`) | ✅ etcd pluggable |
| — *(neither does this)* | **identity guard + crash-consistency** | 🎯 differentiation |

## The Dynamo cost function

`DynamoCostRouter` reproduces the cost function NVIDIA Dynamo's KV router runs.
For each worker:

```text
overlap_credit   = 1.0·device + 0.75·host + 0.25·disk   (HBM / DRAM / SSD hits)
adjusted_prefill = max(0, raw_prefill_blocks − overlap_credit)
cost             = prefill_load_scale · adjusted_prefill + decode_blocks
```

It routes to the lowest-cost worker. A GPU-resident prefix hit is worth 4× an SSD
one — cache locality vs load, as a single number — and a unit test reproduces
Dynamo's own published worked example (costs 18 / 10 / 11 → pick worker 2).

## The distributed read path

The pooled read mirrors Mooncake's Conductor → metadata → Transfer Engine flow:

1. the residency index **locates** which nodes hold the block (`index.locate`);
2. the `NodeRegistry` **resolves** a node id to its transfer address;
3. the **transfer engine** fetches it; the block is admitted locally and served,
   identity-guarded.
