---
title: ART vs LSM storage study
description: Which storage engine fits a KV-cache residency index — prefix-scan latency, write amplification, recovery.
---

The residency / prefix index is written on every KV event and read on every
request (longest reusable prefix), and a persistent control plane needs it on
disk. Its workload is **prefix-heavy** (shared system prompts, RAG docs, agent
sessions) and **write-frequent**. Which storage engine fits?

QuillCache makes the index a pluggable, persistent, measured component behind one
`IndexBackend` trait, and asks the question directly. Two natural designs:

- **ART (Holt)** — a radix/trie. Prefix-native (the prefix *is* the path),
  near-memory point/prefix lookups, and **no compaction write amplification**
  (append-only). SGLang's RadixAttention uses a radix tree in memory for exactly
  this; Holt makes it persistent with a WAL.
- **LSM (RocksDB)** — write-optimized via compaction, but compaction causes write
  amplification and prefix scans are less natural (they span levels).

## Results

Same workload, same trait, via `cargo run --features "rocksdb holt" -- bench-index`:

| backend | ingest | prefix-scan p50 | p99 | write-amp | recovery | on-disk |
| --- | --- | --- | --- | --- | --- | --- |
| memory (flat map) | 706k/s | 494 µs | 1685 µs | — | — | 0 |
| rocksdb (LSM) | 56k/s | 16.8 µs | 29.6 µs | **10.6×** | 4.1 ms | small |
| **holt (ART)** | 55k/s | **9.96 µs** | **13.7 µs** | **1.0×** | **2.6 ms** | larger |

**ART wins** prefix-scan latency (~1.7× over LSM at p50, ~50× over the flat map's
O(N) scan), recovery, and write amplification (append-only, 1×). **LSM wins** the
on-disk footprint (compaction reclaims space). Ingest is comparable between the
two persistent backends — the cost of durability.

So pick ART when prefix-scan latency and recovery dominate (the common case for a
residency index queried per request); pick LSM when disk footprint is the
constraint.

## Write amplification, measured

Write amplification is read from RocksDB's own flush/compaction statistics, not
assumed. The tradeoff is exact and opposite: ART writes each record once (1×) but
keeps everything; LSM rewrites data through compaction (10.6×) but reclaims space.
This is the classic append-only-vs-compaction storage tradeoff, here measured for
a KV-cache residency index — a gap a recently published RocksDB-for-KV-cache
approach left unanalyzed.

## A bottleneck the benchmark caught

An eviction-churn phase (`remove_block` + `put` under cache pressure) surfaced
that `remove_block` was **O(scope)** — given a block hash but not its prefix,
every backend scanned the whole identity scope. A secondary `block_hash → primary
key` reverse index made it an **O(matches)** seek, **100–300× faster** eviction on
the persistent backends. The benchmark working as intended: it found the
bottleneck, then measured the fix.
