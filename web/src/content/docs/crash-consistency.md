---
title: Crash-consistent tier
description: The SSD tier survives a restart with object-first atomic publish + a WAL — proven by test.
---

The SSD tier holds real KV bytes that should survive a restart, so it needs a
durable, **crash-consistent** catalog. Mooncake's pool is mostly volatile DRAM
(rebuilt on restart); QuillCache occupies the seam it leaves open — a durable,
immediately-reusable persistent tier — using the same pattern as the author's
NoKV distributed file system: **object-first atomic publish + a WAL**.

## How a block is committed

```text
demote_to_ssd(block):
  1. write the block file        -> fsync          (object-first: data is durable)
  2. append a Commit{key,file,len,crc} to the WAL
  3. fsync the WAL                                  (the single atomic publish point)
```

A block is "live" *only* once its commit record is fsynced. Eviction is the
reverse — a `Remove` tombstone is appended (and fsynced) before the file is
deleted, so a crash in between leaves a tombstoned orphan for GC, never a live
pointer to a deleted file.

## How recovery works

```text
recover(dir):
  replay the WAL  -> the live commit set (last write per key wins)
  for each commit -> verify the file exists AND matches recorded length + CRC
                     pass -> re-enter the index
                     fail -> drop it (orphan, GC-able)
```

The WAL is framed `[len][crc][payload]`, so a torn write at the crash point fails
its CRC and replay simply stops there — everything before it is intact.

## The invariants, proven by test

`ssd_tier_survives_crash_and_rejects_half_written_or_corrupt_blocks` simulates a
process death (drop the in-memory store; files + WAL remain) and recovers:

- a **complete** block recovers and serves the correct bytes (identity-guarded);
- a **half-written / uncommitted** block (a file with no commit record) is **never served**;
- a **corrupted** block (length / CRC mismatch) is **dropped** on recovery;
- a missing file never becomes a **dangling pointer** — the recovered index has no stale entries.

This is the concrete answer to "why persist the index at all?" — the moment the
pool has a durable tier, you need a durable, crash-consistent catalog to know
which on-disk blocks are complete and safe to reuse after a restart. An in-memory
index plus durable bytes loses that catalog on the crash.
