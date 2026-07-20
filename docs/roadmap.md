# Roadmap

This document defines planned milestones and exit criteria. For current
implementation status, see [status.md](status.md).

## Research Question

When KV is distributed across local HBM, remote GPU, host DRAM, and persistent
pool tiers, how should serving choose among local attention, query routing, KV
staging, sharded attention, and recomputation while respecting identity,
generation, failures, topology, and SLO constraints?

## Scope

The first system targets long-context decode for Llama-style MHA/GQA models. It
uses an existing inference engine for model execution and an external pool for
sealed KV lifetime. Training, MLA, speculative decoding, and CUDA Graph support
follow only after the decode data path is measured.

## Milestones

### M0: Architecture Break

Status: implementation complete.

- remove the built-in store and byte transfer product path;
- establish dependency-clean types, pool, catalog, planner, runtime, attention,
  and tensor-transport modules in one Rust package;
- preserve only a local metadata pool for deterministic tests;
- make implementation status explicit.

Exit: `cargo test --workspace` validates ownership and state-machine invariants.

### M1: Engine-Local Backend

Status: real-model local delegate acceptance complete on Modal L4; the
metadata-only physical-block bridge is implemented and external pool binding
remains open.

- add a vLLM `AttentionBackend` adapter;
- delegate to the existing local kernel;
- translate vLLM block tables and layouts into Loom types;
- produce a Q tensor handle, optional mutable-tail K_new/V_new append, and a
  generation-pinned `KvView`;
- verify output equality and fallback behavior.

Local acceptance exit: complete. One real model decodes through the adapter with
no remote execution and exact native-output equality.

Physical bridge exit: complete on Modal L4. vLLM scheduler allocation metadata
reaches a generation-checked worker registry, real paged-KV tensors are
registered, and each request-bearing CUSTOM forward validates the active
binding generation without device-to-host table readback.

Integration exit: vLLM physical block IDs resolve to lease-covered
`PoolObjectRef` generations without device-to-host table readback.

### M2a: One-Node Route-Q Protocol

Status: implementation complete with a phase-instrumented Modal L4 4K-32K
prefix sweep; real-model and topology-comparable hardware sweeps remain open.

- place sealed prefix shards on a second GPU;
- send Q through CUDA P2P or NCCL;
- execute remote attention over the sealed prefix;
- return output/LSE state and merge with the local active tail;
- compare end-to-end Route-Q and Stage-KV latency under one deterministic
  workload before replacing the reference kernel.

Exit: split execution matches the unsharded reference within dtype tolerance.

### M2b: Paged-KV Executor

Status: executor, two-GPU correctness gate, and CUDA-event phase timing validated
on Modal L4; external page-table binding and Nsight attribution remain open.

- consume generation-pinned page tables without repacking contiguous KV;
- reuse planned FlashInfer wrappers and workspaces across decode steps;
- return output/LSE state without host synchronization;
- report kernel, transfer, and merge time separately.

Exit: the paged executor passes the same two-GPU correctness gate and produces
a hardware-qualified report.

### M2c: Local-Tail Scheduling And Fusion

Status: handwritten Rust/CUDA/PyTorch operator and single-H20 correctness plus
isolated microbenchmark complete; real two-GPU A/B and production dispatch open.

- overlap local active-tail attention with the remote Route-Q round trip on a
  separate engine CUDA stream;
- fuse bounded local-tail attention with the exact remote-state merge;
- preserve sequential, overlap, and fused modes for matched comparisons;
- integrate the selected path with the real engine attention dispatch.

Exit: all three strategies pass the two-GPU full-attention oracle, have
Nsight-attributed latency reports on the same topology, and select a default
from measured end-to-end benefit rather than isolated kernel speedup.

### M3: External Pool

Status: interface only; production adapter not implemented.

- implement the Mooncake `KvPool` adapter;
- publish sealed blocks and consume ordered residency events;
- stage DRAM-resident objects to an attention worker;
- reconcile Holt recovery against live pool generations.

Exit: restart recovery never serves a stale object or handle.

### M4: Cross-Node Data Path

Status: contracts only; transport not implemented.

- implement NIXL/UCX or equivalent registered-device transport;
- batch query and partial-result transfers;
- overlap communication, remote kernels, and local-tail attention;
- add topology and NIC-pressure telemetry.

Exit: end-to-end results report TTFT, TPOT, throughput, goodput, queueing, and
communication bytes against fetch-KV and local-only baselines.

### M5: Heterogeneous Executors

Status: planned.

- add CPU DRAM attention and capability-aware dispatch;
- evaluate GPU, CPU, and staged SSD paths;
- schedule using queue, transfer, kernel, merge, lease, and deadline costs.

Exit: the planner improves measured SLO goodput over static execution policies.

## Claim Discipline

The reference merge, planner, local pool, and HTTP control endpoints are not GPU
performance evidence. Performance claims require a real model, real accelerator,
specified topology, a reproducible workload, and comparisons against equivalent
vLLM/SGLang and external-pool baselines.
