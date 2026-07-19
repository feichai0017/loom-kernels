# Loom Platform Plan v3

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

- remove the built-in store and byte transfer product path;
- establish dependency-clean types, pool, catalog, planner, runtime, attention,
  and tensor-transport modules in one Rust package;
- preserve only a local metadata pool for deterministic tests;
- make implementation status explicit.

Exit: `cargo test --workspace` validates ownership and state-machine invariants.

### M1: Engine-Local Backend

- add a vLLM `AttentionBackend` adapter;
- delegate to the existing local kernel;
- translate vLLM block tables and layouts into Loom types;
- produce a Q tensor handle, optional mutable-tail K_new/V_new append, and a
  generation-pinned `KvView`;
- verify output equality and fallback behavior.

Exit: one real model decodes through the adapter with no remote execution.

### M2: One-Node Route-Q

- place sealed prefix shards on a second GPU;
- send Q through CUDA P2P or NCCL;
- execute remote partial attention;
- return output/LSE statistics and merge with the local active tail.
- compare end-to-end Route-Q and Stage-KV latency under one deterministic
  workload before replacing the reference kernel.

Exit: split execution matches the unsharded reference within dtype tolerance.

### M3: External Pool

- implement the Mooncake `KvPool` adapter;
- publish sealed blocks and consume ordered residency events;
- stage DRAM-resident objects to an attention worker;
- reconcile Holt recovery against live pool generations.

Exit: restart recovery never serves a stale object or handle.

### M4: Cross-Node Data Path

- implement NIXL/UCX or equivalent registered-device transport;
- batch query and partial-result transfers;
- overlap communication, remote kernels, and local-tail attention;
- add topology and NIC-pressure telemetry.

Exit: end-to-end results report TTFT, TPOT, throughput, goodput, queueing, and
communication bytes against fetch-KV and local-only baselines.

### M5: Heterogeneous Executors

- add CPU DRAM attention and capability-aware dispatch;
- evaluate GPU, CPU, and staged SSD paths;
- schedule using queue, transfer, kernel, merge, lease, and deadline costs.

Exit: the planner improves measured SLO goodput over static execution policies.

## Claim Discipline

The reference merge, planner, local pool, and HTTP control endpoints are not GPU
performance evidence. Performance claims require a real model, real accelerator,
specified topology, a reproducible workload, and comparisons against equivalent
vLLM/SGLang and external-pool baselines.
