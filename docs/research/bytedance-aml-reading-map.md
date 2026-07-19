# ByteDance AML / AI Infra Reading Map

This note organizes the local paper set into a study path for Loom. The
goal is not to read papers as isolated summaries. Each paper should produce one
of three artifacts: a design update, an experiment, or an interview-ready
systems explanation.

## Study Frame

AI infra for LLMs has one recurring pattern:

1. Find the real bottleneck: compute, HBM, interconnect, storage, scheduler, or
   safety.
2. Split the system along resource boundaries.
3. Hide communication or I/O behind useful compute.
4. Feed measured runtime signals back into placement and scheduling.
5. Evaluate with SLO goodput, latency tails, utilization, and cost, not only raw
   throughput.

Loom uses the same pattern: the external pool owns sealed KV objects, and
the runtime is designed to choose where core attention executes relative to
those objects. The current adapter still executes attention locally.

## Local Paper Groups

### 1. Module-Level Disaggregation And Communication Overlap

Read first because this is closest to ByteDance AML / Seed systems style.

| Paper | Local file | Core idea | Loom connection |
| --- | --- | --- | --- |
| MegaScale-Infer | `papers/megascale-infer-sigcomm25-2504.02263.pdf` | Disaggregate attention and FFN for MoE serving; use ping-pong pipeline and M2N communication. | Motivates separate model and attention workers plus workload-specific communication. |
| Flux | `papers/flux-comm-overlap-2406.06858.pdf` | Fine-grained communication/computation overlap through kernel fusion. | Motivates measuring exposed Q/O communication after overlap. |
| Comet | `papers/comet-moe-overlap-mlsys25-2502.19811.pdf` | Data-dependency-aware task rescheduling for MoE overlap. | Guides dependency-aware local/remote partial execution and merge. |

What to learn:

- why lower FLOPs can still waste GPUs;
- how to identify memory-bound versus compute-bound modules;
- when splitting modules creates enough batching to recover utilization;
- why custom communication sometimes beats generic collectives;
- how pipeline depth trades latency for throughput.

Loom tasks:

- implement one-node Route-Q and exact output-plus-LSE merge;
- compare Local, RouteQuery, StageKv, and Sharded under one workload;
- measure queueing, Q/O transport, kernels, merge, and exposed wait separately.

### 2. KV Cache Memory And Storage Tiers

| Paper | Local file | Core idea | Loom connection |
| --- | --- | --- | --- |
| CXL KV Cache | `papers/cxl-kv-cache-storage-neurips24.pdf` | Use CXL memory as a KV cache tier under TTFT SLO. | Extends external-pool placement and attention cost candidates beyond HBM/DRAM. |
| Tutti | `papers/tutti-ssd-kv-2605.03375.pdf` | Make SSD-backed KV practical using GPU-centric bulk I/O and slack-aware scheduling. | Guides SSD-backed `KvPool` and StageKv policy. |
| InstInfer | `papers/instinfer-in-storage-attention-2409.04992.pdf` | Move long-context attention near storage. | Direct baseline for future `NearStorage` executors. |
| LMCache | `papers/lmcache-2510.09665.pdf` | Production KV cache layer for vLLM/SGLang. | External-pool integration baseline. |

What to learn:

- memory hierarchy matters only through latency, bandwidth, capacity, and SLO;
- larger tier is useful only when fetch beats recompute;
- SSD/CXL paths need batching and slack-aware scheduling to avoid GPU stalls;
- storage systems papers matter to serving when KV becomes external state.

Loom tasks:

- add real `KvPool` adapters and tier-specific cost measurements;
- compare RouteQuery, StageKv, and recompute boundaries;
- keep CXL/NVMe as unvalidated future memory domains until hardware tests exist.

### 3. KV Cache Safety And Identity

| Paper | Local file | Core idea | Loom connection |
| --- | --- | --- | --- |
| Prompt Leakage | `papers/prompt-leakage-kvcache-sharing-ndss25.pdf` | KV cache sharing can leak prompts across tenants. | Direct motivation for `IdentityScope` and safe reuse refusal. |

What to learn:

- KV cache is not just a performance object; it can encode user-private state;
- content hash equality is not enough for safe reuse;
- tenant, model, tokenizer, and adapter identity belong in the reuse contract.

Loom tasks:

- add a threat-model note for identity-aware reuse;
- expose identity and generation refusal metrics from the runtime;
- create a demo where content matches but identity mismatch refuses reuse.

### 4. Serving Baselines

| Paper | Local file | Core idea | Loom connection |
| --- | --- | --- | --- |
| DistServe | `papers/distserve-osdi24-2401.09670.pdf` | P/D disaggregation for goodput. | SLO-goodput evaluation baseline. |
| Mooncake | `papers/mooncake-fast25-2407.00079.pdf` | KVCache-centric disaggregated architecture. | External `KvPool` and transfer baseline. |
| PagedAttention / vLLM | `papers/pagedattention-vllm-sosp23-2309.06180.pdf` | Paged KV memory management. | Current engine-adapter and block-table boundary. |
| SGLang | `papers/sglang-radixattention-2312.07104.pdf` | Radix prefix reuse. | Future engine adapter and prefix identity source. |
| NIXL benchmark | `papers/nixl-gpudirect-benchmark-iit-2025.pdf` | GPUDirect performance depends on topology and I/O size. | Avoids assuming RDMA/GDS is always faster. |

## Reading Template

For each paper, answer these in the notes:

1. What resource bottleneck does the paper identify?
2. What boundary does it introduce or move?
3. What scheduling or communication mechanism hides the bottleneck?
4. Which metrics prove the claim?
5. Which assumptions may not hold for Loom?
6. What design or experiment should Loom add?

## Current Reading Order

1. MegaScale-Infer: module-level disaggregation and ping-pong pipeline. Done in
   `docs/research/megascale-infer-notes.md`.
2. Flux: fine-grained communication overlap. Done in
   `docs/research/flux-comet-overlap-notes.md`.
3. Comet: dependency-aware task rescheduling. Done in
   `docs/research/flux-comet-overlap-notes.md`.
4. CXL KV Cache: memory tiering under TTFT SLO.
5. Prompt Leakage: safe reuse and identity governance.
6. Mooncake + Dynamo/NIXL references: production serving and transfer baseline.
