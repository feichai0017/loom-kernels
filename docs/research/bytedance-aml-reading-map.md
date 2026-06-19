# ByteDance AML / AI Infra Reading Map

This note organizes the local paper set into a study path for QuillCache. The
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

QuillCache should use the same pattern. It treats KV cache as inference state:
observable, routable, transferable, persistent, and identity-governed.

## Local Paper Groups

### 1. Module-Level Disaggregation And Communication Overlap

Read first because this is closest to ByteDance AML / Seed systems style.

| Paper | Local file | Core idea | QuillCache connection |
| --- | --- | --- | --- |
| MegaScale-Infer | `papers/megascale-infer-sigcomm25-2504.02263.pdf` | Disaggregate attention and FFN for MoE serving; use ping-pong pipeline and M2N communication. | Extends our co-scheduler thinking beyond P/D into module-level resource pools. |
| Flux | `papers/flux-comm-overlap-2406.06858.pdf` | Fine-grained communication/computation overlap through kernel fusion. | Supports the layer-wise transfer overlap thesis. |
| Comet | `papers/comet-moe-overlap-mlsys25-2502.19811.pdf` | Data-dependency-aware task rescheduling for MoE overlap. | Suggests `comm_stall_ms` and overlap efficiency as first-class telemetry. |

What to learn:

- why lower FLOPs can still waste GPUs;
- how to identify memory-bound versus compute-bound modules;
- when splitting modules creates enough batching to recover utilization;
- why custom communication sometimes beats generic collectives;
- how pipeline depth trades latency for throughput.

QuillCache tasks:

- add predicted-versus-actual transfer/TTFT experiment;
- expose overlap efficiency in `/v1/state.co_scheduler`;
- teach the co-scheduler to reason about queueing, transfer, and compute as one
  budget.

### 2. KV Cache Memory And Storage Tiers

| Paper | Local file | Core idea | QuillCache connection |
| --- | --- | --- | --- |
| CXL KV Cache | `papers/cxl-kv-cache-storage-neurips24.pdf` | Use CXL memory as a KV cache tier under TTFT SLO. | Extends `CacheTier` and cost model beyond HBM/DRAM/SSD. |
| Tutti | `papers/tutti-ssd-kv-2605.03375.pdf` | Make SSD-backed KV practical using GPU-centric bulk I/O and slack-aware scheduling. | Guides future DiskTier/GDS design and transfer scheduling. |
| InstInfer | `papers/instinfer-in-storage-attention-2409.04992.pdf` | Move long-context attention near storage. | A stronger baseline than naive SSD fetch for very long contexts. |
| LMCache | `papers/lmcache-2510.09665.pdf` | Production KV cache layer for vLLM/SGLang. | Direct external baseline for connector and offload behavior. |

What to learn:

- memory hierarchy matters only through latency, bandwidth, capacity, and SLO;
- larger tier is useful only when fetch beats recompute;
- SSD/CXL paths need batching and slack-aware scheduling to avoid GPU stalls;
- storage systems papers matter to serving when KV becomes external state.

QuillCache tasks:

- extend `TransferObservation` with tier-specific measured costs;
- build reuse-versus-transfer-versus-recompute plots;
- document CXL/NVMe as future tiers without claiming hardware validation.

### 3. KV Cache Safety And Identity

| Paper | Local file | Core idea | QuillCache connection |
| --- | --- | --- | --- |
| Prompt Leakage | `papers/prompt-leakage-kvcache-sharing-ndss25.pdf` | KV cache sharing can leak prompts across tenants. | Direct motivation for `IdentityScope` and safe reuse refusal. |

What to learn:

- KV cache is not just a performance object; it can encode user-private state;
- content hash equality is not enough for safe reuse;
- tenant, model, tokenizer, and adapter identity belong in the reuse contract.

QuillCache tasks:

- add a threat-model note for identity-aware reuse;
- keep identity refusal metrics visible in gateway state and Prometheus;
- create a demo where content matches but identity mismatch refuses reuse.

### 4. Serving Baselines

| Paper | Local file | Core idea | QuillCache connection |
| --- | --- | --- | --- |
| DistServe | `papers/distserve-osdi24-2401.09670.pdf` | P/D disaggregation for goodput. | Basis for `EngineRole`, P/D planning, and SLO goodput. |
| Mooncake | `papers/mooncake-fast25-2407.00079.pdf` | KVCache-centric disaggregated architecture. | Store/transfer/master decomposition baseline. |
| PagedAttention / vLLM | `papers/pagedattention-vllm-sosp23-2309.06180.pdf` | Paged KV memory management. | Engine boundary and connector correctness. |
| SGLang | `papers/sglang-radixattention-2312.07104.pdf` | Radix prefix reuse. | Prefix index and residency design. |
| NIXL benchmark | `papers/nixl-gpudirect-benchmark-iit-2025.pdf` | GPUDirect performance depends on topology and I/O size. | Avoids assuming RDMA/GDS is always faster. |

## Reading Template

For each paper, answer these in the notes:

1. What resource bottleneck does the paper identify?
2. What boundary does it introduce or move?
3. What scheduling or communication mechanism hides the bottleneck?
4. Which metrics prove the claim?
5. Which assumptions may not hold for QuillCache?
6. What design or experiment should QuillCache add?

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
