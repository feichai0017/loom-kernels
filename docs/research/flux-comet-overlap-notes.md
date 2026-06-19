# Flux And Comet Overlap Notes

Papers:

- `papers/flux-comm-overlap-2406.06858.pdf`
- `papers/comet-moe-overlap-mlsys25-2502.19811.pdf`

Question for QuillCache: what should a KV-cache control plane learn from
kernel-level and MoE-level communication/computation overlap systems?

## Shared Lesson

Flux, Comet, and MegaScale-Infer all make the same systems move:

1. Measure the exposed communication, not just total runtime.
2. Split work at the dependency boundary.
3. Make the unit of scheduling smaller than the original operator.
4. Keep compute efficiency from collapsing while exposing overlap.
5. Tune the overlap using runtime shape, topology, and queueing signals.

For QuillCache, the equivalent unit is not a GEMM tile or an expert token. It is
a KV layer, block, or prefix segment that becomes consumable before the whole
KV object has arrived.

## Flux Digest

### Bibliographic Info

- Title: Flux: Fast Software-based Communication Overlap on GPUs through Kernel
  Fusion
- Venue/year: arXiv 2024
- Problem area: tensor-parallel communication overlap for LLM training and
  inference

### One-Sentence Thesis

When tensor-parallel communication is dependent on GEMM output or input,
stream-level chunking loses GPU efficiency and timing control; decomposing the
work into GEMM-aligned tiles and fusing communication/wait logic into the kernel
can reduce exposed communication time.

### System Boundary

- Authoritative state: tensor-parallel shards, GEMM inputs/outputs, peer pointers
  or NVSHMEM-accessible remote buffers.
- Derived state: tile readiness signals, tile-to-rank mapping, autotuned kernel
  variants, overlap efficiency.
- Writer path: fused GEMM epilogue writes ReduceScatter/AlltoAll tiles, or host
  transfers AllGather tiles and sets readiness signals.
- Reader path: GEMM tiles consume local or arrived remote tiles after readiness
  checks.
- Recovery path: none. Flux is a performance kernel path, not a durable state
  system.
- Failure assumptions: GPU peer access, NVSHMEM or equivalent transport, tuned
  kernels, and stable hardware support.

### Mechanism

- Define effective communication time:
  `overall_time - best_non_split_gemm_time`.
- Define overlap efficiency from effective communication time versus the
  non-overlapped baseline.
- Over-decompose communication and computation into fine-grained tiles.
- Fuse ReduceScatter/AlltoAll communication into GEMM epilogues.
- For AllGather, fuse wait logic into GEMM and let host/device communication set
  tile readiness signals.
- Use tile-coordinate swizzling, instruction selection, communication-order
  selection, and autotuning across GPU/interconnect combinations.

### Supported Claims

- Property claim: fine-grained fused overlap can reduce exposed communication
  without splitting one GEMM into many inefficient kernels.
- Performance claim: the paper reports up to 1.24x training speedup over
  Megatron-LM, and up to 1.66x prefill / 1.30x decoding speedups over vLLM.
- Operational claim: Flux is evaluated across A100 PCIe, A100 NVLink, and H800
  NVLink environments.

### What The Evaluation Really Covers

- Workloads: GPT-3 175B and Llama-2 70B training, prefill, and decode; operation
  microbenchmarks for ReduceScatter and AllGather.
- Baselines: PyTorch/NCCL non-overlap, Megatron-LM, vLLM, TransformerEngine.
- Important caveat: speedup depends on both communication fraction and overlap
  efficiency. If communication is only a small fraction, high overlap efficiency
  may still produce small end-to-end gains.
- Weak point: very small `m` cases can be worse than the non-overlapped baseline,
  especially in decode-like shapes where the fused kernel has too little useful
  work to hide latency.

## Comet Digest

### Bibliographic Info

- Title: Comet: Fine-grained Computation-communication Overlapping for
  Mixture-of-Experts
- Venue/year: MLSys 2025
- Problem area: MoE communication/computation overlap

### One-Sentence Thesis

MoE overlap needs dependency-aware scheduling because token dispatch, expert
GEMM, and token combine have irregular runtime data dependencies; shared-tensor
decomposition plus adaptive thread-block assignment can hide communication while
preserving expert compute efficiency.

### System Boundary

- Authoritative state: MoE token routing, expert placement, shared tensors
  between dispatch/GEMM/combine stages, model parallelism metadata.
- Derived state: decomposed shared-tensor tiles, rescheduled GroupGEMM order,
  profiled thread-block assignment metadata.
- Writer path: dispatch and expert computations produce shared tensor slices.
- Reader path: downstream expert/combination stages consume slices when their
  dependencies are satisfied.
- Recovery path: none. Comet optimizes execution of one MoE layer; durable
  recovery is outside scope.
- Failure assumptions: runtime token distribution is dynamic but bounded by
  profiled kernel variants; GPU transport and NVSHMEM buffers are available.

### Mechanism

- Model MoE layer execution as producer-consumer pipelines:
  communication-to-computation and computation-to-communication.
- Identify shared tensors between producer and consumer stages.
- Decompose shared tensors along dimensions that match both communication and
  computation dependencies.
- Reschedule GroupGEMM order so consumers can start before the full producer
  operator finishes.
- Use thread-block specialization to isolate communication blocks from
  computation blocks.
- Pick the communication/computation block split from preprofiled variants,
  based on model shape, runtime token count, and parallelism.

### Supported Claims

- Property claim: fine-grained dependency resolution reduces non-overlapped
  pipeline bubbles without degrading expert compute efficiency.
- Performance claim: the paper reports 1.96x speedup for a single MoE layer and
  1.71x average end-to-end speedup.
- Operational claim: Comet is described as deployed in production clusters with
  ten-thousand-scale GPUs and saving millions of GPU hours.

### What The Evaluation Really Covers

- Workloads: Mixtral-8x7B, Qwen2-MoE, and Phi3.5-MoE with varying token lengths,
  expert counts, top-k, and parallelism.
- Hardware: H800 NVLink cluster and L20 PCIe cluster.
- Baselines: Megatron-CUTLASS, Megatron-TE, FasterMoE, and Tutel.
- Important metric: Comet reports hiding 86.5% of communication latency on
  average in the detailed MoE-layer breakdown.
- Weak point: the design relies on custom CUDA/C++ kernels, NVSHMEM buffers, and
  profiled kernel variants. This is not a portable control-plane interface.

## Comparison For QuillCache

| System | Boundary | Scheduling unit | Runtime signal | What QuillCache borrows |
| --- | --- | --- | --- | --- |
| Flux | Tensor-parallel GEMM + communication | GEMM tile | effective communication time, overlap efficiency | measure exposed transfer and separate it from compute |
| Comet | MoE dispatch/expert/combine pipeline | shared-tensor slice / thread block | hidden communication %, optimal comm/compute block split | dependency-aware readiness and adaptive resource split |
| QuillCache | KV-cache routing, transfer, and tier placement | KV layer, block, or prefix segment | time-to-first-layer, full transfer, queue depth, overlap efficiency | schedule KV as a pipeline stage, not a blob copy |

## Design Implications For QuillCache

1. Add exposed-transfer accounting.
   `overlap_efficiency_pct` is useful, but Flux suggests also tracking an
   effective exposed transfer time:
   `exposed_transfer_ms = full_transfer_ms - overlap_saved_ms`.

2. Treat readiness as a dependency graph.
   Comet's shared-tensor analysis maps to a KV readiness graph:
   layer 0 unlocks decode start, later layers unlock continued decode, and a
   prefix is only routable when identity and tier constraints pass.

3. Separate transport speed from overlap quality.
   A faster backend can still be bad if it delays first-layer readiness or
   creates queueing. A slower backend can be acceptable if most of its transfer
   is hidden.

4. Tune in-flight depth with measurements.
   Flux and Comet both show that finer granularity helps only until scheduling
   overhead or compute underutilization dominates. QuillCache should sweep
   `max_inflight` and plot first-layer latency, full-transfer latency, queue
   depth, and overlap efficiency.

5. Do not claim kernel-level novelty.
   Flux and Comet own the CUDA-kernel side. QuillCache's research position is a
   vendor-neutral control plane that consumes these kinds of signals across real
   vLLM/SGLang/KV backends.

## Concrete Follow-Up Experiments

P0: Add exposed transfer time to gateway/co-scheduler state. Done.

- `exposed_transfer_ms = full_transfer_ms - overlap_saved_ms`.
- This is closer to Flux's effective communication-time framing.

P1: Build a transfer-depth sweep.

- Sweep `max_inflight = 1, 2, 4, 8`.
- Record time-to-first-layer, full-transfer, overlap efficiency, bandwidth, and
  queue depth.
- Use local TCP first, then repeat with cloud GPU transfer backends when
  available.

P2: Build predicted-versus-actual TTFT plots.

- Compare planner-estimated transfer time against connector-reported telemetry.
- Label points by backend, tier, byte size, and in-flight depth.
- The graph should answer whether QuillCache's routing cost model predicts real
  first-token behavior.

P3: Add dependency-aware action reasons.

- Instead of only saying `TuneTransferDepth`, explain which boundary is bad:
  first layer is late, exposed transfer is high, queue depth is high, or
  bandwidth is low.

## Interview Takeaway

For ByteDance AML-style AI infra, the pattern is:

> Do not optimize "communication" as one number. Identify the exposed part,
> understand the dependency that prevents overlap, reduce the scheduling unit,
> and verify that compute efficiency does not collapse.

QuillCache applies that pattern at the KV-cache control-plane level: it turns KV
movement into measured, dependency-aware pipeline state that routing and
placement can react to.
