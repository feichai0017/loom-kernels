# Flux And Comet Overlap Notes

Papers:

- `papers/flux-comm-overlap-2406.06858.pdf`
- `papers/comet-moe-overlap-mlsys25-2502.19811.pdf`

Question for Loom: what should a distributed attention runtime learn from
kernel-level and MoE-level communication/computation overlap systems?

## Shared Lesson

Flux, Comet, and MegaScale-Infer all make the same systems move:

1. Measure the exposed communication, not just total runtime.
2. Split work at the dependency boundary.
3. Make the unit of scheduling smaller than the original operator.
4. Keep compute efficiency from collapsing while exposing overlap.
5. Tune the overlap using runtime shape, topology, and queueing signals.

For Loom, the equivalent unit is a query batch, KV shard, or partial
softmax result that can execute before all remote work finishes.

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

## Comparison For Loom

| System | Boundary | Scheduling unit | Runtime signal | What Loom borrows |
| --- | --- | --- | --- | --- |
| Flux | Tensor-parallel GEMM + communication | GEMM tile | effective communication time, overlap efficiency | measure exposed transfer and separate it from compute |
| Comet | MoE dispatch/expert/combine pipeline | shared-tensor slice / thread block | hidden communication %, optimal comm/compute block split | dependency-aware readiness and adaptive resource split |
| Loom | distributed attention over externally owned KV | query batch, KV shard, partial result | queue, transport, kernel, merge, and exposed wait time | schedule local and remote partial attention as one dependency graph |

## Design Implications For Loom

1. Measure exposed communication.
   Record how much Route-Q transport remains visible after remote attention and
   local-tail execution overlap. Raw link bandwidth does not answer that.

2. Treat readiness as a dependency graph.
   A leased sealed prefix can run remotely while the active tail runs locally;
   merge starts only after shape-compatible partial statistics are ready.

3. Separate transport speed from overlap quality.
   A faster backend can still be bad if it delays first-layer readiness or
   creates queueing. A slower backend can be acceptable if most of its transfer
   is hidden.

4. Tune remote in-flight depth with measurements.
   Flux and Comet both show that finer granularity helps only until scheduling
   overhead or compute underutilization dominates. Loom should sweep the
   number of concurrent remote partials and report queue, transport, kernel,
   merge, and end-to-end decode time.

5. Do not claim kernel-level novelty.
   The initial research claim is execution placement and exact distributed
   attention semantics. A future CUDA kernel must earn a separate claim through
   kernel-level baselines.

## Concrete Follow-Up Experiments

P0: Build the one-node Route-Q path.

- Run the sealed prefix on a second GPU and the active tail on the model GPU.
- Return output-plus-LSE attention states and compare the merged output with
  unsharded attention.

P1: Build a remote-depth sweep.

- Sweep `max_inflight = 1, 2, 4, 8`.
- Record queue, Q/O transfer, remote kernel, local-tail kernel, merge, and
  exposed wait time.
- Start with CUDA P2P or NCCL, then repeat across nodes with a registered-device
  transport.

P2: Build predicted-versus-actual decode-time plots.

- Compare planner estimates against executor and transport telemetry.
- Label points by execution mode, topology, KV length, and in-flight depth.
- The graph should answer whether Loom's routing cost model predicts real
  decode behavior.

## Interview Takeaway

For ByteDance AML-style AI infra, the pattern is:

> Do not optimize "communication" as one number. Identify the exposed part,
> understand the dependency that prevents overlap, reduce the scheduling unit,
> and verify that compute efficiency does not collapse.

Loom applies that pattern to distributed core attention: local and remote
partial attention, Q/O transport, and exact merge form one measured dependency
graph.
