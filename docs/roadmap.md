# Roadmap

## Permanent Product Boundary

Loom does not implement matrix multiplication. Dense, quantized, sparse, and
grouped GEMM belong to cuBLASLt, CUTLASS, FlashInfer, or another
engine-selected vendor backend. Loom may prepare or consume their buffers and
fuse memory-bound work immediately around them, but it will not hide a second
matrix core behind a Loom API.

A new direction enters this roadmap only when all three statements are true:

1. the cost is memory traffic, launch overhead, layout conversion, or scheduling
   metadata rather than matrix arithmetic;
2. a named inference-engine path has a real gap that Loom can enter without
   copying tensors into a private format;
3. a real model or serving workload can close an engine, memory, or end-to-end
   exit gate.

Microbenchmark opportunity alone is not admission.

## Execution Order After K0.7

K0.7's first native-wheel matrix row is complete. Publication remains an
explicit release action, not an engineering prerequisite for starting the next
feature. New feature work follows this order:

| Order | Track | First deliverable | Required system proof |
| --- | --- | --- | --- |
| 1 | KV-cache compression | FP8 KV write/read boundary with explicit scale layout | lower cache bytes and higher admitted context or batch size without unacceptable quality or TPOT loss |
| 2 | Complete sampling tail | fused penalties, top-k/top-p, renormalization, and deterministic RNG | seeded token parity or a declared statistical contract plus an order-reversed engine win |
| 3 | KV-cache movement | block copy/gather/scatter/compact/remap for prefix reuse and preemption | fewer launches or less movement time in a real scheduler path |
| 4 | Profile-gated speculative extensions | tree/stochastic/KV boundaries only after profiling exposes material non-GEMM cost | a named draft/target model pair improves decode latency or throughput |
| 5 | Quantization plumbing | scale, pack/unpack, dequant/requant, and layout transitions around vendor GEMM | one named quantized model removes an HBM pass or temporary tensor |
| 6 | MoE routing and movement | top-k routing, histogram/prefix sum, permutation, and inverse permutation | lower model-level MoE latency while grouped GEMM remains vendor-owned |
| 7 | Minimal Rust decode proof | zero-copy Rust orchestration over vendor-produced tensors and Loom operators | one deterministic decode step uses borrowed memory and stream ownership without becoming an inference engine |

## K0: Backend Foundation

Status: complete.

- backend-independent Rust contracts and CPU oracle;
- safe CUDA resource ownership and C ABI;
- reproducible correctness and latency report format.

## K0.5: Publishable Rust Distribution

Status: complete for the Rust source crates in `1.0.0-alpha.1`.

- independent `loom-kernels`, `loom-cuda-sys`, and `loom-cuda` package
  metadata with versioned registry dependencies;
- handwritten CUDA sources packaged inside `loom-cuda-sys`, so an extracted
  crate does not depend on repository-relative files;
- package-specific READMEs, changelog, Cargo archive checks in CI, and a pure
  Rust H2D → CUDA → D2H oracle smoke example;
- clean archive rebuild of `loom-kernels` plus CUDA-enabled archive rebuild of
  `loom-cuda-sys` on NVIDIA H20;
- source-adapter Python metadata at `1.0.0a1`, which established the package
  name and entry point before K0.7 added native distribution.

Exit: a downstream Rust consumer can build the published source crates and run
an oracle-checked CUDA path without cloning the repository.

## K0.6: Engine-Owned Runtime Interop

Status: complete for the framework operator surface after
`v1.0.0-alpha.1`.

- ~~generic safe backend over owned or borrowed CUDA streams~~;
- ~~sealed read/write device-memory traits shared by owned buffers and borrowed
  tensor views~~;
- ~~zero-copy H20 oracle smoke on a borrowed stream and borrowed allocations,
  including non-destruction of framework-owned resources~~;
- ~~route one real framework adapter through the safe Rust boundary instead of
  calling the raw C ABI directly~~ — PyTorch/vLLM Add+RMSNorm and PyTorch
  RMSNorm+dynamic-FP8 now enter `loom-cuda-bridge` with actual buffer lengths
  and borrowed current-stream ownership;
- ~~validate external current-stream ordering, CUDA Graph capture, and engine
  fallback behavior through that Rust-owned path~~ — the H20 gate covers all
  three dtypes, odd widths, `torch.compile`, CUDA Graph replay, Add+RMSNorm
  vLLM IR invocation, and Rust-side invalid-buffer rejection for both paths.
- ~~move one proven decode-tail engine path through checked Rust~~ — contiguous
  greedy+sampled-logprob now uses typed borrowed Rust views, exact buffer
  lengths, disjoint-output validation, and the framework current stream;
- ~~route every remaining PyTorch operator through the same boundary~~ —
  standalone RMSNorm, activation/FP8, padded logits, selected-token logprobs,
  Min-P, RoPE+paged-KV, and base/split-K paged decode now use explicit Rust
  layout contracts; the ctypes, direct-CUDA, and unchecked dispatcher paths
  were removed as a breaking change.

Exit: an inference-engine call reaches checked Rust dispatch using its existing
tensor memory and CUDA stream, with no hidden copy, allocation, or ownership
transfer.

## K0.7: Framework Compatibility And Binary Distribution

Status: complete for the first Linux x86_64, CUDA 13.1, SM90 matrix row.

- ~~qualify the next vLLM minor without weakening adapter gates~~ — official
  vLLM 0.24.0 and 0.25.1 packages each pass the complete 192-test H20 GPU suite;
- ~~centralize runtime version admission and package metadata~~ — supported
  range is `vllm>=0.24,<0.26`, with registration-time series checks;
- ~~document the current binary boundary and Stable ABI decision~~ — the
  dispatcher target, runtime range, and revalidation rules are explicit;
- ~~replace the entire production dispatcher with PyTorch's Stable ABI~~ —
  every schema and kernel uses the boxed Stable ABI with a declared PyTorch
  2.10 target; the temporary probe and previous ATen dispatcher were deleted;
- ~~validate one binary across two PyTorch minor releases~~ — the exact H20
  `.so` built with PyTorch 2.11 passes on 2.10 and 2.11, including complete
  vLLM 0.24/0.25 suites on the qualified 2.11 stack;
- ~~automate the CUDA/PyTorch/Python matrix artifact~~ —
  `python/build_wheel.py` builds from a clean revision, packages exactly the
  two native libraries, emits their manifest, audits ELF/RPATH/symbols, and
  refuses an accidental source-only wheel;
- ~~prove repository-free H20 clean installs~~ — one exact
  `py3-none-linux_x86_64` artifact passes fresh Python 3.11 venv gates on
  PyTorch 2.10/2.11 and vLLM 0.24/0.25, including `pip check`, package-local
  library loading, BF16 smoke, and the applicable complete suites.

Exit: a qualified binary artifact installs without a repository checkout, uses
a declared PyTorch ABI boundary, and passes the same framework and H20 gates as
the source build. The first row reached this exit; it has not been published to
a package index.

## K1: Useful Normalization Family

Status: in progress.

1. ~~vectorized FP16 and BF16 RMSNorm~~ — H20 correctness gate complete;
2. ~~fused residual Add+RMSNorm~~ — double in-place H20 gate complete;
3. ~~RMSNorm plus dynamic per-token FP8 output quantization~~ — H20 and named
   vLLM bitwise/performance gates complete; INT8 remains planned;
4. ~~named vLLM baseline and engine integration~~ — IR provider, compilation,
   CUDA Graph, and synthetic-Qwen2 generate-loop gates complete;
5. ~~native CUDA/LibTorch wheel and clean-install matrix gate~~; a production
   model/workload gate remains.

Exit: one fused path improves a real decode workload, not only a microbenchmark.

## K2: MLP Activation And Quantization

Status: in progress.

1. ~~split-half SiLU-and-Mul for F32/FP16/BF16~~ — Rust, CUDA, PyTorch,
   vLLM layer override, and H20 compatibility gates complete;
2. ~~SiLU-and-Mul plus dynamic per-block FP8 output quantization~~ — groups
   64/128, exact vLLM compatibility, compiler-fusion registration, and H20
   named-baseline gates complete; pinned Qwen2.5 online-FP8 compilation,
   path-hit, CUDA Graph, exact-token, and order-reversed engine gates are also
   complete, while the measured 0.5B end-to-end result remains at parity;
3. dynamic INT8 output quantization when a named model path requires it;
4. GELU/GELU-tanh and gated variants admitted by model coverage;
5. explicit handoff to engine-selected vendor GEMM, with Loom limited to
   memory-bound bias, activation, and quantization boundaries around it.

Exit: a fused activation+quantization path removes an HBM round trip and
improves a real model workload. Standalone SiLU parity alone does not close it.

## K2.5: Quantization Plumbing Around Vendor GEMM

Status: planned.

- per-token, per-channel, and per-block scale reduction for FP8 and INT8;
- pack/unpack and layout conversion for engine-selected quantized kernels;
- dequantize, requantize, scale conversion, and scale-layout transpose;
- fuse adjacent activation, normalization, or cache movement only when it
  removes a measured launch or HBM round trip;
- keep matrix multiplication and its tuning entirely in the vendor backend.

Exit: a named quantized model path passes bitwise or declared-tolerance gates,
records the vendor GEMM unchanged on both sides, and improves an engine-level
latency, memory, or temporary-allocation metric.

## K3: KV-Cache Update Family

Status: in progress.

- ~~RoPE plus paged-KV write~~ — Rust/CUDA/PyTorch, packed-QKV and NHD/HND
  layouts, vLLM compiler fusion, H20 named baseline, and exact-token Qwen2.5
  engine gates complete; operator benefit is measurable, model-level benefit
  remains open;
- FP8 KV quantize-on-write and dequantize-on-read with an explicit per-head or
  per-block scale contract is the next cache deliverable; INT8 follows only
  when a named engine/model path requires it;
- append/copy with layout conversion for engine-native paged caches;
- block copy, swap, gather, scatter, compact, and remap for prefix caching,
  preemptive scheduling, beam movement, and cache defragmentation;
- expose no private cache ownership: engine allocations, page tables, streams,
  and lifetime remain borrowed.

Exit: a real engine shows lower cache bytes and a larger admitted context or
batch size for compression, or lower scheduler movement time for relocation,
while preserving token/quality gates and reporting TPOT impact explicitly.

## K4: Decode Tail

Status: in progress.

- ~~greedy argmax plus sampled-token raw logprob~~ — Rust oracle, safe
  CUDA/C ABI, PyTorch, checked-Rust contiguous dispatch, and narrow vLLM
  0.24/0.25 integration complete; the vLLM 0.24 H20 named baseline and both
  real-engine provider orders show exact token/rank parity and material
  latency/TPOT benefit;
- ~~general selected-token raw logprob and rank~~ — vLLM continues to own
  penalties, top-k/top-p, RNG, and token selection; Rust/CUDA/PyTorch plus
  order-reversed Qwen2.5 H20 gates show exact token/rank parity and material
  operator and end-to-end benefit;
- ~~in-place min-p filtering~~ — Rust/CUDA/PyTorch and a vLLM 0.24/0.25 opt-in
  are complete; H20 evidence selects Loom only for at least 32 rows and a
  65,536+ vocabulary, while smaller shapes fall back because the
  one-block-per-row kernel is slower there;
- fused logits bias, temperature, masking, bad-word suppression, and sparse
  repetition/presence/frequency penalties;
- top-k/top-p filtering, renormalization, and deterministic counter-based RNG
  sampling without a host round trip;
- top-k logprobs without a full-vocabulary probability tensor.

Exit: fewer launches and temporary tensors with identical token results. The
selected-logprob exit gates are closed for pure greedy and engine-owned general
sampling requests with `logprobs=0`; owning the selection kernels remains open.

## K4.5: Speculative Decoding Support

Status: real-engine path complete; performance exit open and further extensions
are profile-gated.

- ~~verify flattened ragged greedy drafts and compact accepted/bonus tokens~~ —
  Rust contract and CPU oracle, one-warp handwritten CUDA, safe borrowed-Rust
  dispatch, PyTorch current-stream/compile/graph coverage, and explicit vLLM
  0.24/0.25 registration are complete; all 15 H20 benchmark shapes are
  bit-exact and reduce verifier-level latency by `9.2-11.3%`;
- ~~run a named draft/target model through isolated native and Loom
  providers~~ — Qwen2.5-1.5B target plus Qwen2.5-0.5B draft on vLLM 0.24
  preserves exact native/Loom speculative tokens and statistics, records
  `714/714` measured Loom calls per order, and isolates target/native/Loom in
  separate processes;
- profile result: the verifier is only `0.048-0.200%` of batch latency,
  native/Loom end-to-end ratios cross parity under order reversal, and this
  speculative configuration is `3.18-4.97x` slower than target-only. Do not
  spend the next milestone on verifier micro-optimization;

The remaining speculative boundaries require a new named workload that shows
material metadata, sampling, or KV-management cost:

- construct batched draft-verification metadata and tree/branch masks consumed
  by an engine-selected attention backend;
- implement stochastic residual-distribution acceptance/rejection using an
  explicit counter-based RNG state contract;
- update caller-owned sequence/KV metadata without host round trips;
- add cache commit/rollback or slot-remap primitives only where the selected
  engine exposes that boundary;
- keep draft/target model GEMM and verification attention in vendor libraries.

Exit: one named draft/target model pair reaches Loom from a real engine,
preserves the engine's declared sampling distribution and seeded behavior,
records path hits, and improves end-to-end decode latency or throughput in both
provider orders. A standalone acceptance-kernel benchmark does not close this
milestone. The current Qwen2.5 gate closes invocation and equivalence, but
explicitly does not close the performance clause.

## K5: MoE Routing And Movement

Status: planned.

- top-k routing, renormalization, and expert mapping;
- token histogram, prefix sum, permutation, and inverse permutation;
- caller-owned metadata and buffer handoff into the engine-selected grouped
  GEMM, with no Loom matrix implementation;
- weighted expert-output reduction and shared/routed output fusion when they
  remove measured memory traffic.

Exit: routing and movement reduce model-level MoE latency on a named engine.
The vendor grouped GEMM is identical on both sides of the comparison.

## K6: Attention

Status: in progress.

- ~~paged MQA/GQA base contract and CPU oracle~~ — one query per request,
  native paged KV, MQA/GQA mapping, and block-table validation are fixed;
- ~~first handwritten short-context CUDA candidate~~ — F32/FP16/BF16 C ABI,
  safe Rust, current-stream PyTorch, randomized oracle, compile/graph gates,
  and an H20 FA3 comparison are complete;
- ~~GQA-packed 32/64-token optimization~~ — two/four query heads reuse each
  paged K/V load; compile-time partial tails support odd GQA ratios without
  adding hot-loop guards to full groups;
- ~~native vLLM cache layout and broad short-context qualification~~ — the C
  ABI accepts interleaved K/V block strides; a 156-case shape sweep and focused
  132-case batch sweep identify the exact winning envelope;
- ~~measured-shape vLLM 0.24/0.25 adapter with explicit FA3 fallback~~ — the
  opt-in route is limited to FP16/BF16 Hq/Hkv 32/8, D128, block 16/32,
  batch <=128, context <=32; 0.25 compatibility and the 0.24 direct-backend
  and stable-output synthetic-engine gates pass;
- pretrained-model gate and broader head geometry — the first Qwen2.5 `14/2`,
  D64 attempt hit the engine but failed exact-token and latency gates, so it
  remains intentionally unrouted;
- ~~tiled split-K/LSE optimization for 128-1024 tokens~~ — explicit
  caller-owned Rust/C workspace, stable partial-state merge, CUDA Graph-safe
  PyTorch dispatch, and H20 legacy/FA3 gates are complete for D128 batches
  1-8; it materially improves Loom but does not widen the vLLM route because
  FA3 remains faster;
- vendor attention integration where it wins;
- distributed split-KV/LSE merge, sliding-window variants, and MLA when a
  consumer exists.

Exit: hardware-qualified engine evidence determines admission; prior Loom
Attention prototype code is not carried forward automatically.

## K7: Communication-Aware Fusion

Status: planned after reproducible single-GPU and multi-GPU engine baselines.

- tensor-parallel reduction plus residual/norm epilogues;
- sharded-vocabulary sampling and selected-logprob merge;
- expert-parallel dispatch/combine integration.

Exit: end-to-end TP or EP goodput improves under an equivalent NCCL/transport
baseline; local adapters do not count as distributed evidence.

## K8: Engine-Neutral Rust Decode Proof

Status: planned after one new post-K0.7 feature reaches an engine.

- accept vendor- or engine-produced CUDA tensors through borrowed Rust device
  memory and a non-owning stream;
- chain a minimal decode slice such as cache update, logits processing,
  sampling, and token output through the existing safe Loom APIs;
- use a callback or external boundary for every GEMM and model-owned attention
  operation;
- allocate no private copy of framework tensors and own no scheduler, model
  weights, tokenizer, or KV-cache lifetime.

Exit: a reproducible Rust example performs one deterministic zero-copy decode
step, matches a reference token and state update, survives external-stream and
CUDA Graph gates where applicable, and demonstrates that Loom is engine-neutral
without growing into an inference engine.

The complete intended surface, including profile-gated layout primitives, is
tracked in the [operator catalog](operator-catalog.md).
