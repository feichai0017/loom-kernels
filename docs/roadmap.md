# Roadmap

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
- source-adapter Python wheel metadata at `1.0.0a1`; portable, automated
  CUDA/LibTorch binary wheels remain a later distribution milestone.

Exit: a downstream Rust consumer can build the published source crates and run
an oracle-checked CUDA path without cloning the repository.

## K0.6: Engine-Owned Runtime Interop

Status: in progress; the runtime foundation is included in the GitHub
prerelease `v1.0.0-alpha.1`.

- ~~generic safe backend over owned or borrowed CUDA streams~~;
- ~~sealed read/write device-memory traits shared by owned buffers and borrowed
  tensor views~~;
- ~~zero-copy H20 oracle smoke on a borrowed stream and borrowed allocations,
  including non-destruction of framework-owned resources~~;
- route one real framework adapter through the safe Rust boundary instead of
  calling the raw C ABI directly;
- validate external current-stream ordering, CUDA Graph capture, and engine
  fallback behavior through that Rust-owned path.

Exit: an inference-engine call reaches checked Rust dispatch using its existing
tensor memory and CUDA stream, with no hidden copy, allocation, or ownership
transfer.

## K1: Useful Normalization Family

Status: in progress.

1. ~~vectorized FP16 and BF16 RMSNorm~~ — H20 correctness gate complete;
2. ~~fused residual Add+RMSNorm~~ — double in-place H20 gate complete;
3. ~~RMSNorm plus dynamic per-token FP8 output quantization~~ — H20 and named
   vLLM bitwise/performance gates complete; INT8 remains planned;
4. ~~named vLLM baseline and engine integration~~ — IR provider, compilation,
   CUDA Graph, and synthetic-Qwen2 generate-loop gates complete;
5. ~~source-adapter wheel metadata and isolated-install smoke gate~~; automated
   CUDA/LibTorch binary wheels and a production model/workload gate remain.

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
5. vendor GEMM integration with bias, activation, and quantization epilogues.

Exit: a fused activation+quantization path removes an HBM round trip and
improves a real model workload. Standalone SiLU parity alone does not close it.

## K3: KV-Cache Update Family

Status: in progress.

- ~~RoPE plus paged-KV write~~ — Rust/CUDA/PyTorch, packed-QKV and NHD/HND
  layouts, vLLM compiler fusion, H20 named baseline, and exact-token Qwen2.5
  engine gates complete; operator benefit is measurable, model-level benefit
  remains open;
- append/copy with layout conversion;
- FP8/INT8 quantize and dequantize;
- gather/scatter for paged cache movement.

Exit: fewer HBM passes and lower TPOT in a real engine.

## K4: Decode Tail

Status: in progress.

- ~~greedy argmax plus sampled-token raw logprob~~ — Rust oracle, safe
  CUDA/C ABI, PyTorch, and narrow vLLM 0.24 integration complete; H20 named
  baseline and both real-engine provider orders show exact token/rank parity
  and material latency/TPOT benefit;
- ~~general selected-token raw logprob and rank~~ — vLLM continues to own
  penalties, top-k/top-p, RNG, and token selection; Rust/CUDA/PyTorch plus
  order-reversed Qwen2.5 H20 gates show exact token/rank parity and material
  operator and end-to-end benefit;
- ~~in-place min-p filtering~~ — Rust/CUDA/PyTorch and a vLLM 0.24 opt-in are
  complete; H20 evidence selects Loom only for at least 32 rows and a 65,536+
  vocabulary, while smaller shapes fall back because the one-block-per-row
  kernel is slower there;
- fused logits bias, masking, bad-word suppression, and history penalties;
- top-k/top-p filtering, renormalization, and deterministic RNG sampling;
- top-k logprobs.

Exit: fewer launches and temporary tensors with identical token results. The
selected-logprob exit gates are closed for pure greedy and engine-owned general
sampling requests with `logprobs=0`; owning the selection kernels remains open.

## K5: MoE Routing And Movement

Status: planned.

- top-k routing, renormalization, and expert mapping;
- token histogram, prefix sum, permutation, and inverse permutation;
- grouped-GEMM vendor dispatch and fused expert-output reduction.

Exit: routing and movement reduce model-level MoE latency on a named engine.

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
- ~~measured-shape vLLM 0.24 adapter with explicit FA3 fallback~~ — the opt-in
  route is limited to FP16/BF16 Hq/Hkv 32/8, D128, block 16/32, batch <=128,
  context <=32; direct backend and stable-output synthetic-engine gates pass;
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

The complete intended surface, including profile-gated layout primitives, is
tracked in the [operator catalog](operator-catalog.md).
