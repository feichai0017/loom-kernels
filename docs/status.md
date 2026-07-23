# Implementation Status

## Implemented

- Rust workspace split into public contracts, safe CUDA backend, one checked
  framework bridge, and an internal kernel-launch FFI;
- `DType`, contiguous `TensorSpec`, normalization operator specs,
  `OperatorSpec`, and backend capability query;
- F64-accumulation F32/FP16/BF16 RMSNorm and fused Add+RMSNorm CPU oracles;
- owned CUDA streams, device buffers, and events, plus non-owning external
  stream handles and typed read-only/exclusive device-memory views shared by
  every safe operator entrypoint;
- checked handwritten F32 plus pair-vectorized FP16/BF16 RMSNorm dispatch;
- double in-place Add+RMSNorm with Rust-exclusive mutable-buffer contracts;
- aligned 128-bit pack8 Add+RMSNorm with pair and scalar fallbacks;
- three-pass RMSNorm plus dynamic per-token FP8 E4M3FN quantization, with
  pack4 output conversion and scalar fallback;
- checked safe-Rust, caller-allocated PyTorch out, and convenience allocating
  entrypoints for RMSNorm+FP8;
- split-half SiLU-and-Mul contracts and CPU oracles for F32/FP16/BF16;
- handwritten SiLU-and-Mul with aligned 16-byte packs and scalar fallback;
- FP16/BF16 SiLU-and-Mul fused directly into dynamic per-block FP8 E4M3FN,
  with groups 64/128, optional scale upper bound, and row/group-major scales;
- NeoX and interleaved RoPE contracts, partial rotary dimensions, and a fused
  RoPE+paged-KV write oracle for native NHD/HND cache layouts;
- F32/FP16/BF16 fused CUDA with explicit packed-QKV token/head strides,
  independent rotation/cache token counts, and strided paged-cache writes;
- greedy-sampling contract and CPU oracles with first-index tie breaking,
  sampled-token log-softmax, and explicit finite-logit precondition;
- one-block-per-row F32/FP16/BF16 CUDA fusion of argmax, online logsumexp,
  selected logprob, and vLLM-compatible maximum-tie rank, including padded
  vocabulary row strides;
- arbitrary selected-token contracts and CPU oracles with int64 IDs, F32 raw
  logprobs, and vLLM-compatible tie-aware ranks;
- one-block-per-row F32/FP16/BF16 CUDA selected-token normalization and rank,
  preserving caller-owned sampling policy without a full-vocabulary F32
  logprob tensor;
- in-place F32/FP16/BF16 Min-P contracts, CPU oracles, handwritten CUDA, C ABI,
  PyTorch mutation schemas, and an opt-in vLLM 0.24/0.25 processor override
  that cancels the softmax denominator instead of allocating probability and
  mask tensors;
- base paged MQA/GQA decode attention for one query per request: Rust contract
  and stable CPU oracle, F32/FP16/BF16 handwritten CUDA/C ABI, safe Rust
  entrypoints, dense-inner NHD cache indirection with explicit outer block
  strides, distinct K/V widths, a two/four-query-head GQA specialization that
  reuses paged K/V loads, compile-time full/partial tail groups for odd GQA
  ratios, D64 decode-Q register caching, and a current-stream caller-allocated
  PyTorch operator capped at 1,024 tokens;
- a D128 long-context split-K implementation with caller-owned F32
  `(max, denominator, numerator)` workspace, stable LSE merge, explicit safe
  Rust sizing/execution APIs, two/four-head GQA packing, and CUDA Graph-safe
  PyTorch temporary ownership while preserving the original allocation-free
  C ABI;
- a single boxed LibTorch Stable ABI dispatcher targeting PyTorch 2.10 and
  using the current CUDA stream; all ten semantic operators route through
  `loom-cuda-bridge` into borrowed safe Rust dispatch with explicit storage
  spans and layouts;
- a source-adapter Python wheel with explicit framework extras, project
  metadata, license/readme payloads, and a CI install/entry-point smoke gate;
- a `loom_cuda` vLLM IR provider with exact-contract admission and an opt-in vLLM
  `SiluAndMul` out-of-tree layer replacement, plus an opt-in activation-quant
  fusion-table replacement, RoPE+KV compiler-pass adapter, and pure-greedy
  general selected-token sampled-logprob fast paths, and a measured-shape
  Min-P override plus a native-cache short-context paged-decode override for
  vLLM 0.24 and 0.25;
- per-operator JSON correctness/latency benchmarks and named vLLM baselines.

## Validated

- local formatting, clippy, tests, and release build;
- the Python adapter wheel built, installed into an isolated environment,
  passed dependency checks, imported, and exposed its vLLM entry point;
- CUDA-feature clippy on `forge-gas1`;
- NVIDIA H20 correctness for six shapes from `1x1` through `16x8192` and a
  `64x4096` batch;
- maximum absolute error `4.77e-7` against the CPU oracle;
- `8x4096` post-cleanup seven-sample isolated median `7.27 us`.
- FP16 and BF16 pair paths plus odd-size scalar fallbacks validated on H20;
- `8x4096` seven-sample medians: FP16 `4.75 us`, BF16 `4.71 us`;
- maximum FP16 absolute error `0.001953125`; BF16 cases were exact against the
  quantized-output oracle for the tested deterministic inputs.
- fused Add+RMSNorm CUDA-feature Clippy, release build, and all tests passed;
- fused `8x4096` H20 medians after pack8 optimization: FP16 `3.061 us` and
  BF16 `2.914 us` over 15 samples and 2000 launches per sample; the F32 path
  remains `9.062 us` in its earlier gate;
- fused residual outputs were exact against the oracle in every tested dtype
  and shape; maximum normalized-output error was `4.77e-7` for F32,
  `4.88e-4` for FP16, and zero for the tested BF16 cases;
- fused FP16/BF16 scalar fallbacks passed at `3x127`, and low-precision
  large-shape execution passed at `16x8192`.
- PyTorch external-stream, mutation-schema/FakeTensor, `torch.compile`, and
  CUDA Graph tests passed with the Stable ABI dispatcher;
- one exact dispatcher binary built with PyTorch 2.11.0+cu130 passed without
  recompilation on PyTorch 2.10.0+cu128; the 2.11 vLLM 0.24 and 0.25.1
  environments each passed 192 tests, while the vLLM-free 2.10 environment
  passed 123 applicable tests with 44 vLLM-dependent skips and two deselected
  vLLM-reference cases;
- the dispatcher exposes only `aoti_torch_*`/`torch_*` PyTorch symbol families,
  has no ATen/c10 C++ or raw CUDA launch dependency, and is protected by a
  source-level CI boundary;
- the checked normalization Rust bridge built and linked on H20, passed
  CUDA-feature Clippy and its address-range unit test, rejected short and
  overlapping buffers before launch, and recorded Add+RMSNorm path hits from
  direct PyTorch and vLLM IR plus RMSNorm+FP8 path hits from direct PyTorch;
- greedy+sampled-logprob now builds and links through the checked Rust bridge
  for both contiguous and padded row-strided logits, rejects short/overlapping
  regions before submission, and passes external-stream, compile, graph, and
  vLLM adapter tests;
- official vLLM 0.24.0 and 0.25.1 packages each passed the complete 192-test
  H20 Python GPU suite on Torch 2.11.0+cu130; the 0.25.1 process loaded its own
  `vllm/_C_stable_libtorch.abi3.so`, and the focused greedy/vLLM gate passed
  40 tests;
- Loom and vLLM's CUDA provider were bitwise identical for BF16 at `1x4096`,
  `8x4096`, `128x4096`, and `8x8192`;
- through the same vLLM IR dispatch, order-reversed H20 runs measured Loom
  `8.260-8.261 us` versus `vllm_c` `9.133-9.202 us` in eager execution, and
  Loom `2.769-2.779 us` versus `vllm_c` `2.866-2.877 us` in CUDA Graph replay;
- the provider completed vLLM 0.24 compilation, graph capture, and repeated
  generation with the normal Qwen2 model runner and a synthetic random model.
- RMSNorm+FP8 F32/FP16/BF16 outputs and F32 row scales were bitwise identical
  to vLLM 0.24 at `8x4096`, `3x127`, and `2x128` on H20;
- the 24-test normalization dispatcher suite passed with RMSNorm+FP8 coverage
  for external streams,
  dispatcher mutation, out-buffer reuse, fullgraph compilation, and CUDA Graph
  capture/replay;
- order-reversed BF16 `8x4096` named-baseline runs measured Loom
  `5.311-5.430 us` versus vLLM `7.994-8.052 us` through eager C++ dispatch,
  and Loom `3.947-4.044 us` versus vLLM `4.246-4.274 us` under graph replay.
- SiLU-and-Mul F32, FP16, and BF16 kernels passed CPU-oracle checks at
  `8x11008`, and the BF16 scalar fallback passed at `3x127`; observed raw
  medians were `3.975 us`, `4.279 us`, `4.392 us`, and `2.932 us`
  respectively;
- SiLU-and-Mul matched vLLM bitwise for F32/FP16/BF16 over representative,
  odd-width, and higher-rank inputs; the full Python suite passed 43 tests,
  including external streams, schema/FakeTensor checks, fullgraph compilation,
  CUDA Graph capture/replay, and vLLM layer dispatch;
- order-reversed BF16 `8x11008` named-baseline runs put Loom and vLLM within
  `0.1%` under CUDA Graph replay (`3.932-4.060 us` versus
  `3.935-4.061 us`). Eager dispatch changed materially with run order, so no
  standalone SiLU speedup is claimed;
- with the opt-in vLLM layer override, a synthetic Qwen2 engine completed
  model compilation, CUDA Graph capture, and generation on H20;
- SiLU-and-Mul+block-FP8 matched vLLM's fused operator exactly for FP16/BF16,
  groups 64/128, representative and higher-rank inputs, row/group-major scale
  layouts, and optional scale upper bounds;
- the complete H20 Python suite passed 62 tests; the 27 focused activation and
  vLLM tests also covered external streams, mutation schema/FakeTensor,
  `torch.compile`, auto-functionalization, CUDA Graph replay, and fusion-table
  registration; vLLM's official activation-quant pattern matcher rewrote the
  composed graph to Loom with one match;
- raw `8x11008` medians were `3.268 us`/`3.060 us` for FP16 group 64/128 and
  `3.262 us`/`2.952 us` for BF16 group 64/128, with zero byte or scale errors;
- order-reversed BF16 group-128 named-baseline runs measured
  `1.216-1.231x` eager speedup ratios (`17.7-18.8%` lower latency) and
  `1.037-1.082x` CUDA Graph ratios (`3.6-7.5%` lower latency) against vLLM's
  semantically identical fused operator.
- a pinned Qwen2.5-0.5B-Instruct checkpoint completed vLLM 0.24 online
  `fp8_per_block` loading, compilation, activation-quant rewriting, CUDA Graph
  capture, and repeated generation through Loom on H20;
- baseline-first and Loom-first runs matched every generated token ID across
  `1x128x128`, `8x128x128`, and `32x128x64` request shapes; both providers
  recorded two compiler-pattern matches, while the process-local path probe
  recorded zero Loom launches for the baseline and 1584 for Loom;
- order-reversed end-to-end batch-latency ratios ranged from `0.9991x` to
  `1.0043x`. This closes real-model invocation and correctness, but is parity
  rather than evidence of model-level TTFT, TPOT, or throughput improvement.
- fused RoPE+paged-KV matched vLLM's separate rotary and
  `reshape_and_cache_flash` operations across F32/FP16/BF16, NeoX/interleaved,
  NHD/HND cache layouts, partial RoPE, negative slots, external streams,
  packed-QKV views, and padded tensors with a shorter slot mapping;
- the complete H20 Python suite passed 94 tests; the Rust core passed 23
  contract/oracle tests, and the CUDA-feature workspace passed formatting,
  Clippy, release checking, plus two safe-wrapper tests against CPU oracles;
- on H20 BF16 Qwen2.5-style shapes, the fused dispatcher path measured roughly
  `2.30-2.40x` faster than vLLM's two-op path for 1-512 tokens. Ratios narrowed
  to `1.686x`, `1.240x`, `1.145x`, and `1.088x` at 1024, 2048, 4096, and 8192
  tokens respectively, confirming that this is a decode-oriented fusion;
- isolated baseline-first and Loom-first Qwen2.5-0.5B engine runs matched every
  generated token for `1x32x64` and `8x32x64`. The launch probe recorded zero
  Loom submissions in baseline processes and 552 in Loom processes;
- order-reversed engine batch-latency ratios ranged from `0.9957x` to
  `1.0180x`. Invocation and correctness are proven, but the end-to-end result
  crosses parity with provider order and does not establish a speedup.
- greedy argmax+sampled-logprob PyTorch tests cover F32/FP16/BF16, Qwen's
  151,936-token vocabulary, explicit ties, padded rows, external streams,
  FakeTensor/schema validation, `torch.compile`, and CUDA Graph replay; the 30
  focused operator/vLLM tests pass;
- against vLLM's exact `compute_logprobs + greedy_sample + gather_logprobs(0)`
  BF16 path on H20, token IDs and tie-aware ranks were exact and maximum
  logprob error was `9.54e-7`; 1-128 row speedup ratios were `3.16-4.35x`;
- baseline-first and Loom-first Qwen2.5-0.5B runs for `1x32x64`, `8x32x64`,
  and `32x32x32` matched every generated token, sampled-token rank, and
  sampled logprob within `1.32e-6`. Each baseline recorded zero Loom launches
  and each fused process recorded 1120;
- across both provider orders, real-engine batch-latency ratios were
  `1.129-1.250x` and TPOT ratios were `1.147-1.257x`. This establishes a
  model-level win for the deliberately narrow pure-greedy `logprobs=0` path,
  not for general sampling.
- selected-token PyTorch tests cover arbitrary IDs/ranks, F32/FP16/BF16,
  Qwen's 151,936-token vocabulary, ties, padded rows, external streams,
  FakeTensor/schema validation, `torch.compile`, and CUDA Graph replay;
- the current complete H20 Python suite passes 192 tests; the Rust core passes
  30 contract/oracle tests, and the CUDA-feature workspace passes formatting,
  Clippy, release build, plus seven safe-wrapper CPU-oracle tests;
- the final shared-library audit exposes 15 versioned bridge symbols and no
  raw CUDA launch symbols; the PyTorch shim depends on those same 15 bridge
  symbols and no raw launch symbol;
- against vLLM's exact `compute_logprobs + gather_logprobs(0)` path for the
  same caller-selected BF16 IDs, ranks were exact and maximum logprob error was
  `9.54e-7`; 1-128 row H20 speedup ratios were `2.77-3.78x`;
- order-reversed Qwen2.5-0.5B top-k/top-p runs matched every sampled token,
  rank, and raw logprob within `1.20e-6`. Baseline processes recorded zero
  Loom launches and Loom processes recorded 1440 selected-token launches;
- across both provider orders, top-k/top-p batch-latency ratios were
  `1.044-1.125x` and TPOT ratios were `1.054-1.130x`. vLLM still owns masks,
  penalties, top-k/top-p, RNG, and selection; Loom accelerates only the raw
  sampled-token logprob/rank tail.
- Min-P F32 masks and retained logits matched vLLM 0.24 exactly for 1, 8, 32,
  and 128 rows over a 151,936-token vocabulary. Loom used no tensor-sized
  temporary allocation versus `0.76-97.24 MB` for the composed baseline;
- Min-P latency ratios were `0.714x`, `0.771x`, `1.104x`, and `1.885x` for
  1, 8, 32, and 128 rows. The vLLM adapter therefore requires at least 32 rows
  and a 65,536-token vocabulary, falling back below either threshold. A second
  65,536-vocabulary sweep measured `1.35x` and `2.35x` at 32 and 128 rows;
- paged-decode focused tests pass 46/46 across F32/FP16/BF16, MQA/GQA,
  odd GQA tail groups, shuffled physical blocks, partial pages, odd head sizes,
  distinct value widths, native vLLM-interleaved cache strides, external
  streams, schema/FakeTensor, `torch.compile`, long-context split-K/LSE, and
  launch telemetry; the
  paged-decode/vLLM gate passes 34 focused tests and the safe Rust F32 wrapper
  plus its caller-owned split-K workspace path match the CPU oracle on H20;
- a 156-case native-interleaved layout sweep spans 13 dtype/head/block shapes,
  three batches, and four contexts with maximum absolute error `0.015625`;
  only 82 cases beat FA3 and 74 lose, confirming shape-dependent routing;
- a focused 132-case Hq/Hkv `32/8`, head-size-128 sweep qualifies FP16/BF16,
  block 16/32, and batches 1-128. All 44 context-16 cases are at least `1.42x`
  and all 44 context-32 cases at least `1.15x` under CUDA Graph replay;
  context 64 remains mixed;
- the opt-in vLLM 0.24 route admits only that context-32-and-below envelope.
  Against `FlashAttentionImpl.forward`, all 24 admitted cases win
  (`1.154-2.374x`, median `1.478x`, CUDA Graph); all 12 context-64 cases execute
  FA3 with a `1.001x` median graph ratio. Eager fallback retains about `3.7%`
  Python wrapper overhead in this isolated method benchmark;
- the 16-case BF16/block-16 long-context split-K gate spans batches 1/2/4/8
  and contexts 128/256/512/1,024. Every CUDA Graph case beats Loom's legacy
  single-CTA path (`1.140-6.223x`, median `2.497x`) with maximum FA3 absolute
  error `0.00390625`; FP16 and block-32 cross-checks also win every legacy
  comparison. FA3 remains faster overall, so no long-context vLLM route was
  added;
- a 72-case odd-GQA `14/2`, D64 sweep passes at maximum absolute error
  `0.015625`. All 36 context-16 cases win under CUDA Graph replay and 31/36
  context-32 cases win; block-16 batches 24/32 remain below FA3;
- an experimental pretrained Qwen2.5-0.5B route recorded `0/408` baseline/Loom
  host submissions, but only two of five cases preserved every generated token
  and Loom was about 3-5% slower end to end. The adapter change was rejected;
- order-reversed one-layer stable-output synthetic Qwen2 engine gates match
  every token, record zero Loom launches in baseline processes and 18 only in
  Loom processes, and exercise actual FA3 scheduler metadata plus native
  interleaved cache strides. End-to-end ratios remain process-order sensitive,
  so this proves engine invocation but not model-level acceleration;

See the [F32 report](results/h20-rms-norm-f32-smoke-20260721.json) and
[low-precision report](results/h20-rms-norm-low-precision-20260721.json), plus
the [fused Add+RMSNorm report](results/h20-add-rms-norm-20260721.json) and
[vLLM IR integration report](results/h20-vllm-ir-add-rms-norm-20260721.json).
The [RMSNorm+dynamic-FP8 report](results/h20-rms-norm-dynamic-fp8-20260721.json)
contains its exact contract, raw CUDA results, and order-reversed vLLM
comparison. The [SiLU-and-Mul report](results/h20-silu-and-mul-20260721.json)
records exact vLLM compatibility, graph parity, eager instability, and the
engine smoke gate. The
[SiLU-and-Mul+dynamic-block-FP8 report](results/h20-silu-and-mul-dynamic-fp8-20260721.json)
records the fused contract, exact vLLM comparison, raw CUDA results, and
order-reversed named baseline. The
[Qwen2.5 FP8 engine report](results/h20-vllm-qwen25-05b-fp8-engine-20260722.json)
records the pinned checkpoint, path-hit evidence, exact-token gates, and
order-reversed end-to-end parity result.
The [RoPE+paged-KV report](results/h20-rope-paged-kv-20260722.json),
[large-token sweep](results/h20-rope-paged-kv-large-20260722.json), and
[Qwen2.5 engine gate](results/h20-vllm-qwen25-rope-paged-kv-engine-20260722.json)
separate operator-level benefit from real-engine invocation and end-to-end
parity.
The [greedy sampled-logprob operator report](results/h20-greedy-sample-logprobs-20260722.json)
and order-reversed
[baseline-first](results/h20-vllm-greedy-logprobs-baseline-first-20260722.json)
and [Loom-first](results/h20-vllm-greedy-logprobs-loom-first-20260722.json)
engine reports record the first qualified end-to-end acceleration.
The [selected-token operator report](results/h20-selected-token-logprobs-20260722.json)
and top-k/top-p order-reversed
[baseline-first](results/h20-vllm-selected-logprobs-baseline-first-20260722.json)
and [Loom-first](results/h20-vllm-selected-logprobs-loom-first-20260722.json)
engine reports extend that result without moving sampling policy into Loom.
The [Min-P operator report](results/h20-min-p-filter-20260722.json) records the
exact-mask gate, memory reduction, small-batch regression, and crossover point;
the [65,536-token sweep](results/h20-min-p-filter-vocab65536-20260722.json)
records the lower vocabulary boundary behind the vLLM routing threshold.
The original
[paged decode-attention report](results/h20-paged-decode-attention-20260722.json)
records separate-cache bring-up. The
[native-interleaved shape sweep](results/h20-paged-decode-interleaved-shape-sweep-20260722.json),
[focused Qwen-shape batch sweep](results/h20-paged-decode-qwen-batch-sweep-20260722.json),
and [vLLM backend report](results/h20-vllm-paged-decode-backend-20260722.json)
define the routing boundary. The synthetic-Qwen
[baseline-first](results/h20-vllm-paged-decode-engine-baseline-first-20260722.json)
and [Loom-first](results/h20-vllm-paged-decode-engine-loom-first-20260722.json)
reports prove isolated real-engine invocation and preserve the neutral
end-to-end conclusion.
The [odd-GQA operator sweep](results/h20-paged-decode-odd-gqa-20260722.json),
[non-regression backend gate](results/h20-vllm-paged-decode-tail-gqa-backend-20260722.json),
and [rejected pretrained-Qwen experiment](results/h20-vllm-qwen25-paged-decode-rejected-20260722.json)
record the partial-tail extension, preservation of the existing route, and the
reason `14/2`, D64 is not exposed through vLLM.
The long-context [BF16/block-16 split-K report](results/h20-paged-decode-split-k-20260722.json),
[FP16 cross-check](results/h20-paged-decode-split-k-f16-20260722.json), and
[block-32 cross-check](results/h20-paged-decode-split-k-block32-20260722.json)
record the stable LSE merge, legacy speedup, and explicit decision to retain
FA3 for the engine's 128-1,024-token path.

## Not Yet Proven

- fused Add+RMSNorm or RMSNorm+FP8 model-level benefit;
- SiLU-and-Mul+FP8 model-level benefit on a workload where the exposed
  activation-quant boundary is material;
- RoPE+paged-KV model-level TTFT/TPOT or throughput benefit beyond the current
  exact-token engine integration gate;
- Min-P real-model invocation and end-to-end serving benefit;
- Loom-owned logits preprocessing, top-k/top-p, stochastic sampling, and
  general top-k logprob integration;
- speculative draft verification, acceptance/rejection, token compaction, and
  an end-to-end draft/target engine win;
- FP8/INT8 KV-cache compression with measured cache bytes, quality, admitted
  context/batch, and TPOT;
- prefix-cache/preemption KV movement and compaction in a real scheduler path;
- MoE routing/permutation/combine benefit around an unchanged vendor grouped
  GEMM;
- an engine-neutral zero-copy Rust decode-step proof;
- a paged decode-attention pretrained-model route that passes token/quality and
  end-to-end gates; the attempted Qwen2.5 `14/2`, D64 route was rejected;
- an FA3-competitive paged-decode kernel at 1,024 tokens and batches above one;
- integration into SGLang or a Rust-native engine path;
- larger production-model and serving-workload validation;
- automated native Python/PyTorch/CUDA matrix wheels, clean-install gates, or
  binary compatibility beyond the tested PyTorch 2.10/2.11 runtimes;
- serving-scale concurrency, goodput, and memory improvement.
