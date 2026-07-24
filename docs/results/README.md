# H20 evidence index

Machine-readable correctness and performance artifacts for Loom Kernels. All
results in this directory were captured on NVIDIA H20 unless the artifact says
otherwise.

[Documentation](../README.md) · [Implementation status](../status.md) · [Benchmark page](https://feichai0017.github.io/loom-kernels/benchmarks/)

> [!IMPORTANT]
> Operator latency, dispatcher latency, CUDA Graph replay, engine latency, and
> serving performance are different claims. The decision column below states
> the narrow conclusion supported by each result set.

## Evidence ladder

| Level | What it proves |
| --- | --- |
| Correctness | The accelerator agrees with the declared oracle and contract |
| Operator | A warmed kernel or fused boundary beats a named equivalent baseline |
| Engine | A real framework/engine invokes Loom and preserves outputs |
| Serving | TTFT, TPOT, throughput, memory, or goodput improves under load |

## Compatibility

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Native Python ABI2 matrix wheel | [current clean-install H20 gate](h20-native-wheel-clean-install-abi2-20260724.json) | One exact `py3-none-linux_x86_64` wheel contains the two Loom `.so` files and passes 225 tests with each vLLM minor plus 138 applicable tests on PyTorch 2.10. It is not published. |
| Historical ABI1 matrix wheel | [predecessor clean-install H20 gate](h20-native-wheel-clean-install-20260723.json) | Preserved as the earlier 192/123-test artifact; ABI 2 is the current wheel boundary. |
| LibTorch Stable ABI across PyTorch minors | [two-minor H20 binary gate](h20-libtorch-stable-abi-20260723.json) | The source-built predecessor established the PyTorch 2.10 Stable ABI target and same-binary 2.10/2.11 boundary; the packaged clean-install result is the row above. |
| Pre-Stable-ABI single Rust bridge | [breaking-change H20 gate](h20-single-rust-bridge-compatibility-20260723.json) | Historical revision `cb5feaf` first proved all ten framework families on the Rust-owned path and passed 191 tests on each vLLM minor. The current dispatcher result is the row above. |
| Historical partial-bridge baseline | [pre-unification 0.24/0.25 gate](h20-vllm-compatibility-rust-bridge-20260723.json) | Preserved as historical evidence for revision `3ae4210`; its raw-ABI routing description does not apply to the current architecture. |

## Normalization and activation

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| RMSNorm | [F32 bring-up](h20-rms-norm-f32-smoke-20260721.json) · [FP16/BF16 paths](h20-rms-norm-low-precision-20260721.json) | Handwritten CUDA correctness and low-precision vector paths are qualified |
| Add+RMSNorm | [Operator gate](h20-add-rms-norm-20260721.json) · [vLLM IR gate](h20-vllm-ir-add-rms-norm-20260721.json) | Double in-place fusion and current-stream engine dispatch are supported |
| RMSNorm→FP8 | [Operator gate](h20-rms-norm-dynamic-fp8-20260721.json) · [Qwen2.5 engine gate](h20-vllm-qwen25-05b-fp8-engine-20260722.json) | Exact path invocation is proven; real-model latency remains at parity |
| SiLU-and-Mul | [Operator and engine gate](h20-silu-and-mul-20260721.json) | Compatible and engine-valid; CUDA Graph latency is at parity |
| SiLU-and-Mul→block FP8 | [Fused operator gate](h20-silu-and-mul-dynamic-fp8-20260721.json) · [Qwen2.5 engine gate](h20-vllm-qwen25-05b-fp8-engine-20260722.json) | Operator-level advantage; exact real-model invocation; end-to-end parity |

## RoPE and paged-KV write

| Result set | Current conclusion |
| --- | --- |
| [Decode-sized operator sweep](h20-rope-paged-kv-20260722.json) · [large-token sweep](h20-rope-paged-kv-large-20260722.json) | Fusion wins most strongly at decode-sized token counts and narrows with larger batches |
| [Baseline-first engine gate](h20-vllm-qwen25-rope-paged-kv-engine-20260722.json) · [Loom-first engine gate](h20-vllm-qwen25-rope-paged-kv-engine-loom-first-20260722.json) | Exact tokens and Loom path hits are proven; order reversal crosses parity, so no model-level speedup is claimed |
| [Static FP8 E4M3 cache-write gate](h20-fp8-kv-cache-write-20260724.json) | Exact vLLM cache bytes, framework/clean-wheel coverage, `2x` BF16-to-FP8 physical storage ratio, and `1.317-1.378x` operator ratios are qualified. Both engine orders preserve tokens and hit Loom; latency is order-sensitive, so the native-vs-FP8 quality/capacity/serving gate remains open. |

## Sampling and log probabilities

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Greedy + sampled logprob | [Operator gate](h20-greedy-sample-logprobs-20260722.json) · [baseline first](h20-vllm-greedy-logprobs-baseline-first-20260722.json) · [Loom first](h20-vllm-greedy-logprobs-loom-first-20260722.json) | Exact tokens/ranks and an order-stable real-engine win for pure greedy `logprobs=0` |
| Selected-token logprob + rank | [Operator gate](h20-selected-token-logprobs-20260722.json) · [baseline first](h20-vllm-selected-logprobs-baseline-first-20260722.json) · [Loom first](h20-vllm-selected-logprobs-loom-first-20260722.json) | vLLM-owned top-k/top-p sampling preserves exact tokens/ranks and shows an order-stable engine win |
| Min-P | [151,936-vocabulary sweep](h20-min-p-filter-20260722.json) · [65,536-vocabulary boundary](h20-min-p-filter-vocab65536-20260722.json) | The crossover is shape-dependent; the adapter routes only qualified larger rows/vocabularies |

## Speculative decoding

| Boundary | Result set | Current conclusion |
| --- | --- | --- |
| Greedy verify + accepted/bonus compaction | [15-case H20 gate](h20-greedy-speculative-verify-20260723.json) | Bit-exact with vLLM 0.24 across batches 1-256 and draft lengths 1/4/8; `1.101-1.128x` operator-level ratio. The source suite also passes 202 tests on vLLM 0.24 and 0.25.1. No end-to-end model claim. |
| Real draft/target engine gate | [native first](h20-vllm-qwen25-speculative-native-first-20260723.json) · [Loom first](h20-vllm-qwen25-speculative-loom-first-20260723.json) | Qwen2.5-1.5B target plus 0.5B draft on vLLM 0.24 preserves exact native/Loom tokens and draft statistics with `714/714` measured Loom calls in each order. Loom's verifier boundary is `1.026-1.133x` faster but only `0.048-0.200%` of batch latency; end-to-end native/Loom ratios cross parity and speculative decode is `3.18-4.97x` slower than target-only in these cases. |

The target-only and speculative providers use different target-model execution
shapes. At batch 32, both speculative providers follow the same deterministic
trajectory while two of 32 target-only requests diverge after generated token
51 or 53; batch 1 and 8 match fully. The reports retain those mismatches and
make target-only equality informational. Exact native-vLLM versus Loom
speculative output is the correctness gate.

## Paged-decode attention

| Evidence set | Result set | Current conclusion |
| --- | --- | --- |
| Bring-up | [Separate-cache report](h20-paged-decode-attention-20260722.json) | Base Rust/CUDA/PyTorch contract and correctness path |
| Native-layout breadth | [156-case interleaved sweep](h20-paged-decode-interleaved-shape-sweep-20260722.json) · [Qwen 32/8 batch sweep](h20-paged-decode-qwen-batch-sweep-20260722.json) | Performance is geometry-dependent; only the measured short envelope is admitted |
| vLLM short route | [Backend gate](h20-vllm-paged-decode-backend-20260722.json) · [baseline-first engine gate](h20-vllm-paged-decode-engine-baseline-first-20260722.json) · [Loom-first engine gate](h20-vllm-paged-decode-engine-loom-first-20260722.json) | All 24 admitted backend cases win; synthetic-engine path and stable tokens are proven, not pretrained-model acceleration |
| Odd GQA experiment | [72-case sweep](h20-paged-decode-odd-gqa-20260722.json) · [rejected Qwen2.5 route](h20-vllm-qwen25-paged-decode-rejected-20260722.json) · [32/8 non-regression gate](h20-vllm-paged-decode-tail-gqa-backend-20260722.json) | The broader Qwen2.5 route failed token and latency gates and was removed; the existing route remains qualified |
| Local split-K/LSE | [BF16 block-16](h20-paged-decode-split-k-20260722.json) · [FP16](h20-paged-decode-split-k-f16-20260722.json) · [BF16 block-32](h20-paged-decode-split-k-block32-20260722.json) | Faster than legacy Loom across the tested long-context matrix; FA3 remains the engine fallback |

## Reproducing a claim

1. Use the commit, GPU, software versions, shapes, warm-up, iterations, and
   sample counts recorded in the JSON artifact.
2. Run correctness before timing.
3. Keep the named baseline and dispatch boundary unchanged.
4. Reverse provider order for engine comparisons.
5. Report regressions and rejected routes alongside wins.
