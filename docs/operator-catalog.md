# LLM Inference Operator Catalog

Loom Kernels aims to cover the common operator surface of production LLM
inference, but it is not a general tensor framework. An operator belongs here
when an inference engine needs a stable backend boundary, or when fusion,
layout knowledge, quantization, or launch reduction can improve a measured
workload.

The catalog deliberately does not promise handwritten replacements for every
pointwise expression. Dense, quantized, sparse, and grouped GEMM always use
cuBLASLt, CUTLASS, FlashInfer, or an engine-selected vendor implementation.
Loom owns only memory-bound preparation, movement, fusion, and epilogues around
that matrix core.

Catalog admission requires a memory/layout/scheduling bottleneck, a named
engine gap, and a real model or serving exit gate. A plausible CUDA kernel or
isolated microbenchmark is not sufficient.

## Status Legend

| State | Meaning |
| --- | --- |
| supported | contract, oracle, CUDA, framework adapter, and H20 evidence exist |
| in progress | source path exists, but required hardware or engine evidence is still open |
| next | admitted to the immediate implementation queue |
| planned | useful surface with a named engine consumer, ordered after `next` |
| profile-gated | implemented only when a real workload shows material cost |
| vendor-backed | the engine/vendor owns the base primitive; Loom may expose only an adjacent memory-bound boundary |

## Normalization And Quantization

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| RMSNorm | P0 | supported | standalone normalization |
| residual Add+RMSNorm | P0 | supported | residual update plus normalization |
| RMSNorm+dynamic per-token FP8 | P0 | supported | normalization plus quantized GEMM input |
| RMSNorm+dynamic INT8 | P0 | next | normalization plus INT8 GEMM input |
| LayerNorm and Add+LayerNorm | P2 | profile-gated | models that actually use LayerNorm |
| static/dynamic per-token, per-channel, and per-block quantization | P0/P1 | planned | FP8/INT8/INT4 scale production and packing around vendor GEMM |
| dequantize, requantize, pack/unpack, and scale conversion | P1 | planned | layout-aware transitions between vendor kernels |

## MLP Activations And Linear Epilogues

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| split-half SiLU-and-Mul (SwiGLU) | P0 | supported | standard Llama/Qwen MLP activation |
| SiLU-and-Mul+dynamic per-block FP8 | P0 | supported | activation directly into FP8 down projection |
| SiLU-and-Mul+dynamic INT8 | P0 | next | activation directly into INT8 down projection |
| GELU, GELU-tanh, GELU-and-Mul, GeGLU, ReGLU | P1 | planned | model-specific gated MLPs |
| bias+activation and bias+gated activation | P1 | planned | GEMM output epilogues |
| vendor GEMM handoff+bias+activation/quantization | P0 | vendor-backed | unchanged cuBLASLt/CUTLASS matrix core with a measured Loom pre/post boundary |
| quantized linear and grouped linear | — | vendor-backed | explicitly engine-owned FP8/INT8/INT4 matrix core; Loom does not implement it |

## Embedding, Position, And KV Cache

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| NeoX and interleaved RoPE, partial rotary dimensions | P0 | next | in-place Q/K position encoding; currently exposed through the fused cache-write path |
| RoPE+paged-KV write | P0 | supported | position encoding without materializing another K pass |
| paged-KV reshape/store/append | P0 | next | engine tensor to cache layout |
| KV block copy, swap, gather, scatter, compact, and remap | P0 | next | prefix reuse, preemption, beam movement, and cache compaction |
| RoPE+paged-KV write to static FP8 E4M3 | P0 | in progress | implementation, exact-byte, clean-wheel, operator, and engine-invocation gates pass; native-vs-FP8 quality/capacity/serving value remains open |
| dynamic FP8 per-token-head scale/write | P1 | planned | separate engine scale-cache contract only when a named backend requires it |
| INT8 KV quantize/dequantize with scale update | P1 | planned | admitted only by a named engine/model cache contract |
| embedding gather and parallel-vocabulary embedding | P1 | profile-gated | lookup plus dtype/layout conversion |

## Logits, Sampling, And Log Probabilities

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| logits bias, temperature, masks, and bad-word suppression | P0 | next | one logits preprocessing pass |
| repetition, presence, and frequency penalties | P0 | next | sparse history-aware update |
| greedy argmax+sampled-token raw logprob | P0 | supported | one-pass selection, normalization, gather, and tie-aware rank |
| general selected-token raw logprob+rank | P0 | supported | engine-owned sampling followed by one-pass normalization and tie-aware rank |
| in-place min-p filtering | P0 | supported | row-max threshold without probability or mask tensors; vLLM route is H20 shape-gated |
| deterministic counter-based RNG sampling | P0 | next | seeded token selection without host round trips |
| top-k, top-p, and renormalization | P0 | next | fused candidate filtering and sampling |
| top-k logprobs | P0 | next | avoid full-vocabulary probability tensors when multiple candidates are returned |
| sharded-vocabulary top-k/logsumexp merge | P1 | planned | tensor-parallel token selection |
| structured-output bitmask application | P1 | profile-gated | grammar mask plus logits processing |

## Speculative Decoding Support

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| batched draft-verification metadata and tree/branch mask construction | P0 | profile-gated | prepare one vendor-attention verification call without host metadata loops |
| greedy draft verification+accepted/bonus-token compaction | P0 | supported | flattened ragged comparison and compact emission without device-to-host control flow |
| stochastic rejection sampling | P0 | profile-gated | residual-distribution acceptance/rejection with explicit counter-based RNG state |
| speculative KV slot commit, rollback, and remap | P0 | profile-gated | caller-owned cache metadata movement when an engine exposes the boundary |
| draft/target model GEMM and verification attention | — | vendor-backed | engine-selected matrix and attention backends; never reimplemented by Loom |

## Mixture Of Experts

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| softmax/sigmoid top-k routing and grouped top-k | P1 | planned | score, select, renormalize, and expert map |
| expert histogram, prefix sum, token permutation, and alignment | P1 | planned | dispatch preparation without host work |
| inverse permutation and weighted expert reduction | P1 | planned | combine routed expert outputs |
| grouped GEMM and quantized grouped GEMM | — | vendor-backed | explicitly engine-owned vendor matrix core fed by Loom-permuted buffers |
| shared-expert gate and routed/shared output fusion | P1 | planned | reduce temporary expert tensors |

## Attention

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| paged MQA/GQA decode attention | P1 | supported | GQA-packed Rust/CUDA/PyTorch path, D128 caller-owned split-K workspace for 128-1,024 tokens at batch <=8, plus an opt-in H20-qualified vLLM route for native interleaved FP16/BF16 Hq/Hkv 32/8, D128, block 16/32, batch <=128, context <=32 |
| ragged prefill attention | P1 | vendor-backed | FlashAttention/FlashInfer selected by evidence |
| local split-KV state and numerically stable LSE merge | P1 | supported | D128 long-context decode with explicit F32 workspace |
| distributed split-KV/LSE merge | P1 | planned | context-parallel attention after transport qualification |
| sliding-window, ALiBi, soft-cap, and causal variants | P1 | planned | standard attention contract options |
| MLA paged decode and latent-cache transforms | P1 | planned | DeepSeek-style inference path |
| speculative verification attention | — | vendor-backed | engine-selected attention consumes Loom-prepared verification metadata |

## Communication-Aware Operators

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| all-reduce+residual+RMSNorm | P2 | planned | tensor-parallel post-attention/MLP path |
| reduce-scatter/all-gather+quantization epilogues | P2 | planned | communication payload reduction |
| tensor-parallel logits top-k merge | P1 | planned | distributed sampling without full gather |
| expert-parallel dispatch/combine | P2 | planned | token movement plus permutation metadata |

Collectives will wrap NCCL or another qualified transport. Loom will not claim
a distributed speedup from a same-process adapter or from a local kernel alone.

## Layout And Internal Primitives

Cast, transpose, concatenate/split, pad/unpad, gather/scatter, reductions,
prefix sums, packing, and block copies are profile-gated. They may be shared
internal building blocks, but become public operators only when an engine needs
the boundary or an isolated implementation is measurably useful.

## Implementation Order

1. Finish system-level qualification for the static FP8 E4M3 quantize-on-write
   path by proving cache bytes, admitted context/batch, quality, TTFT, and TPOT
   together; H20 operator and clean-wheel qualification are complete.
2. Complete the sampling tail with fused preprocessing, penalties, top-k/top-p,
   renormalization, deterministic RNG, and top-k logprobs.
3. Add KV block movement for a real prefix-cache, preemption, or compaction
   call site.
4. Return to tree metadata, stochastic speculative rejection, or KV
   commit/remap only when a named engine profile shows material cost. The
   current real-model gate puts verification below `0.2%` of batch latency.
5. Fill quantization scale/pack/layout gaps only around an unchanged vendor
   GEMM path.
6. Add MoE routing, histogram/prefix sum, permutation, and combine; grouped
   GEMM remains entirely engine/vendor-owned.
7. Build the engine-neutral Rust decode proof after one new feature reaches an
   engine.
8. Broaden paged decode or communication-aware fusion only when profiling and
   reproducible engine workloads justify it.

Every item advances independently through the admission gates in the
[operator-library design](design/operator-library.md). Catalog membership is a
product direction, not a performance or production-readiness claim.
