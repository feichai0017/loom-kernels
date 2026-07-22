# LLM Inference Operator Catalog

Loom Kernels aims to cover the common operator surface of production LLM
inference, but it is not a general tensor framework. An operator belongs here
when an inference engine needs a stable backend boundary, or when fusion,
layout knowledge, quantization, or launch reduction can improve a measured
workload.

The catalog deliberately does not promise handwritten replacements for every
pointwise expression. Dense GEMM should normally use cuBLASLt, CUTLASS, or an
engine-selected vendor implementation; Loom owns the Rust contract, dispatch,
valuable epilogues, and evidence around it.

## Status Legend

| State | Meaning |
| --- | --- |
| supported | contract, oracle, CUDA, framework adapter, and H20 evidence exist |
| next | admitted to the immediate implementation queue |
| planned | useful surface with a named engine consumer, ordered after `next` |
| profile-gated | implemented only when a real workload shows material cost |
| vendor-backed | Loom exposes or fuses the boundary but does not reimplement the base primitive |

## Normalization And Quantization

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| RMSNorm | P0 | supported | standalone normalization |
| residual Add+RMSNorm | P0 | supported | residual update plus normalization |
| RMSNorm+dynamic per-token FP8 | P0 | supported | normalization plus quantized GEMM input |
| RMSNorm+dynamic INT8 | P0 | next | normalization plus INT8 GEMM input |
| LayerNorm and Add+LayerNorm | P2 | profile-gated | models that actually use LayerNorm |
| static/dynamic per-token, per-channel, and per-block quantization | P0/P1 | planned | FP8/INT8/INT4 scale production and packing |
| dequantize, requantize, and scale conversion | P1 | planned | layout-aware transitions between kernels |

## MLP Activations And Linear Epilogues

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| split-half SiLU-and-Mul (SwiGLU) | P0 | supported | standard Llama/Qwen MLP activation |
| SiLU-and-Mul+dynamic per-block FP8 | P0 | supported | activation directly into FP8 down projection |
| SiLU-and-Mul+dynamic INT8 | P0 | next | activation directly into INT8 down projection |
| GELU, GELU-tanh, GELU-and-Mul, GeGLU, ReGLU | P1 | planned | model-specific gated MLPs |
| bias+activation and bias+gated activation | P1 | planned | GEMM output epilogues |
| GEMM+bias+activation/quantization | P0 | vendor-backed | cuBLASLt/CUTLASS base GEMM with Loom-owned epilogue |
| quantized linear and grouped linear dispatch | P1 | vendor-backed | engine-selected FP8/INT8/INT4 matrix core |

## Embedding, Position, And KV Cache

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| NeoX and interleaved RoPE, partial rotary dimensions | P0 | next | in-place Q/K position encoding; currently exposed through the fused cache-write path |
| RoPE+paged-KV write | P0 | supported | position encoding without materializing another K pass |
| paged-KV reshape/store/append | P0 | planned | engine tensor to cache layout |
| KV block copy, swap, gather, scatter, compact, and remap | P0 | planned | cache movement and prefix reuse |
| FP8/INT8 KV quantize/dequantize with scale update | P0 | planned | compressed cache read/write |
| embedding gather and parallel-vocabulary embedding | P1 | profile-gated | lookup plus dtype/layout conversion |

## Logits, Sampling, And Log Probabilities

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| logits bias, temperature, masks, and bad-word suppression | P0 | planned | one logits preprocessing pass |
| repetition, presence, and frequency penalties | P0 | planned | sparse history-aware update |
| greedy argmax+sampled-token raw logprob | P0 | supported | one-pass selection, normalization, gather, and tie-aware rank |
| general selected-token raw logprob+rank | P0 | supported | engine-owned sampling followed by one-pass normalization and tie-aware rank |
| deterministic RNG sampling | P0 | planned | token selection without host round trips |
| top-k, top-p, min-p, and renormalization | P0 | planned | fused candidate filtering and sampling |
| top-k logprobs | P0 | planned | avoid full-vocabulary probability tensors when multiple candidates are returned |
| sharded-vocabulary top-k/logsumexp merge | P1 | planned | tensor-parallel token selection |
| structured-output bitmask application | P1 | profile-gated | grammar mask plus logits processing |

## Mixture Of Experts

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| softmax/sigmoid top-k routing and grouped top-k | P1 | planned | score, select, renormalize, and expert map |
| expert histogram, prefix sum, token permutation, and alignment | P1 | planned | dispatch preparation without host work |
| inverse permutation and weighted expert reduction | P1 | planned | combine routed expert outputs |
| grouped GEMM and quantized grouped GEMM | P1 | vendor-backed | vendor matrix core plus stable Rust dispatch |
| shared-expert gate and routed/shared output fusion | P1 | planned | reduce temporary expert tensors |

## Attention

| Operator | Priority | State | Intended fusion boundary |
| --- | --- | --- | --- |
| paged MQA/GQA decode attention | P1 | planned | decode over engine-owned paged KV |
| ragged prefill attention | P1 | vendor-backed | FlashAttention/FlashInfer selected by evidence |
| split-KV state and numerically stable LSE merge | P1 | planned | long-context and distributed attention |
| sliding-window, ALiBi, soft-cap, and causal variants | P1 | planned | standard attention contract options |
| MLA paged decode and latent-cache transforms | P1 | planned | DeepSeek-style inference path |
| speculative/tree attention masks | P2 | profile-gated | engine-specific speculative decoding |

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

1. Extend the proven engine-owned selected-token path with fused logits
   preprocessing, top-k/top-p, and deterministic RNG sampling where profiling
   shows additional value.
2. Optimize the real-engine RoPE+paged-KV boundary only where profiling shows
   TPOT materiality; keep its current parity result explicit.
3. Add MoE routing/movement before attempting a full grouped-GEMM stack.
4. Add paged decode attention against engine-owned cache contracts.
5. Add INT8 fused boundaries only for a named engine/model consumer.
6. Attempt communication-aware fusion only after single-GPU operators and
   real tensor-parallel workloads are reproducible.

Every item advances independently through the admission gates in the
[operator-library design](design/operator-library.md). Catalog membership is a
product direction, not a performance or production-readiness claim.
