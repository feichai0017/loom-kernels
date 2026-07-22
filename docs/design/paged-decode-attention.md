# Paged Decode Attention Contract

The first Loom paged-attention boundary is deliberately narrower than a full
FlashAttention replacement. It models the latency-sensitive decode case where
an inference engine owns scheduling and KV-cache lifetime, has already written
the current token's K/V values, and submits one query token per active request.

## Logical Tensors

- query: `[sequences, query_heads, head_size]`;
- key cache: `[num_blocks, block_size, kv_heads, head_size]`;
- value cache: `[num_blocks, block_size, kv_heads, value_head_size]`;
- block tables: `[sequences, max_blocks_per_sequence]`;
- sequence lengths: `[sequences]`, including the current cached token;
- output: `[sequences, query_heads, value_head_size]`.

This matches the logical NHD cache consumed by vLLM 0.24. A framework adapter
may preserve an HND physical stride order without materializing a new cache.
`query_heads` must be divisible by `kv_heads`; consecutive groups of query
heads share one KV head for MQA/GQA.

For logical position `p`, the physical token is selected by
`block_tables[sequence, p / block_size]` and `p % block_size`. The base score
and output are:

```text
score(p) = scale * dot(query, key_cache[p])
output   = sum(softmax(score)[p] * value_cache[p])
```

The Rust CPU oracle uses a stable max-subtracted softmax and validates every
active block ID before touching output. Unused block-table entries may contain
negative sentinels; active entries may not.

## First-Phase Scope

The base contract includes F32, FP16, and BF16 native KV caches, standard
causal decode, MQA/GQA, block indirection, and distinct key/value head widths.
It intentionally excludes:

- multi-token speculative or chunked-prefill queries;
- sliding windows, ALiBi, logits soft caps, attention sinks, or custom masks;
- FP8/INT8 KV cache scaling;
- cascade/common-prefix and decode-context-parallel execution;
- distributed transport or cross-device KV ownership.

Those options become separate contract fields only after the base kernel and a
named engine path are correct. They will not be hidden behind silent fallback
inside the Rust operator.

## Implemented Short-Context Kernel

The first handwritten CUDA implementation assigns one 256-thread block to
each `(sequence, query_head)` pair. Eight warps compute Q/K dot products over
the paged cache, a block reduction performs stable max-subtracted softmax in
F32, and threads accumulate independent value dimensions. The dynamic score
buffer deliberately caps this path at 1,024 tokens.

The C ABI and PyTorch boundary accept contiguous int32 block tables and
sequence lengths, matching vLLM's live metadata. Their active values are
trusted engine metadata: the host-supplied maximum length and active block IDs
must be valid. The safe Rust entrypoints preserve the same shapes and dtypes
with checked device-buffer sizes. No implicit device-to-host validation or
fallback is introduced on the launch path.

## Qualification Sequence

1. Rust contract and CPU oracle, including invalid metadata and GQA mapping;
2. a PyTorch reference cross-check over randomized block tables and lengths;
3. short-context one-pass CUDA and long-context split-K/LSE candidates;
4. current-stream PyTorch schema, FakeTensor, compile, and CUDA Graph gates;
5. vLLM 0.24 FlashAttention logical-layout adapter with explicit fallback;
6. H20 comparison against the engine-selected FA3/FlashInfer implementation;
7. real-model TPOT, throughput, and KV-memory evidence.

Steps 1-4 and the operator-level part of step 6 are complete. Randomized
PyTorch tests cover MQA/GQA, partial final blocks, shuffled physical blocks,
distinct value widths, F32/FP16/BF16, external streams, FakeTensor/schema,
`torch.compile`, and launch telemetry. On NVIDIA H20 all 16 focused tests and
the 144-test Python suite pass.

The named BF16 FA3 matrix establishes a narrow performance envelope rather
than a blanket replacement. At context 16, CUDA Graph speedups are `1.43x`,
`1.44x`, and `2.04x` for batches 1, 8, and 32. At context 32 only batch 32
remains ahead (`1.11x`); at 64-512 tokens the first kernel is slower than FA3.
Therefore step 5 remains open and vLLM automatic routing is intentionally not
enabled. The next CUDA work is a tiled/split-K design for the 32-128-token
region, followed by an explicit measured-shape fallback and real-model gate.
