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

This matches the logical NHD cache consumed by vLLM 0.24/0.25. The inner
`[block_size, kv_heads, head_size]` dimensions must be dense, while the outer
block stride is explicit. Consequently, both separate K/V allocations and the
two noncontiguous views of vLLM's physical
`[blocks, 2, block_size, kv_heads, head_size]` cache are accepted without a
copy. `query_heads` must be divisible by `kv_heads`; consecutive groups of
query heads share one KV head for MQA/GQA.

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

## Implemented CUDA Kernels

The base handwritten CUDA implementation assigns one 256-thread block to each
`(sequence, query_head)` pair. Eight warps compute Q/K dot products over the
paged cache, a block reduction performs stable max-subtracted softmax in F32,
and threads accumulate independent value dimensions.

For GQA contexts above 16 tokens, a packed specialization assigns one block to
two query heads that share a KV head. It resolves each paged token once, uses
pair loads for Q/K where alignment permits, computes both softmax rows, and
loads each V element once for both output heads. A final partial group is
guarded explicitly, so odd GQA ratios such as Qwen2.5-0.5B's `14/2` geometry
do not fall back to one block per query head. Full and partial groups are
separate compile-time specializations: the established even-ratio path keeps
an unguarded hot loop. Head-size 64 also caches each lane's fixed decode Q pair
in registers instead of reloading it for every context position.

When the grid contains at least 128 `(sequence, kv_head)` work items and the
GQA ratio is divisible by four, the H20-qualified dispatch packs four query
heads instead. A partial four-head tail requires at least 256 resulting packed
blocks; smaller grids keep two-head packing to retain parallelism. The dynamic
score and token-offset buffers serve the one-pass path; the public contract
deliberately caps every implementation at 1,024 tokens.

For D128 with equal K/V widths, batches up to eight, and maximum contexts from
128 through 1,024, the Stable ABI dispatcher selects a two-stage split-K path.
Stage one partitions the logical KV sequence into at most 16 contiguous tiles
and writes one F32 state per packed query head:

```text
partial_state = (local_max, local_exp_sum, unnormalized_output[Dv])
```

Stage two finds the global maximum and rescales every partial numerator and
denominator by `exp(local_max - global_max)` before summation. This is the
stable LSE merge; averaging independently normalized tile outputs would be
incorrect. Dispatch keeps at most 64 tokens in a tile while targeting at least
128 partial CTAs. It packs two GQA heads by default and uses four only for the
measured higher-work regime `batch * max_context >= 4096`.

The allocation-free one-pass launch ABI remains the default. Split-K sizing and
execution accept caller-owned F32 workspace; safe Rust exposes the same
explicit ownership contract. The Stable ABI dispatcher obtains a temporary
tensor from the framework allocator, so CUDA Graph capture owns a stable
allocation while graph replay performs only the two captured kernel launches.
Shapes outside the qualified split-K predicate use the one-pass kernel.

The C ABI carries independent K/V block strides and accepts contiguous int32
block tables and sequence lengths, matching vLLM's live metadata. Their active
values are trusted engine metadata: the host-supplied maximum length and active
block IDs must be valid. Safe Rust owns contiguous buffers and therefore passes
their canonical block strides; PyTorch may pass native interleaved views. No
implicit device-to-host validation or fallback is introduced on the launch
path.

## Qualified vLLM Route

The vLLM 0.24/0.25 adapter is opt-in with
`LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1` or explicit
`register_vllm_paged_decode_attention()`. It replaces
`FlashAttentionImpl.forward` only for the measured envelope:

- FP16 or BF16 native KV cache;
- Hq/Hkv `32/8`, query/value head size 128, block size 16 or 32;
- one causal decoder query per active sequence, batch 1-128;
- maximum sequence length 1-32;
- no sliding window, ALiBi, soft cap, sinks, cascade/common prefix,
  quantized KV, KV sharing, multimodal prefix mask, or DCP.

FA3's AOT scheduler tensor is an execution hint and does not block Loom; the
adapter still rejects every semantic feature outside the list above. All
other calls execute the original vLLM method. This is a version-gated engine
integration, not a new global attention backend.

## Qualification Sequence

1. Rust contract and CPU oracle, including invalid metadata and GQA mapping;
2. a PyTorch reference cross-check over randomized block tables and lengths;
3. short-context one-pass CUDA and long-context split-K/LSE candidates;
4. current-stream PyTorch schema, FakeTensor, compile, and CUDA Graph gates;
5. vLLM 0.24/0.25 FlashAttention logical-layout adapter with explicit fallback;
6. H20 comparison against the engine-selected FA3/FlashInfer implementation;
7. real-model TPOT, throughput, and KV-memory evidence.

Steps 1-6 are complete for the narrow route above. Randomized PyTorch tests
cover MQA/GQA, partial final blocks, shuffled physical blocks, odd GQA tail
groups, odd head sizes, distinct value widths, F32/FP16/BF16, external
streams, FakeTensor/schema, `torch.compile`, CUDA Graph replay,
vLLM-interleaved cache strides, caller-owned split-K workspace, stable LSE
merge, and launch telemetry. On NVIDIA H20 all 46 focused paged-decode tests,
the 34-test paged-decode/vLLM gate, the safe Rust CUDA tests, and the current
225-test Python suite pass.

The native-interleaved 156-case shape sweep establishes why the route is
narrow: 82 cases beat FA3 and 74 lose. A focused 132-case Hq/Hkv `32/8`
qualification covers FP16/BF16, block 16/32, and batches 1-128. Every
context-16 case reaches at least `1.42x` and every context-32 case at least
`1.15x`; context 64 is mixed. Through the real vLLM method boundary, all 24
admitted cases win (`1.154-2.374x`, median `1.478x`, CUDA Graph), while all 12
context-64 cases fall back with a `1.001x` median graph ratio.

The odd-GQA H20 sweep is deliberately not a new engine route. Across 72
`14/2`, D64 cases, correctness passes with maximum absolute error `0.015625`.
All 36 context-16 cases beat FA3 under CUDA Graph replay; 31 of 36 context-32
cases win, with block-16 batches 24/32 forming the main regression pocket.
An experimental Qwen2.5-0.5B route reached the real engine (`0` baseline and
`408` Loom host submissions), but exact generated tokens matched in only two
of five cases and Loom batch latency was about 3-5% higher. The route was
therefore rejected and is absent from the adapter. This separates a useful
general CUDA improvement from an unqualified production integration.

The long-context split-K report covers BF16, interleaved block-16 cache,
batches 1/2/4/8, and contexts 128/256/512/1,024. All 16 CUDA Graph cases beat
Loom's prior single-CTA path: speedups range from `1.140x` to `6.223x` with a
`2.497x` median and maximum absolute FA3 error `0.00390625`. At batch one,
Loom reaches `90.0%`, `93.4%`, and `95.4%` of FA3 at 128, 256, and 512 tokens,
then falls to `66.4%` at 1,024. FP16 and block-32 cross-checks show the same
legacy wins, but FA3 remains faster overall. The vLLM route therefore stays at
context 32 and falls back to FA3 for every long-context call.

A one-layer stable-output synthetic Qwen2 gate proves actual `LLM.generate`,
FA3 metadata, native interleaved cache, compilation, and CUDA Graph path hits:
baseline processes record zero Loom submissions, Loom processes record 18,
and generated tokens match. The fixture keeps nonzero Q/K/V projections but
zeros the attention/MLP output projections and makes token zero a stable LM
head winner; it is intentionally a path gate, not pretrained-model numerical
evidence. Order-reversed end-to-end ratios are not stable enough for a
model-level claim. The rejected Qwen2.5 experiment supplies pretrained-model
evidence, but not an admissible route. Step 7 remains open for a geometry whose
token/quality gate and end-to-end serving measurement both pass. The next
attention work targets the remaining 1,024-token merge gap, broader head
geometries, and vendor fallback rather than widening the engine route from a
microbenchmark alone.
