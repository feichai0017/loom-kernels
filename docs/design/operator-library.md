# Loom Kernels Architecture

## Product Boundary

Loom Kernels is an operator backend that can be embedded into Rust-native or
existing LLM inference engines. It does not own model scheduling, weights,
tokenization, KV lifetime, or request serving.

## Layers

| Layer | Responsibility |
| --- | --- |
| `loom-kernels` | dtype, shape/layout, aliasing, capability, and reference contracts |
| `loom-cuda-sys` | stable C ABI, CUDA compilation, and packaged handwritten kernels |
| `loom-cuda` | safe owned/borrowed CUDA resources, validation, dispatch, and benchmarks |
| `loom-cuda-bridge` | panic-contained checked C entrypoints into borrowed Rust dispatch |
| engine adapters | translate engine tensors/streams without owning engine policy |

CPU references never call accelerator code. Backends report unsupported
contracts explicitly; they do not silently copy, cast, or fall back.

## Engine-Owned CUDA Resources

`loom-cuda` supports two execution modes behind the same checked operator
methods:

| Resource | Standalone Rust | Embedded engine |
| --- | --- | --- |
| stream | owned `CudaStream` | non-owning `CudaStreamRef` |
| read-only memory | owned `DeviceBuffer<T>` | borrowed `DeviceSlice<'a, T>` |
| writable memory | owned `DeviceBuffer<T>` | exclusive `DeviceSliceMut<'a, T>` |

`CudaBackend` is generic over the stream handle. All safe operator entrypoints
accept sealed read/write device-memory traits, so owned and borrowed storage use
the same dtype, length, and shape validation. Borrowed handles do not allocate,
copy, synchronize, or destroy framework resources.

Raw stream and device-pointer construction is intentionally `unsafe`: an
adapter must prove the active CUDA device/context, storage dtype, pointer
alignment, lifetime, exclusivity, and stream ordering. Once that narrow
boundary is crossed, ordinary callers cannot invent alternative trait
implementations or pass read-only storage as a mutable output. Kernel launches
remain asynchronous.

The pure-Rust H20 smoke test exercises this zero-copy path. Add+RMSNorm also
closes the first real framework gate: the C++ PyTorch dispatcher passes tensor
pointers, actual element counts, and PyTorch's current stream through
`loom-cuda-bridge`, which constructs borrowed Rust views and calls the same
safe `CudaBackend` method. The bridge validates lengths, alignment, address
overflow, and non-overlap, contains Rust panics behind a status ABI, and keeps a
thread-local detailed error. It does not allocate, copy, synchronize, free, or
destroy framework resources.

This is deliberately an operator-by-operator migration. RMSNorm+FP8, activation,
RoPE/KV, sampling, Min-P, and paged-decode framework paths still call the raw
CUDA C ABI directly.

## Add+RMSNorm Contract

The fused normalization path follows the double in-place convention used by
LLM inference engines:

1. materialize `residual = input + residual` in the storage dtype;
2. compute the RMS statistic from that materialized residual;
3. overwrite `input = RMSNorm(residual, weight, epsilon)`.

The safe Rust entrypoint requires separate exclusive borrows for `input` and
`residual`, plus a shared borrow for `weight`. Owned `DeviceBuffer` values are
not cloneable, so ordinary safe callers cannot accidentally alias these three
allocations. Raw C ABI callers must obey the documented non-overlap rule.

FP16 and BF16 fused kernels use aligned 128-bit, eight-element packs when the
row shape and all pointers permit it. An aligned two-element path handles other
even sizes, while odd or unaligned shapes select the scalar implementation.
Launches are asynchronous on the caller-selected backend stream.

Because repeated in-place calls change nonzero operands, the standalone
benchmark separates correctness from timing. Correctness uses one nonzero
launch; latency uses zero-valued input/residual buffers, which are a stable
fixed point and execute the same branch-free kernel path. Reset copies are not
included in the kernel latency.

## RMSNorm+Dynamic-FP8 Contract

The quantized normalization path consumes contiguous F32, FP16, or BF16 input
and a matching one-dimensional weight. It writes FP8 E4M3FN values with the
same logical shape plus one F32 dequantization scale per flattened row. The
scale is `max(absmax / 448, 1 / (448 * 512))`, so zero rows remain valid.

For FP16 and BF16, both the normalized intermediate and weighted value follow
the input scalar arithmetic boundaries before FP8 conversion. This detail is
part of the public compatibility contract, not an implementation accident.
The CUDA kernel uses three passes—RMS reduction, weighted absmax reduction,
and quantization—and follows the caller's stream without synchronizing.

Safe Rust and the PyTorch out variant require caller-owned output and scale
buffers. The convenience Python API allocates those buffers once per call;
engine and benchmark paths should use the out variant to reuse memory.

## SiLU-And-Mul Contract

The SwiGLU activation consumes a contiguous tensor whose final dimension is
`2 * width`. The first half is the gate and the second half is the up branch:

```text
output[..., index] = silu(input[..., index]) * input[..., width + index]
```

F32, FP16, and BF16 are supported. Low-precision compatibility includes the
storage-dtype rounding point used by vLLM: the SiLU activation is rounded to
the input dtype before multiplication, then the product is rounded into the
output dtype. The output is separately allocated, has the same prefix shape,
and has final dimension `width`; input/output overlap is forbidden.

Aligned rows use 16-byte packs (four F32 or eight FP16/BF16 elements), while
odd widths and unaligned pointers use a scalar path. Both safe Rust and
PyTorch launch asynchronously on the caller's current stream.

The vLLM out-of-tree layer replacement is explicitly opt-in because its graph
latency is currently at parity with vLLM's native CUDA operator. Compatibility
and engine integration are useful coverage, but do not justify silently
changing an installed engine. The next performance-motivated boundary is
SiLU-and-Mul fused with dynamic output quantization.

## SiLU-And-Mul+Dynamic-Block-FP8 Contract

The fused quantized path accepts contiguous FP16 or BF16 input with the same
split-half `[rows, 2 * width]` layout. It produces FP8 E4M3FN output with shape
`[rows, width]` and one F32 dequantization scale for every 64 or 128 adjacent
output elements. `width` must be divisible by the selected group size.

Unlike the standalone compatibility operator, this fusion does not materialize
or round a FP16/BF16 SiLU intermediate. Gate activation, multiplication by the
up branch, group absmax, and division by scale use F32 before the final FP8
conversion. Each scale is:

```text
max(min(absmax / 448, optional_scale_upper_bound), 1 / (448 * 512))
```

The public Rust and Python APIs emit contiguous row-major scales with logical
shape `[rows, width / group_size]`. The vLLM compatibility operator additionally
accepts the same logical shape backed by group-major strides and its optional
same-device F32 scale upper bound. Output, scales, and input storage must not
overlap; all launches use the caller's current CUDA stream.

The CUDA mapping assigns one thread block to each row/group pair and holds one
fused value per thread in a register across the absmax reduction. This removes
the temporary low-precision activation tensor and the second kernel launch of
the composed path. Because the composed path rounds that temporary tensor, it
is a useful performance comparison but not an exact semantic baseline; vLLM's
own fused per-block operator is the compatibility baseline.

## Sampling And Selected-Logprob Contracts

The decode-tail operator consumes finite rank-2 F32, FP16, or BF16 logits with
a unit vocabulary stride and an explicit, possibly padded row stride. For each
row it returns the lowest token index attaining the maximum, that token's F32
raw log-softmax value, and an `int64` sampled-token rank. The rank deliberately
matches vLLM 0.24: it counts values greater than or equal to the selected
value, so tied maximum logits produce a rank greater than one.

One CUDA block performs first-index argmax, online logsumexp, and maximum-tie
counting in the same vocabulary pass. This avoids materializing the full F32
logprob tensor and replaces vLLM's separate log-softmax, argmax, gather, and
rank work. Launches follow the caller's current stream; token IDs, logprobs,
and ranks are separately allocated outputs.

The vLLM adapter is intentionally narrower than the CUDA primitive. It only
intercepts vLLM 0.24 requests where every row is greedy, `max_num_logprobs` is
zero, raw logprobs are requested, and masks, penalties, bad words, thinking
state, and argmax-changing processors are inactive. Other requests execute
vLLM's original sampler unchanged.

The complementary `selected_token_logprobs` contract accepts one caller-owned
int64 token ID per row and returns only that token's F32 raw logprob plus its
tie-aware int64 rank. One CUDA block loads the selected raw logit, computes an
online logsumexp over the row, and counts logits greater than or equal to the
selected value. It never materializes `[rows, vocab_size]` F32 logprobs.

Its vLLM 0.24 adapter deliberately does not own sampling policy. vLLM still
converts logits to F32, applies masks/processors/penalties and temperature,
runs greedy or random top-k/top-p selection, and consumes RNG in its original
order. Loom runs afterward against the preserved BF16/FP16 raw logits only for
`raw_logprobs` requests with `max_num_logprobs == 0`; all-greedy batches retain
the narrower fused argmax path. F32 logits, top-k logprob lists, specific-token
lists, processed-logprob modes, and version-mismatched vLLM builds fall back.

## Operator Contract

Every operator contract must make these properties explicit:

- input/output dtype and accumulation dtype;
- logical shape, physical layout, strides, and alignment;
- aliasing and in-place mutation rules;
- stream and synchronization semantics;
- temporary workspace ownership and lifetime;
- supported shape range and deterministic fallback behavior.

## Admission Gates

An operator joins the supported surface only after closing six independent
gates:

1. validated contract and invalid-input tests;
2. deterministic CPU or high-precision oracle;
3. accelerator correctness over edge and representative shapes;
4. warmed repeated measurements against a named baseline;
5. invocation from a real inference-engine execution path;
6. TTFT, TPOT, throughput, memory, or goodput benefit on the motivating workload.

Kernel latency closes gate 4 only when the baseline and measurement protocol
are equivalent. It does not close engine integration or end-to-end value.

## Implementation Policy

- Handwrite memory-bound and fusion-sensitive kernels.
- Reuse cuBLAS/cuBLASLt or equivalent for base GEMM and own valuable epilogues.
- Keep tuning decisions keyed by device, dtype, layout, and shape.
- Preserve one stable Rust contract across CUDA implementations.
- Add another backend only after a real consumer and benchmark justify it.

The complete product surface is tracked in the
[LLM inference operator catalog](../operator-catalog.md); items in that catalog
still have to pass these gates one by one.
