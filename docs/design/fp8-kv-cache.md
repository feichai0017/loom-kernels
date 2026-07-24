# FP8 KV-Cache Write Contract

Loom's first KV-cache compression boundary fuses three memory-bound steps:
in-place RoPE on Q/K, paged placement, and static FP8 E4M3 quantization of K/V.
Attention, cache allocation, page tables, scale selection, and cache lifetime
remain owned by the inference engine.

## Why This Boundary

vLLM 0.24 and 0.25 map `fp8` and `fp8_e4m3` KV caches to byte storage. Their
`reshape_and_cache_flash` operator accepts one K scale and one V scale either
per layer or per KV head. FlashAttention and FlashInfer consume that compressed
cache directly with the same scales.

Loom therefore does not add a full-cache dequantization pass. Such a pass would
restore the HBM traffic that compression removes. The useful NVIDIA gap is the
existing RoPE+cache-write fusion: Loom extends that single operator to emit the
engine's native FP8 bytes instead of adding a second public cache operator.

## Contract

| Buffer | Logical layout | Dtype |
| --- | --- | --- |
| query | `[tokens, query_heads, head_size]` | F32, FP16, or BF16 |
| key | `[tokens, kv_heads, head_size]` | same as query |
| value | `[tokens, kv_heads, head_size]` | same as query |
| packed K/V cache | `[blocks, 2, block_size, kv_heads, head_size]` | source dtype or uint8 FP8 storage |
| K/V scales | `[1]` or `[kv_heads]` | contiguous F32 on the same CUDA device |
| positions and slots | `[tokens]` and `[cache_tokens]` | contiguous int64 |

The Rust contract makes the physical encoding explicit:

```text
KvCacheEncoding::Native
KvCacheEncoding::Fp8E4M3Fn(PerTensor)
KvCacheEncoding::Fp8E4M3Fn(PerHead)
```

For FP8 storage and head `h`, Loom writes:

```text
key_cache   = fp8_e4m3fn(rotated_key / key_scale[h])
value_cache = fp8_e4m3fn(value       / value_scale[h])
```

The scale index is zero for the per-tensor form. Scales are positive and
finite, and the consuming attention backend reconstructs values with the
corresponding multiplicative scale. Negative slots skip the cache write while
Q/K still receive RoPE. Non-negative slots must be unique and in range.

Packed QKV token strides and NHD/HND cache strides remain explicit. The
framework boundary takes the engine's single packed K/V allocation instead of
two mutable views into the same storage. That keeps auto-functionalization
correct on PyTorch 2.10/2.11 while the checked Rust bridge still receives
explicit borrowed K/V pointers and strides. Loom borrows every allocation and
PyTorch's current stream; it performs no copies, hidden allocation, host
synchronization, or cache ownership.

The CPU oracle validates scale values directly. The borrowed CUDA bridge
validates their dtype, device, storage length, and aliasing, but treats positive
finite device values as an engine precondition: copying scales to the host just
to inspect them would add a synchronization point to the decode path.

## One Execution Path

```text
PyTorch/vLLM rope_paged_kv_write_
  -> boxed LibTorch Stable ABI dispatcher
  -> bridge ABI v2
  -> checked borrowed Rust views
  -> CudaBackend native-or-FP8 dispatch
  -> one handwritten RoPE+quantize+write CUDA kernel
```

The PyTorch schema always carries K/V scale tensors because it mirrors vLLM's
cache-write call site. Native storage ignores their values. The Rust-native
API keeps native and FP8 cache element types distinct, so a byte cache cannot
accidentally enter the native path.

## vLLM Admission

The opt-in RoPE+KV compiler adapter admits:

- native `auto`, F32, FP16, and BF16 cache storage;
- exactly `fp8` or `fp8_e4m3` compressed storage;
- per-tensor or per-KV-head static F32 scales;
- FlashAttention or FlashInfer causal decoder layers without KV sharing.

`configure_vllm_rope_paged_kv()` keeps both `rotary_embedding` and
`quant_fp8` opaque to the compiler before enabling vLLM's official RoPE+KV
fusion pass. The latter is required for static FP8 query quantization to remain
matchable; Loom still replaces only the fused RoPE+cache-write boundary.

It deliberately rejects FP8 E5M2, FP8 per-token-head, INT8, NVFP4,
TurboQuant, MLA-specific layouts, and dynamic scale production. Unsupported
contracts retain vLLM's original RoPE and cache-write path.

## Deliberate Exclusions

- GEMM and attention kernels;
- a full-cache dequantize-on-read operator;
- dynamic per-token-head scale caches;
- runtime amax reduction or scale calibration;
- cache allocation, eviction, compaction, transfer, or scheduling;
- INT8, E5M2, NVFP4, and model-specific packed cache formats.

These become separate work only when a named engine path exposes a
memory-bound gap that cannot be handled by its selected attention backend.

## Qualification Gates

The implementation is complete through Rust contracts and CPU oracles, safe
CUDA dispatch, the checked bridge, the Stable ABI PyTorch operator, vLLM
registration, and a repository-free ABI2 wheel. Revision
`a2f37666ed31aa8781a26e150980a75f9f569171` closed the first four H20 gates:

1. exact FP8 bytes versus vLLM for FP16/BF16, per-tensor/per-head scales,
   packed QKV, padding, and NHD/HND layouts;
2. current-stream, FakeTensor, `torch.compile`, CUDA Graph, and bridge
   telemetry coverage;
3. an operator comparison against vLLM's separate RoPE plus
   `reshape_and_cache_flash`;
4. clean-install wheel tests on vLLM 0.24 and 0.25;
5. **Open:** a pretrained-model native-versus-FP8 gate reporting generated
   quality, cache bytes, admitted context or batch size, TTFT, and TPOT.

The first four gates prove implementation and integration; the raw result is
[recorded here](../results/h20-fp8-kv-cache-write-20260724.json). They show
exact vLLM E4M3 bytes, a `2x` BF16-to-FP8 physical cache-storage ratio at the
operator boundary, faster fused submissions across the measured sweep, and
exact real-engine tokens with Loom path hits. Only the fifth gate can support
a system-level KV-compression value claim, so this family remains
`in progress` rather than `supported`.
