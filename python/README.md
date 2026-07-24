# Loom Kernels · Python

Native current-stream PyTorch operators and narrow, opt-in vLLM 0.24/0.25
integration for [Loom Kernels](https://github.com/feichai0017/loom-kernels).

[Project README](../README.md) · [Integration guide](../docs/guides/vllm-ir-provider.md) · [Operator catalog](../docs/operator-catalog.md)

> [!IMPORTANT]
> The current bridge-ABI-2 native wheel is H20-qualified but is not published
> to a package index yet. A source-only wheel is intentionally unsupported:
> `pip wheel ./python` fails unless `build_wheel.py` has staged both native
> libraries and their manifest.

## Qualified artifact

The current matrix row is:

| Axis | Qualified value |
| --- | --- |
| Artifact | `py3-none-linux_x86_64` |
| CUDA build | toolkit 13.1, `sm_90` |
| PyTorch runtime | `>=2.10,<2.12` through a 2.10 Stable ABI target |
| Python runtime tested on H20 | 3.11 |
| vLLM extra | `>=0.24,<0.26` |
| Native payload | `libloom_cuda_bridge.so`, `libloom_kernels_torch.so` |

The qualified artifact's build tag encodes bridge ABI 2:
`2cu131torch210sm90`. The exact H20 artifact, binary audit, and three
repository-free clean-install gates are recorded in the
[native-wheel evidence](../docs/results/h20-native-wheel-clean-install-abi2-20260724.json).
The same wheel passes 225 tests with each supported vLLM minor and 138
applicable tests in the vLLM-free PyTorch 2.10 environment. It includes static
FP8 E4M3 KV quantize-on-write and is bound to source revision
`a2f37666ed31aa8781a26e150980a75f9f569171`.

The older `1cu131torch210sm90` ABI-1 wheel remains historical evidence only.
`build_wheel.py` uses the ABI-specific tag so incompatible bridge signatures
cannot overwrite or masquerade as one another. No native Python wheel has been
published.

## Install a built wheel

The wheel has a hard PyTorch dependency because its dispatcher is not useful
without PyTorch. vLLM and tests remain explicit extras:

```bash
python3 -m venv .venv-loom
.venv-loom/bin/pip install \
  'dist/loom_kernels-1.0.0a1-2cu131torch210sm90-py3-none-linux_x86_64.whl[test]'

# Add the supported vLLM integration when needed.
.venv-loom/bin/pip install \
  'dist/loom_kernels-1.0.0a1-2cu131torch210sm90-py3-none-linux_x86_64.whl[vllm,test]' \
  'vllm>=0.24,<0.26'
```

No repository checkout, `PYTHONPATH`, `LD_LIBRARY_PATH`, or external library
override is used at runtime. The installed package reads
`loom_kernels/lib/native.json`, validates the PyTorch range and bridge ABI,
verifies both library hashes, and loads only its packaged dispatcher.

```python
import loom_kernels

print(loom_kernels.native_build_info())
```

## Build the matrix wheel

Use a clean Linux x86_64 checkout with Cargo, CUDA, ELF inspection tools, and a
CUDA-enabled PyTorch build:

```bash
python3 -m venv .venv-wheel
.venv-wheel/bin/pip install \
  'setuptools>=80,<82' 'wheel>=0.45' build 'torch>=2.10,<2.12'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-wheel/bin/python python/build_wheel.py \
  --cuda-home /usr/local/cuda-13.1 \
  --archs 90 \
  --wheel-dir dist
```

`build_wheel.py` is the only binary-wheel entrypoint. It builds the Rust CUDA
bridge, builds the boxed LibTorch Stable ABI dispatcher, rejects ATen/c10 C++
and raw CUDA-launch dependencies, verifies `$ORIGIN` loading, writes the
revision/toolkit/SM/runtime manifest, and checks the final archive contains
exactly the two Loom `.so` files.

## Source development

Editable source work remains available without creating a distributable
source wheel:

```bash
python3 -m venv .venv-dev
.venv-dev/bin/pip install -e 'python[test]'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-dev/bin/python python/build_native.py
CUDA_HOME=/usr/local/cuda-13.1 \
  .venv-dev/bin/python python/build_torch_extension.py
```

Source checkouts discover the paired libraries only under repository
`build/`. Installed wheels discover them only under `loom_kernels/lib/`.
Every operator, including padded logits and strided paged-cache views, enters
checked borrowed Rust dispatch. There is no ctypes, ATen dispatcher twin, or
direct raw-CUDA framework path.

## Direct PyTorch use

```python
import torch

from loom_kernels import (
    greedy_sample_logprobs,
    greedy_speculative_verify,
    min_p_filter_,
    rope_paged_kv_write_,
    selected_token_logprobs,
    silu_and_mul_dynamic_fp8,
)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)

token_ids, logprobs, ranks = greedy_sample_logprobs(logits)
logprobs, ranks = selected_token_logprobs(logits, sampled_ids_i64)
verified_ids, accepted_lengths, emitted_lengths = greedy_speculative_verify(
    flattened_draft_ids_i32,
    flattened_target_argmax_ids_i64,
    bonus_ids_i32,
    inclusive_cumulative_draft_lengths_i32,
    max_draft_tokens,
)
min_p_filter_(sampling_logits_f32, min_p_f32)

# Native caches ignore the scale values. FP8 uint8 caches use either one
# calibrated F32 scale or one scale per KV head.
cache_scales = torch.ones(1, device=query.device, dtype=torch.float32)
rope_paged_kv_write_(
    query,
    key,
    value,
    positions_i64,
    cos_sin_cache,
    packed_kv_cache,
    cache_scales,
    cache_scales,
    slot_mapping_i64,
    is_neox=True,
)
```

All CUDA calls use PyTorch's current stream. Out variants accept caller-owned
buffers for capture-safe reuse. Public APIs are inference-only and reject
tensors that require gradients.

## Exported operator families

| Family | Python entry points |
| --- | --- |
| Normalization | `rms_norm`, `rms_norm_out`, `add_rms_norm_`, `rms_norm_dynamic_fp8`, `rms_norm_dynamic_fp8_out` |
| Activation | `silu_and_mul`, `silu_and_mul_out`, `silu_and_mul_dynamic_fp8`, `silu_and_mul_dynamic_fp8_out` |
| Position and KV | `rope_paged_kv_write_` for native or static FP8 E4M3 paged caches |
| Decode tail | `greedy_sample_logprobs`, `selected_token_logprobs`, `min_p_filter_` |
| Speculative decode | `greedy_speculative_verify` |
| Attention | `paged_decode_attention`, `paged_decode_attention_out` |

The base paged-decode API accepts one contiguous `[B, Hq, D]` query,
dense-inner NHD paged K/V views, and contiguous int32 block tables and sequence
lengths. It directly accepts K/V views from vLLM's
`[blocks, 2, block, Hkv, D]` storage.

`rope_paged_kv_write_` accepts F32/FP16/BF16 sources and one packed
`[blocks, 2, block, Hkv, D]` cache allocation with either the matching source
dtype or `torch.uint8` FP8 E4M3 storage. The single mutable cache tensor is the
real vLLM allocation and remains functionalization-safe across PyTorch
2.10/2.11; separate K/V mutable-view arguments are not supported. K/V scales
are contiguous CUDA F32 tensors with one element or one element per KV head.
Dynamic per-token-head scales, E5M2, INT8, and NVFP4 are not silently coerced
into this contract.

## vLLM opt-ins

| Route | Enable |
| --- | --- |
| Add+RMSNorm IR provider | `ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]}` |
| Standalone SiLU-and-Mul | `LOOM_KERNELS_ENABLE_SILU_AND_MUL=1` |
| SiLU-and-Mul→block FP8 | `LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1` |
| RoPE+paged-KV compiler pass | `configure_vllm_rope_paged_kv(...)` |
| Short paged decode | `LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1` |
| Greedy sampled logprob | `register_vllm_greedy_sample_logprobs()` |
| Greedy speculative verify | `register_vllm_greedy_speculative_verify()` |
| Selected-token logprob | `register_vllm_selected_token_logprobs()` |
| Min-P processor | `LOOM_KERNELS_ENABLE_MIN_P=1` |

Every route checks its exact dtype, shape, layout, and semantic contract. An
unsupported request runs the original vLLM path instead of being copied,
cast, or reshaped into eligibility.

The [compatibility matrix](../docs/compatibility.md) records the qualified
PyTorch/vLLM versions and binary distribution boundary. Build details and
validation commands live in the
[vLLM provider guide](../docs/guides/vllm-ir-provider.md).
