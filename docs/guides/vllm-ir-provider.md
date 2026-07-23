# vLLM IR Provider

Loom Kernels can replace fused residual Add+RMSNorm implementations in vLLM
0.24 and 0.25 through the vLLM IR provider registry. The integration is
inference-only, mutates both tensors in place, launches on PyTorch's current
CUDA stream, and survives vLLM compilation and CUDA Graph capture.

The same package also provides an opt-in out-of-tree replacement for vLLM's
standard `SiluAndMul` layer. It is not enabled merely by installing the plugin:
the current H20 result establishes exact compatibility and graph parity, not a
performance win.

A second opt-in replaces vLLM's fused SiLU-and-Mul plus dynamic symmetric
per-block FP8 implementations for group sizes 64 and 128. This boundary is
bitwise compatible with vLLM's fused operator and has an operator-level H20
advantage. It has also completed a pinned Qwen2.5 online-FP8 engine gate with
direct compiler-match and launch evidence; that small-model end-to-end result
is at parity rather than a demonstrated speedup.

A third opt-in uses the existing RoPE+KV compiler fusion pass in vLLM 0.24 and
0.25 with Loom's CUDA implementation for FlashAttention and FlashInfer native
or static FP8 E4M3 caches. It preserves packed-QKV token/head strides, NHD or
HND cache strides, negative slots, the shorter slot mapping used with padded
engine inputs, and vLLM's `[1]` or `[num_kv_heads]` K/V scales. E5M2, dynamic
per-token-head scales, INT8, NVFP4, and model-specific cache formats are
deliberately declined.

A fourth explicit registration replaces only the pure-greedy `logprobs=0`
sampler tail in vLLM 0.24 and 0.25. It fuses argmax, sampled-token raw logprob,
and tie-aware rank without materializing a full-vocabulary F32 logprob tensor.
Unlike the parity-only integrations above, pinned Qwen2.5-0.5B H20 runs show
an order-stable end-to-end latency and TPOT improvement for this narrow
request contract.

A fifth registration extends the same idea to general sampling without taking
over policy: vLLM still applies masks, penalties, temperature, top-k/top-p,
and RNG, while Loom computes only the chosen token's raw logprob and rank from
the preserved BF16/FP16 logits. Pinned top-k/top-p H20 runs show exact tokens
and ranks plus an order-stable end-to-end improvement.

A sixth opt-in replaces only a measured short-context slice of vLLM's
FlashAttention decode method. Loom reads vLLM's interleaved native KV cache
directly and routes every unsupported shape or semantic feature to the
original FA3 method.

A seventh explicit registration replaces vLLM's deterministic all-greedy
speculative rejection kernel. It consumes the engine's flattened ragged draft
metadata, verifies target argmax IDs, and compacts the accepted prefix plus
mismatch or bonus token. Stochastic rejection and every model, attention,
GEMM, RNG, scheduler, and KV-cache policy remain engine-owned.

The registered contract is:

```text
residual = input + residual
input = RMSNorm(residual, weight, epsilon)
```

## Compatibility

The supported package interval is `vllm>=0.24,<0.26`. The qualified native
wheel passes 192 H20 tests with each minor. The subsequent greedy speculative
revision passed an expanded 202-test suite on both official vLLM 0.24.0 and
0.25.1 packages. The current source additionally adds FP8 KV quantize-on-write;
its new H20 matrix is still open and it is not a published native wheel.
Existing model-level performance artifacts were captured on 0.24.0 and are
not automatically performance claims for 0.25.1.
See the
[compatibility matrix](../compatibility.md) and
[native-wheel gate](../results/h20-native-wheel-clean-install-20260723.json).

## Build and install

Build the matrix artifact from a clean Linux x86_64 checkout with a
CUDA-enabled PyTorch:

```bash
python3 -m venv .venv-wheel
.venv-wheel/bin/pip install \
  'setuptools>=80,<82' 'wheel>=0.45' build 'torch>=2.10,<2.12'

CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  .venv-wheel/bin/python python/build_wheel.py \
  --cuda-home /usr/local/cuda-13.1 \
  --archs 90 \
  --wheel-dir dist

python3 -m venv .venv-vllm
.venv-vllm/bin/pip install \
  'dist/loom_kernels-1.0.0a1-2cu131torch210sm90-py3-none-linux_x86_64.whl[vllm,test]' \
  'vllm>=0.24,<0.26'
```

The wheel contains the single native backend, `libloom_cuda_bridge.so`, and
the boxed LibTorch Stable ABI dispatcher, `libloom_kernels_torch.so`, targeting
PyTorch 2.10. Installed packages validate the matrix manifest and both hashes,
then load only that package-local pair. Every admitted operator passes physical
buffer spans, strides, and PyTorch's current stream through the Rust bridge
into safe borrowed dispatch. There is no Python/ctypes fallback, ATen
dispatcher twin, unchecked twin, direct C++-to-CUDA route, or external
dispatcher override.

The first qualified artifact is not published to a package index. Editable
source development remains documented in the
[Python README](../../python/README.md#source-development), but it cannot
produce a source-only wheel.

## Direct PyTorch Use

```python
from loom_kernels.torch_ops import add_rms_norm_

output, updated_residual = add_rms_norm_(
    input_tensor,
    residual,
    weight,
    1.0e-5,
)

from loom_kernels import (
    greedy_sample_logprobs,
    greedy_speculative_verify,
    paged_decode_attention_out,
    rope_paged_kv_write_,
    silu_and_mul,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_out,
    silu_and_mul_out,
    selected_token_logprobs,
)

output = silu_and_mul(gate_and_up)
silu_and_mul_out(gate_and_up, reusable_output)

fp8_output, block_scales = silu_and_mul_dynamic_fp8(
    gate_and_up_bf16,
    group_size=128,
)

token_ids, sampled_logprobs, sampled_ranks = greedy_sample_logprobs(logits)
sampled_logprobs, sampled_ranks = selected_token_logprobs(logits, token_ids_i64)
verified_ids, accepted_lengths, emitted_lengths = greedy_speculative_verify(
    flattened_draft_ids_i32,
    flattened_target_argmax_ids_i64,
    bonus_ids_i32,
    inclusive_cumulative_draft_lengths_i32,
    max_draft_tokens,
)
silu_and_mul_dynamic_fp8_out(
    gate_and_up_bf16,
    reusable_fp8_output,
    reusable_block_scales,
    group_size=128,
)

rope_paged_kv_write_(
    query,
    key,
    value,
    positions,
    cos_sin_cache,
    packed_kv_cache,
    key_scales_f32,
    value_scales_f32,
    slot_mapping,
    is_neox=True,
)

paged_decode_attention_out(
    decode_query,
    interleaved_kv_cache[:, 0],
    interleaved_kv_cache[:, 1],
    block_table_i32,
    sequence_lengths_i32,
    reusable_attention_output,
    max_sequence_length=32,
)
```

Add+RMSNorm and standalone SiLU-and-Mul tensors must be contiguous CUDA tensors
using their documented matching F32, FP16, or BF16 dtype. The dynamic-block-FP8
path accepts FP16/BF16 input, group size 64 or 128, and a width divisible by the
group. `weight` must be one-dimensional and match the final normalization
dimension. The RoPE+KV path accepts native caches or uint8 FP8 E4M3 caches,
with contiguous CUDA F32 K/V scales shaped `[1]` or `[num_kv_heads]`. Checked
public operators reject gradients and aliasing.

## vLLM Use

Installing the Python package exposes a `vllm.general_plugins` entry point.
Select Loom for only the supported IR operation:

```python
from vllm import LLM

engine = LLM(
    model="/path/to/model",
    ir_op_priority={"fused_add_rms_norm": ["loom_cuda"]},
)
```

vLLM appends its native fallback to the priority list. Loom declines tensors
outside its contiguous same-dtype contract, weighted RMSNorm calls without a
normal variance size, and unsupported devices.

To replace vLLM's standard SwiGLU layer as well, opt in before the engine
process starts:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL=1 python your_vllm_service.py
```

Python embedding code can instead call
`loom_kernels.vllm.register_vllm_silu_and_mul()` explicitly before constructing
the model. The replacement supports contiguous CUDA F32/FP16/BF16 input with
an even final dimension and preserves vLLM's output dtype and rounding.

To replace the activation-quant fusion table entries for dynamic symmetric FP8
groups 64 and 128, enable the separate opt-in before vLLM imports its model:

```bash
LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8=1 python your_vllm_service.py
```

Embedding code can call
`loom_kernels.vllm.register_vllm_silu_and_mul_dynamic_fp8()` explicitly. The
replacement uses vLLM's mutable custom-op schema, including an optional F32
scale upper bound and row-major or transposed scale storage. Registration is
intentionally version-specific to the vLLM 0.24/0.25 activation-quant compiler
pass;
unsupported versions should leave the opt-in unset.

To enable fused RoPE+paged-KV on vLLM 0.24/0.25 CUDA, configure the compilation
object before constructing the engine:

```python
from vllm import LLM
from loom_kernels.vllm import configure_vllm_rope_paged_kv

engine = LLM(
    model="/path/to/model",
    compilation_config=configure_vllm_rope_paged_kv(max_token_num=256),
)
```

The helper explicitly enables `+rotary_embedding` and `+quant_fp8`, keeps the
cache update in the compiled graph, registers Loom on the
FlashAttention/FlashInfer backend classes, and enables fusion only through 256
tokens by default. Keeping static FP8 query quant opaque is required for the
FlashAttention FP8 graph to match the official fusion pass. The threshold is
intentional: the H20 advantage is largest for decode-sized batches and narrows
as long prefill becomes compute-bound. The adapter targets vLLM's
version-specific compiler contract and native or static FP8 E4M3 cache dtype.

To enable the measured paged-decode route, opt in before vLLM constructs the
engine:

```bash
LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION=1 python your_vllm_service.py
```

Embedding code can instead call
`loom_kernels.vllm.register_vllm_paged_decode_attention()` explicitly. The
fast path requires FP16/BF16 native KV, Hq/Hkv `32/8`, head size 128, block
size 16 or 32, one causal decoder token per sequence, batch 1-128, and maximum
context 1-32. Sliding windows, ALiBi, soft caps, sinks, cascade/common prefix,
DCP, KV sharing, quantized cache, and multimodal prefix masks all execute the
original `FlashAttentionImpl.forward`. FA3 AOT scheduler metadata is allowed
because it affects only FA3's kernel scheduling, not attention semantics.

To enable the pure-greedy sampled-logprob fast path, register it before engine
construction:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_greedy_sample_logprobs

assert register_vllm_greedy_sample_logprobs() == "greedy_sample_logprobs"
engine = LLM(model="/path/to/model")
```

The adapter only intercepts requests whose sampler contract is all-greedy,
uses raw logprobs, and asks for `logprobs=0`. It also requires no penalties,
allowed-token mask, bad words, per-request logprob token IDs, thinking-budget
state, or active argmax-changing logits processor. F32/FP16/BF16 logits may
have padded rows but require unit vocabulary stride. Every unsupported case
runs the original vLLM sampler; speculative bonus-token sampling is also
declined. Registration is version-gated to vLLM 0.24/0.25. Both contiguous and
padded logits enter the same checked Rust bridge with an explicit row stride.

To replace vLLM's deterministic speculative verifier, register it before
constructing the engine:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_greedy_speculative_verify

assert (
    register_vllm_greedy_speculative_verify()
    == "greedy_speculative_verify"
)
engine = LLM(model="/path/to/model")
```

The hook intercepts only `sampling_metadata.all_greedy` with standard,
non-synthetic rejection semantics. vLLM computes target argmax and owns draft
generation, bonus-token selection, attention, GEMM, scheduler state, and every
stochastic path. Loom consumes contiguous flattened int32 draft IDs, matching
int64 target IDs, int32 bonus IDs shaped `[requests, 1]`, and inclusive int32
cumulative draft lengths. Unsupported contracts call the original vLLM
function. Registration is explicit because the current gates prove exact
operator behavior, lower verifier latency, and real draft/target engine
invocation, but not end-to-end speculative decode acceleration.

To preserve vLLM's full sampling policy but avoid its full-vocabulary raw
log-softmax output, use the general registration instead:

```python
from vllm import LLM
from loom_kernels.vllm import register_vllm_selected_token_logprobs

assert register_vllm_selected_token_logprobs() == "selected_token_logprobs"
engine = LLM(model="/path/to/model")
```

This registration includes the narrower greedy registration, so pure-greedy
batches keep the fused argmax path. Non-greedy and mixed batches qualify when
vLLM 0.24/0.25 requests raw `logprobs=0` from BF16/FP16 logits and does not
request specific-token or top-k logprob lists. vLLM executes its original F32
processors and sampler first; Loom then scans the preserved raw logits for the
selected int64 IDs. F32 logits and processed-logprob modes conservatively fall
back because vLLM may mutate their storage in place.

The provider can only replace a graph-visible activation-quant boundary. On
the tested H20 stack, vLLM's automatic `fp8_per_block` selection uses a
FlashInfer/DeepGEMM linear kernel that accepts BF16 and performs activation
quantization inside GEMM. That path contains no separate node for Loom to
replace. The engine A/B therefore fixes `linear_backend="cutlass"`, enables
the `quant_fp8` custom op, and enables `fuse_act_quant` for both providers. The
GEMM is identical on both sides; only the fused activation-quant operator
changes.

To verify selection without starting an engine:

```bash
.venv-vllm/bin/python - <<'PY'
from loom_kernels.vllm import provider_metadata, register_vllm_ir

register_vllm_ir()
print(provider_metadata())
PY
```

## Validation

```bash
.venv-vllm/bin/pytest -q python/tests

.venv-vllm/bin/python benchmarks/vllm_greedy_speculative_verify.py \
  --batches 1,8,32,128,256 --draft-lengths 1,4,8 \
  --warmup 30 --iterations 300 --samples 9 \
  --output /tmp/greedy-speculative-verify.json

.venv-vllm/bin/python benchmarks/vllm_engine_speculative_decode.py \
  --tested-revision "$(git rev-parse HEAD)" \
  --target-model /path/to/Qwen2.5-1.5B-Instruct \
  --target-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --draft-model /path/to/Qwen2.5-0.5B-Instruct \
  --draft-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --spec-tokens 4 --prompt-mode natural \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --boundary-profile-repeats 3 \
  --gpu-memory-utilization 0.6 --provider-order native-first \
  --result-json /tmp/speculative-native-first.json

# Repeat with --provider-order loom-first and a distinct result path.

.venv-vllm/bin/python benchmarks/vllm_ir_add_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_engine_add_rms_norm.py \
  --model build/synthetic-qwen2-h4096-l4 \
  --provider loom_cuda --batch-size 8 \
  --input-len 128 --output-len 128

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul.py \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_silu_and_mul_dynamic_fp8.py \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 2000 --samples 15 \
  --provider-order forward

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-fp8-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-fp8-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_rope_paged_kv.py \
  --dtype bf16 --layout NHD --tokens 1,8,32,128,256,512 \
  --warmup 100 --iterations 2000 --repeats 5

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --warmup 2 --repeats 5 \
  --provider-order baseline-first \
  --result-json /tmp/qwen25-rope-kv-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --warmup 2 --repeats 5 \
  --provider-order loom-first \
  --result-json /tmp/qwen25-rope-kv-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_paged_decode_shape_sweep.py \
  --batches 1,8,32 --contexts 16,32,64,128 \
  --cache-storage vllm-interleaved \
  --output /tmp/paged-decode-shape-sweep.json

.venv-vllm/bin/python benchmarks/vllm_paged_decode_backend.py \
  --batches 1,8,32 --contexts 16,32,64 \
  --dtypes bf16,f16 --block-sizes 16,32 \
  --output /tmp/paged-decode-backend.json

.venv-vllm/bin/python benchmarks/create_synthetic_qwen2.py \
  --output build/synthetic-qwen2-h4096-l1-stable --layers 1 \
  --hidden-size 4096 --intermediate-size 4096 \
  --attention-heads 32 --kv-heads 8 --max-position-embeddings 64 \
  --stable-token-zero

.venv-vllm/bin/python benchmarks/vllm_engine_paged_decode.py \
  --model build/synthetic-qwen2-h4096-l1-stable \
  --case 1x16x16 --case 8x16x16 --case 32x16x16 \
  --provider-order baseline-first \
  --result-json /tmp/paged-decode-engine.json

.venv-vllm/bin/python benchmarks/vllm_rope_paged_kv.py \
  --dtype bf16 --cache-dtype fp8 --scale-mode per-tensor \
  --layouts NHD,HND --tokens 1,2,4,8,16,32,64,128 \
  --output /tmp/rope-paged-kv-fp8.json

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --kv-cache-dtype fp8 --provider-order baseline-first \
  --result-json /tmp/qwen25-rope-paged-kv-fp8.json

.venv-vllm/bin/python benchmarks/vllm_greedy_sample_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 5 --provider-order baseline-first \
  --result-json /tmp/qwen25-greedy-logprobs-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 5 --provider-order loom-first \
  --result-json /tmp/qwen25-greedy-logprobs-loom-first.json

.venv-vllm/bin/python benchmarks/vllm_selected_token_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/qwen25-selected-logprobs-baseline-first.json

.venv-vllm/bin/python benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order loom-first \
  --result-json /tmp/qwen25-selected-logprobs-loom-first.json
```

The microbenchmark compares `loom_cuda` and `vllm_c` through the same vLLM IR
eager dispatcher and CUDA Graph replay. It warms the GPU before each provider
to avoid clock-state order bias. The engine benchmark uses the normal Qwen2
model runner, compilation, scheduler, KV cache, and graph capture, but the
repository's generated checkpoint contains random weights and is not a
pretrained model.

On H20 with vLLM 0.24.0, Loom and `vllm_c` were bitwise identical for the
tested BF16 shapes. Both differ slightly from vLLM IR's FP32-add formal
reference because the CUDA path materializes the BF16 residual before its RMS
reduction. See the
[H20 integration report](../results/h20-vllm-ir-add-rms-norm-20260721.json).

For SiLU-and-Mul, F32/FP16/BF16 and odd-width fallback tests were bitwise equal
to vLLM. Order-reversed CUDA Graph medians were within 0.1%, while eager
dispatch was sensitive to run order. The synthetic Qwen2 engine completed
compilation, graph capture, and generation with the opt-in replacement. See
the [H20 SiLU-and-Mul report](../results/h20-silu-and-mul-20260721.json).

For SiLU-and-Mul+block-FP8, Loom was bitwise identical to vLLM's fused
operator for both supported input dtypes and group sizes. On BF16 `8x11008`,
order-reversed runs showed `1.216-1.231x` eager speedup ratios
(`17.7-18.8%` lower latency) and `1.037-1.082x` CUDA Graph ratios
(`3.6-7.5%` lower latency). The composed vLLM SiLU-then-quantize path is slower
but rounds an intermediate BF16 tensor, so it is not the exact correctness
baseline. See the
[H20 fused activation-quant report](../results/h20-silu-and-mul-dynamic-fp8-20260721.json).

The real-model gate pins Qwen2.5-0.5B-Instruct, online-quantizes it with
vLLM's `fp8_per_block` mode, and runs each provider in a fresh process with an
isolated compile cache. Both provider orders matched every generated token,
each compiler graph recorded two activation-quant matches, and the launch
probe recorded 1584 Loom submissions only in the Loom process. Across the
three cases, batch-latency ratios ranged from `0.9991x` to `1.0043x`, so this
is integration and correctness evidence rather than a model-level performance
claim. See the
[H20 Qwen2.5 engine report](../results/h20-vllm-qwen25-05b-fp8-engine-20260722.json).

For RoPE+paged-KV, FP16/BF16 results were bitwise equal to vLLM's separate
rotary and cache-write operators across packed-QKV, padding, partial rotary,
NHD/HND, and both pairing styles; F32 remained within the qualified tolerance.
BF16 Qwen2.5-style dispatcher ratios were roughly `2.30-2.40x` for 1-512
tokens, then narrowed to `1.088x` at 8192 tokens. Two provider orders on the
real Qwen2.5-0.5B engine matched every generated token and recorded 552 Loom
host submissions only in Loom processes. End-to-end batch-latency ratios
ranged from `0.9957x` to `1.0180x`, so the correct conclusion is engine
integration plus operator-level benefit, not model-level acceleration. See the
[operator report](../results/h20-rope-paged-kv-20260722.json),
[large-token sweep](../results/h20-rope-paged-kv-large-20260722.json), and
[engine report](../results/h20-vllm-qwen25-rope-paged-kv-engine-20260722.json).
Those artifacts cover native caches only. The static FP8 E4M3 source path
reuses the same operator and vLLM compiler pass, but has no accepted H20 result
yet; its exact-byte, clean-wheel, quality, cache-capacity, and TPOT gates remain
open. See the [FP8 KV-cache contract](../design/fp8-kv-cache.md).

For paged decode, the native-interleaved
[156-case shape sweep](../results/h20-paged-decode-interleaved-shape-sweep-20260722.json)
has 82 FA3 wins and 74 losses. The focused
[132-case batch sweep](../results/h20-paged-decode-qwen-batch-sweep-20260722.json)
qualifies both low-precision dtypes and block sizes across batches 1-128:
every context-16/32 case wins. The
[backend report](../results/h20-vllm-paged-decode-backend-20260722.json)
confirms all 24 routed cases at `1.154-2.374x` CUDA Graph speedup and graph-parity
fallback for 12 context-64 cases. Order-reversed stable-output synthetic-Qwen
[baseline-first](../results/h20-vllm-paged-decode-engine-baseline-first-20260722.json)
and [Loom-first](../results/h20-vllm-paged-decode-engine-loom-first-20260722.json)
runs match tokens and record zero/18 Loom submissions. Their latency ratios
are process-order sensitive. The stable fixture preserves nonzero Q/K/V work
but zeros the downstream projection and forces a robust token-zero winner, so
the result proves integration rather than pretrained-model numerics or speedup.
The later [odd-GQA sweep](../results/h20-paged-decode-odd-gqa-20260722.json)
passes 72 Qwen2.5-style `14/2`, D64 cases, but the
[pretrained-model experiment](../results/h20-vllm-qwen25-paged-decode-rejected-20260722.json)
matched every generated token in only two of five cases and was 3-5% slower.
That profile is intentionally absent from the adapter; the
[non-regression gate](../results/h20-vllm-paged-decode-tail-gqa-backend-20260722.json)
keeps the existing `32/8`, D128 route at 24/24 wins.

For greedy sampled logprobs, Loom matched vLLM's token IDs and tie-aware ranks
exactly over a 151,936-token BF16 vocabulary; maximum sampled-logprob error was
`9.54e-7`. The fused operator measured `3.16-4.35x` faster for 1-128 rows. Two
isolated provider orders on Qwen2.5-0.5B matched every generated token and
rank, recorded 1120 Loom submissions only in each Loom process, and measured
`1.129-1.250x` batch-latency plus `1.147-1.257x` TPOT ratios. See the
[operator report](../results/h20-greedy-sample-logprobs-20260722.json),
[baseline-first engine report](../results/h20-vllm-greedy-logprobs-baseline-first-20260722.json),
and [Loom-first engine report](../results/h20-vllm-greedy-logprobs-loom-first-20260722.json).

For general selected-token logprobs, caller-selected IDs covered ranks from
288 through 151,842 over the same 151,936-token BF16 vocabulary. Ranks were
exact, maximum logprob error was `9.54e-7`, and the operator measured
`2.77-3.78x` faster for 1-128 rows. Baseline-first and Loom-first Qwen2.5
top-k/top-p runs preserved every vLLM-selected token and rank, recorded 1440
Loom submissions only in each Loom process, and measured `1.044-1.125x`
batch-latency plus `1.054-1.130x` TPOT ratios. See the
[operator report](../results/h20-selected-token-logprobs-20260722.json),
[baseline-first engine report](../results/h20-vllm-selected-logprobs-baseline-first-20260722.json),
and [Loom-first engine report](../results/h20-vllm-selected-logprobs-loom-first-20260722.json).

For deterministic greedy speculative verification, Loom matches vLLM's
flattened ragged rejection output bit-for-bit, including zero-draft requests,
first mismatches, full acceptance, and bonus-token emission. All 15 H20 cases
across batches 1-256 and draft lengths 1/4/8 measured `1.101-1.128x` against
vLLM 0.24's exact Triton verifier through equivalent allocating Python calls.
Both vLLM minors pass the expanded 202-test source suite. See the
[H20 verifier report](../results/h20-greedy-speculative-verify-20260723.json).

The process-isolated real-engine gate uses vLLM 0.24 with a pinned
Qwen2.5-1.5B target and Qwen2.5-0.5B draft. Native and Loom speculative paths
match every generated token and acceptance counter in both provider orders;
each Loom run records `714/714` measured verifier launches. Post-timing CUDA
events show a `1.026-1.133x` verifier-boundary ratio, but the verifier accounts
for only `0.048-0.200%` of batch latency. Native/Loom end-to-end ratios cross
parity under order reversal, while speculative decode is `3.18-4.97x` slower
than target-only for the measured cases. See the
[native-first](../results/h20-vllm-qwen25-speculative-native-first-20260723.json)
and [Loom-first](../results/h20-vllm-qwen25-speculative-loom-first-20260723.json)
reports.

The target-only baseline uses different target execution shapes. At batch 32,
two target-only trajectories diverge from both mutually exact speculative
providers after token 51 or 53; the raw reports retain those differences.
Provider correctness is therefore exact native-vLLM versus Loom speculative
equivalence. vLLM's dummy sampler warm-up uses a non-greedy metadata fixture,
so lifetime fallback telemetry is informational; measured Loom launches must
equal measured rejection calls.

## Opt-In Min-P Filtering

```bash
LOOM_KERNELS_ENABLE_MIN_P=1 python your_service.py
```

vLLM 0.24 promotes sampling logits to F32 before its processors. Loom replaces
the allocating `softmax + amax + compare + masked_fill` sequence with the
equivalent in-place threshold `logit < row_max + log(min_p)`. The adapter uses
Loom only for at least 32 rows and a vocabulary of at least 65,536 tokens. It
calls the original vLLM processor for smaller shapes because H20 evidence shows
that the current one-block-per-row kernel is slower there.

The [Qwen-vocabulary report](../results/h20-min-p-filter-20260722.json) records
exact masks, retained logits, temporary memory, all raw samples, and the row
crossover. The
[65,536-token boundary report](../results/h20-min-p-filter-vocab65536-20260722.json)
independently validates the lower vocabulary gate. The override is opt-in and
has not yet completed a real-model end-to-end gate.

The compatible arithmetic and schema follow vLLM 0.24's
[fused CUDA implementation](https://github.com/vllm-project/vllm/blob/v0.24.0/csrc/libtorch_stable/quantization/fused_kernels/fused_silu_mul_block_quant.cu)
and its documented
[fusion mechanism](https://docs.vllm.ai/en/v0.23.0/design/fusions/).

The provider API follows vLLM's
[IR design](https://docs.vllm.ai/en/v0.22.1/design/vllm_ir/) and the mutable
dispatcher follows PyTorch's
[LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and [custom-operator contract](https://docs.pytorch.org/docs/stable/library.html).

## Current Limits

- Linux and CUDA only;
- the first native artifact is qualified only for Linux x86_64, CUDA 13.1,
  SM90, and H20 Python 3.11; it is not published;
- inference-only mutation, with no autograd implementation;
- one selectable IR provider (`fused_add_rms_norm`), one opt-in out-of-tree
  layer replacement (`SiluAndMul`), and one vLLM-version-specific
  activation-quant fusion-table replacement, plus a vLLM 0.24/0.25-specific
  RoPE+native/static-FP8-KV compiler-pass adapter, greedy/general selected-token
  sampled-logprob sampler overrides, a shape-gated Min-P override, and a
  measured-shape FlashAttention paged-decode override;
- the activation-quant provider requires a graph-visible quantization boundary;
  it does not intercept vLLM's fused BF16-input FlashInfer/DeepGEMM path;
- the isolated operator is faster on H20 and real-model invocation is proven,
  but no model-level speedup has been established for either FP8 activation
  fusion or RoPE+paged-KV;
- vLLM-owned penalties, masks, top-k/top-p, and stochastic sampling can feed
  the selected-token path, but Loom does not accelerate those stages yet;
  Min-P is the first separately qualified processor, while top-k logprob lists
  and non-raw modes still fall back;
- paged decode is limited to the exact H20-qualified 32/8-head, D128,
  context-at-most-32 envelope; pretrained-model and serving-scale evidence plus
  competitive 128-1,024-token kernels remain open.
