# Loom Kernels

[Website](https://feichai0017.github.io/loom-kernels/) ·
[Operator catalog](https://feichai0017.github.io/loom-kernels/docs/operators/) ·
[H20 evidence](https://feichai0017.github.io/loom-kernels/benchmarks/)

Loom Kernels is a Rust-first high-performance operator library for LLM
inference. It provides backend-independent contracts and CPU references,
checked accelerator dispatch, handwritten CUDA kernels, and reproducible
correctness/performance gates.

The project is intentionally not an inference engine or tensor framework. It
targets the small set of decode-critical operators where fusion, layout-aware
execution, or lower dispatch overhead can create measurable engine value.

## Current Status

- `loom-kernels`: dtype, tensor, capability, normalization, quantization,
  split-half SiLU-and-Mul, RoPE/KV, logits processing, sampling, and base paged
  decode-attention contracts plus CPU oracles;
- `loom-cuda`: safe CUDA stream/buffer/event ownership and checked dispatch;
- `loom-cuda-sys`: dependency-light raw C ABI;
- handwritten F32 plus vectorized FP16/BF16 RMSNorm, fused Add+RMSNorm,
  dynamic per-token FP8 E4M3FN output quantization, SiLU-and-Mul, and fused
  SiLU-and-Mul+dynamic per-block FP8 validated on NVIDIA H20;
- PyTorch current-stream custom operators and a vLLM 0.24 Add+RMSNorm IR
  provider validated through compilation, CUDA Graph capture, and a real Qwen2
  engine generate loop;
- an opt-in vLLM `SiluAndMul` layer replacement is bitwise compatible and
  engine-valid, but graph latency is at parity, so no speedup is claimed;
- an opt-in vLLM 0.24 activation-quant fusion replacement covers dynamic
  symmetric FP8 groups 64/128 and is bitwise compatible with vLLM's fused
  kernel; pinned Qwen2.5-0.5B online-FP8 runs now prove compiler path hits,
  CUDA Graph execution, and exact generation parity, while end-to-end latency
  remains at parity;
- NeoX/interleaved RoPE and fused RoPE+paged-KV write now span Rust contracts,
  CPU oracles, safe Rust/C ABI, handwritten F32/FP16/BF16 CUDA, PyTorch, and an
  opt-in vLLM 0.24 FlashAttention/FlashInfer integration. Packed-QKV source
  strides plus NHD/HND cache strides are preserved without materialization;
- fused greedy argmax plus sampled-token raw logprob spans Rust, safe CUDA/C
  ABI, handwritten F32/FP16/BF16 CUDA, PyTorch, and a narrow opt-in vLLM 0.24
  sampler fast path. On H20 it is `3.16-4.35x` faster than vLLM's exact
  decode-tail sequence for 1-128 Qwen-sized rows; order-reversed real
  Qwen2.5-0.5B runs show `1.129-1.250x` batch-latency ratios with exact tokens
  and sampled-token ranks;
- general selected-token raw logprob keeps vLLM responsible for masks,
  penalties, temperature, top-k/top-p, RNG, and token selection, then replaces
  only its full-vocabulary F32 `log_softmax + gather/rank` tail. On H20 the
  operator is `2.77-3.78x` faster for 1-128 Qwen-sized rows; order-reversed
  Qwen2.5 top-k/top-p runs show exact tokens/ranks and `1.044-1.125x`
  batch-latency ratios;
- in-place Min-P filtering uses the equivalent
  `logit < row_max + log(min_p)` threshold, avoiding full probability and mask
  tensors. The opt-in vLLM route is performance-gated: H20 measurements use
  Loom only for at least 32 rows and a 65,536-token vocabulary, with smaller
  decode batches falling back to vLLM;
- paged MQA/GQA decode attention spans Rust contract/oracle, safe Rust/C ABI,
  handwritten F32/FP16/BF16 CUDA, and a current-stream PyTorch out API. H20
  tests pass against randomized PyTorch and vLLM FA3 references. The first
  kernel beats FA3 consistently at context 16, but loses at 64+ tokens, so no
  automatic vLLM replacement is enabled yet;

## Workspace

| Path | Responsibility |
| --- | --- |
| `crates/loom-kernels` | public contracts, capability queries, and CPU references |
| `crates/loom-cuda` | safe Rust CUDA backend and benchmarks |
| `crates/loom-cuda-sys` | raw CUDA C ABI and build plumbing |
| `cuda` | handwritten CUDA kernels |
| `python` | PyTorch dispatcher bridge and vLLM IR provider |
| `benchmarks` | named external baselines |
| `docs/results` | hardware-qualified validation artifacts |
| `website` | Astro documentation and project site |

## Operator Priorities

| Priority | Operator family | Why it matters |
| --- | --- | --- |
| P0 | RMSNorm, Add+RMSNorm, Norm+Quant | memory-bound decode kernels with useful fusion boundaries |
| P0 | RoPE+KV write, KV append/layout/quantization | removes extra HBM passes around KV-cache updates |
| P0 | SwiGLU/GELU fused epilogues | combines activation, multiply, bias, and quantization |
| P0 | sampling and selected-token logprob | reduces decode-tail launches and temporary tensors |
| P1 | paged decode attention | first short-context CUDA path is qualified; optimize 32-128 tokens before a shape-gated engine route |
| P1 | MoE top-k, permutation, grouped dispatch | routing and movement often dominate small expert batches |
| P1 | quantized GEMM epilogues | wrap vendor GEMM and own the fusion, not another basic GEMM |
| P2 | communication-aware fusions | RMSNorm/all-reduce and TP epilogues after single-GPU evidence |

See the [complete operator catalog](docs/operator-catalog.md),
[operator library design](docs/design/operator-library.md), and
[roadmap](docs/roadmap.md).

## Build And Test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release

cd website
npm ci
npm run build
```

On a CUDA host:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench rms_norm -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7
```

The inference-engine-style fused path uses the explicit double in-place
contract `residual = input + residual`, followed by
`input = RMSNorm(residual, weight)`:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench add_rms_norm -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 50 --iterations 1000 --samples 9
```

RMSNorm plus dynamic per-token FP8 uses caller-owned output buffers and emits
one F32 dequantization scale per row:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench rms_norm_dynamic_fp8 -- \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15
```

Split-half SiLU-and-Mul accepts `[... , 2 * width]` and produces
`silu(gate) * up` with shape `[... , width]`:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench silu_and_mul -- \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15
```

SiLU-and-Mul plus dynamic per-block FP8 removes the low-precision activation
intermediate and emits one F32 scale per 64- or 128-element output group:

```bash
CUDA_HOME=/usr/local/cuda-13.1 LOOM_CUDA_ARCHS=90 \
  cargo bench -p loom-cuda --features cuda \
  --bench silu_and_mul_dynamic_fp8 -- \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 1000 --samples 9
```

These programs live under `crates/loom-cuda/benches`, not `src/bin`: they are
validation tools rather than installable product executables. `harness = false`
keeps their JSON CLI behavior while preserving the correct Cargo target
boundary.

The benchmark checks the GPU result against the CPU oracle before reporting
CUDA-event latency. The named PyTorch baselines are:

```bash
python3 benchmarks/pytorch_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7

python3 benchmarks/pytorch_add_rms_norm.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 20 --iterations 100 --samples 7

PYTHONPATH=python/src python3 benchmarks/vllm_rms_norm_dynamic_fp8.py \
  --dtype bf16 --rows 8 --hidden-size 4096 \
  --warmup 100 --iterations 2000 --samples 15

PYTHONPATH=python/src python3 benchmarks/vllm_silu_and_mul.py \
  --dtype bf16 --rows 8 --width 11008 \
  --warmup 100 --iterations 2000 --samples 15

PYTHONPATH=python/src python3 benchmarks/vllm_silu_and_mul_dynamic_fp8.py \
  --dtype bf16 --rows 8 --width 11008 --group-size 128 \
  --warmup 100 --iterations 2000 --samples 15

.venv-vllm/bin/python benchmarks/vllm_engine_fp8_ab.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x128x128 --case 8x128x128 --case 32x128x64 \
  --provider-order baseline-first --result-json /tmp/loom-fp8-ab.json

PYTHONPATH=python/src .venv-vllm/bin/python benchmarks/vllm_rope_paged_kv.py \
  --dtype bf16 --layout NHD --tokens 1,8,32,128,256,512 \
  --warmup 100 --iterations 2000 --repeats 5

PYTHONPATH=python/src .venv-vllm/bin/python \
  benchmarks/vllm_paged_decode_attention.py \
  --dtype bf16 --batches 1,8,32 --contexts 16,32,64,128,256,512 \
  --warmup 30 --iterations 200 --samples 7

.venv-vllm/bin/python benchmarks/vllm_engine_rope_paged_kv.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --warmup 2 --repeats 5 \
  --provider-order baseline-first --result-json /tmp/loom-rope-kv-ab.json

PYTHONPATH=python/src .venv-vllm/bin/python \
  benchmarks/vllm_greedy_sample_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

PYTHONPATH=python/src .venv-vllm/bin/python \
  benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 5 --provider-order baseline-first \
  --result-json /tmp/loom-greedy-logprobs-ab.json

PYTHONPATH=python/src .venv-vllm/bin/python \
  benchmarks/vllm_selected_token_logprobs.py \
  --rows 1,2,4,8,16,32,64,128 --vocab-size 151936 --dtype bf16 \
  --warmup 100 --iterations 1000 --repeats 7

PYTHONPATH=python/src .venv-vllm/bin/python \
  benchmarks/vllm_engine_greedy_logprobs.py \
  --model /path/to/Qwen2.5-0.5B-Instruct --sampling-mode top-k-top-p \
  --case 1x32x64 --case 8x32x64 --case 32x32x32 \
  --warmup 2 --repeats 7 --provider-order baseline-first \
  --result-json /tmp/loom-selected-logprobs-ab.json
```

The H20 reports cover
[F32 bring-up](docs/results/h20-rms-norm-f32-smoke-20260721.json) and
[FP16/BF16 vectorization](docs/results/h20-rms-norm-low-precision-20260721.json),
the [fused Add+RMSNorm gate](docs/results/h20-add-rms-norm-20260721.json), and
the [vLLM IR integration](docs/results/h20-vllm-ir-add-rms-norm-20260721.json),
plus the
[RMSNorm+dynamic-FP8 gate](docs/results/h20-rms-norm-dynamic-fp8-20260721.json)
and the
[SiLU-and-Mul compatibility gate](docs/results/h20-silu-and-mul-20260721.json),
plus the
[fused SiLU-and-Mul+block-FP8 gate](docs/results/h20-silu-and-mul-dynamic-fp8-20260721.json).
The
[Qwen2.5 FP8 engine gate](docs/results/h20-vllm-qwen25-05b-fp8-engine-20260722.json)
records the pinned real checkpoint, compiler matches, direct Loom launch
evidence, exact generated tokens, and order-reversed end-to-end measurements.
The fused operator is faster than `vllm_c` in the qualified microbenchmark;
the real engine run proves integration but does not show a measurable
end-to-end speedup. Standalone SiLU-and-Mul is graph-parity coverage. Its
activation-plus-FP8 boundary has an order-stable operator-level advantage and
a real-model correctness gate, but still needs a workload with measurable
model-level benefit.

The RoPE+paged-KV
[operator report](docs/results/h20-rope-paged-kv-20260722.json) records roughly
`2.30-2.40x` lower dispatcher latency than vLLM's separate RoPE and cache-write
ops for 1-512 tokens; the advantage narrows to `1.09x` at 8192 tokens in the
[large-token report](docs/results/h20-rope-paged-kv-large-20260722.json). The
[real Qwen2.5 engine gate](docs/results/h20-vllm-qwen25-rope-paged-kv-engine-20260722.json)
proves exact tokens and direct Loom launches. Order reversal moves end-to-end
ratios across parity, so no model-level speedup is claimed yet.

The greedy decode-tail
[operator report](docs/results/h20-greedy-sample-logprobs-20260722.json)
compares one Loom launch with vLLM's full `log_softmax`, argmax, selected-value
gather, and rank path over a 151,936-token BF16 vocabulary. Token IDs and
tie-aware ranks are exact, maximum logprob error is `9.54e-7`, and the H20
ratio is `3.16-4.35x` for 1-128 rows. The order-reversed
[baseline-first](docs/results/h20-vllm-greedy-logprobs-baseline-first-20260722.json)
and [Loom-first](docs/results/h20-vllm-greedy-logprobs-loom-first-20260722.json)
Qwen2.5 engine gates both pass. Across batches 1, 8, and 32, batch-latency
ratios are `1.129-1.250x` and TPOT ratios are `1.147-1.257x`, with 1120 Loom
path hits only in each fused process. This claim is deliberately limited to
pure greedy requests asking for only the sampled token's raw logprob.

The general selected-token
[operator report](docs/results/h20-selected-token-logprobs-20260722.json)
uses caller-selected IDs spanning the vocabulary and compares against vLLM's
exact full-logprob path. Ranks are exact, maximum logprob error is `9.54e-7`,
and the H20 ratio is `2.77-3.78x` for 1-128 rows. In order-reversed Qwen2.5
top-k/top-p engine gates, vLLM still owns sampling and RNG; every token and
rank matches, 1440 Loom launches occur only in Loom processes, batch-latency
ratios span `1.044-1.125x`, and TPOT ratios span `1.054-1.130x`. See the
[baseline-first](docs/results/h20-vllm-selected-logprobs-baseline-first-20260722.json)
and [Loom-first](docs/results/h20-vllm-selected-logprobs-loom-first-20260722.json)
reports. This does not claim Loom accelerates top-k/top-p selection itself.

The Min-P
[operator report](docs/results/h20-min-p-filter-20260722.json) compares the
exact vLLM 0.24 F32 processor over a 151,936-token vocabulary. Loom removes
`0.76-97.24 MB` of per-call probability/mask temporaries. It is slower for 1
and 8 rows, `1.10x` faster at 32 rows, and `1.89x` faster at 128 rows, which is
why the engine adapter has an explicit measured-shape fallback rather than a
blanket replacement. A separate
[65,536-token boundary sweep](docs/results/h20-min-p-filter-vocab65536-20260722.json)
measures `1.35x` at 32 rows and `2.35x` at 128 rows, validating the lower
vocabulary gate.

The paged-decode
[operator report](docs/results/h20-paged-decode-attention-20260722.json)
compares the first handwritten kernel directly with vLLM 0.24 FA3 over batch
1/8/32 and context 16-512. CUDA Graph ratios at context 16 are
`1.43x/1.44x/2.04x`; only batch 32 remains ahead at context 32 (`1.11x`), and
all 64+ cases lose. This is evidence for a narrow short-context kernel and the
next optimization target, not evidence for replacing the engine backend.

For the Python build and engine configuration, see the
[vLLM IR provider guide](docs/guides/vllm-ir-provider.md).

## License

MIT
