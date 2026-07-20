# Rust/CUDA Fused Tail Attention

## What Is Implemented

Loom now has an optional handwritten CUDA path for the small active-tail part
of Route-Q. The remote worker still returns its prefix attention state
`(O_remote, LSE_remote)`. On the engine GPU, one CUDA kernel consumes Q, local
tail K/V, and that remote state and produces the exact combined state.

The fused kernel uses this identity directly:

```text
local_logit_j = scale * dot(Q, K_tail_j)
LSE = logaddexp(LSE_remote, logsumexp_j(local_logit_j))
O = exp(LSE_remote - LSE) * O_remote
    + sum_j(exp(local_logit_j - LSE) * V_tail_j)
```

It therefore avoids materializing the intermediate local-tail output and LSE
and removes the second state-merge kernel launch.

## Code Boundaries

| Path | Role |
| --- | --- |
| `cuda/include/loom_cuda.h` | dependency-light C ABI |
| `cuda/src/attention_kernels.cu` | FP16/BF16 tail, merge, and fused kernels |
| `crates/loom-cuda-sys` | raw Rust FFI plus CUDA build/link plumbing |
| `crates/loom-cuda` | generation-pinned tensor validation, safe contract, CPU oracle, and benchmark |
| `python/csrc/loom_cuda_ops.cpp` | PyTorch dispatcher registration on the caller's current CUDA stream |
| `python/src/loom_attention/cuda_ops.py` | lazy Python API |

CUDA remains opt-in. Normal macOS and CPU CI do not need a CUDA toolkit and
continue to build the default `loom-attention` package only.

## Build The Rust CUDA Path

On a Linux CUDA host with `nvcc` and Cargo in `PATH`:

```bash
CUDA_HOME=/usr/local/cuda-13.1 \
LOOM_CUDA_ARCHS=90 \
cargo build --release \
  -p loom-cuda \
  --features cuda \
  --bin loom-fused-tail-bench
```

`LOOM_CUDA_ARCHS` is a comma-separated list of SM numbers. Its default is
`80,89,90`; setting `90` keeps an H20-only build smaller.

Run the isolated benchmark:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/compat \
target/release/loom-fused-tail-bench \
  --rows 4 \
  --query-heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --tail-tokens 16 \
  --dtype fp16 \
  --warmup 100 \
  --iterations 1000 \
  --samples 20
```

The baseline launches a local-tail state kernel followed by a two-state merge.
The candidate launches one fused kernel. Inputs and outputs are allocated before
timing; CUDA events measure only the repeated kernel region.

## Build The PyTorch Operator

Build against the exact PyTorch and CUDA installation used by the engine:

```bash
CUDA_HOME=/usr/local/cuda-13.1 \
TORCH_CUDA_ARCH_LIST=9.0 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/compat \
python3 python/setup_cuda.py build_ext --inplace
```

Then run the one-GPU correctness gate:

```bash
PYTHONPATH=python/src:python/tests \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:/usr/local/cuda-13.1/lib64:/usr/local/cuda-13.1/compat \
python3 -m integration.fused_tail_smoke \
  --rows 4 \
  --prefix-tokens 512 \
  --tail-tokens 16 \
  --report build/h20-fused-tail-smoke.json
```

The NVSHMEM directory above is required by the PyTorch build currently installed
on `forge-gas1`; omit or adjust it on hosts whose PyTorch package resolves all
of its libraries without that path.

The gate compares the custom operator with reference attention over concatenated
prefix and tail KV for both FP16 and BF16. It also exercises the three Route-Q
strategies with prepopulated remote state:

| Strategy | Engine-side order |
| --- | --- |
| `sequential` | receive remote state, compute local tail, merge |
| `overlap` | launch local tail on a side stream while waiting for remote state, then merge |
| `fused` | receive remote state, run one handwritten tail-plus-merge kernel |

The overlap path establishes explicit stream dependencies and allocator lifetime
records. It is a separate optimization from fusion: overlap preserves the local
state so it can run concurrently with the remote worker, while fusion delays
local work until the remote state arrives but removes an intermediate tensor and
kernel launch.

Select a strategy in the two-GPU gate with:

```bash
PYTHONPATH=python/src:python/tests \
CUDA_VISIBLE_DEVICES=0,1 \
python3 -m integration.two_gpu_smoke run \
  --attention-backend flashinfer-paged \
  --route-strategy overlap \
  --report build/two-gpu-smoke/overlap.json
```

Use `sequential`, `overlap`, and `fused` under the same workload for a valid
A/B/C comparison. The `fused` option requires the PyTorch CUDA extension above.

## H20 Result

On the single NVIDIA H20 exposed by `forge-gas1`, CUDA 13.1 and Rust 1.97.1,
the isolated kernel benchmark measured:

| Rows | Dtype | Two kernels | Fused | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | FP16 | 11.271 us | 10.041 us | 1.123x |
| 1 | BF16 | 11.069 us | 9.935 us | 1.114x |
| 4 | FP16 | 11.369 us | 10.054 us | 1.131x |
| 4 | BF16 | 11.629 us | 10.234 us | 1.136x |
| 16 | FP16 | 13.744 us | 11.670 us | 1.178x |
| 16 | BF16 | 13.752 us | 11.926 us | 1.153x |

The PyTorch full-attention gate passed. At rows=4, the fused output's maximum
absolute error was `7.63e-6` for FP16 and `6.10e-5` for BF16; maximum LSE error
was `4.77e-7` for both.

The complete machine-readable result is
[h20-fused-tail-2026-07-20.json](../results/h20-fused-tail-2026-07-20.json).

## Current Limits

- FP16 and BF16 only;
- `head_dim <= 256` and `tail_tokens <= 64`;
- GQA requires `query_heads % kv_heads == 0`;
- all decode rows currently share one contiguous tail K/V segment; distinct
  per-sequence paged tails are not represented by this ABI;
- FP32 accumulation, one 128-thread block per decode-row/query-head pair;
- no masks, dropout, training/backward, MLA, or CUDA Graph capture yet;
- no production vLLM dispatch or external-pool integration yet.

The H20 measurement is an isolated one-GPU kernel result. That host cannot
measure real NCCL overlap because it exposes only one GPU. A two-GPU run must
still validate `overlap` and `fused` end to end, followed by Nsight attribution,
before either becomes the default Route-Q strategy.
