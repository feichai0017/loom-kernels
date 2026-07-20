# Loom

Loom is a disaggregated core-attention runtime for externally managed KV.
vLLM or SGLang keeps model execution; Mooncake, LMCache, or another `KvPool`
owns sealed KV objects; Loom moves Q to a suitable attention worker,
executes near the historical KV, and returns the attention output. K_new/V_new
travel only to the worker that owns the mutable tail.

## Boundary

| Component | Owns |
| --- | --- |
| vLLM / SGLang | batching, weights, QKV projection, RoPE, FFN, sampling |
| Loom | generation-pinned KV views, attention plans, tensor transport, execution and exact merge |
| external `KvPool` | sealed KV allocation, placement, replication, eviction, durability |
| Holt catalog | persistent `prefix -> PoolObjectRef` metadata, revalidated after recovery |

The control service is a slow path. Per-layer execution uses node-local state and
never synchronously queries the global controller or Holt.

## Workspace

| Path | Responsibility |
| --- | --- |
| `crates/loom-attention` | runtime, pool, catalog, planner, transport, and attention contracts |
| `crates/loom-cuda-sys` | optional raw C ABI bindings and `nvcc` build plumbing |
| `crates/loom-cuda` | checked Rust CUDA executor, CPU oracle, and fused-tail benchmark |
| `cuda` | handwritten FP16/BF16 local-tail attention and exact-merge kernels |
| `loom-control` | slow-path catalog/scheduler binary from the `loom-attention` package |
| `loom-worker` | attention-worker control binary from the `loom-attention` package |
| `python/src/loom_attention` | installable vLLM adapters, metadata contracts, attention-state executors, and lazy CUDA-op API |
| `python/csrc` | optional PyTorch dispatcher shim for the shared CUDA kernels |
| `python/tests` | unit and adapter contract tests |
| `python/tests/integration` | CUDA smoke tests and two-GPU benchmarks excluded from the wheel |

## Current Status

The Rust lifecycle contracts, Holt catalog, planner, real-model vLLM
observer/delegate, node-local physical-block binding registry, output-plus-LSE
merge, NCCL Route-Q harness, and generation-pinned FlashInfer paged-KV executor
are implemented and covered by tests. An optional Rust/CUDA local-tail plus
exact-merge operator is also implemented and has an isolated H20 correctness
and microbenchmark report. Modal L4 reports for the vLLM adapter and
physical-block bridge, plus the phase-instrumented 4K-32K two-GPU sweep, are
recorded under `docs/results`.
Binding external `PoolObjectRef` values to those vLLM slots, the Mooncake
load/save adapter, cross-node transport, Nsight-level attribution, and broader
hardware evaluation are not implemented yet. See the
[implementation status](docs/status.md) for exact boundaries.

## Build

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
PYTHONPATH=python/src:python/tests \
  python3 -m unittest discover -s python/tests -v
```

On a Linux CUDA host, build and benchmark the optional handwritten fused path:

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo run --release -p loom-cuda --features cuda \
  --bin loom-fused-tail-bench -- --rows 4 --dtype fp16
```

See the [Rust/CUDA fused-tail guide](docs/guides/rust-cuda-fused-tail.md) for
the PyTorch build, Route-Q strategy A/B commands, and measured H20 boundary.

On a Linux CUDA host with vLLM installed, run the M1 acceptance gate:

```bash
PYTHONPATH=python/src:python/tests \
  python3 -m integration.vllm_smoke compare \
  --report build/vllm-smoke/report.json
```

Inspect the M2 payload asymmetry without CUDA, then run it on a Linux host with
two NVIDIA GPUs:

```bash
PYTHONPATH=python/src:python/tests \
  python3 -m integration.two_gpu_smoke plan --prefix-tokens 4096
PYTHONPATH=python/src:python/tests \
  python3 -m integration.two_gpu_smoke run \
  --prefix-tokens 4096 \
  --attention-backend flashinfer-paged \
  --route-strategy sequential \
  --page-size 16 \
  --report build/two-gpu-smoke/report.json
```

Run the control endpoints:

```bash
cargo run -p loom-attention --bin loom-control -- --bind 127.0.0.1:8080
cargo run -p loom-attention --bin loom-worker -- --bind 127.0.0.1:8090
```

Start with the [documentation index](docs/README.md).

## License

MIT
