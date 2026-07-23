# Contributing to Loom Kernels

Loom accepts changes that make a real inference boundary clearer, safer, or
faster. A kernel is not considered complete because it compiles or wins one
microbenchmark.

## Choose work

Start with the [operator catalog](docs/operator-catalog.md) and
[roadmap](docs/roadmap.md). For a new operator, open an operator proposal before
writing a large implementation. Name:

- the inference engine and call site;
- the exact tensor, layout, aliasing, and stream contract;
- the current named baseline;
- the workload where the boundary is material;
- the correctness and performance exit gates.

Dense, quantized, sparse, and grouped GEMM implementations are out of scope and
must remain engine/vendor-owned. Loom is for memory-bound operators, layout and
scheduling transitions, quantization plumbing, decode-tail work, and measured
fusions around an unchanged vendor matrix kernel.

## Development loop

The dependency-light workspace must stay healthy on a machine without CUDA:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo check --workspace --release
python3 -m compileall -q benchmarks python
```

On an NVIDIA host, build the checked Rust path and PyTorch adapter:

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo test -p loom-cuda --features cuda --release

CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo test -p loom-cuda-bridge --features cuda --release

CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  python python/build_native.py
CUDA_HOME=/usr/local/cuda \
  python python/build_torch_extension.py
python -m pytest -q python/tests
```

## Operator checklist

1. Define the Rust contract and deterministic CPU oracle.
2. Reject zero sizes, overflow, invalid layouts, invalid dtypes, and forbidden
   aliasing before launch.
3. Add a safe `loom-cuda` entrypoint over owned and borrowed memory.
4. Add the smallest framework adapter that preserves the caller's current
   stream and allocation ownership.
5. Cover scalar tails, representative production shapes, external streams,
   FakeTensor/opcheck, `torch.compile`, and CUDA Graph capture where applicable.
6. Benchmark against the exact engine or framework operation being replaced.
7. Record negative, parity, fallback, and accepted results without changing the
   baseline after seeing the numbers.

## Pull requests

Keep each pull request to one contract or one migration slice. Include:

- what engine path changes;
- which contracts still fall back;
- exact local and GPU test commands;
- correctness tolerances and any token/rank parity result;
- raw JSON evidence for a performance claim;
- documentation updates when support or compatibility changes.

Generated build outputs, virtual environments, model weights, and profiler
captures do not belong in the repository.
