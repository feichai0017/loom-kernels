# Changelog

Loom Kernels follows Semantic Versioning. The Rust crates use Cargo's SemVer
spelling; Python source-adapter metadata uses the equivalent PEP 440 spelling.

## 1.0.0-alpha.1 — 2026-07-22

First public alpha of Loom Kernels as a Rust-first CUDA operator backend for
LLM inference.

GitHub tag and Release name: `v1.0.0-alpha.1`. Cargo packages use the matching
Semantic Versioning spelling `1.0.0-alpha.1`.

### Included

- backend-independent Rust contracts, capability queries, and deterministic
  CPU oracles;
- safe Rust CUDA streams, buffers, events, checked dispatch, and a raw C ABI;
- non-owning `CudaStreamRef`, `DeviceSlice`, and `DeviceSliceMut` adapters for
  zero-copy execution over framework-controlled streams and device memory;
- sealed read/write memory traits shared by every owned and borrowed safe Rust
  operator entrypoint;
- handwritten CUDA for normalization/quantization, SwiGLU, RoPE plus paged-KV
  writes, decode-tail sampling and logprobs, Min-P, and paged decode attention;
- opt-in PyTorch and vLLM 0.24 adapters with explicit shape and policy gates;
- H20 correctness, framework, engine, and named-baseline evidence kept as
  machine-readable artifacts;
- self-contained Cargo source archives and a pure Rust CUDA smoke example that
  covers both owned and borrowed runtime resources on NVIDIA H20.

### Alpha boundaries

- APIs and admitted shape envelopes may change before 1.0 stable;
- CUDA is opt-in and requires a local NVIDIA toolkit at build time;
- Python packaging is source-adapter metadata, not a portable CUDA/LibTorch
  binary wheel;
- unsupported engine contracts intentionally fall back to the native backend.
