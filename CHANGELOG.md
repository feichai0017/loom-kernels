# Changelog

Loom Kernels follows Semantic Versioning. The Rust crates use Cargo's SemVer
spelling; Python package metadata uses the equivalent PEP 440 spelling.

## Unreleased

### Breaking

- replaced the mixed Python/ctypes, direct C++ CUDA, and partial Rust-bridge
  framework stack with one required
  `PyTorch -> C++ dispatcher -> Rust bridge -> safe Rust -> CUDA` path;
- removed `_native.py`, every `*_unchecked` dispatcher operator,
  `LOOM_KERNELS_CUDA_LIBRARY`, `libloom_kernels_cuda.so`, `adapter_backend()`,
  and the per-operator telemetry functions;
- replaced telemetry with `Operator`, `launch_count(operator)`, and
  `reset_launch_count(operator)`;
- changed Rust CUDA entrypoints for row-strided logits, paged decode, RoPE/KV,
  and activation FP8 to require explicit physical-layout objects;
- changed `PagedDecodeAttentionSpec::new` to accept
  `max_sequence_length` independently from block-table capacity;
- replaced the production ATen/c10 dispatcher with one boxed LibTorch Stable
  ABI implementation targeting PyTorch 2.10; no old dispatcher or experimental
  probe remains;
- removed `LOOM_KERNELS_TORCH_LIBRARY`: installed wheels load only their
  package-local native pair, while editable source checkouts use repository
  `build/`.

### Added

- complete bridge coverage for RMSNorm, Add+RMSNorm, RMSNorm+FP8,
  SiLU-and-Mul, SiLU-and-Mul+FP8, RoPE+paged-KV, greedy/selected logprobs,
  Min-P, and base/split-K paged decode;
- explicit Rust physical-layout contracts for padded logits, packed QKV,
  NHD/HND caches, interleaved cache storage, scale layout, and FP8 scale upper
  bounds;
- standalone PyTorch `rms_norm` and `rms_norm_out` APIs;
- vLLM 0.25 support, an explicit compatibility matrix, H20 0.24/0.25 GPU-suite
  evidence, contribution guidance, and structured issue forms;
- a two-minor H20 binary gate proving the exact same dispatcher `.so` on
  PyTorch 2.10 and 2.11, plus a CI guard against unstable ATen/c10 C++ APIs;
- a clean-revision native wheel builder that packages exactly the Rust CUDA
  bridge and Stable ABI dispatcher, emits and validates their matrix manifest
  and hashes, and rejects accidental source-only wheels;
- fresh H20 wheel-install evidence for PyTorch 2.10/2.11 and vLLM 0.24/0.25.
- deterministic greedy speculative verification and accepted/bonus-token
  compaction over vLLM-compatible flattened ragged metadata, with Rust/CUDA/
  PyTorch coverage, explicit vLLM 0.24/0.25 registration, and H20 evidence.
- a process-isolated vLLM draft/target benchmark with exact native/Loom token
  and acceptance gates, measured launch coverage, post-timing CUDA boundary
  profiling, provider-order reversal, and pinned Qwen2.5 H20 evidence.

### Fixed

- source-checkout library discovery now follows the packaged
  `crates/loom-cuda-sys/cuda` layout after removal of the legacy root `cuda`
  directory;
- constrained the wheel build backend to setuptools 80–81 to match PyTorch
  2.11's build dependency range.

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
