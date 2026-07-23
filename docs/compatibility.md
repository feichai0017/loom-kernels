# Compatibility and distribution

Loom separates source compatibility, GPU validation, engine compatibility, and
binary portability. A green row below applies only to the stated boundary.

## Qualified matrix

| Component | Qualified version | Boundary | Evidence |
| --- | --- | --- | --- |
| Rust | current stable toolchain | format, Clippy, tests, release checks, source crate archives | GitHub CI |
| CUDA | 13.1, `sm_90` | `loom-cuda`, `loom-cuda-sys`, and `loom-cuda-bridge` build and execute | NVIDIA H20 gate |
| PyTorch | 2.11.0+cu130 | source-built dispatcher, current stream, `torch.compile`, FakeTensor/opcheck, CUDA Graph replay | 183-test H20 suite |
| vLLM | 0.24.0 | all registered adapters plus the existing operator and real-engine evidence | [evidence index](results/README.md) |
| vLLM | 0.25.1 | official wheel import, registered adapters, dispatcher behavior, and the complete 183-test H20 suite | [compatibility gate](results/h20-vllm-compatibility-rust-bridge-20260723.json) |

The 0.25.1 gate proves that the current adapters and CUDA paths execute against
the official vLLM wheel. It does not retroactively transfer the 0.24
model-level latency results to 0.25.1. A new engine benchmark is required before
making a 0.25.1 performance claim.

Python package metadata therefore accepts:

```text
vllm>=0.24,<0.26
```

Versions outside that interval are not supported. Loom's optional registration
functions also check the installed vLLM series before patching engine classes
or compiler tables.

## Current binary boundary

The published Rust crates are self-contained source distributions. The Python
wheel currently contains Python adapters only; users build these native
libraries against their local CUDA and PyTorch installations:

- `libloom_cuda_bridge.so` — Rust contracts, borrowed safe dispatch, and the
  internal handwritten CUDA launch layer;
- `libloom_kernels_torch.so` — PyTorch dispatcher shim.

The dispatcher currently includes standard ATen/LibTorch headers and uses
`TORCH_LIBRARY`, so it is not yet a PyTorch-version-independent binary. Do not
label a locally built library or wheel as Stable ABI compatible.

## Stable ABI plan

PyTorch documents a [LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and stable registration APIs for PyTorch 2.10 and newer. Loom will adopt it in
four independently testable steps:

1. keep tensor semantics and CUDA execution behind `loom-cuda-bridge`, so the
   framework shim remains translation-only;
2. prototype one boxed Stable ABI operator for contiguous Add+RMSNorm without
   changing the existing dispatcher;
3. prove current-device/current-stream access, mutation schemas, FakeTensor,
   `torch.compile`, and CUDA Graph behavior across two PyTorch minor releases;
4. only then build a CUDA/PyTorch matrix wheel and declare a minimum
   `TORCH_TARGET_VERSION`.

If the Stable ABI cannot express a required CUDA stream or tensor operation,
Loom will publish per-PyTorch binary wheels instead of weakening the runtime
contract. Source builds remain the supported fallback throughout the migration.

## What must be revalidated

| Change | Minimum gate |
| --- | --- |
| Rust contract or aliasing rule | CPU oracle, invalid-input tests, safe CUDA wrapper |
| CUDA kernel | edge shapes, representative shapes, external stream, CUDA Graph |
| PyTorch dispatcher | opcheck/FakeTensor, mutation schema, `torch.compile`, full GPU suite |
| vLLM minor release | official wheel import, all adapter tests, explicit fallback tests |
| Performance claim | named baseline, warmed samples, correctness first, provider-order reversal for engines |
| Binary wheel claim | clean install on every published Python/PyTorch/CUDA matrix entry |
