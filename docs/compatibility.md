# Compatibility and distribution

Loom separates source compatibility, GPU validation, engine compatibility, and
binary portability. A green row below applies only to the stated boundary.

## Qualified matrix

| Component | Qualified version | Boundary | Evidence |
| --- | --- | --- | --- |
| Rust | current stable toolchain | format, Clippy, tests, release checks, source crate archives | GitHub CI |
| CUDA | 13.1, `sm_90` | `loom-cuda`, `loom-cuda-sys`, and `loom-cuda-bridge` build and execute | NVIDIA H20 gate |
| Python | 3.11.2 | clean native-wheel install; the `py3-none` artifact does not use the CPython C API | [ABI2 native-wheel gate](results/h20-native-wheel-clean-install-abi2-20260724.json) |
| PyTorch | 2.10.0+cu128 | the exact wheel built on 2.11 loads without recompilation; 138 applicable Loom tests pass | [ABI2 native-wheel gate](results/h20-native-wheel-clean-install-abi2-20260724.json) |
| PyTorch | 2.11.0+cu130 | clean wheel install, current stream, `torch.compile`, FakeTensor/opcheck, and CUDA Graph replay | [ABI2 native-wheel gate](results/h20-native-wheel-clean-install-abi2-20260724.json) |
| vLLM | 0.24.0 | clean wheel install and all 225 registered-adapter/operator tests | [ABI2 native-wheel gate](results/h20-native-wheel-clean-install-abi2-20260724.json) |
| vLLM | 0.25.1 | clean install from the official wheel and all 225 registered-adapter/operator tests | [ABI2 native-wheel gate](results/h20-native-wheel-clean-install-abi2-20260724.json) |

The current exact wheel includes greedy speculative verification and static
FP8 E4M3 KV quantize-on-write through bridge ABI 2. Both vLLM minors pass the
same expanded 225-test suite. The separate
[FP8 KV evidence](results/h20-fp8-kv-cache-write-20260724.json) closes the
exact-byte, framework, operator, clean-wheel, and real-engine invocation gates;
pretrained native-versus-FP8 quality, admitted capacity, TTFT, and TPOT remain
an open system-value gate.

The process-isolated Qwen2.5 draft/target engine benchmark is qualified on
vLLM 0.24 only; its [native-first](results/h20-vllm-qwen25-speculative-native-first-20260723.json)
and [Loom-first](results/h20-vllm-qwen25-speculative-loom-first-20260723.json)
reports prove invocation and provider equivalence, not acceleration. The
0.25.1 gate proves that the current adapters and CUDA paths execute against the
official vLLM wheel. It does not retroactively transfer any 0.24 model-level
latency result to 0.25.1. A new engine benchmark is required before making a
0.25.1 performance claim.

Python package metadata therefore requires or accepts:

```text
torch>=2.10,<2.12
vllm>=0.24,<0.26
```

Versions outside that interval are not supported. Loom's optional registration
functions also check the installed vLLM series before patching engine classes
or compiler tables.

## Current native-wheel boundary

The published Rust crates remain self-contained source distributions. The
current qualified Python artifact is
`loom_kernels-1.0.0a1-2cu131torch210sm90-py3-none-linux_x86_64.whl`.
It is built only through `python/build_wheel.py` from a clean Git revision and
contains exactly:

- `libloom_cuda_bridge.so` — Rust contracts, borrowed safe dispatch, and the
  internal handwritten CUDA launch layer;
- `libloom_kernels_torch.so` — boxed LibTorch Stable ABI dispatcher.

`native.json` records their hashes, Git revision, CUDA 13.1 toolkit, SM90
target, bridge ABI, and PyTorch runtime range. Installed wheels load only this
package-local pair. `PYTHONPATH`, `LD_LIBRARY_PATH`, and an external dispatcher
override were absent from every clean gate.

The earlier `1cu131torch210sm90` ABI-1 artifact remains historical evidence.
The ABI-specific build tag prevents incompatible bridge signatures from
colliding; ABI 2 is now the only current artifact boundary.

The wheel is Python-ABI-independent (`py3-none`) because neither native library
uses the CPython C API. Its platform tag remains the conservative
`linux_x86_64`; auditwheel 6.7 found a `manylinux_2_34_x86_64` symbol floor,
but Loom does not claim an earlier manylinux baseline. H20 runtime validation
currently covers Python 3.11 only.

The artifact is not published to PyPI or a GitHub release. This is a qualified
build/install boundary, not a claim that `pip install loom-kernels` can fetch a
native wheel from a public index.

## Current Stable ABI boundary

PyTorch documents a [LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
and stable registration APIs for PyTorch 2.10 and newer. Loom's single
production dispatcher now uses that boundary:

- all schemas use boxed Stable ABI registration;
- tensor metadata, allocations, pointers, device guards, and the current CUDA
  stream use stable headers or AOTI C shims;
- all eleven semantic operators continue into `loom-cuda-bridge`; the dispatcher
  has no ATen/c10 C++ symbol dependency and consumes no raw CUDA launch symbol;
- the public Python APIs and vLLM admission predicates reject tensors requiring
  gradients. No autograd kernel is advertised;
- the temporary Add+RMSNorm probe and the previous ATen dispatcher were deleted
  after the production migration passed.

`python/build_wheel.py` now automates the first CUDA/PyTorch/Python matrix row,
audits its ELF boundary, and rejects accidental source-only wheels. One exact
artifact passed fresh-venv gates on PyTorch 2.10/2.11 and vLLM 0.24/0.25.
Publishing that artifact remains a separate, explicit release action.

## What must be revalidated

| Change | Minimum gate |
| --- | --- |
| Rust contract or aliasing rule | CPU oracle, invalid-input tests, safe CUDA wrapper |
| CUDA kernel | edge shapes, representative shapes, external stream, CUDA Graph |
| PyTorch dispatcher | opcheck/FakeTensor, mutation schema, `torch.compile`, full GPU suite |
| vLLM minor release | official wheel import, all adapter tests, explicit fallback tests |
| Performance claim | named baseline, warmed samples, correctness first, provider-order reversal for engines |
| Binary wheel claim | clean install on every published Python/PyTorch/CUDA matrix entry |
