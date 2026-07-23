# Code Layout

Loom is organized by operator domain at every semantic layer. A contributor
should be able to trace one operator vertically from its public contract to its
engine adapter without searching through unrelated operators.

## Dependency Direction

```text
engine integration
    -> Stable ABI dispatcher
    -> checked Rust C bridge
    -> safe Rust CUDA backend
    -> internal CUDA launch ABI
    -> handwritten CUDA

public Rust contracts and CPU references define every layer above
```

Dependencies only point downward. CUDA code does not know about PyTorch or
vLLM, the safe backend does not know about framework tensors, and engine
adapters do not duplicate Rust contract validation.

## Top-Level Responsibilities

| Path | Owns | Must not own |
| --- | --- | --- |
| `crates/loom-kernels` | Public specs, dtypes, capabilities, errors, and CPU oracles | CUDA resources or framework policy |
| `crates/loom-cuda` | Borrowed/owned device memory, streams, layouts, and safe launches | Raw framework objects or engine fallback |
| `crates/loom-cuda-bridge` | Checked C ABI, pointer spans, aliasing, panic containment, and telemetry | Tensor allocation or CUDA kernel logic |
| `crates/loom-cuda-sys` | Internal launch declarations, build plumbing, and CUDA implementations | Public semantic contracts |
| `python/csrc` | Stable ABI schemas, tensor/stream extraction, and boxed PyTorch wrappers | Direct CUDA launches or semantic fallbacks |
| `python/src/loom_kernels/vllm` | Version/shape gates and narrowly scoped engine registration | Generic PyTorch operator definitions |
| `benchmarks` | Named baselines and reproducible measurements | Product APIs |
| `docs/results` | Accepted machine-readable hardware evidence | Unqualified performance claims |

## Domain Alignment

The same domain vocabulary is used across Rust, the bridge, PyTorch, tests,
and vLLM, while a filename suffix identifies each private Rust layer:

- `<domain>.rs` is the public contract and CPU oracle;
- `<domain>_dispatch.rs` is safe CUDA validation and launch dispatch;
- `<domain>_bridge.rs` is the checked C entrypoint over borrowed storage;
- CUDA files use the concrete algorithm or fusion name.

This preserves vertical alignment without leaving several editor tabs,
compiler diagnostics, or review comments named only `norm.rs`. Every private
layer file also starts with a module comment that states its responsibility.

| Domain | Contract and oracle | Safe CUDA dispatch | Checked Rust bridge | Raw CUDA | PyTorch dispatcher | vLLM adapter |
| --- | --- | --- | --- | --- | --- | --- |
| normalization | `norm.rs` | `norm_dispatch.rs` | `cuda/norm_bridge.rs` | `rms_norm.cu` · `add_rms_norm.cu` · `rms_norm_quant.cu` | `norm.cpp` | IR registration in `vllm/__init__.py` |
| activation and output quantization | `activation.rs` | `activation_dispatch.rs` | `cuda/activation_bridge.rs` | `silu_and_mul.cu` · `silu_and_mul_quant.cu` | `activation.cpp` | `vllm/activation.py` |
| logits processing | `logits.rs` | `logits_dispatch.rs` | `cuda/logits_bridge.rs` | `min_p.cu` | `logits.cpp` | `vllm/logits.py` |
| sampling and log probabilities | `sampling.rs` | `sampling_dispatch.rs` | `cuda/sampling_bridge.rs` | `greedy_sample.cu` | `sampling.cpp` | `vllm/sampling.py` |
| speculative decoding | `speculative.rs` | `speculative_dispatch.rs` | `cuda/speculative_bridge.rs` | `greedy_speculative_verify.cu` | `speculative.cpp` | `vllm/speculative.py` |
| RoPE and KV write | `rope_kv.rs` | `rope_kv_dispatch.rs` | `cuda/rope_kv_bridge.rs` | `rope_paged_kv.cu` | `rope_kv.cpp` | `vllm/rope_kv.py` |
| decode attention | `attention.rs` | `attention_dispatch.rs` | `cuda/attention_bridge.rs` | `paged_decode_attention.cu` | `attention.cpp` | `vllm/attention.py` |

Cross-domain infrastructure has explicit names:

- `contract.rs`, `element.rs`, and `quantization.rs` define shared public
  invariants;
- `backend.rs` defines the public backend capability interface;
- `cuda_backend.rs`, `runtime.rs`, and `layout.rs` own concrete safe CUDA
  infrastructure;
- `cuda/mod.rs` owns shared bridge dtype dispatch, borrowed-region validation,
  status mapping, and launch telemetry;
- `python/csrc/common.h` owns only Stable ABI types and shared tensor/stream
  extraction;
- `python/csrc/torch_ops.cpp` owns schemas only;
- `python/src/loom_kernels/vllm/_runtime.py` owns version and environment
  policy, while `vllm/__init__.py` is the public integration facade.

Tests mirror the public domain under `crates/loom-kernels/src/tests`. The
`*_tests.rs` suffix prevents test modules from colliding with production module
names under strict Clippy checks.

## CUDA File Granularity

Rust and adapter files follow semantic domains. CUDA files follow cohesive
algorithms and fusion boundaries instead of an arbitrary line limit. For
example, base decode, split-K partial reduction, and LSE merge remain together
in `paged_decode_attention.cu` because they share layouts, scheduling choices,
and numerical invariants.

A CUDA file should split only when a component has an independent contract,
build target, or tuning lifecycle. File length alone is not a reason to
separate tightly coupled kernels.

## Rules for New Work

1. Add or extend the public spec and CPU oracle in `loom-kernels`.
2. Add the matching safe backend method in
   `loom-cuda/src/<domain>_dispatch.rs`.
3. Add one checked bridge entrypoint in
   `loom-cuda-bridge/src/cuda/<domain>_bridge.rs`.
4. Add the internal launch declaration and the cohesive CUDA implementation.
5. If PyTorch consumes it, add the schema once in `torch_ops.cpp` and the
   wrapper plus dispatch registration in `<domain>.cpp`.
6. If an engine consumes it, add policy only to that engine's domain module.
7. Add domain tests and hardware evidence at the level of the claim.

Do not add `utils.rs`, a second execution path, ctypes fallbacks, unchecked
bridge twins, direct C++ CUDA calls, legacy aliases, or compatibility wrappers.
A shared module is justified only by an invariant used by multiple domains.

## When to Split a File

Split a file when it contains multiple independent reasons to change, such as
engine policy mixed with tensor extraction or unrelated operator state in one
module. Keep it together when the pieces form one algorithm, share numerical
invariants, and must be qualified as a unit.

This rule keeps the vertical operator trace obvious without turning the
repository into hundreds of one-function files.
