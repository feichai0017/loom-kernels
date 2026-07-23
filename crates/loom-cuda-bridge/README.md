# loom-cuda-bridge

Checked C entrypoints into Loom Kernels' safe Rust CUDA runtime.

This crate is the narrow boundary used by framework adapters that already own
CUDA tensors and a current stream. The adapter passes raw pointers, element
counts, and the stream handle once; Rust constructs non-owning typed views,
validates the operator contract, and launches asynchronously without copying,
allocating device memory, synchronizing, or taking ownership.

The bridge exposes one dtype-generic entrypoint for each semantic operator:
RMSNorm, Add+RMSNorm, RMSNorm+dynamic FP8, SiLU-and-Mul,
SiLU-and-Mul+dynamic FP8, greedy sampled logprobs, selected-token logprobs,
Min-P filtering, RoPE+paged-KV write, and paged decode attention. Explicit
layout descriptors cover padded logits, strided Q/K/V tensors, native paged
caches, and base or split-K decode. The split-K decision remains inside safe
Rust dispatch.

Framework code has no alternate CUDA route. The PyTorch shim calls only this
ABI; the bridge calls safe `loom-cuda`; and only `loom-cuda` reaches the
internal launch ABI in `loom-cuda-sys`.

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo build -p loom-cuda-bridge --features cuda --release
```

Raw entrypoints are inherently unsafe for their C/C++ caller: pointers must
refer to correctly typed allocations on the active CUDA context, remain alive
until stream work completes, and obey the documented aliasing contract.

Licensed under MIT.
