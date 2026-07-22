# Loom Kernels Python adapters

This package exposes PyTorch current-stream operators and narrow vLLM 0.24
integration points for [Loom Kernels](https://github.com/feichai0017/loom-kernels).

The source wheel contains the Python adapter layer. Build the CUDA C ABI and
LibTorch dispatcher bridge from a repository checkout before GPU use:

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  python python/build_native.py

CUDA_HOME=/usr/local/cuda \
  python python/build_torch_extension.py
```

Repository checkouts discover both libraries under `build/`. A packaged or
externally managed deployment can set `LOOM_KERNELS_CUDA_LIBRARY` and
`LOOM_KERNELS_TORCH_LIBRARY` to their absolute paths. Automated binary wheels
are not published yet.

The base paged-decode API is available as
`loom_kernels.paged_decode_attention` and `paged_decode_attention_out`. It
accepts one contiguous `[B, Hq, D]` query, native NHD paged K/V caches, and
contiguous int32 block tables/sequence lengths. The first CUDA kernel supports
F32/FP16/BF16 through 1,024 tokens. H20 evidence favors it only for extremely
short contexts, so it is not registered as an automatic vLLM replacement.

The vLLM 0.24 Min-P processor replacement is opt-in with
`LOOM_KERNELS_ENABLE_MIN_P=1`. Its H20-qualified fast path requires at least 32
rows and a 65,536-token vocabulary; smaller shapes use vLLM's original path.

Install the framework dependencies needed by the consumer:

```bash
pip install -e 'python[torch]'
pip install -e 'python[vllm,test]'
```

See the full
[vLLM provider guide](https://github.com/feichai0017/loom-kernels/blob/main/docs/guides/vllm-ir-provider.md)
for supported contracts, opt-ins, and validation commands.
