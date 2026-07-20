"""Lazy entry point for Loom's optional Torch CUDA custom operator."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXTENSION_LOADED = False


def load_cuda_extension() -> None:
    """Load and register the compiled ``torch.ops.loom`` implementation."""
    global _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return
    try:
        import_module("loom_attention._cuda_ops")
    except ImportError as error:
        raise RuntimeError(
            "Loom CUDA extension is not built; run "
            "'python python/setup_cuda.py build_ext --inplace' in the target "
            "PyTorch/CUDA environment"
        ) from error
    _EXTENSION_LOADED = True


def fused_tail_attention_merge(
    query: Any,
    tail_key: Any,
    tail_value: Any,
    remote_output: Any,
    remote_lse: Any,
    *,
    scale: float,
) -> tuple[Any, Any]:
    """Compute local-tail attention and merge it with a remote O/LSE state."""
    load_cuda_extension()
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Loom CUDA operators require PyTorch") from error
    return torch.ops.loom.fused_tail_attention_merge(
        query,
        tail_key,
        tail_value,
        remote_output,
        remote_lse,
        float(scale),
    )


__all__ = ["fused_tail_attention_merge", "load_cuda_extension"]
