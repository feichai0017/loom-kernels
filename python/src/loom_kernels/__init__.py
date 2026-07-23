"""Framework adapters for Loom Kernels."""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0a1"


def add_rms_norm_(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and execute the PyTorch Add+RMSNorm adapter."""
    from .torch_ops import add_rms_norm_ as implementation

    return implementation(*args, **kwargs)


def rms_norm(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute inference RMSNorm."""
    from .torch_ops import rms_norm as implementation

    return implementation(*args, **kwargs)


def rms_norm_out(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute RMSNorm into caller-owned output."""
    from .torch_ops import rms_norm_out as implementation

    return implementation(*args, **kwargs)


def rms_norm_dynamic_fp8(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute fused RMSNorm and dynamic per-token FP8 quantization."""
    from .torch_ops import rms_norm_dynamic_fp8 as implementation

    return implementation(*args, **kwargs)


def rms_norm_dynamic_fp8_out(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute the caller-allocated RMSNorm plus FP8 out variant."""
    from .torch_ops import rms_norm_dynamic_fp8_out as implementation

    return implementation(*args, **kwargs)


def rope_paged_kv_write_(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute fused in-place RoPE plus paged K/V cache write."""
    from .torch_ops import rope_paged_kv_write_ as implementation

    return implementation(*args, **kwargs)


def greedy_sample_logprobs(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute fused greedy selection and sampled-token logprob."""
    from .torch_ops import greedy_sample_logprobs as implementation

    return implementation(*args, **kwargs)


def selected_token_logprobs(*args: Any, **kwargs: Any) -> Any:
    """Lazily normalize and rank one caller-selected token per logits row."""
    from .torch_ops import selected_token_logprobs as implementation

    return implementation(*args, **kwargs)


def min_p_filter_(*args: Any, **kwargs: Any) -> Any:
    """Lazily apply in-place min-p filtering without materializing softmax."""
    from .torch_ops import min_p_filter_ as implementation

    return implementation(*args, **kwargs)


def paged_decode_attention(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute base paged MQA/GQA decode attention."""
    from .torch_ops import paged_decode_attention as implementation

    return implementation(*args, **kwargs)


def paged_decode_attention_out(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute paged decode attention into caller-owned output."""
    from .torch_ops import paged_decode_attention_out as implementation

    return implementation(*args, **kwargs)


def silu_and_mul(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute fused split-half SiLU-and-Mul."""
    from .torch_ops import silu_and_mul as implementation

    return implementation(*args, **kwargs)


def silu_and_mul_out(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute caller-allocated split-half SiLU-and-Mul."""
    from .torch_ops import silu_and_mul_out as implementation

    return implementation(*args, **kwargs)


def silu_and_mul_dynamic_fp8(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute fused SwiGLU and dynamic per-block FP8."""
    from .torch_ops import silu_and_mul_dynamic_fp8 as implementation

    return implementation(*args, **kwargs)


def silu_and_mul_dynamic_fp8_out(*args: Any, **kwargs: Any) -> Any:
    """Lazily execute the caller-allocated activation plus FP8 variant."""
    from .torch_ops import silu_and_mul_dynamic_fp8_out as implementation

    return implementation(*args, **kwargs)


__all__ = [
    "__version__",
    "add_rms_norm_",
    "greedy_sample_logprobs",
    "min_p_filter_",
    "paged_decode_attention",
    "paged_decode_attention_out",
    "rms_norm",
    "rms_norm_dynamic_fp8",
    "rms_norm_dynamic_fp8_out",
    "rms_norm_out",
    "rope_paged_kv_write_",
    "selected_token_logprobs",
    "silu_and_mul",
    "silu_and_mul_dynamic_fp8",
    "silu_and_mul_dynamic_fp8_out",
    "silu_and_mul_out",
]
