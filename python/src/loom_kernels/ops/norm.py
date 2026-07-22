"""Normalization predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import math

import torch

from ._common import _DTYPE_NAMES


def _dispatch():
    from .. import _torch_dispatch

    return _torch_dispatch


def supports_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Shape/dtype predicate shared with the vLLM IR provider."""
    del epsilon
    return bool(
        variance_size is None
        and weight is not None
        and input_tensor.device.type == "cuda"
        and residual.device == input_tensor.device
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and residual.dtype == input_tensor.dtype
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.shape == residual.shape
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def supports_vllm_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Minimal hot-path predicate for tensors already governed by vLLM IR."""
    del epsilon
    return bool(
        variance_size is None
        and weight is not None
        and input_tensor.dtype in _DTYPE_NAMES
        and residual.dtype == input_tensor.dtype
        and weight.dtype == input_tensor.dtype
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def supports_rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> bool:
    """Return whether Loom can fuse RMSNorm with per-token FP8 quantization."""
    return bool(
        weight is not None
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and input_tensor.device.type == "cuda"
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and weight.is_contiguous()
    )



def _validate_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    if not supports_add_rms_norm(input_tensor, residual, weight, epsilon):
        raise ValueError(
            "Loom Add+RMSNorm requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes and a 1D hidden-size weight"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    if input_tensor.requires_grad or residual.requires_grad or weight.requires_grad:
        raise ValueError("Loom Add+RMSNorm is an inference-only operator")

    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, hidden_size


def _validate_dynamic_fp8_inputs(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    if not supports_rms_norm_dynamic_fp8(input_tensor, weight, epsilon):
        raise ValueError(
            "Loom RMSNorm+FP8 requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes and a 1D hidden-size weight"
        )
    if input_tensor.requires_grad or weight.requires_grad:
        raise ValueError("Loom RMSNorm+FP8 is an inference-only operator")

    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, hidden_size


def _validate_dynamic_fp8_buffers(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    dtype, rows, hidden_size = _validate_dynamic_fp8_inputs(
        input_tensor, weight, epsilon
    )
    if (
        output.device != input_tensor.device
        or output.dtype != torch.float8_e4m3fn
        or output.shape != input_tensor.shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Loom RMSNorm+FP8 output must be a same-device contiguous "
            "torch.float8_e4m3fn tensor matching the input shape"
        )
    if (
        scales.device != input_tensor.device
        or scales.dtype != torch.float32
        or scales.shape != (rows, 1)
        or not scales.is_contiguous()
    ):
        raise ValueError(
            "Loom RMSNorm+FP8 scales must be a same-device contiguous F32 "
            "tensor with shape [rows, 1]"
        )
    return dtype, rows, hidden_size



def add_rms_norm_(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update input/residual in place and return those same tensor objects."""
    _dispatch()._add_rms_norm_mut(input_tensor, residual, weight, float(epsilon))
    return input_tensor, residual


def rms_norm_dynamic_fp8_out(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write fused RMSNorm and per-token FP8 results into caller-owned buffers."""
    _dispatch()._rms_norm_dynamic_fp8(input_tensor, weight, output, scales, float(epsilon))
    return output, scales


def rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 E4M3FN output and one F32 dequantization scale per row."""
    _, rows, _ = _validate_dynamic_fp8_inputs(input_tensor, weight, epsilon)
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), device=input_tensor.device, dtype=torch.float32)
    return rms_norm_dynamic_fp8_out(
        input_tensor, weight, output, scales, float(epsilon)
    )


def mutable_custom_op():
    """Expose the registered op definition for torch.library.opcheck."""
    return _dispatch()._add_rms_norm_mut


def dynamic_fp8_custom_op():
    """Expose the registered FP8 op definition for torch.library.opcheck."""
    return _dispatch()._rms_norm_dynamic_fp8


def dynamic_fp8_unchecked_custom_op():
    """Expose the raw-byte out variant for dispatcher schema validation."""
    return _dispatch()._rms_norm_dynamic_fp8_unchecked
