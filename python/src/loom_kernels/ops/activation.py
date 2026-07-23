"""Activation predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _silu_and_mul, _silu_and_mul_dynamic_fp8
from ._common import _DTYPE_NAMES


def supports_silu_and_mul(input_tensor: torch.Tensor) -> bool:
    """Return whether Loom supports split-half SiLU-and-Mul for this input."""
    return bool(
        input_tensor.device.type == "cuda"
        and input_tensor.dtype in _DTYPE_NAMES
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and input_tensor.shape[-1] % 2 == 0
        and input_tensor.is_contiguous()
        and not input_tensor.requires_grad
    )


def supports_silu_and_mul_dynamic_fp8(
    input_tensor: torch.Tensor, group_size: int
) -> bool:
    """Return whether Loom supports fused SwiGLU and block FP8."""
    if input_tensor.dim() < 1 or input_tensor.shape[-1] % 2 != 0:
        return False
    width = input_tensor.shape[-1] // 2
    return bool(
        input_tensor.device.type == "cuda"
        and input_tensor.dtype in (torch.float16, torch.bfloat16)
        and input_tensor.numel() > 0
        and group_size in (64, 128)
        and width % group_size == 0
        and input_tensor.is_contiguous()
        and not input_tensor.requires_grad
    )


def _validate_silu_and_mul_input(
    input_tensor: torch.Tensor,
) -> int:
    if not supports_silu_and_mul(input_tensor):
        raise ValueError(
            "Loom SiLU-and-Mul requires a non-empty contiguous F32/FP16/BF16 "
            "CUDA tensor with an even last dimension"
        )
    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    if rows > 0xFFFF_FFFF or width > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return width


def _validate_silu_and_mul_dynamic_fp8_input(
    input_tensor: torch.Tensor,
    group_size: int,
) -> tuple[int, int, int]:
    if not supports_silu_and_mul_dynamic_fp8(input_tensor, group_size):
        raise ValueError(
            "Loom SiLU-and-Mul+FP8 requires a non-empty contiguous FP16/BF16 "
            "CUDA tensor, group size 64 or 128, and a divisible output width"
        )
    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    group_count = width // group_size
    if rows > 0xFFFF_FFFF or width > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return rows, width, group_count


def silu_and_mul_out(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write split-half `silu(gate) * up` into a caller-owned tensor."""
    _silu_and_mul(input_tensor, output)
    return output


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Return split-half `silu(input[..., :d]) * input[..., d:]`."""
    width = _validate_silu_and_mul_input(input_tensor)
    output = torch.empty(
        (*input_tensor.shape[:-1], width),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    return silu_and_mul_out(input_tensor, output)


def silu_and_mul_dynamic_fp8_out(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write fused SwiGLU and dynamic block-FP8 into caller buffers."""
    _silu_and_mul_dynamic_fp8(input_tensor, output, scales, int(group_size))
    return output, scales


def silu_and_mul_dynamic_fp8(
    input_tensor: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 SwiGLU output and row-major per-block F32 scales."""
    rows, width, group_count = _validate_silu_and_mul_dynamic_fp8_input(
        input_tensor, group_size
    )
    output = torch.empty(
        (*input_tensor.shape[:-1], width),
        device=input_tensor.device,
        dtype=torch.float8_e4m3fn,
    )
    scales = torch.empty(
        (rows, group_count), device=input_tensor.device, dtype=torch.float32
    )
    return silu_and_mul_dynamic_fp8_out(
        input_tensor, output, scales, int(group_size)
    )
