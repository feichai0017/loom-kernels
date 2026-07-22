"""Activation predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from ._common import _DTYPE_NAMES


def _dispatch():
    from .. import _torch_dispatch

    return _torch_dispatch


def supports_silu_and_mul(input_tensor: torch.Tensor) -> bool:
    """Return whether Loom supports split-half SiLU-and-Mul for this input."""
    return bool(
        input_tensor.device.type == "cuda"
        and input_tensor.dtype in _DTYPE_NAMES
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and input_tensor.shape[-1] % 2 == 0
        and input_tensor.is_contiguous()
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
    )



def _validate_silu_and_mul_input(
    input_tensor: torch.Tensor,
) -> tuple[str, int, int]:
    if not supports_silu_and_mul(input_tensor):
        raise ValueError(
            "Loom SiLU-and-Mul requires a non-empty contiguous F32/FP16/BF16 "
            "CUDA tensor with an even last dimension"
        )
    if input_tensor.requires_grad:
        raise ValueError("Loom SiLU-and-Mul is an inference-only operator")

    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    if rows > 0xFFFF_FFFF or width > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, width


def _validate_silu_and_mul_buffers(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> tuple[str, int, int]:
    dtype, rows, width = _validate_silu_and_mul_input(input_tensor)
    expected_shape = (*input_tensor.shape[:-1], width)
    if (
        output.device != input_tensor.device
        or output.dtype != input_tensor.dtype
        or output.shape != expected_shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Loom SiLU-and-Mul output must be a same-device contiguous tensor "
            "with matching dtype and half the input last dimension"
        )
    return dtype, rows, width


def _validate_silu_and_mul_dynamic_fp8_input(
    input_tensor: torch.Tensor,
    group_size: int,
) -> tuple[str, int, int, int]:
    if not supports_silu_and_mul_dynamic_fp8(input_tensor, group_size):
        raise ValueError(
            "Loom SiLU-and-Mul+FP8 requires a non-empty contiguous FP16/BF16 "
            "CUDA tensor, group size 64 or 128, and a divisible output width"
        )
    if input_tensor.requires_grad:
        raise ValueError("Loom SiLU-and-Mul+FP8 is an inference-only operator")

    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    group_count = width // group_size
    if rows > 0xFFFF_FFFF or width > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, width, group_count


def _validate_silu_and_mul_dynamic_fp8_buffers(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> tuple[str, int, int]:
    dtype, rows, width, group_count = _validate_silu_and_mul_dynamic_fp8_input(
        input_tensor, group_size
    )
    expected_shape = (*input_tensor.shape[:-1], width)
    if (
        output.device != input_tensor.device
        or output.dtype != torch.float8_e4m3fn
        or output.shape != expected_shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Loom SiLU-and-Mul+FP8 output must be a same-device contiguous "
            "torch.float8_e4m3fn tensor with half the input last dimension"
        )
    if (
        scales.device != input_tensor.device
        or scales.dtype != torch.float32
        or scales.shape != (rows, group_count)
        or not scales.is_contiguous()
    ):
        raise ValueError(
            "Loom SiLU-and-Mul+FP8 scales must be same-device contiguous F32 "
            "with shape [rows, width / group_size]"
        )
    return dtype, rows, width



def silu_and_mul_out(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write split-half `silu(gate) * up` into a caller-owned tensor."""
    _dispatch()._silu_and_mul(input_tensor, output)
    return output


def silu_and_mul(input_tensor: torch.Tensor) -> torch.Tensor:
    """Return split-half `silu(input[..., :d]) * input[..., d:]`."""
    _, _, width = _validate_silu_and_mul_input(input_tensor)
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
    _dispatch()._silu_and_mul_dynamic_fp8(input_tensor, output, scales, int(group_size))
    return output, scales


def silu_and_mul_dynamic_fp8(
    input_tensor: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 SwiGLU output and row-major per-block F32 scales."""
    _, rows, width, group_count = _validate_silu_and_mul_dynamic_fp8_input(
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


def silu_and_mul_custom_op():
    """Expose the checked SiLU-and-Mul operator for torch.library.opcheck."""
    return _dispatch()._silu_and_mul


def silu_and_mul_dynamic_fp8_custom_op():
    """Expose checked fused activation+FP8 for torch.library.opcheck."""
    return _dispatch()._silu_and_mul_dynamic_fp8


def silu_and_mul_dynamic_fp8_unchecked_custom_op():
    """Expose unchecked fused activation+FP8 for compilation tests."""
    return _dispatch()._silu_and_mul_dynamic_fp8_unchecked


def vllm_silu_and_mul_per_block_fp8_launch_count() -> int:
    """Return host submissions through vLLM's Loom activation-FP8 boundary.

    CUDA Graph replay does not return to the host dispatcher, so this counter
    proves that Loom participated in graph construction or eager execution; it
    is not a count of graph replays.
    """
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(
        torch.ops.loom_kernels.vllm_silu_and_mul_per_block_fp8_launch_count()
    )


def reset_vllm_silu_and_mul_per_block_fp8_launch_count() -> None:
    """Reset host-side activation-FP8 launch telemetry."""
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_vllm_silu_and_mul_per_block_fp8_launch_count()
