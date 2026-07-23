"""Logits-processing predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _min_p_filter
from ._common import _DTYPE_NAMES


def supports_min_p_filter(logits: torch.Tensor, min_p: torch.Tensor) -> bool:
    """Return whether tensors match Loom's in-place min-p CUDA boundary."""
    if logits.dim() != 2 or min_p.dim() not in (1, 2):
        return False
    min_p_shape_matches = bool(
        (min_p.dim() == 1 and min_p.shape[0] == logits.shape[0])
        or (
            min_p.dim() == 2
            and min_p.shape[0] == logits.shape[0]
            and min_p.shape[1] == 1
        )
    )
    return bool(
        logits.device.type == "cuda"
        and logits.dtype in _DTYPE_NAMES
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.shape[0] <= 0xFFFF_FFFF
        and logits.shape[1] <= 0xFFFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and min_p.device == logits.device
        and min_p.dtype == torch.float32
        and min_p_shape_matches
        and min_p.is_contiguous()
        and not logits.requires_grad
        and not min_p.requires_grad
    )


def _validate_min_p_filter(
    logits: torch.Tensor,
    min_p: torch.Tensor,
) -> None:
    if not supports_min_p_filter(logits, min_p):
        raise ValueError(
            "Loom min-p filtering requires non-empty rank-2 F32/FP16/BF16 "
            "CUDA logits with unit vocabulary stride and same-device "
            "contiguous F32 probabilities shaped [rows] or [rows, 1]"
        )
def min_p_filter_(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """Filter logits in place using each row's max-probability ratio."""
    _validate_min_p_filter(logits, min_p)
    _min_p_filter(logits, min_p)
    return logits
