"""Sampling predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import (
    _greedy_sample_logprobs,
    _selected_token_logprobs,
)
from ._common import _DTYPE_NAMES


def supports_greedy_sample_logprobs(logits: torch.Tensor) -> bool:
    """Return whether logits match the deterministic greedy CUDA boundary."""
    return bool(
        logits.device.type == "cuda"
        and logits.dtype in _DTYPE_NAMES
        and logits.dim() == 2
        and logits.shape[0] > 0
        and logits.shape[1] > 0
        and logits.shape[0] <= 0xFFFF_FFFF
        and logits.shape[1] <= 0x7FFF_FFFF
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and not logits.requires_grad
    )


def supports_selected_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> bool:
    """Return whether one selected token per logits row can be normalized."""
    return bool(
        supports_greedy_sample_logprobs(logits)
        and token_ids.device == logits.device
        and token_ids.dtype == torch.int64
        and token_ids.dim() == 1
        and token_ids.shape[0] == logits.shape[0]
        and token_ids.is_contiguous()
        and not token_ids.requires_grad
    )


def _validate_greedy_sample_logits(
    logits: torch.Tensor,
) -> None:
    if not supports_greedy_sample_logprobs(logits):
        raise ValueError(
            "Loom greedy sampling requires finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride, "
            "non-overlapping rows, and no gradients"
        )
def _validate_selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> None:
    if not supports_selected_token_logprobs(logits, token_ids):
        raise ValueError(
            "Loom selected-token logprobs require finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride and one "
            "same-device contiguous int64 token ID per row; token IDs must "
            "be in vocabulary range"
        )
def greedy_sample_logprobs(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return greedy IDs, sampled logprobs, and vLLM-compatible tie ranks."""
    _validate_greedy_sample_logits(logits)
    return _greedy_sample_logprobs(logits)


def selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized logprobs and ranks for one selected token per row."""
    _validate_selected_token_logprobs(logits, token_ids)
    return _selected_token_logprobs(logits, token_ids)
