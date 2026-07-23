"""RoPE and paged-KV predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _rope_paged_kv_write
from ._common import _DTYPE_NAMES


def supports_rope_paged_kv_write(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> bool:
    """Return whether tensors match the native-cache RoPE+paged-write ABI."""
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        return False
    if query.numel() == 0 or key.numel() == 0 or value.numel() == 0:
        return False
    if cos_sin_cache.dim() != 2:
        return False
    rotary_dim = cos_sin_cache.shape[1]
    return bool(
        query.device.type == "cuda"
        and key.device == query.device
        and value.device == query.device
        and positions.device == query.device
        and cos_sin_cache.device == query.device
        and key_cache.device == query.device
        and value_cache.device == query.device
        and slot_mapping.device == query.device
        and query.dtype in _DTYPE_NAMES
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and cos_sin_cache.dtype == query.dtype
        and key_cache.dtype == query.dtype
        and value_cache.dtype == query.dtype
        and positions.dtype == torch.int64
        and slot_mapping.dtype == torch.int64
        and query.shape[0] == key.shape[0] == value.shape[0]
        and query.shape[2] == key.shape[2]
        and key.shape[1] == value.shape[1]
        and value.shape[2] > 0
        and positions.dim() == 1
        and positions.numel() == query.shape[0]
        and slot_mapping.dim() == 1
        and slot_mapping.numel() <= query.shape[0]
        and cos_sin_cache.shape[0] > 0
        and rotary_dim > 0
        and rotary_dim % 2 == 0
        and rotary_dim <= query.shape[2]
        and key_cache.dim() == 4
        and value_cache.dim() == 4
        and key_cache.shape[0] > 0
        and key_cache.shape[1] > 0
        and key_cache.shape[2:] == key.shape[1:]
        and value_cache.shape[:3]
        == (key_cache.shape[0], key_cache.shape[1], value.shape[1])
        and value_cache.shape[3] == value.shape[2]
        and query.stride(2) == 1
        and key.stride(2) == 1
        and value.stride(2) == 1
        and all(stride > 0 for stride in query.stride()[:2])
        and all(stride > 0 for stride in key.stride()[:2])
        and all(stride > 0 for stride in value.stride()[:2])
        and positions.is_contiguous()
        and cos_sin_cache.is_contiguous()
        and slot_mapping.is_contiguous()
        and key_cache.stride(3) == 1
        and value_cache.stride(3) == 1
        and all(stride > 0 for stride in key_cache.stride()[:3])
        and all(stride > 0 for stride in value_cache.stride()[:3])
        and not query.requires_grad
        and not key.requires_grad
        and not value.requires_grad
        and not cos_sin_cache.requires_grad
    )

def rope_paged_kv_write_(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate Q/K in place and scatter rotated K plus V into paged caches."""
    _rope_paged_kv_write(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        bool(is_neox),
    )
    return query, key, key_cache, value_cache
