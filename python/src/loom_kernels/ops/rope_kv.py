"""RoPE and paged-KV predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from ._common import _DTYPE_NAMES


def _dispatch():
    from .. import _torch_dispatch

    return _torch_dispatch


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
    )

def _validate_rope_paged_kv_write(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> tuple[
    str,
    tuple[int, ...],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]:
    if not supports_rope_paged_kv_write(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
    ):
        raise ValueError(
            "Loom RoPE+paged-KV requires unit-dim-stride rank-3 native Q/K/V, "
            "int64 metadata, a contiguous cosine/sine cache, and logical "
            "[blocks, block_size, kv_heads, dim] cache views with unit dim stride"
        )
    if any(
        tensor.requires_grad for tensor in (query, key, value, cos_sin_cache)
    ):
        raise ValueError("Loom RoPE+paged-KV is an inference-only operator")

    dimensions = (
        query.shape[0],
        slot_mapping.numel(),
        query.shape[1],
        key.shape[1],
        query.shape[2],
        value.shape[2],
        cos_sin_cache.shape[1],
        cos_sin_cache.shape[0],
        key_cache.shape[0],
        key_cache.shape[1],
    )
    if any(dimension > 0xFFFF_FFFF for dimension in dimensions):
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return (
        _DTYPE_NAMES[query.dtype],
        dimensions,
        tuple(query.stride()[:2]),
        tuple(key.stride()[:2]),
        tuple(value.stride()[:2]),
        tuple(key_cache.stride()[:3]),
        tuple(value_cache.stride()[:3]),
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
    _dispatch()._rope_paged_kv_write(
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


def rope_paged_kv_write_custom_op():
    """Expose checked fused RoPE+paged-KV for dispatcher validation."""
    return _dispatch()._rope_paged_kv_write


def rope_paged_kv_write_unchecked_custom_op():
    """Expose the hot-path fused RoPE+paged-KV dispatcher implementation."""
    return _dispatch()._rope_paged_kv_write_unchecked


def rope_paged_kv_write_launch_count() -> int:
    """Return host submissions through Loom's fused RoPE+paged-KV op.

    CUDA Graph replay does not return to the host dispatcher, so this proves
    graph construction or eager execution rather than counting graph replays.
    """
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(torch.ops.loom_kernels.rope_paged_kv_write_launch_count())


def reset_rope_paged_kv_write_launch_count() -> None:
    """Reset host-side RoPE+paged-KV launch telemetry."""
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_rope_paged_kv_write_launch_count()
