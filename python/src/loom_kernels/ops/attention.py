"""Paged decode-attention predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import math

import torch

from ._common import _DTYPE_NAMES


PAGED_DECODE_MAX_CONTEXT = 1024


def _dispatch():
    from .. import _torch_dispatch

    return _torch_dispatch


def supports_paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_sequence_length: int,
) -> bool:
    """Return whether tensors match the first native paged-decode kernel."""
    if (
        query.dim() != 3
        or key_cache.dim() != 4
        or value_cache.dim() != 4
        or block_tables.dim() != 2
        or sequence_lengths.dim() != 1
        or query.numel() == 0
        or key_cache.numel() == 0
        or value_cache.numel() == 0
        or isinstance(max_sequence_length, bool)
        or not isinstance(max_sequence_length, int)
    ):
        return False
    sequences, query_heads, head_size = query.shape
    num_blocks, block_size, kv_heads, key_head_size = key_cache.shape
    return bool(
        query.device.type == "cuda"
        and key_cache.device == query.device
        and value_cache.device == query.device
        and block_tables.device == query.device
        and sequence_lengths.device == query.device
        and query.dtype in _DTYPE_NAMES
        and key_cache.dtype == query.dtype
        and value_cache.dtype == query.dtype
        and block_tables.dtype == torch.int32
        and sequence_lengths.dtype == torch.int32
        and query.is_contiguous()
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and block_tables.is_contiguous()
        and sequence_lengths.is_contiguous()
        and head_size == key_head_size
        and value_cache.shape[:3] == (num_blocks, block_size, kv_heads)
        and value_cache.shape[3] > 0
        and query_heads % kv_heads == 0
        and block_tables.shape[0] == sequences
        and block_tables.shape[1] > 0
        and sequence_lengths.shape[0] == sequences
        and 0 < max_sequence_length <= PAGED_DECODE_MAX_CONTEXT
        and max_sequence_length <= block_tables.shape[1] * block_size
        and all(
            dimension <= 0xFFFF_FFFF
            for dimension in (
                sequences,
                query_heads,
                kv_heads,
                head_size,
                value_cache.shape[3],
                num_blocks,
                block_size,
                block_tables.shape[1],
            )
        )
        and sequences * query_heads <= 0x7FFF_FFFF
    )


def _validate_paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    output: torch.Tensor,
    max_sequence_length: int,
    scale: float,
) -> tuple[str, tuple[int, ...]]:
    if not supports_paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        max_sequence_length=max_sequence_length,
    ):
        raise ValueError(
            "Loom paged decode attention requires contiguous same-device "
            "F32/FP16/BF16 query [B,Hq,D], native NHD K/V caches, int32 "
            "block tables/sequence lengths, Hq divisible by Hkv, and "
            "max_sequence_length in [1, 1024]"
        )
    if output.device != query.device or output.dtype != query.dtype:
        raise ValueError("paged decode output must share query device and dtype")
    expected_output_shape = (
        query.shape[0],
        query.shape[1],
        value_cache.shape[3],
    )
    if tuple(output.shape) != expected_output_shape or not output.is_contiguous():
        raise ValueError(
            f"paged decode output must be contiguous with shape {expected_output_shape}"
        )
    if any(tensor.requires_grad for tensor in (query, key_cache, value_cache)):
        raise ValueError("paged decode attention is an inference-only operator")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("paged decode attention scale must be finite and positive")
    if any(
        torch._C._overlaps(output, tensor)
        for tensor in (
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
        )
    ):
        raise ValueError("paged decode output storage must not overlap its inputs")

    dimensions = (
        query.shape[0],
        query.shape[1],
        key_cache.shape[2],
        query.shape[2],
        value_cache.shape[3],
        key_cache.shape[0],
        key_cache.shape[1],
        block_tables.shape[1],
        max_sequence_length,
    )
    return _DTYPE_NAMES[query.dtype], dimensions


def paged_decode_attention_out(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    output: torch.Tensor,
    *,
    max_sequence_length: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Execute base paged decode attention into caller-owned output storage."""
    if scale is None:
        scale = query.shape[-1] ** -0.5
    _dispatch()._paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        int(max_sequence_length),
        float(scale),
    )
    return output


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_sequence_length: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Execute base paged MQA/GQA for one query token per sequence."""
    if query.dim() != 3 or value_cache.dim() != 4:
        raise ValueError("paged decode query/value cache must have ranks 3 and 4")
    output = torch.empty(
        (query.shape[0], query.shape[1], value_cache.shape[3]),
        device=query.device,
        dtype=query.dtype,
    )
    return paged_decode_attention_out(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        max_sequence_length=max_sequence_length,
        scale=scale,
    )


def paged_decode_attention_custom_op():
    """Expose the checked dispatcher operator for validation and integration."""
    return _dispatch()._paged_decode_attention


def paged_decode_attention_unchecked_custom_op():
    """Expose the validated hot-path dispatcher implementation."""
    return _dispatch()._paged_decode_attention_unchecked


def paged_decode_attention_launch_count() -> int:
    """Return host submissions through the C++ dispatcher bridge."""
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(torch.ops.loom_kernels.paged_decode_attention_launch_count())


def reset_paged_decode_attention_launch_count() -> None:
    """Reset host-side paged decode launch telemetry."""
    if _dispatch()._EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_paged_decode_attention_launch_count()
