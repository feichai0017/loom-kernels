"""Paged decode-attention predicates, validation, and public PyTorch APIs."""

from __future__ import annotations

import torch

from .._torch_dispatch import _paged_decode_attention
from ._common import _DTYPE_NAMES


PAGED_DECODE_MAX_CONTEXT = 1024


def _has_dense_nhd_inner_strides(tensor: torch.Tensor) -> bool:
    """Return whether only the outer cache-block stride may contain gaps."""
    if tensor.dim() != 4:
        return False
    _, block_size, heads, width = tensor.shape
    block_elements = block_size * heads * width
    return bool(
        tensor.stride(3) == 1
        and tensor.stride(2) == width
        and tensor.stride(1) == heads * width
        and tensor.stride(0) >= block_elements
        and tensor.stride(0) <= 0xFFFF_FFFF_FFFF_FFFF
    )


def supports_paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    max_sequence_length: int,
) -> bool:
    """Return whether tensors match the native paged-decode kernel family."""
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
        and _has_dense_nhd_inner_strides(key_cache)
        and _has_dense_nhd_inner_strides(value_cache)
        and block_tables.is_contiguous()
        and sequence_lengths.is_contiguous()
        and not query.requires_grad
        and not key_cache.requires_grad
        and not value_cache.requires_grad
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
    _paged_decode_attention(
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
