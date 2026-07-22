"""Stable PyTorch operator facade for Loom Kernels.

Implementations live in domain modules under :mod:`loom_kernels.ops`; this
module preserves the original import path used by vLLM, tests, and benchmarks.
"""

from __future__ import annotations

from ._torch_dispatch import (
    _ADAPTER_BACKEND,
    _EXTENSION_PATH,
    _add_rms_norm_mut,
    _add_rms_norm_mut_unchecked,
    _greedy_sample_logprobs,
    _greedy_sample_logprobs_unchecked,
    _min_p_filter,
    _min_p_filter_unchecked,
    _rms_norm_dynamic_fp8,
    _rms_norm_dynamic_fp8_unchecked,
    _rope_paged_kv_write,
    _rope_paged_kv_write_unchecked,
    _selected_token_logprobs,
    _selected_token_logprobs_unchecked,
    _silu_and_mul,
    _silu_and_mul_dynamic_fp8,
    _silu_and_mul_dynamic_fp8_unchecked,
    _silu_and_mul_unchecked,
)
from .ops.activation import (
    reset_vllm_silu_and_mul_per_block_fp8_launch_count,
    silu_and_mul,
    silu_and_mul_custom_op,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_custom_op,
    silu_and_mul_dynamic_fp8_out,
    silu_and_mul_dynamic_fp8_unchecked_custom_op,
    silu_and_mul_out,
    supports_silu_and_mul,
    supports_silu_and_mul_dynamic_fp8,
    vllm_silu_and_mul_per_block_fp8_launch_count,
)
from .ops.norm import (
    add_rms_norm_,
    dynamic_fp8_custom_op,
    dynamic_fp8_unchecked_custom_op,
    mutable_custom_op,
    rms_norm_dynamic_fp8,
    rms_norm_dynamic_fp8_out,
    supports_add_rms_norm,
    supports_rms_norm_dynamic_fp8,
    supports_vllm_add_rms_norm,
)
from .ops.logits import (
    min_p_filter_,
    min_p_filter_custom_op,
    min_p_filter_launch_count,
    min_p_filter_unchecked_custom_op,
    reset_min_p_filter_launch_count,
    supports_min_p_filter,
)
from .ops.rope_kv import (
    reset_rope_paged_kv_write_launch_count,
    rope_paged_kv_write_,
    rope_paged_kv_write_custom_op,
    rope_paged_kv_write_launch_count,
    rope_paged_kv_write_unchecked_custom_op,
    supports_rope_paged_kv_write,
)
from .ops.sampling import (
    greedy_sample_logprobs,
    greedy_sample_logprobs_custom_op,
    greedy_sample_logprobs_launch_count,
    reset_greedy_sample_logprobs_launch_count,
    reset_selected_token_logprobs_launch_count,
    selected_token_logprobs,
    selected_token_logprobs_custom_op,
    selected_token_logprobs_launch_count,
    supports_greedy_sample_logprobs,
    supports_selected_token_logprobs,
)


def adapter_backend() -> str:
    """Return the active dispatcher bridge implementation."""
    return _ADAPTER_BACKEND


__all__ = [
    "adapter_backend",
    "add_rms_norm_",
    "dynamic_fp8_custom_op",
    "dynamic_fp8_unchecked_custom_op",
    "greedy_sample_logprobs",
    "greedy_sample_logprobs_custom_op",
    "greedy_sample_logprobs_launch_count",
    "mutable_custom_op",
    "min_p_filter_",
    "min_p_filter_custom_op",
    "min_p_filter_launch_count",
    "min_p_filter_unchecked_custom_op",
    "rms_norm_dynamic_fp8",
    "rms_norm_dynamic_fp8_out",
    "rope_paged_kv_write_",
    "rope_paged_kv_write_custom_op",
    "rope_paged_kv_write_launch_count",
    "rope_paged_kv_write_unchecked_custom_op",
    "reset_rope_paged_kv_write_launch_count",
    "reset_greedy_sample_logprobs_launch_count",
    "reset_min_p_filter_launch_count",
    "reset_selected_token_logprobs_launch_count",
    "reset_vllm_silu_and_mul_per_block_fp8_launch_count",
    "silu_and_mul",
    "silu_and_mul_custom_op",
    "silu_and_mul_dynamic_fp8",
    "silu_and_mul_dynamic_fp8_custom_op",
    "silu_and_mul_dynamic_fp8_out",
    "silu_and_mul_dynamic_fp8_unchecked_custom_op",
    "silu_and_mul_out",
    "selected_token_logprobs",
    "selected_token_logprobs_custom_op",
    "selected_token_logprobs_launch_count",
    "supports_add_rms_norm",
    "supports_greedy_sample_logprobs",
    "supports_min_p_filter",
    "supports_selected_token_logprobs",
    "supports_rms_norm_dynamic_fp8",
    "supports_rope_paged_kv_write",
    "supports_silu_and_mul",
    "supports_silu_and_mul_dynamic_fp8",
    "supports_vllm_add_rms_norm",
    "vllm_silu_and_mul_per_block_fp8_launch_count",
]
