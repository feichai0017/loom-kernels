"""PyTorch facade for Loom's Rust-owned CUDA operator path."""

from __future__ import annotations

from .ops.activation import (
    silu_and_mul,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_out,
    silu_and_mul_out,
    supports_silu_and_mul,
    supports_silu_and_mul_dynamic_fp8,
)
from .ops.attention import (
    PAGED_DECODE_MAX_CONTEXT,
    paged_decode_attention,
    paged_decode_attention_out,
    supports_paged_decode_attention,
)
from .ops.logits import (
    min_p_filter_,
    supports_min_p_filter,
)
from .ops.norm import (
    add_rms_norm_,
    rms_norm,
    rms_norm_dynamic_fp8,
    rms_norm_dynamic_fp8_out,
    rms_norm_out,
    supports_add_rms_norm,
    supports_rms_norm,
    supports_rms_norm_dynamic_fp8,
)
from .ops.rope_kv import (
    rope_paged_kv_write_,
    supports_rope_paged_kv_write,
)
from .ops.sampling import (
    greedy_sample_logprobs,
    selected_token_logprobs,
    supports_greedy_sample_logprobs,
    supports_selected_token_logprobs,
)
from .ops.telemetry import (
    Operator,
    bridge_abi_version,
    launch_count,
    reset_launch_count,
)


__all__ = [
    "Operator",
    "PAGED_DECODE_MAX_CONTEXT",
    "add_rms_norm_",
    "bridge_abi_version",
    "greedy_sample_logprobs",
    "launch_count",
    "min_p_filter_",
    "paged_decode_attention",
    "paged_decode_attention_out",
    "reset_launch_count",
    "rms_norm",
    "rms_norm_dynamic_fp8",
    "rms_norm_dynamic_fp8_out",
    "rms_norm_out",
    "rope_paged_kv_write_",
    "selected_token_logprobs",
    "silu_and_mul",
    "silu_and_mul_dynamic_fp8",
    "silu_and_mul_dynamic_fp8_out",
    "silu_and_mul_out",
    "supports_add_rms_norm",
    "supports_greedy_sample_logprobs",
    "supports_min_p_filter",
    "supports_paged_decode_attention",
    "supports_rms_norm",
    "supports_rms_norm_dynamic_fp8",
    "supports_rope_paged_kv_write",
    "supports_selected_token_logprobs",
    "supports_silu_and_mul",
    "supports_silu_and_mul_dynamic_fp8",
]
