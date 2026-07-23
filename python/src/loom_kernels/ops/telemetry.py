"""Uniform Rust-bridge launch telemetry."""

from __future__ import annotations

from enum import IntEnum

from .._torch_dispatch import (
    _bridge_abi_version,
    _bridge_launch_count,
    _reset_bridge_launch_count,
)


class Operator(IntEnum):
    RMS_NORM = 0
    ADD_RMS_NORM = 1
    RMS_NORM_DYNAMIC_FP8 = 2
    SILU_AND_MUL = 3
    SILU_AND_MUL_DYNAMIC_FP8 = 4
    ROPE_PAGED_KV_WRITE = 5
    GREEDY_SAMPLE_LOGPROBS = 6
    SELECTED_TOKEN_LOGPROBS = 7
    MIN_P_FILTER = 8
    PAGED_DECODE_ATTENTION = 9


def _operator_id(operator: Operator) -> int:
    if not isinstance(operator, Operator):
        raise TypeError("operator must be a loom_kernels.Operator")
    return int(operator)


def bridge_abi_version() -> int:
    """Return the loaded Rust bridge ABI version."""
    return int(_bridge_abi_version())


def launch_count(operator: Operator) -> int:
    """Return successful submissions for one semantic operator."""
    return int(_bridge_launch_count(_operator_id(operator)))


def reset_launch_count(operator: Operator) -> None:
    """Reset successful-submission telemetry for one semantic operator."""
    _reset_bridge_launch_count(_operator_id(operator))
