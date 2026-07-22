"""PyTorch custom-operator registration for Loom Kernels."""

from __future__ import annotations

import math

import torch

from ._native import (
    launch_add_rms_norm,
    launch_greedy_sample_logprobs,
    launch_rms_norm_dynamic_fp8,
    launch_rope_paged_kv_write,
    launch_selected_token_logprobs,
    launch_silu_and_mul,
    launch_silu_and_mul_dynamic_fp8,
)
from ._torch_extension import load_torch_extension


_DTYPE_NAMES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}


def supports_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Shape/dtype predicate shared with the vLLM IR provider."""
    del epsilon
    return bool(
        variance_size is None
        and weight is not None
        and input_tensor.device.type == "cuda"
        and residual.device == input_tensor.device
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and residual.dtype == input_tensor.dtype
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.shape == residual.shape
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def supports_vllm_add_rms_norm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Minimal hot-path predicate for tensors already governed by vLLM IR."""
    del epsilon
    return bool(
        variance_size is None
        and weight is not None
        and input_tensor.dtype in _DTYPE_NAMES
        and residual.dtype == input_tensor.dtype
        and weight.dtype == input_tensor.dtype
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
    )


def supports_rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> bool:
    """Return whether Loom can fuse RMSNorm with per-token FP8 quantization."""
    return bool(
        weight is not None
        and math.isfinite(epsilon)
        and epsilon > 0.0
        and input_tensor.device.type == "cuda"
        and weight.device == input_tensor.device
        and input_tensor.dtype in _DTYPE_NAMES
        and weight.dtype == input_tensor.dtype
        and input_tensor.dim() >= 1
        and input_tensor.numel() > 0
        and weight.dim() == 1
        and weight.shape[0] == input_tensor.shape[-1]
        and input_tensor.is_contiguous()
        and weight.is_contiguous()
    )


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


def _validate(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    if not supports_add_rms_norm(input_tensor, residual, weight, epsilon):
        raise ValueError(
            "Loom Add+RMSNorm requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes and a 1D hidden-size weight"
        )
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    if input_tensor.requires_grad or residual.requires_grad or weight.requires_grad:
        raise ValueError("Loom Add+RMSNorm is an inference-only operator")

    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, hidden_size


def _validate_dynamic_fp8_inputs(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    if not supports_rms_norm_dynamic_fp8(input_tensor, weight, epsilon):
        raise ValueError(
            "Loom RMSNorm+FP8 requires same-device contiguous CUDA tensors, "
            "matching F32/FP16/BF16 dtypes and a 1D hidden-size weight"
        )
    if input_tensor.requires_grad or weight.requires_grad:
        raise ValueError("Loom RMSNorm+FP8 is an inference-only operator")

    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    if rows > 0xFFFF_FFFF or hidden_size > 0xFFFF_FFFF:
        raise ValueError("tensor shape exceeds the Loom CUDA ABI")
    return _DTYPE_NAMES[input_tensor.dtype], rows, hidden_size


def _validate_dynamic_fp8_buffers(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
) -> tuple[str, int, int]:
    dtype, rows, hidden_size = _validate_dynamic_fp8_inputs(
        input_tensor, weight, epsilon
    )
    if (
        output.device != input_tensor.device
        or output.dtype != torch.float8_e4m3fn
        or output.shape != input_tensor.shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Loom RMSNorm+FP8 output must be a same-device contiguous "
            "torch.float8_e4m3fn tensor matching the input shape"
        )
    if (
        scales.device != input_tensor.device
        or scales.dtype != torch.float32
        or scales.shape != (rows, 1)
        or not scales.is_contiguous()
    ):
        raise ValueError(
            "Loom RMSNorm+FP8 scales must be a same-device contiguous F32 "
            "tensor with shape [rows, 1]"
        )
    return dtype, rows, hidden_size


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


def _validate_greedy_sample_logits(
    logits: torch.Tensor,
) -> tuple[str, int, int, int]:
    if not supports_greedy_sample_logprobs(logits):
        raise ValueError(
            "Loom greedy sampling requires finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride, "
            "non-overlapping rows, and no gradients"
        )
    return (
        _DTYPE_NAMES[logits.dtype],
        logits.shape[0],
        logits.shape[1],
        logits.stride(0),
    )


def _validate_selected_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> tuple[str, int, int, int]:
    if not supports_selected_token_logprobs(logits, token_ids):
        raise ValueError(
            "Loom selected-token logprobs require finite, non-empty rank-2 "
            "F32/FP16/BF16 CUDA logits with unit vocabulary stride and one "
            "same-device contiguous int64 token ID per row; token IDs must "
            "be in vocabulary range"
        )
    return (
        _DTYPE_NAMES[logits.dtype],
        logits.shape[0],
        logits.shape[1],
        logits.stride(0),
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


_EXTENSION_PATH = load_torch_extension()

if _EXTENSION_PATH is None:

    @torch.library.custom_op(
        "loom_kernels::add_rms_norm_mut",
        mutates_args={"input_tensor", "residual"},
        device_types="cuda",
    )
    def _add_rms_norm_mut(
        input_tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> None:
        dtype, rows, hidden_size = _validate(input_tensor, residual, weight, epsilon)
        device_index = input_tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_add_rms_norm(
                dtype,
                input_tensor.data_ptr(),
                residual.data_ptr(),
                weight.data_ptr(),
                rows,
                hidden_size,
                epsilon,
                stream.cuda_stream,
            )

    @torch.library.custom_op(
        "loom_kernels::rms_norm_dynamic_fp8",
        mutates_args={"output", "scales"},
        device_types="cuda",
    )
    def _rms_norm_dynamic_fp8(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        scales: torch.Tensor,
        epsilon: float,
    ) -> None:
        dtype, rows, hidden_size = _validate_dynamic_fp8_buffers(
            input_tensor, weight, output, scales, epsilon
        )
        device_index = input_tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_rms_norm_dynamic_fp8(
                dtype,
                input_tensor.data_ptr(),
                weight.data_ptr(),
                output.data_ptr(),
                scales.data_ptr(),
                rows,
                hidden_size,
                epsilon,
                stream.cuda_stream,
            )

    @torch.library.custom_op(
        "loom_kernels::silu_and_mul",
        mutates_args={"output"},
        device_types="cuda",
    )
    def _silu_and_mul(
        input_tensor: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        dtype, rows, width = _validate_silu_and_mul_buffers(input_tensor, output)
        device_index = input_tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_silu_and_mul(
                dtype,
                input_tensor.data_ptr(),
                output.data_ptr(),
                rows,
                width,
                stream.cuda_stream,
            )

    @torch.library.custom_op(
        "loom_kernels::silu_and_mul_dynamic_fp8",
        mutates_args={"output", "scales"},
        device_types="cuda",
    )
    def _silu_and_mul_dynamic_fp8(
        input_tensor: torch.Tensor,
        output: torch.Tensor,
        scales: torch.Tensor,
        group_size: int,
    ) -> None:
        dtype, rows, width = _validate_silu_and_mul_dynamic_fp8_buffers(
            input_tensor, output, scales, group_size
        )
        device_index = input_tensor.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_silu_and_mul_dynamic_fp8(
                dtype,
                input_tensor.data_ptr(),
                output.data_ptr(),
                scales.data_ptr(),
                rows,
                width,
                group_size,
                stream.cuda_stream,
            )

    @torch.library.custom_op(
        "loom_kernels::greedy_sample_logprobs",
        mutates_args=(),
        device_types="cuda",
    )
    def _greedy_sample_logprobs(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype, rows, vocab_size, row_stride = _validate_greedy_sample_logits(logits)
        token_ids = torch.empty(rows, device=logits.device, dtype=torch.int32)
        logprobs = torch.empty(rows, device=logits.device, dtype=torch.float32)
        ranks = torch.empty(rows, device=logits.device, dtype=torch.int64)
        device_index = logits.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_greedy_sample_logprobs(
                dtype,
                logits.data_ptr(),
                token_ids.data_ptr(),
                logprobs.data_ptr(),
                ranks.data_ptr(),
                rows,
                vocab_size,
                row_stride,
                stream.cuda_stream,
            )
        return token_ids, logprobs, ranks

    @_greedy_sample_logprobs.register_fake
    def _greedy_sample_logprobs_fake(
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if logits.dim() != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("greedy sampling logits must be non-empty rank-2")
        rows = logits.shape[0]
        return (
            torch.empty(rows, device=logits.device, dtype=torch.int32),
            torch.empty(rows, device=logits.device, dtype=torch.float32),
            torch.empty(rows, device=logits.device, dtype=torch.int64),
        )

    @torch.library.custom_op(
        "loom_kernels::selected_token_logprobs",
        mutates_args=(),
        device_types="cuda",
    )
    def _selected_token_logprobs(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype, rows, vocab_size, row_stride = _validate_selected_token_logprobs(
            logits, token_ids
        )
        logprobs = torch.empty(rows, device=logits.device, dtype=torch.float32)
        ranks = torch.empty(rows, device=logits.device, dtype=torch.int64)
        device_index = logits.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_selected_token_logprobs(
                dtype,
                logits.data_ptr(),
                token_ids.data_ptr(),
                logprobs.data_ptr(),
                ranks.data_ptr(),
                rows,
                vocab_size,
                row_stride,
                stream.cuda_stream,
            )
        return logprobs, ranks

    @_selected_token_logprobs.register_fake
    def _selected_token_logprobs_fake(
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("selected-token logits must be non-empty rank-2")
        if token_ids.dim() != 1 or token_ids.shape[0] != logits.shape[0]:
            raise ValueError("selected token IDs must contain one value per row")
        rows = logits.shape[0]
        return (
            torch.empty(rows, device=logits.device, dtype=torch.float32),
            torch.empty(rows, device=logits.device, dtype=torch.int64),
        )

    @torch.library.custom_op(
        "loom_kernels::rope_paged_kv_write_",
        mutates_args={"query", "key", "key_cache", "value_cache"},
        device_types="cuda",
    )
    def _rope_paged_kv_write(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        is_neox: bool,
    ) -> None:
        (
            dtype,
            dimensions,
            query_strides,
            key_strides,
            value_strides,
            key_cache_strides,
            value_cache_strides,
        ) = (
            _validate_rope_paged_kv_write(
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                key_cache,
                value_cache,
                slot_mapping,
            )
        )
        device_index = query.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_rope_paged_kv_write(
                dtype,
                query.data_ptr(),
                key.data_ptr(),
                value.data_ptr(),
                positions.data_ptr(),
                cos_sin_cache.data_ptr(),
                key_cache.data_ptr(),
                value_cache.data_ptr(),
                slot_mapping.data_ptr(),
                *dimensions,
                query_strides,
                key_strides,
                value_strides,
                key_cache_strides,
                value_cache_strides,
                is_neox,
                stream.cuda_stream,
            )

    _ADAPTER_BACKEND = "python-ctypes"
    _add_rms_norm_mut_unchecked = _add_rms_norm_mut
    _rms_norm_dynamic_fp8_unchecked = _rms_norm_dynamic_fp8
    _silu_and_mul_unchecked = _silu_and_mul
    _silu_and_mul_dynamic_fp8_unchecked = _silu_and_mul_dynamic_fp8
    _greedy_sample_logprobs_unchecked = _greedy_sample_logprobs
    _selected_token_logprobs_unchecked = _selected_token_logprobs
    _rope_paged_kv_write_unchecked = _rope_paged_kv_write
else:
    _add_rms_norm_mut = torch.ops.loom_kernels.add_rms_norm_mut.default
    _add_rms_norm_mut_unchecked = (
        torch.ops.loom_kernels.add_rms_norm_mut_unchecked.default
    )
    _rms_norm_dynamic_fp8 = torch.ops.loom_kernels.rms_norm_dynamic_fp8.default
    _rms_norm_dynamic_fp8_unchecked = (
        torch.ops.loom_kernels.rms_norm_dynamic_fp8_unchecked.default
    )
    _silu_and_mul = torch.ops.loom_kernels.silu_and_mul.default
    _silu_and_mul_unchecked = torch.ops.loom_kernels.silu_and_mul_unchecked.default
    _silu_and_mul_dynamic_fp8 = (
        torch.ops.loom_kernels.silu_and_mul_dynamic_fp8.default
    )
    _silu_and_mul_dynamic_fp8_unchecked = (
        torch.ops.loom_kernels.silu_and_mul_dynamic_fp8_unchecked.default
    )
    _greedy_sample_logprobs = (
        torch.ops.loom_kernels.greedy_sample_logprobs.default
    )
    _greedy_sample_logprobs_unchecked = _greedy_sample_logprobs
    _selected_token_logprobs = (
        torch.ops.loom_kernels.selected_token_logprobs.default
    )
    _selected_token_logprobs_unchecked = _selected_token_logprobs
    _rope_paged_kv_write = torch.ops.loom_kernels.rope_paged_kv_write_.default
    _rope_paged_kv_write_unchecked = (
        torch.ops.loom_kernels.rope_paged_kv_write_unchecked_.default
    )
    _ADAPTER_BACKEND = "cpp-dispatch"


def add_rms_norm_(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update input/residual in place and return those same tensor objects."""
    _add_rms_norm_mut(input_tensor, residual, weight, float(epsilon))
    return input_tensor, residual


def rms_norm_dynamic_fp8_out(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write fused RMSNorm and per-token FP8 results into caller-owned buffers."""
    _rms_norm_dynamic_fp8(input_tensor, weight, output, scales, float(epsilon))
    return output, scales


def rms_norm_dynamic_fp8(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 E4M3FN output and one F32 dequantization scale per row."""
    _, rows, _ = _validate_dynamic_fp8_inputs(input_tensor, weight, epsilon)
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), device=input_tensor.device, dtype=torch.float32)
    return rms_norm_dynamic_fp8_out(
        input_tensor, weight, output, scales, float(epsilon)
    )


def silu_and_mul_out(
    input_tensor: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write split-half `silu(gate) * up` into a caller-owned tensor."""
    _silu_and_mul(input_tensor, output)
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
    _silu_and_mul_dynamic_fp8(input_tensor, output, scales, int(group_size))
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


def mutable_custom_op():
    """Expose the registered op definition for torch.library.opcheck."""
    return _add_rms_norm_mut


def dynamic_fp8_custom_op():
    """Expose the registered FP8 op definition for torch.library.opcheck."""
    return _rms_norm_dynamic_fp8


def dynamic_fp8_unchecked_custom_op():
    """Expose the raw-byte out variant for dispatcher schema validation."""
    return _rms_norm_dynamic_fp8_unchecked


def silu_and_mul_custom_op():
    """Expose the checked SiLU-and-Mul operator for torch.library.opcheck."""
    return _silu_and_mul


def silu_and_mul_dynamic_fp8_custom_op():
    """Expose checked fused activation+FP8 for torch.library.opcheck."""
    return _silu_and_mul_dynamic_fp8


def silu_and_mul_dynamic_fp8_unchecked_custom_op():
    """Expose unchecked fused activation+FP8 for compilation tests."""
    return _silu_and_mul_dynamic_fp8_unchecked


def rope_paged_kv_write_custom_op():
    """Expose checked fused RoPE+paged-KV for dispatcher validation."""
    return _rope_paged_kv_write


def rope_paged_kv_write_unchecked_custom_op():
    """Expose the hot-path fused RoPE+paged-KV dispatcher implementation."""
    return _rope_paged_kv_write_unchecked


def greedy_sample_logprobs_custom_op():
    """Expose fused greedy sampling for dispatcher and FakeTensor checks."""
    return _greedy_sample_logprobs


def selected_token_logprobs_custom_op():
    """Expose selected-token normalization for dispatcher/FakeTensor checks."""
    return _selected_token_logprobs


def vllm_silu_and_mul_per_block_fp8_launch_count() -> int:
    """Return host submissions through vLLM's Loom activation-FP8 boundary.

    CUDA Graph replay does not return to the host dispatcher, so this counter
    proves that Loom participated in graph construction or eager execution; it
    is not a count of graph replays.
    """
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(
        torch.ops.loom_kernels.vllm_silu_and_mul_per_block_fp8_launch_count()
    )


def reset_vllm_silu_and_mul_per_block_fp8_launch_count() -> None:
    """Reset host-side activation-FP8 launch telemetry."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_vllm_silu_and_mul_per_block_fp8_launch_count()


def rope_paged_kv_write_launch_count() -> int:
    """Return host submissions through Loom's fused RoPE+paged-KV op.

    CUDA Graph replay does not return to the host dispatcher, so this proves
    graph construction or eager execution rather than counting graph replays.
    """
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(torch.ops.loom_kernels.rope_paged_kv_write_launch_count())


def reset_rope_paged_kv_write_launch_count() -> None:
    """Reset host-side RoPE+paged-KV launch telemetry."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_rope_paged_kv_write_launch_count()


def greedy_sample_logprobs_launch_count() -> int:
    """Return host submissions through the fused greedy sampling boundary."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(torch.ops.loom_kernels.greedy_sample_logprobs_launch_count())


def reset_greedy_sample_logprobs_launch_count() -> None:
    """Reset host-side greedy sampling launch telemetry."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_greedy_sample_logprobs_launch_count()


def selected_token_logprobs_launch_count() -> int:
    """Return host submissions through selected-token normalization."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    return int(torch.ops.loom_kernels.selected_token_logprobs_launch_count())


def reset_selected_token_logprobs_launch_count() -> None:
    """Reset host-side selected-token logprob launch telemetry."""
    if _EXTENSION_PATH is None:
        raise RuntimeError("launch telemetry requires the C++ dispatcher bridge")
    torch.ops.loom_kernels.reset_selected_token_logprobs_launch_count()


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
    "rms_norm_dynamic_fp8",
    "rms_norm_dynamic_fp8_out",
    "rope_paged_kv_write_",
    "rope_paged_kv_write_custom_op",
    "rope_paged_kv_write_launch_count",
    "rope_paged_kv_write_unchecked_custom_op",
    "reset_rope_paged_kv_write_launch_count",
    "reset_greedy_sample_logprobs_launch_count",
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
    "supports_selected_token_logprobs",
    "supports_rms_norm_dynamic_fp8",
    "supports_rope_paged_kv_write",
    "supports_silu_and_mul",
    "supports_silu_and_mul_dynamic_fp8",
    "supports_vllm_add_rms_norm",
    "vllm_silu_and_mul_per_block_fp8_launch_count",
]
