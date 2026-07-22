"""Register dispatcher operators and select the C++ or ctypes bridge."""

from __future__ import annotations

import torch

from ._native import (
    launch_add_rms_norm,
    launch_greedy_sample_logprobs,
    launch_min_p_filter,
    launch_rms_norm_dynamic_fp8,
    launch_rope_paged_kv_write,
    launch_selected_token_logprobs,
    launch_silu_and_mul,
    launch_silu_and_mul_dynamic_fp8,
)
from ._torch_extension import load_torch_extension
from .ops.activation import (
    _validate_silu_and_mul_buffers,
    _validate_silu_and_mul_dynamic_fp8_buffers,
)
from .ops.logits import _validate_min_p_filter
from .ops.norm import _validate_add_rms_norm, _validate_dynamic_fp8_buffers
from .ops.rope_kv import _validate_rope_paged_kv_write
from .ops.sampling import (
    _validate_greedy_sample_logits,
    _validate_selected_token_logprobs,
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
        dtype, rows, hidden_size = _validate_add_rms_norm(input_tensor, residual, weight, epsilon)
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
        "loom_kernels::min_p_filter_",
        mutates_args={"logits"},
        device_types="cuda",
    )
    def _min_p_filter(logits: torch.Tensor, min_p: torch.Tensor) -> None:
        dtype, rows, vocab_size, row_stride = _validate_min_p_filter(logits, min_p)
        device_index = logits.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        with torch.cuda.device(device_index):
            stream = torch.cuda.current_stream(device_index)
            launch_min_p_filter(
                dtype,
                logits.data_ptr(),
                min_p.data_ptr(),
                rows,
                vocab_size,
                row_stride,
                stream.cuda_stream,
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
    _min_p_filter_unchecked = _min_p_filter
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
    _min_p_filter = torch.ops.loom_kernels.min_p_filter_.default
    _min_p_filter_unchecked = (
        torch.ops.loom_kernels.min_p_filter_unchecked_.default
    )
    _rope_paged_kv_write = torch.ops.loom_kernels.rope_paged_kv_write_.default
    _rope_paged_kv_write_unchecked = (
        torch.ops.loom_kernels.rope_paged_kv_write_unchecked_.default
    )
    _ADAPTER_BACKEND = "cpp-dispatch"
