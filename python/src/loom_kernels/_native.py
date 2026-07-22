"""Lazy ctypes binding to the Loom Kernels CUDA C ABI."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import threading


class NativeLibraryError(RuntimeError):
    """The native library is missing or rejected a launch."""


_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_PATH: Path | None = None


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (parent / "cuda").is_dir():
            return parent
    return None


def _candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("LOOM_KERNELS_CUDA_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.append(Path(__file__).resolve().parent / "lib" / "libloom_kernels_cuda.so")
    repository = _repository_root()
    if repository is not None:
        candidates.append(repository / "build" / "libloom_kernels_cuda.so")
    return candidates


def native_library_path() -> Path | None:
    """Return the first existing native-library candidate without loading it."""
    for candidate in _candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def native_available() -> bool:
    return native_library_path() is not None


def _load() -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_PATH
    if _LIBRARY is not None:
        return _LIBRARY

    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        path = native_library_path()
        if path is None:
            searched = ", ".join(str(candidate) for candidate in _candidates())
            raise NativeLibraryError(
                "Loom CUDA shared library was not found; run "
                f"`python python/build_native.py` or set LOOM_KERNELS_CUDA_LIBRARY. "
                f"Searched: {searched}"
            )
        try:
            library = ctypes.CDLL(str(path))
        except OSError as error:
            raise NativeLibraryError(f"failed to load {path}: {error}") from error

        pointer = ctypes.c_void_p
        common_arguments = [
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            pointer,
        ]
        for name in (
            "loom_cuda_add_rms_norm_f32",
            "loom_cuda_add_rms_norm_f16",
            "loom_cuda_add_rms_norm_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = common_arguments
            function.restype = ctypes.c_int
        dynamic_fp8_arguments = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_float,
            pointer,
        ]
        for name in (
            "loom_cuda_rms_norm_dynamic_fp8_f32",
            "loom_cuda_rms_norm_dynamic_fp8_f16",
            "loom_cuda_rms_norm_dynamic_fp8_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = dynamic_fp8_arguments
            function.restype = ctypes.c_int
        silu_and_mul_arguments = [
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            pointer,
        ]
        for name in (
            "loom_cuda_silu_and_mul_f32",
            "loom_cuda_silu_and_mul_f16",
            "loom_cuda_silu_and_mul_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = silu_and_mul_arguments
            function.restype = ctypes.c_int
        silu_and_mul_dynamic_fp8_arguments = [
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            pointer,
            ctypes.c_uint32,
            pointer,
        ]
        for name in (
            "loom_cuda_silu_and_mul_dynamic_fp8_f16",
            "loom_cuda_silu_and_mul_dynamic_fp8_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = silu_and_mul_dynamic_fp8_arguments
            function.restype = ctypes.c_int
        greedy_sample_arguments = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            pointer,
        ]
        for name in (
            "loom_cuda_greedy_sample_logprobs_f32",
            "loom_cuda_greedy_sample_logprobs_f16",
            "loom_cuda_greedy_sample_logprobs_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = greedy_sample_arguments
            function.restype = ctypes.c_int
        selected_token_logprobs_arguments = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            pointer,
        ]
        for name in (
            "loom_cuda_selected_token_logprobs_f32",
            "loom_cuda_selected_token_logprobs_f16",
            "loom_cuda_selected_token_logprobs_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = selected_token_logprobs_arguments
            function.restype = ctypes.c_int
        rope_paged_kv_arguments = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint32,
            pointer,
        ]
        for name in (
            "loom_cuda_rope_paged_kv_write_f32",
            "loom_cuda_rope_paged_kv_write_f16",
            "loom_cuda_rope_paged_kv_write_bf16",
        ):
            function = getattr(library, name)
            function.argtypes = rope_paged_kv_arguments
            function.restype = ctypes.c_int
        library.loom_cuda_status_string.argtypes = [ctypes.c_int]
        library.loom_cuda_status_string.restype = ctypes.c_char_p
        _LIBRARY = library
        _LIBRARY_PATH = path
        return library


def launch_add_rms_norm(
    dtype: str,
    input_pointer: int,
    residual_pointer: int,
    weight_pointer: int,
    rows: int,
    hidden_size: int,
    epsilon: float,
    stream_pointer: int,
) -> None:
    """Submit Add+RMSNorm to an existing CUDA stream."""
    function_name = {
        "f32": "loom_cuda_add_rms_norm_f32",
        "f16": "loom_cuda_add_rms_norm_f16",
        "bf16": "loom_cuda_add_rms_norm_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(input_pointer),
        ctypes.c_void_p(residual_pointer),
        ctypes.c_void_p(weight_pointer),
        rows,
        hidden_size,
        epsilon,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            f"Loom CUDA Add+RMSNorm launch failed with status {status}: {message}"
        )


def launch_rms_norm_dynamic_fp8(
    dtype: str,
    input_pointer: int,
    weight_pointer: int,
    output_pointer: int,
    scales_pointer: int,
    rows: int,
    hidden_size: int,
    epsilon: float,
    stream_pointer: int,
) -> None:
    """Submit RMSNorm plus dynamic per-token FP8 to an existing CUDA stream."""
    function_name = {
        "f32": "loom_cuda_rms_norm_dynamic_fp8_f32",
        "f16": "loom_cuda_rms_norm_dynamic_fp8_f16",
        "bf16": "loom_cuda_rms_norm_dynamic_fp8_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(input_pointer),
        ctypes.c_void_p(weight_pointer),
        ctypes.c_void_p(output_pointer),
        ctypes.c_void_p(scales_pointer),
        rows,
        hidden_size,
        epsilon,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            "Loom CUDA RMSNorm+FP8 launch failed with status "
            f"{status}: {message}"
        )


def launch_silu_and_mul(
    dtype: str,
    input_pointer: int,
    output_pointer: int,
    rows: int,
    width: int,
    stream_pointer: int,
) -> None:
    """Submit split-half SiLU-and-Mul to an existing CUDA stream."""
    function_name = {
        "f32": "loom_cuda_silu_and_mul_f32",
        "f16": "loom_cuda_silu_and_mul_f16",
        "bf16": "loom_cuda_silu_and_mul_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(input_pointer),
        ctypes.c_void_p(output_pointer),
        rows,
        width,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            f"Loom CUDA SiLU-and-Mul launch failed with status {status}: {message}"
        )


def launch_silu_and_mul_dynamic_fp8(
    dtype: str,
    input_pointer: int,
    output_pointer: int,
    scales_pointer: int,
    rows: int,
    width: int,
    group_size: int,
    stream_pointer: int,
) -> None:
    """Submit fused SwiGLU and dynamic per-block FP8 to a CUDA stream."""
    function_name = {
        "f16": "loom_cuda_silu_and_mul_dynamic_fp8_f16",
        "bf16": "loom_cuda_silu_and_mul_dynamic_fp8_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(input_pointer),
        ctypes.c_void_p(output_pointer),
        ctypes.c_void_p(scales_pointer),
        rows,
        width,
        group_size,
        None,
        0,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            "Loom CUDA SiLU-and-Mul+FP8 launch failed with status "
            f"{status}: {message}"
        )


def launch_greedy_sample_logprobs(
    dtype: str,
    logits_pointer: int,
    token_ids_pointer: int,
    logprobs_pointer: int,
    ranks_pointer: int,
    rows: int,
    vocab_size: int,
    row_stride: int,
    stream_pointer: int,
) -> None:
    """Submit fused greedy selection and sampled-token logprob."""
    function_name = {
        "f32": "loom_cuda_greedy_sample_logprobs_f32",
        "f16": "loom_cuda_greedy_sample_logprobs_f16",
        "bf16": "loom_cuda_greedy_sample_logprobs_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(logits_pointer),
        ctypes.c_void_p(token_ids_pointer),
        ctypes.c_void_p(logprobs_pointer),
        ctypes.c_void_p(ranks_pointer),
        rows,
        vocab_size,
        row_stride,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            "Loom CUDA greedy sampling launch failed with status "
            f"{status}: {message}"
        )


def launch_selected_token_logprobs(
    dtype: str,
    logits_pointer: int,
    token_ids_pointer: int,
    logprobs_pointer: int,
    ranks_pointer: int,
    rows: int,
    vocab_size: int,
    row_stride: int,
    stream_pointer: int,
) -> None:
    """Submit selected-token normalization and rank computation."""
    function_name = {
        "f32": "loom_cuda_selected_token_logprobs_f32",
        "f16": "loom_cuda_selected_token_logprobs_f16",
        "bf16": "loom_cuda_selected_token_logprobs_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(logits_pointer),
        ctypes.c_void_p(token_ids_pointer),
        ctypes.c_void_p(logprobs_pointer),
        ctypes.c_void_p(ranks_pointer),
        rows,
        vocab_size,
        row_stride,
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            "Loom CUDA selected-token logprob launch failed with status "
            f"{status}: {message}"
        )


def launch_rope_paged_kv_write(
    dtype: str,
    query_pointer: int,
    key_pointer: int,
    value_pointer: int,
    positions_pointer: int,
    cos_sin_cache_pointer: int,
    key_cache_pointer: int,
    value_cache_pointer: int,
    slot_mapping_pointer: int,
    tokens: int,
    cache_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    value_head_size: int,
    rotary_dim: int,
    max_position: int,
    num_blocks: int,
    block_size: int,
    query_strides: tuple[int, int],
    key_strides: tuple[int, int],
    value_strides: tuple[int, int],
    key_cache_strides: tuple[int, int, int],
    value_cache_strides: tuple[int, int, int],
    is_neox: bool,
    stream_pointer: int,
) -> None:
    """Submit fused RoPE plus a strided paged K/V write."""
    function_name = {
        "f32": "loom_cuda_rope_paged_kv_write_f32",
        "f16": "loom_cuda_rope_paged_kv_write_f16",
        "bf16": "loom_cuda_rope_paged_kv_write_bf16",
    }.get(dtype)
    if function_name is None:
        raise NativeLibraryError(f"unsupported native dtype: {dtype}")

    library = _load()
    function = getattr(library, function_name)
    status = function(
        ctypes.c_void_p(query_pointer),
        ctypes.c_void_p(key_pointer),
        ctypes.c_void_p(value_pointer),
        ctypes.c_void_p(positions_pointer),
        ctypes.c_void_p(cos_sin_cache_pointer),
        ctypes.c_void_p(key_cache_pointer),
        ctypes.c_void_p(value_cache_pointer),
        ctypes.c_void_p(slot_mapping_pointer),
        tokens,
        cache_tokens,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        rotary_dim,
        max_position,
        num_blocks,
        block_size,
        *query_strides,
        *key_strides,
        *value_strides,
        *key_cache_strides,
        *value_cache_strides,
        int(is_neox),
        ctypes.c_void_p(stream_pointer),
    )
    if status != 0:
        raw_message = library.loom_cuda_status_string(status)
        message = raw_message.decode("utf-8") if raw_message else "unknown status"
        raise NativeLibraryError(
            "Loom CUDA RoPE+paged-KV launch failed with status "
            f"{status}: {message}"
        )
