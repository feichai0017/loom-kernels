from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import rope_paged_kv_write_


def make_cos_sin_cache(
    max_position: int, rotary_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.outer(positions, inverse_frequency)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(dtype)


def make_cache(
    num_blocks: int,
    block_size: int,
    kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_blocks, 2, block_size, kv_heads, head_size)
    if layout == "NHD":
        combined = torch.empty(shape, device="cuda", dtype=dtype)
    elif layout == "HND":
        block_stride = 2 * block_size * kv_heads * head_size
        kv_stride = block_size * kv_heads * head_size
        combined = torch.empty_strided(
            shape,
            (
                block_stride,
                kv_stride,
                head_size,
                block_size * head_size,
                1,
            ),
            device="cuda",
            dtype=dtype,
        )
    else:
        raise ValueError(layout)
    combined.fill_(-7.0)
    key_cache, value_cache = combined.unbind(1)
    return combined, key_cache, value_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_rope_paged_kv_matches_vllm(dtype, is_neox, layout):
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    torch.manual_seed(73)
    tokens = 5
    query_heads = 4
    kv_heads = 2
    head_size = 16
    value_head_size = 16
    rotary_dim = 8
    max_position = 32
    num_blocks = 3
    block_size = 8

    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=dtype
    )
    key = torch.randn((tokens, kv_heads, head_size), device="cuda", dtype=dtype)
    value = torch.randn(
        (tokens, kv_heads, value_head_size), device="cuda", dtype=dtype
    )
    positions = torch.tensor([0, 3, 5, 7, 11], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 7, -1, 15, 22], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(max_position, rotary_dim, dtype)

    expected_query = query.clone()
    expected_key = key.clone()
    expected_combined, expected_key_cache, expected_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )
    actual_query = query.clone()
    actual_key = key.clone()
    actual_combined, actual_key_cache, actual_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )

    torch.ops._C.rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        is_neox,
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        expected_key,
        value,
        expected_key_cache,
        expected_value_cache,
        slots,
        "auto",
        scale,
        scale,
    )

    returned = rope_paged_kv_write_(
        actual_query,
        actual_key,
        value,
        positions,
        cos_sin_cache,
        actual_key_cache,
        actual_value_cache,
        slots,
        is_neox,
    )
    torch.cuda.synchronize()

    assert all(
        actual is expected
        for actual, expected in zip(
            returned,
            (actual_query, actual_key, actual_key_cache, actual_value_cache),
            strict=True,
        )
    )
    tolerance = {
        torch.float32: (1.0e-5, 1.0e-6),
        torch.float16: (1.0e-3, 1.0e-3),
        torch.bfloat16: (1.0e-2, 1.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual_query, expected_query, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        actual_key, expected_key, rtol=tolerance[0], atol=tolerance[1]
    )
    # Compare the combined physical allocations so untouched padding and both
    # logical NHD/HND stride paths are covered as well.
    torch.testing.assert_close(
        actual_combined, expected_combined, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_accepts_packed_qkv_and_short_slot_mapping():
    pytest.importorskip("vllm")
    import vllm._custom_ops  # noqa: F401 - registers vLLM dispatcher ops

    torch.manual_seed(97)
    tokens, query_heads, kv_heads, head_size = 5, 4, 2, 16
    q_width = query_heads * head_size
    kv_width = kv_heads * head_size
    packed = torch.randn(
        (tokens, q_width + 2 * kv_width),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([0, 1, 2, 3, 4], device="cuda", dtype=torch.int64)
    # vLLM leaves Q/K/V padded while the cache update only receives real slots.
    slots = torch.tensor([0, 7, 9], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(16, head_size, torch.bfloat16)

    expected_packed = packed.clone()
    expected_query, expected_key, expected_value = expected_packed.split(
        (q_width, kv_width, kv_width), dim=-1
    )
    expected_query = expected_query.view(tokens, query_heads, head_size)
    expected_key = expected_key.view(tokens, kv_heads, head_size)
    expected_value = expected_value.view(tokens, kv_heads, head_size)
    actual_packed = packed.clone()
    actual_query, actual_key, actual_value = actual_packed.split(
        (q_width, kv_width, kv_width), dim=-1
    )
    actual_query = actual_query.view(tokens, query_heads, head_size)
    actual_key = actual_key.view(tokens, kv_heads, head_size)
    actual_value = actual_value.view(tokens, kv_heads, head_size)
    assert not actual_query.is_contiguous()
    assert actual_query.stride(0) == actual_key.stride(0) == actual_value.stride(0)

    expected_combined, expected_key_cache, expected_value_cache = make_cache(
        2, 8, kv_heads, head_size, torch.bfloat16, "NHD"
    )
    actual_combined, actual_key_cache, actual_value_cache = make_cache(
        2, 8, kv_heads, head_size, torch.bfloat16, "NHD"
    )
    torch.ops._C.rotary_embedding(
        positions,
        expected_query,
        expected_key,
        head_size,
        cos_sin_cache,
        True,
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    torch.ops._C_cache_ops.reshape_and_cache_flash(
        expected_key,
        expected_value,
        expected_key_cache,
        expected_value_cache,
        slots,
        "auto",
        scale,
        scale,
    )

    rope_paged_kv_write_(
        actual_query,
        actual_key,
        actual_value,
        positions,
        cos_sin_cache,
        actual_key_cache,
        actual_value_cache,
        slots,
        True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_packed, expected_packed, rtol=1.0e-2, atol=1.0e-2)
    torch.testing.assert_close(
        actual_combined, expected_combined, rtol=1.0e-2, atol=1.0e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rope_paged_kv_uses_the_current_external_stream():
    tokens, query_heads, kv_heads, head_size = 2, 2, 1, 8
    query = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=torch.float16
    )
    key = torch.randn((tokens, kv_heads, head_size), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    positions = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    slots = torch.tensor([0, 5], device="cuda", dtype=torch.int64)
    cos_sin_cache = make_cos_sin_cache(8, head_size, torch.float16)
    _, key_cache, value_cache = make_cache(
        1, 8, kv_heads, head_size, torch.float16, "NHD"
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        rope_paged_kv_write_(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            slots,
            True,
        )
    stream.synchronize()

    assert torch.isfinite(query).all()
    assert torch.isfinite(key).all()
    torch.testing.assert_close(key_cache[0, 0], key[0])
    torch.testing.assert_close(key_cache[0, 5], key[1])
    torch.testing.assert_close(value_cache[0, 0], value[0])
    torch.testing.assert_close(value_cache[0, 5], value[1])
