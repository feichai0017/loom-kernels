from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    PAGED_DECODE_MAX_CONTEXT,
    adapter_backend,
    paged_decode_attention,
    paged_decode_attention_launch_count,
    paged_decode_attention_out,
    paged_decode_attention_unchecked_custom_op,
    reset_paged_decode_attention_launch_count,
    supports_paged_decode_attention,
)


def reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Readable PyTorch oracle that follows logical-to-physical block maps."""
    sequences, query_heads, _ = query.shape
    block_size = key_cache.shape[1]
    kv_heads = key_cache.shape[2]
    queries_per_kv = query_heads // kv_heads
    kv_for_query = torch.arange(query_heads, device=query.device) // queries_per_kv
    results = []
    for sequence in range(sequences):
        length = int(sequence_lengths[sequence].item())
        active_blocks = math.ceil(length / block_size)
        physical_blocks = block_tables[sequence, :active_blocks].long()
        keys = key_cache[physical_blocks].reshape(-1, kv_heads, query.shape[-1])[
            :length
        ]
        values = value_cache[physical_blocks].reshape(
            -1, kv_heads, value_cache.shape[-1]
        )[:length]
        per_query_keys = keys[:, kv_for_query].float()
        scores = torch.einsum(
            "hd,thd->ht", query[sequence].float(), per_query_keys
        )
        weights = torch.softmax(scores * scale, dim=-1)
        per_query_values = values[:, kv_for_query].float()
        results.append(torch.einsum("ht,thv->hv", weights, per_query_values))
    return torch.stack(results).to(query.dtype)


def make_case(
    *,
    dtype: torch.dtype,
    sequences: int,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    value_head_size: int,
    block_size: int,
    lengths: list[int],
    seed: int,
    interleaved_cache: bool = False,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    max_sequence_length = max(lengths)
    max_blocks = math.ceil(max_sequence_length / block_size)
    num_blocks = sequences * max_blocks + 7
    query = torch.randn(
        (sequences, query_heads, head_size), device="cuda", dtype=dtype
    )
    if interleaved_cache:
        if head_size != value_head_size:
            raise ValueError("interleaved cache requires matching K/V widths")
        kv_cache = torch.randn(
            (num_blocks, 2, block_size, kv_heads, head_size),
            device="cuda",
            dtype=dtype,
        )
        key_cache, value_cache = kv_cache.unbind(1)
    else:
        key_cache = torch.randn(
            (num_blocks, block_size, kv_heads, head_size),
            device="cuda",
            dtype=dtype,
        )
        value_cache = torch.randn(
            (num_blocks, block_size, kv_heads, value_head_size),
            device="cuda",
            dtype=dtype,
        )
    permutation = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
    block_tables = permutation[: sequences * max_blocks].reshape(
        sequences, max_blocks
    )
    block_tables = block_tables.to(torch.int32).contiguous()
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return (
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
    )


CASES = [
    # MQA, partial final blocks, and distinct value width.
    (3, 8, 1, 32, 24, 16, [1, 17, 31]),
    # GQA with a fully occupied final block mixed with short sequences.
    (4, 8, 2, 64, 64, 16, [16, 33, 64, 7]),
    # More KV heads and the larger engine block size.
    (3, 12, 4, 80, 48, 32, [63, 129, 255]),
    # Exercise the upper half of the initial short-context envelope.
    (2, 8, 2, 128, 96, 32, [257, 511]),
    # Force the four-query-head GQA path at its 128 KV-work-item threshold.
    (16, 32, 8, 32, 24, 16, list(range(17, 33))),
    # Exercise the scalar Q/K fallback inside the packed GQA kernel.
    (2, 8, 2, 33, 17, 8, [17, 23]),
    # Qwen2.5-style 7:1 GQA exercises the partial tail in the two-head path.
    (2, 14, 2, 64, 64, 16, [17, 32]),
    # Reach the partial four-head packed-grid threshold with the same tail.
    (64, 14, 2, 64, 64, 16, list(range(17, 33)) * 4),
    # Long-context split-K with the common 4:1 GQA mapping.
    (1, 32, 8, 128, 128, 16, [128]),
    # Stable LSE merge over ragged split-K partials.
    (4, 32, 8, 128, 128, 16, [512, 501, 489, 477]),
    # High-workload four-query-head packing and vLLM's larger cache block.
    (8, 32, 8, 128, 128, 32, list(range(491, 499))),
    # Odd 7:1 GQA exercises the guarded split-K tail group.
    (2, 14, 2, 128, 128, 16, [129, 256]),
    # MHA uses one active head in each two-head split-K group.
    (1, 8, 8, 128, 128, 16, [257]),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES)
def test_paged_decode_matches_randomized_pytorch_oracle(dtype, case):
    (
        sequences,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        block_size,
        lengths,
    ) = case
    tensors = make_case(
        dtype=dtype,
        sequences=sequences,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_size=head_size,
        value_head_size=value_head_size,
        block_size=block_size,
        lengths=lengths,
        seed=307,
    )
    query, key_cache, value_cache, block_tables, sequence_lengths = tensors
    max_sequence_length = int(sequence_lengths.max().item())
    scale = query.shape[-1] ** -0.5
    expected = reference(*tensors, scale)

    actual = paged_decode_attention(
        *tensors,
        max_sequence_length=max_sequence_length,
        scale=scale,
    )
    torch.cuda.synchronize()

    assert adapter_backend() == "cpp-dispatch"
    tolerance = {
        torch.float32: (3.0e-4, 3.0e-5),
        torch.float16: (3.0e-3, 3.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_paged_decode_accepts_vllm_interleaved_kv_stride(dtype):
    lengths = list(range(17, 33))
    tensors = make_case(
        dtype=dtype,
        sequences=16,
        query_heads=32,
        kv_heads=8,
        head_size=32,
        value_head_size=32,
        block_size=16,
        lengths=lengths,
        seed=353,
        interleaved_cache=True,
    )
    query, key_cache, value_cache, block_tables, sequence_lengths = tensors
    scale = query.shape[-1] ** -0.5
    expected = reference(*tensors, scale)

    assert not key_cache.is_contiguous()
    assert key_cache.stride(0) == 2 * key_cache[0].numel()
    assert supports_paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        max_sequence_length=max(lengths),
    )
    actual = paged_decode_attention(
        *tensors,
        max_sequence_length=max(lengths),
        scale=scale,
    )
    torch.cuda.synchronize()

    tolerance = {
        torch.float32: (3.0e-4, 3.0e-5),
        torch.float16: (3.0e-3, 3.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_decode_uses_current_stream_and_caller_output():
    tensors = make_case(
        dtype=torch.bfloat16,
        sequences=3,
        query_heads=8,
        kv_heads=2,
        head_size=64,
        value_head_size=48,
        block_size=16,
        lengths=[9, 23, 47],
        seed=401,
    )
    query, _, value_cache, _, sequence_lengths = tensors
    scale = query.shape[-1] ** -0.5
    expected = reference(*tensors, scale)
    output = torch.empty(
        (query.shape[0], query.shape[1], value_cache.shape[-1]),
        device="cuda",
        dtype=query.dtype,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        returned = paged_decode_attention_out(
            *tensors,
            output,
            max_sequence_length=int(sequence_lengths.max().item()),
            scale=scale,
        )
    stream.synchronize()

    assert returned is output
    torch.testing.assert_close(output, expected, rtol=2.0e-2, atol=2.0e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_decode_schema_survives_opcheck_and_torch_compile():
    tensors = make_case(
        dtype=torch.float32,
        sequences=2,
        query_heads=4,
        kv_heads=2,
        head_size=16,
        value_head_size=12,
        block_size=16,
        lengths=[7, 21],
        seed=503,
    )
    query, _, value_cache, _, sequence_lengths = tensors
    output = torch.empty(
        (query.shape[0], query.shape[1], value_cache.shape[-1]),
        device="cuda",
        dtype=query.dtype,
    )
    arguments = (*tensors, output, int(sequence_lengths.max().item()), 0.25)
    torch.library.opcheck(
        paged_decode_attention_unchecked_custom_op(),
        arguments,
        test_utils=("test_schema", "test_faketensor"),
    )

    @torch.compile(fullgraph=True)
    def compiled(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tables: torch.Tensor,
        lengths: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops.loom_kernels.paged_decode_attention_unchecked(
            q, k, v, tables, lengths, out, 21, 0.25
        )
        return out

    expected = reference(*tensors, 0.25)
    actual = compiled(*tensors, output)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=3.0e-4, atol=3.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_paged_decode_contract_and_launch_telemetry():
    tensors = make_case(
        dtype=torch.float16,
        sequences=2,
        query_heads=4,
        kv_heads=1,
        head_size=32,
        value_head_size=32,
        block_size=16,
        lengths=[5, 19],
        seed=607,
    )
    reset_paged_decode_attention_launch_count()
    paged_decode_attention(*tensors, max_sequence_length=19)
    torch.cuda.synchronize()
    assert paged_decode_attention_launch_count() == 1

    query, key_cache, value_cache, block_tables, sequence_lengths = tensors
    with pytest.raises(RuntimeError, match="1024"):
        paged_decode_attention(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            max_sequence_length=PAGED_DECODE_MAX_CONTEXT + 1,
        )
    with pytest.raises(RuntimeError, match="int32"):
        paged_decode_attention(
            query,
            key_cache,
            value_cache,
            block_tables.long(),
            sequence_lengths,
            max_sequence_length=19,
        )


def test_paged_decode_support_predicate_rejects_cpu_tensors():
    query = torch.empty((1, 2, 4))
    key_cache = torch.empty((1, 8, 1, 4))
    value_cache = torch.empty_like(key_cache)
    tables = torch.zeros((1, 1), dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    assert not supports_paged_decode_attention(
        query,
        key_cache,
        value_cache,
        tables,
        lengths,
        max_sequence_length=8,
    )
