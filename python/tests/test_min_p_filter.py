from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    adapter_backend,
    min_p_filter_,
    min_p_filter_unchecked_custom_op,
)


def reference(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    # vLLM promotes sampling logits to F32 before applying processors. Keep
    # low-precision inputs as a storage-format test, not an FP16/BF16 softmax
    # rounding contract.
    probabilities = torch.softmax(logits.float(), dim=-1)
    maximum = probabilities.amax(dim=-1, keepdim=True)
    invalid = probabilities < maximum * min_p.reshape(-1, 1)
    return logits.clone().masked_fill_(invalid, -float("inf"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17), (7, 4096), (8, 151936)])
def test_min_p_filter_matches_probability_definition(dtype, shape):
    torch.manual_seed(211)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    min_p = torch.linspace(0.0, 0.9, shape[0], device="cuda")
    expected = reference(logits, min_p)

    returned = min_p_filter_(logits, min_p)
    torch.cuda.synchronize()

    assert adapter_backend() == "cpp-dispatch"
    assert returned is logits
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    assert torch.equal(logits[~torch.isneginf(logits)], expected[~torch.isneginf(expected)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_min_p_filter_preserves_ties_and_accepts_column_probabilities():
    logits = torch.tensor(
        [[4.0, 4.0, 3.0, -8.0], [1.0, 2.0, 3.0, 4.0]],
        device="cuda",
    )
    min_p = torch.tensor([[1.0], [0.0]], device="cuda")
    min_p_filter_(logits, min_p)
    torch.cuda.synchronize()

    assert torch.equal(
        logits,
        torch.tensor(
            [
                [4.0, 4.0, -float("inf"), -float("inf")],
                [1.0, 2.0, 3.0, 4.0],
            ],
            device="cuda",
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_min_p_filter_accepts_padded_vocabulary_rows_and_external_stream():
    storage = torch.randn((8, 152064), device="cuda", dtype=torch.bfloat16)
    logits = storage[:, :151936]
    min_p = torch.linspace(0.05, 0.4, 8, device="cuda")
    expected = reference(logits, min_p)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        stream.wait_stream(torch.cuda.default_stream())
        returned = min_p_filter_(logits, min_p)
    stream.synchronize()

    assert returned is logits
    assert logits.stride() == (152064, 1)
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_min_p_filter_schema_survives_opcheck_and_torch_compile():
    logits = torch.randn((3, 257), device="cuda", dtype=torch.float32)
    min_p = torch.tensor([0.0, 0.1, 0.5], device="cuda")
    torch.library.opcheck(
        min_p_filter_unchecked_custom_op(),
        (logits.clone(), min_p),
        test_utils=("test_schema", "test_faketensor"),
    )

    @torch.compile(fullgraph=True)
    def compiled(values: torch.Tensor, probabilities: torch.Tensor):
        torch.ops.loom_kernels.min_p_filter_unchecked_(values, probabilities)
        return values

    inputs = torch.randn((3, 257), device="cuda", dtype=torch.float32)
    expected = reference(inputs, min_p)
    actual = compiled(inputs, min_p)
    torch.cuda.synchronize()
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def test_min_p_filter_rejects_invalid_tensor_contracts():
    with pytest.raises(ValueError, match="CUDA logits"):
        min_p_filter_(
            torch.empty((2, 4), device="cpu"),
            torch.zeros(2, dtype=torch.float32),
        )
    if torch.cuda.is_available():
        logits = torch.empty((2, 4), device="cuda")
        with pytest.raises(ValueError, match="probabilities"):
            min_p_filter_(
                logits,
                torch.zeros((2, 2), device="cuda", dtype=torch.float32),
            )
