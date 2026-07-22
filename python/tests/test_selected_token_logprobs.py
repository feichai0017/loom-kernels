from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    adapter_backend,
    selected_token_logprobs,
    selected_token_logprobs_custom_op,
)


def reference(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_f32 = logits.float()
    selected = logits_f32.gather(-1, token_ids.unsqueeze(-1))
    logprobs = logits_f32.log_softmax(dim=-1).gather(
        -1, token_ids.unsqueeze(-1)
    )
    ranks = (logits_f32 >= selected).sum(dim=-1)
    return logprobs.squeeze(-1), ranks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17), (7, 4096), (8, 151936)])
def test_selected_token_logprobs_matches_pytorch(dtype, shape):
    torch.manual_seed(127)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    token_ids = torch.randint(
        0, shape[1], (shape[0],), device="cuda", dtype=torch.int64
    )
    expected = reference(logits, token_ids)
    actual = selected_token_logprobs(logits, token_ids)
    torch.cuda.synchronize()

    assert adapter_backend() == "cpp-dispatch"
    torch.testing.assert_close(actual[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_counts_ties_at_arbitrary_ranks():
    logits = torch.tensor(
        [[1.0, 4.0, 4.0, -2.0], [3.0, 3.0, 2.0, 3.0]],
        device="cuda",
    )
    token_ids = torch.tensor([0, 2], device="cuda", dtype=torch.int64)
    logprobs, ranks = selected_token_logprobs(logits, token_ids)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        logprobs,
        reference(logits, token_ids)[0],
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert ranks.tolist() == [3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_accepts_padded_vocabulary_rows():
    storage = torch.randn((8, 152064), device="cuda", dtype=torch.bfloat16)
    logits = storage[:, :151936]
    token_ids = torch.tensor(
        [0, 17, 4096, 70000, 100000, 151935, 11, 99],
        device="cuda",
        dtype=torch.int64,
    )
    assert logits.stride() == (152064, 1)

    actual = selected_token_logprobs(logits, token_ids)
    expected = reference(logits, token_ids)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_uses_current_external_stream():
    logits = torch.randn((4, 8192), device="cuda", dtype=torch.float16)
    token_ids = torch.tensor([0, 9, 1024, 8191], device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = selected_token_logprobs(logits, token_ids)
    stream.synchronize()
    expected = reference(logits, token_ids)
    torch.testing.assert_close(actual[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_dispatcher_contract_and_fake_tensor():
    logits = torch.randn((3, 257), device="cuda", dtype=torch.bfloat16)
    token_ids = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int64)
    torch.library.opcheck(
        selected_token_logprobs_custom_op(),
        (logits, token_ids),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_survives_torch_compile():
    @torch.compile(fullgraph=True)
    def compiled(logits: torch.Tensor, token_ids: torch.Tensor):
        return torch.ops.loom_kernels.selected_token_logprobs(logits, token_ids)

    logits = torch.randn((5, 4096), device="cuda", dtype=torch.bfloat16)
    token_ids = torch.tensor([0, 1, 255, 2048, 4095], device="cuda")
    actual = compiled(logits, token_ids)
    expected = reference(logits, token_ids)
    torch.testing.assert_close(actual[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_selected_token_logprobs_cuda_graph_replay():
    logits = torch.randn((8, 4096), device="cuda", dtype=torch.float16)
    token_ids = torch.arange(8, device="cuda", dtype=torch.int64)
    for _ in range(3):
        outputs = selected_token_logprobs(logits, token_ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = selected_token_logprobs(logits, token_ids)
    logits.copy_(torch.randn_like(logits))
    token_ids.copy_(torch.arange(8, device="cuda", dtype=torch.int64) + 8)
    graph.replay()
    torch.cuda.synchronize()

    expected = reference(logits, token_ids)
    torch.testing.assert_close(outputs[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(outputs[1], expected[1])


def test_selected_token_logprobs_rejects_invalid_inputs():
    logits = torch.empty((2, 4), device="cpu")
    with pytest.raises(ValueError, match="CUDA logits"):
        selected_token_logprobs(logits, torch.zeros(2, dtype=torch.int64))
    if torch.cuda.is_available():
        logits = torch.empty((2, 4), device="cuda")
        with pytest.raises(ValueError, match="int64 token ID"):
            selected_token_logprobs(
                logits, torch.zeros(2, device="cuda", dtype=torch.int32)
            )
