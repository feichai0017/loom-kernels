from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    greedy_sample_logprobs,
    launch_count,
    reset_launch_count,
)


def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_f32 = logits.float()
    token_ids = logits_f32.argmax(dim=-1).to(torch.int32)
    logprobs = logits_f32.log_softmax(dim=-1).gather(
        -1, token_ids.long().unsqueeze(-1)
    )
    sampled = logits_f32.gather(-1, token_ids.long().unsqueeze(-1))
    ranks = (logits_f32 >= sampled).sum(dim=-1)
    return token_ids, logprobs.squeeze(-1), ranks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 17), (7, 4096), (8, 151936)])
def test_greedy_sample_logprobs_matches_pytorch(dtype, shape):
    torch.manual_seed(113)
    logits = torch.randn(shape, device="cuda", dtype=dtype)
    expected = reference(logits)
    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    actual = greedy_sample_logprobs(logits)
    torch.cuda.synchronize()

    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 1
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_uses_first_index_for_ties():
    logits = torch.tensor(
        [[1.0, 4.0, 4.0, -2.0], [3.0, 3.0, 2.0, 3.0]],
        device="cuda",
    )
    token_ids, logprobs, ranks = greedy_sample_logprobs(logits)
    torch.cuda.synchronize()
    assert token_ids.tolist() == [1, 0]
    torch.testing.assert_close(
        logprobs,
        reference(logits)[1],
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert ranks.tolist() == [2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_accepts_padded_vocabulary_rows():
    storage = torch.randn((8, 152064), device="cuda", dtype=torch.float32)
    logits = storage[:, :151936]
    assert not logits.is_contiguous()
    assert logits.stride() == (152064, 1)

    reset_launch_count(Operator.GREEDY_SAMPLE_LOGPROBS)
    actual = greedy_sample_logprobs(logits)
    expected = reference(logits)
    torch.cuda.synchronize()
    assert launch_count(Operator.GREEDY_SAMPLE_LOGPROBS) == 1
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_uses_current_external_stream():
    logits = torch.randn((4, 8192), device="cuda")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = greedy_sample_logprobs(logits)
    stream.synchronize()
    expected = reference(logits)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_dispatcher_contract_and_fake_tensor():
    logits = torch.randn((3, 257), device="cuda")
    torch.library.opcheck(
        torch.ops.loom_kernels.greedy_sample_logprobs.default,
        (logits,),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_survives_torch_compile():
    @torch.compile(fullgraph=True)
    def compiled(logits: torch.Tensor):
        return torch.ops.loom_kernels.greedy_sample_logprobs(logits)

    logits = torch.randn((5, 4096), device="cuda")
    actual = compiled(logits)
    expected = reference(logits)
    assert torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_sample_logprobs_cuda_graph_replay():
    logits = torch.randn((8, 4096), device="cuda")
    for _ in range(3):
        outputs = greedy_sample_logprobs(logits)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = greedy_sample_logprobs(logits)
    logits.copy_(torch.randn_like(logits))
    graph.replay()
    torch.cuda.synchronize()

    expected = reference(logits)
    assert torch.equal(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1], rtol=2.0e-5, atol=2.0e-5)
    assert torch.equal(outputs[2], expected[2])


def test_greedy_sample_logprobs_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="rank-2"):
        greedy_sample_logprobs(torch.empty((2, 0), device="cpu"))
