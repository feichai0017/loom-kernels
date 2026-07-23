from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    silu_and_mul,
    silu_and_mul_out,
)


def vllm_reference(input_tensor: torch.Tensor) -> torch.Tensor:
    pytest.importorskip("vllm")
    output = torch.empty(
        (*input_tensor.shape[:-1], input_tensor.shape[-1] // 2),
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    torch.ops._C.silu_and_mul(output, input_tensor)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,width", [(8, 11008), (3, 127), (2, 128)])
def test_silu_and_mul_matches_vllm_on_external_stream(dtype, rows, width):
    torch.manual_seed(53)
    input_tensor = torch.randn(rows, width * 2, device="cuda", dtype=dtype)
    expected = vllm_reference(input_tensor)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output = silu_and_mul(input_tensor)
    stream.synchronize()

    assert output.shape == (rows, width)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_preserves_prefix_shape_and_reuses_output():
    input_tensor = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    pointer = output.data_ptr()

    returned = silu_and_mul_out(input_tensor, output)
    expected = vllm_reference(input_tensor)
    torch.cuda.synchronize()

    assert returned is output
    assert output.data_ptr() == pointer
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_schema_passes_torch_opcheck():
    input_tensor = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    output = torch.empty(2, 128, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops.loom_kernels.silu_and_mul.default,
        (input_tensor, output),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_op_survives_torch_compile():
    def compiled_target(input_tensor, output):
        torch.ops.loom_kernels.silu_and_mul(input_tensor, output)
        return output

    compiled = torch.compile(compiled_target, fullgraph=True)
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.bfloat16)
    expected = vllm_reference(input_tensor)

    actual = compiled(input_tensor, output)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_checked_op_can_be_captured_and_replayed():
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float16)
    expected = vllm_reference(input_tensor)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.silu_and_mul(input_tensor, output)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output, expected)
