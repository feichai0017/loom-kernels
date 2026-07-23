from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    launch_count,
    reset_launch_count,
    silu_and_mul_dynamic_fp8,
    silu_and_mul_dynamic_fp8_out,
)


def vllm_reference(
    input_tensor: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    pytest.importorskip("vllm")
    width = input_tensor.shape[-1] // 2
    rows = input_tensor.numel() // input_tensor.shape[-1]
    input_2d = input_tensor.view(rows, input_tensor.shape[-1])
    output = torch.empty(
        (rows, width), device=input_tensor.device, dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(
        (rows, width // group_size),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    torch.ops._C.silu_and_mul_per_block_quant(
        output, input_2d, scales, group_size, None, False
    )
    return output.view(*input_tensor.shape[:-1], width), scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,width,group_size", [(8, 11008, 128), (3, 128, 64), (2, 256, 128)]
)
def test_silu_and_mul_dynamic_fp8_matches_vllm_on_external_stream(
    dtype, rows, width, group_size
):
    torch.manual_seed(71)
    input_tensor = torch.randn(rows, width * 2, device="cuda", dtype=dtype)
    expected_output, expected_scales = vllm_reference(input_tensor, group_size)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, scales = silu_and_mul_dynamic_fp8(input_tensor, group_size)
    stream.synchronize()

    assert output.shape == (rows, width)
    assert output.dtype == torch.float8_e4m3fn
    assert scales.shape == (rows, width // group_size)
    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_fp8_preserves_prefix_and_reuses_buffers():
    input_tensor = torch.randn(2, 3, 256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, 3, 128, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(6, 2, device="cuda", dtype=torch.float32)
    output_pointer = output.data_ptr()
    scales_pointer = scales.data_ptr()
    expected_output, expected_scales = vllm_reference(input_tensor, 64)

    returned_output, returned_scales = silu_and_mul_dynamic_fp8_out(
        input_tensor, output, scales, 64
    )
    torch.cuda.synchronize()

    assert returned_output is output
    assert returned_scales is scales
    assert output.data_ptr() == output_pointer
    assert scales.data_ptr() == scales_pointer
    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_fp8_schema_declares_both_mutations():
    schema = str(
        torch.ops.loom_kernels.silu_and_mul_dynamic_fp8.default._schema
    )
    assert "Tensor(a!) output" in schema
    assert "Tensor(b!) scales" in schema


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_fp8_rejects_invalid_contracts():
    input_tensor = torch.randn(2, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="FP16/BF16"):
        silu_and_mul_dynamic_fp8(input_tensor, 64)

    input_tensor = input_tensor.to(torch.bfloat16)
    with pytest.raises(ValueError, match="group size"):
        silu_and_mul_dynamic_fp8(input_tensor, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_fp8_survives_torch_compile():
    def compiled_target(input_tensor, output, scales):
        torch.ops.loom_kernels.silu_and_mul_dynamic_fp8(
            input_tensor, output, scales, 128
        )
        return output, scales

    compiled = torch.compile(compiled_target, fullgraph=True)
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 2, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = vllm_reference(input_tensor, 128)

    actual_output, actual_scales = compiled(input_tensor, output, scales)
    torch.cuda.synchronize()

    assert torch.equal(
        actual_output.view(torch.uint8), expected_output.view(torch.uint8)
    )
    assert torch.equal(actual_scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_silu_and_mul_dynamic_fp8_checked_can_be_captured_and_replayed():
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 2, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = vllm_reference(input_tensor, 128)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.silu_and_mul_dynamic_fp8(
            input_tensor, output, scales, 128
        )
    output.fill_(0)
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_scale_transposed", [False, True])
def test_vllm_compatible_schema_supports_scale_layout_and_upper_bound(
    is_scale_transposed,
):
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float8_e4m3fn)
    expected_output = torch.empty_like(output)
    if is_scale_transposed:
        scales = torch.empty(2, 4, device="cuda", dtype=torch.float32).t()
        expected_scales = torch.empty_like(scales)
    else:
        scales = torch.empty(4, 2, device="cuda", dtype=torch.float32)
        expected_scales = torch.empty_like(scales)
    scale_ub = torch.tensor([0.003], device="cuda", dtype=torch.float32)

    torch.ops._C.silu_and_mul_per_block_quant(
        expected_output,
        input_tensor,
        expected_scales,
        128,
        scale_ub,
        is_scale_transposed,
    )
    torch.ops.loom_kernels.silu_and_mul_per_block_fp8(
        output,
        input_tensor,
        scales,
        128,
        scale_ub,
        is_scale_transposed,
    )
    torch.cuda.synchronize()

    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(scales, expected_scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_compatible_launch_counter_proves_host_dispatch():
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 2, device="cuda", dtype=torch.float32)

    reset_launch_count(Operator.SILU_AND_MUL_DYNAMIC_FP8)
    assert launch_count(Operator.SILU_AND_MUL_DYNAMIC_FP8) == 0
    torch.ops.loom_kernels.silu_and_mul_per_block_fp8(
        output, input_tensor, scales, 128, None, False
    )
    torch.cuda.synchronize()

    assert launch_count(Operator.SILU_AND_MUL_DYNAMIC_FP8) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_vllm_auto_functionalized_boundary_survives_torch_compile():
    from torch._higher_order_ops.auto_functionalize import auto_functionalized

    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default

    def compiled_target(input_tensor, output, scales):
        result = auto_functionalized(
            implementation,
            out=output,
            input=input_tensor,
            scales=scales,
            group_size=128,
            scale_ub=None,
            is_scale_transposed=False,
        )
        return result[1], result[2]

    compiled = torch.compile(compiled_target, fullgraph=True)
    input_tensor = torch.randn(4, 512, device="cuda", dtype=torch.float16)
    output = torch.empty(4, 256, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 2, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = vllm_reference(input_tensor, 128)

    actual_output, actual_scales = compiled(input_tensor, output, scales)
    torch.cuda.synchronize()

    assert torch.equal(
        actual_output.view(torch.uint8), expected_output.view(torch.uint8)
    )
    assert torch.equal(actual_scales, expected_scales)
