from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from loom_kernels.torch_ops import (
    Operator,
    add_rms_norm_,
    bridge_abi_version,
    launch_count,
    reset_launch_count,
    rms_norm,
    rms_norm_dynamic_fp8,
    rms_norm_dynamic_fp8_out,
)


def reference(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = input_tensor.float() + residual.float()
    inverse_rms = torch.rsqrt(summed.square().mean(dim=-1, keepdim=True) + epsilon)
    output = (summed * inverse_rms * weight.float()).to(input_tensor.dtype)
    return output, summed.to(input_tensor.dtype)


def vllm_dynamic_fp8_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pytest.importorskip("vllm")
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    rows = input_tensor.numel() // input_tensor.shape[-1]
    scales = torch.empty((rows, 1), device=input_tensor.device, dtype=torch.float32)
    torch.ops._C.rms_norm_dynamic_per_token_quant(
        output, input_tensor, weight, scales, epsilon, None, None
    )
    return output, scales


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 4096), (3, 127)])
def test_add_rms_norm_matches_vllm_semantics_on_external_stream(dtype, shape):
    torch.manual_seed(11)
    epsilon = 1.0e-5
    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected_output, expected_residual = reference(
        input_tensor, residual, weight, epsilon
    )
    input_pointer = input_tensor.data_ptr()
    residual_pointer = residual.data_ptr()

    reset_launch_count(Operator.ADD_RMS_NORM)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, residual_output = add_rms_norm_(
            input_tensor, residual, weight, epsilon
        )
    stream.synchronize()

    assert launch_count(Operator.ADD_RMS_NORM) == 1
    assert output is input_tensor
    assert residual_output is residual
    assert output.data_ptr() == input_pointer
    assert residual_output.data_ptr() == residual_pointer
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)
    tolerance = {
        torch.float32: (1.0e-5, 1.0e-5),
        torch.float16: (2.0e-3, 2.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        output, expected_output, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mutation_schema_passes_torch_opcheck():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops.loom_kernels.add_rms_norm_mut.default,
        (input_tensor, residual, weight, 1.0e-5),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_add_rms_norm_rejects_short_weight():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(127, device="cuda", dtype=torch.float16)

    reset_launch_count(Operator.ADD_RMS_NORM)
    with pytest.raises(RuntimeError, match=r"weight must match"):
        torch.ops.loom_kernels.add_rms_norm_mut(
            input_tensor, residual, weight, 1.0e-5
        )
    assert launch_count(Operator.ADD_RMS_NORM) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_add_rms_norm_rejects_aliasing():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)

    reset_launch_count(Operator.ADD_RMS_NORM)
    with pytest.raises(RuntimeError, match=r"must not overlap"):
        torch.ops.loom_kernels.add_rms_norm_mut(
            input_tensor, input_tensor, weight, 1.0e-5
        )
    assert launch_count(Operator.ADD_RMS_NORM) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_matches_reference(dtype):
    input_tensor = torch.randn(3, 127, device="cuda", dtype=dtype)
    weight = torch.randn(127, device="cuda", dtype=dtype)
    epsilon = 1.0e-5
    expected = (
        input_tensor.float()
        * torch.rsqrt(
            input_tensor.float().square().mean(dim=-1, keepdim=True) + epsilon
        )
        * weight.float()
    ).to(dtype)

    reset_launch_count(Operator.RMS_NORM)
    actual = rms_norm(input_tensor, weight, epsilon)
    torch.cuda.synchronize()

    assert launch_count(Operator.RMS_NORM) == 1
    tolerance = {
        torch.float32: (1.0e-5, 1.0e-5),
        torch.float16: (2.0e-3, 2.0e-3),
        torch.bfloat16: (2.0e-2, 2.0e-2),
    }[dtype]
    torch.testing.assert_close(
        actual, expected, rtol=tolerance[0], atol=tolerance[1]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(8, 4096), (3, 127)])
def test_rms_norm_dynamic_fp8_matches_vllm_on_external_stream(dtype, shape):
    torch.manual_seed(29)
    epsilon = 1.0e-5
    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    expected_output, expected_scales = vllm_dynamic_fp8_reference(
        input_tensor, weight, epsilon
    )

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_FP8)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, scales = rms_norm_dynamic_fp8(input_tensor, weight, epsilon)
    stream.synchronize()

    assert launch_count(Operator.RMS_NORM_DYNAMIC_FP8) == 1
    assert output.dtype == torch.float8_e4m3fn
    assert scales.shape == (input_tensor.numel() // shape[-1], 1)
    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=2.0e-6, atol=1.0e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rms_norm_dynamic_fp8_out_reuses_caller_buffers():
    input_tensor = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty(4, 1, device="cuda", dtype=torch.float32)
    output_pointer = output.data_ptr()
    scales_pointer = scales.data_ptr()

    returned_output, returned_scales = rms_norm_dynamic_fp8_out(
        input_tensor, weight, output, scales, 1.0e-5
    )
    torch.cuda.synchronize()

    assert returned_output is output
    assert returned_scales is scales
    assert output.data_ptr() == output_pointer
    assert scales.data_ptr() == scales_pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_fp8_rejects_short_scales():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, 1, device="cuda", dtype=torch.float32)

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_FP8)
    with pytest.raises(RuntimeError, match=r"shape \[rows, 1\]"):
        torch.ops.loom_kernels.rms_norm_dynamic_fp8(
            input_tensor, weight, output, scales, 1.0e-5
        )
    assert launch_count(Operator.RMS_NORM_DYNAMIC_FP8) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_fp8_rejects_output_alias():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    output = (
        input_tensor.view(torch.uint8)
        .flatten()[: input_tensor.numel()]
        .view(torch.float8_e4m3fn)
        .reshape_as(input_tensor)
    )
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_FP8)
    with pytest.raises(RuntimeError, match=r"must not overlap"):
        torch.ops.loom_kernels.rms_norm_dynamic_fp8(
            input_tensor, weight, output, scales, 1.0e-5
        )
    assert launch_count(Operator.RMS_NORM_DYNAMIC_FP8) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_fp8_schema_declares_both_mutations():
    schema = str(torch.ops.loom_kernels.rms_norm_dynamic_fp8.default._schema)
    assert "Tensor(a!) output" in schema
    assert "Tensor(b!) scales" in schema


def test_bridge_abi_is_current():
    assert bridge_abi_version() == 1


@pytest.mark.parametrize(
    "operator",
    [
        "add_rms_norm_mut_unchecked",
        "rms_norm_dynamic_fp8_unchecked",
        "silu_and_mul_unchecked",
        "silu_and_mul_dynamic_fp8_unchecked",
        "min_p_filter_unchecked_",
        "paged_decode_attention_unchecked",
        "rope_paged_kv_write_unchecked_",
    ],
)
def test_removed_compatibility_schemas_stay_removed(operator):
    with pytest.raises(RuntimeError, match="Could not find schema"):
        torch._C._dispatch_find_schema_or_throw(f"loom_kernels::{operator}", "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_add_rms_norm_op_survives_torch_compile():
    def compiled_target(input_tensor, residual, weight):
        torch.ops.loom_kernels.add_rms_norm_mut(
            input_tensor, residual, weight, 1.0e-5
        )
        return input_tensor, residual

    compiled = torch.compile(compiled_target, fullgraph=True)
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    expected_residual = (input_tensor.float() + residual.float()).to(torch.bfloat16)
    _, residual_output = compiled(input_tensor, residual, weight)
    torch.cuda.synchronize()
    torch.testing.assert_close(residual_output, expected_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_fp8_op_survives_torch_compile():
    def compiled_target(input_tensor, weight, output, scales):
        torch.ops.loom_kernels.rms_norm_dynamic_fp8(
            input_tensor, weight, output, scales, 1.0e-5
        )
        return output, scales

    compiled = torch.compile(compiled_target, fullgraph=True)
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)
    expected_output, expected_scales = vllm_dynamic_fp8_reference(
        input_tensor, weight, 1.0e-5
    )

    actual_output, actual_scales = compiled(input_tensor, weight, output, scales)
    torch.cuda.synchronize()

    assert torch.equal(
        actual_output.view(torch.uint8), expected_output.view(torch.uint8)
    )
    torch.testing.assert_close(
        actual_scales, expected_scales, rtol=2.0e-6, atol=1.0e-8
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_checked_op_can_be_captured_and_replayed():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(input_tensor)
    weight = torch.ones(128, device="cuda", dtype=torch.float16)
    original_input = input_tensor.clone()
    original_residual = residual.clone()
    expected_residual = (input_tensor.float() + residual.float()).to(torch.float16)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.add_rms_norm_mut(
            input_tensor, residual, weight, 1.0e-5
        )
    input_tensor.copy_(original_input)
    residual.copy_(original_residual)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dynamic_fp8_checked_op_can_be_captured_and_replayed():
    input_tensor = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    expected_output, expected_scales = vllm_dynamic_fp8_reference(
        input_tensor, weight, 1.0e-5
    )
    output = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 1, device="cuda", dtype=torch.float32)

    reset_launch_count(Operator.RMS_NORM_DYNAMIC_FP8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.loom_kernels.rms_norm_dynamic_fp8(
            input_tensor, weight, output, scales, 1.0e-5
        )
    output.fill_(0)
    scales.zero_()
    graph.replay()
    torch.cuda.synchronize()

    assert launch_count(Operator.RMS_NORM_DYNAMIC_FP8) == 1
    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    torch.testing.assert_close(scales, expected_scales, rtol=2.0e-6, atol=1.0e-8)
