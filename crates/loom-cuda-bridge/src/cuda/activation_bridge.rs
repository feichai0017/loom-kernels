//! Checked C bridge entrypoints for activation and output quantization.

use super::*;

unsafe fn launch_silu_and_mul<T: Scalar>(
    input: *const T,
    input_elements: u64,
    output: *mut T,
    output_elements: u64,
    rows: u32,
    width: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "SiLU-and-Mul input") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "SiLU-and-Mul output") }?;
    require_disjoint(
        &[("input", input_range), ("output", output_range)],
        "SiLU-and-Mul",
    )?;
    let spec =
        SiluAndMulSpec::new(rows as usize, width as usize, T::DTYPE).map_err(invalid_contract)?;
    T::silu_and_mul(&stream_backend(stream), &input, &mut output, spec)?;
    record_launch(OP_SILU_AND_MUL);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_silu_and_mul_dynamic_fp8<T: Fp8Scalar>(
    input: *const T,
    input_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    scale_upper_bound: *const f32,
    scale_upper_bound_elements: u64,
    rows: u32,
    width: u32,
    group_size: u32,
    scales_transposed: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) =
        unsafe { read_slice(input, input_elements, "SiLU-and-Mul+FP8 input") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "SiLU-and-Mul+FP8 output") }?;
    let (mut scales, scales_range) =
        unsafe { write_slice(scales, scale_elements, "SiLU-and-Mul+FP8 scales") }?;
    let scale_upper_bound = if scale_upper_bound.is_null() {
        if scale_upper_bound_elements != 0 {
            return Err(CudaExecutorError::InvalidContract(
                "null FP8 scale upper bound must have zero elements".into(),
            ));
        }
        None
    } else {
        if scale_upper_bound_elements != 1 {
            return Err(CudaExecutorError::InvalidContract(
                "FP8 scale upper bound must contain exactly one F32 element".into(),
            ));
        }
        let (value, range) = unsafe {
            read_slice(
                scale_upper_bound,
                scale_upper_bound_elements,
                "SiLU-and-Mul+FP8 scale upper bound",
            )
        }?;
        require_disjoint(
            &[
                ("input", input_range),
                ("output", output_range),
                ("scales", scales_range),
                ("scale upper bound", range),
            ],
            "SiLU-and-Mul+FP8",
        )?;
        Some(value)
    };
    if scale_upper_bound.is_none() {
        require_disjoint(
            &[
                ("input", input_range),
                ("output", output_range),
                ("scales", scales_range),
            ],
            "SiLU-and-Mul+FP8",
        )?;
    }
    let scale_layout = match scales_transposed {
        0 => Fp8ScaleLayout::RowMajor,
        1 => Fp8ScaleLayout::GroupMajor,
        _ => {
            return Err(CudaExecutorError::InvalidContract(
                "FP8 scale layout flag must be 0 or 1".into(),
            ))
        }
    };
    let spec =
        SiluAndMulDynamicFp8Spec::new(rows as usize, width as usize, group_size as usize, T::DTYPE)
            .map_err(invalid_contract)?;
    T::silu_and_mul_dynamic_fp8(
        &stream_backend(stream),
        &input,
        &mut output,
        &mut scales,
        spec,
        SiluAndMulDynamicFp8Options {
            scale_upper_bound,
            scale_layout,
        },
    )?;
    record_launch(OP_SILU_AND_MUL_DYNAMIC_FP8);
    Ok(())
}

/// Checked split-half SiLU-and-Mul.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_silu_and_mul(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    rows: u32,
    width: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_silu_and_mul(
                input.cast(),
                input_elements,
                output.cast(),
                output_elements,
                rows,
                width,
                stream,
            )
        )
    })
}

/// Checked SiLU-and-Mul followed by dynamic per-block FP8.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_silu_and_mul_dynamic_fp8(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    scale_upper_bound: *const f32,
    scale_upper_bound_elements: u64,
    rows: u32,
    width: u32,
    group_size: u32,
    scales_transposed: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| match scalar_kind(dtype)? {
        ScalarKind::F16 => unsafe {
            launch_silu_and_mul_dynamic_fp8::<f16>(
                input.cast(),
                input_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                scale_upper_bound,
                scale_upper_bound_elements,
                rows,
                width,
                group_size,
                scales_transposed,
                stream,
            )
        },
        ScalarKind::Bf16 => unsafe {
            launch_silu_and_mul_dynamic_fp8::<bf16>(
                input.cast(),
                input_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                scale_upper_bound,
                scale_upper_bound_elements,
                rows,
                width,
                group_size,
                scales_transposed,
                stream,
            )
        },
        ScalarKind::F32 => Err(CudaExecutorError::InvalidContract(
            "SiLU-and-Mul+FP8 supports FP16 and BF16 input".into(),
        )),
    })
}
