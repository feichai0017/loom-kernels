//! Checked C bridge entrypoints for normalization.

use super::*;

unsafe fn launch_rms_norm<T: Scalar>(
    input: *const T,
    input_elements: u64,
    weight: *const T,
    weight_elements: u64,
    output: *mut T,
    output_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "RMSNorm input") }?;
    let (weight, weight_range) = unsafe { read_slice(weight, weight_elements, "RMSNorm weight") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "RMSNorm output") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("weight", weight_range),
            ("output", output_range),
        ],
        "RMSNorm",
    )?;
    let spec = RmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::rms_norm(&stream_backend(stream), &input, &weight, &mut output, spec)?;
    record_launch(OP_RMS_NORM);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_add_rms_norm<T: Scalar>(
    input: *mut T,
    input_elements: u64,
    residual: *mut T,
    residual_elements: u64,
    weight: *const T,
    weight_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut input, input_range) =
        unsafe { write_slice(input, input_elements, "Add+RMSNorm input") }?;
    let (mut residual, residual_range) =
        unsafe { write_slice(residual, residual_elements, "Add+RMSNorm residual") }?;
    let (weight, weight_range) =
        unsafe { read_slice(weight, weight_elements, "Add+RMSNorm weight") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("residual", residual_range),
            ("weight", weight_range),
        ],
        "Add+RMSNorm",
    )?;
    let spec = AddRmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::add_rms_norm(
        &stream_backend(stream),
        &mut input,
        &mut residual,
        &weight,
        spec,
    )?;
    record_launch(OP_ADD_RMS_NORM);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_rms_norm_dynamic_fp8<T: Scalar>(
    input: *const T,
    input_elements: u64,
    weight: *const T,
    weight_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "RMSNorm+FP8 input") }?;
    let (weight, weight_range) =
        unsafe { read_slice(weight, weight_elements, "RMSNorm+FP8 weight") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "RMSNorm+FP8 output") }?;
    let (mut scales, scales_range) =
        unsafe { write_slice(scales, scale_elements, "RMSNorm+FP8 scales") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("weight", weight_range),
            ("output", output_range),
            ("scales", scales_range),
        ],
        "RMSNorm+FP8",
    )?;
    let spec = RmsNormDynamicFp8Spec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::rms_norm_dynamic_fp8(
        &stream_backend(stream),
        &input,
        &weight,
        &mut output,
        &mut scales,
        spec,
    )?;
    record_launch(OP_RMS_NORM_DYNAMIC_FP8);
    Ok(())
}
/// Checked standalone RMSNorm.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rms_norm(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rms_norm(
                input.cast(),
                input_elements,
                weight.cast(),
                weight_elements,
                output.cast(),
                output_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}

/// Checked in-place residual Add+RMSNorm.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm(
    dtype: u32,
    input: *mut c_void,
    input_elements: u64,
    residual: *mut c_void,
    residual_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_add_rms_norm(
                input.cast(),
                input_elements,
                residual.cast(),
                residual_elements,
                weight.cast(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}

/// Checked RMSNorm followed by dynamic per-token FP8.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rms_norm_dynamic_fp8(
                input.cast(),
                input_elements,
                weight.cast(),
                weight_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}
