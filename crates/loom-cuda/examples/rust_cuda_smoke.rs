use loom_cuda::{
    runtime::{CudaStreamRef, DeviceBuffer, DeviceSlice, DeviceSliceMut},
    CudaBackend, RowStridedLayout,
};
use loom_kernels::{
    add_rms_norm_f32_reference, greedy_sample_logprobs_f32_reference, AddRmsNormSpec, DType,
    GreedySampleLogprobsSpec,
};
use std::error::Error;
use std::io;

fn main() -> Result<(), Box<dyn Error>> {
    let backend = CudaBackend::new()?;
    validate_add_rms_norm(&backend)?;
    validate_greedy_sample_logprobs(&backend)?;
    validate_borrowed_runtime(&backend)?;
    println!(
        "loom-cuda {}: owned and borrowed CUDA runtime oracle checks passed",
        env!("CARGO_PKG_VERSION")
    );
    Ok(())
}

fn validate_borrowed_runtime(owner: &CudaBackend) -> Result<(), Box<dyn Error>> {
    let spec = AddRmsNormSpec::new(2, 4, 1.0e-5, DType::F32)?;
    let input = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -2.0];
    let residual = vec![1.0, 0.25, -0.5, 2.0, 0.5, -0.25, 1.0, 0.75];
    let weight = vec![1.0, 0.75, 1.25, 0.5];

    let mut expected_input = input.clone();
    let mut expected_residual = residual.clone();
    add_rms_norm_f32_reference(&mut expected_input, &mut expected_residual, &weight, spec)?;

    let mut device_input = DeviceBuffer::from_slice(&input)?;
    let mut device_residual = DeviceBuffer::from_slice(&residual)?;
    let device_weight = DeviceBuffer::from_slice(&weight)?;

    let input_pointer = device_input.as_device_slice_mut().as_mut_ptr();
    let residual_pointer = device_residual.as_device_slice_mut().as_mut_ptr();
    let weight_pointer = device_weight.as_device_slice().as_ptr();

    {
        // This is the same boundary a framework adapter uses: it lends Loom
        // its current stream and tensor storage without transferring ownership.
        let stream = unsafe { CudaStreamRef::from_raw(owner.stream().raw()) };
        let backend = CudaBackend::from_stream(stream);
        let mut input_view = unsafe { DeviceSliceMut::from_raw_parts(input_pointer, input.len())? };
        let mut residual_view =
            unsafe { DeviceSliceMut::from_raw_parts(residual_pointer, residual.len())? };
        let weight_view = unsafe { DeviceSlice::from_raw_parts(weight_pointer, weight.len())? };

        backend.add_rms_norm_f32(&mut input_view, &mut residual_view, &weight_view, spec)?;
        backend.stream().synchronize()?;
    }

    // The borrowed backend did not destroy the external stream or allocations.
    owner.stream().synchronize()?;
    assert_close(
        "borrowed Add+RMSNorm output",
        &device_input.copy_to_vec()?,
        &expected_input,
        2.0e-5,
    )?;
    assert_close(
        "borrowed Add+RMSNorm residual",
        &device_residual.copy_to_vec()?,
        &expected_residual,
        1.0e-6,
    )?;
    Ok(())
}

fn validate_add_rms_norm(backend: &CudaBackend) -> Result<(), Box<dyn Error>> {
    let spec = AddRmsNormSpec::new(2, 4, 1.0e-5, DType::F32)?;
    let input = vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -2.0];
    let residual = vec![1.0, 0.25, -0.5, 2.0, 0.5, -0.25, 1.0, 0.75];
    let weight = vec![1.0, 0.75, 1.25, 0.5];

    let mut expected_input = input.clone();
    let mut expected_residual = residual.clone();
    add_rms_norm_f32_reference(&mut expected_input, &mut expected_residual, &weight, spec)?;

    let mut device_input = DeviceBuffer::from_slice(&input)?;
    let mut device_residual = DeviceBuffer::from_slice(&residual)?;
    let device_weight = DeviceBuffer::from_slice(&weight)?;
    backend.add_rms_norm_f32(
        &mut device_input,
        &mut device_residual,
        &device_weight,
        spec,
    )?;
    backend.stream().synchronize()?;

    assert_close(
        "Add+RMSNorm output",
        &device_input.copy_to_vec()?,
        &expected_input,
        2.0e-5,
    )?;
    assert_close(
        "Add+RMSNorm residual",
        &device_residual.copy_to_vec()?,
        &expected_residual,
        1.0e-6,
    )?;
    Ok(())
}

fn validate_greedy_sample_logprobs(backend: &CudaBackend) -> Result<(), Box<dyn Error>> {
    let spec = GreedySampleLogprobsSpec::new(2, 5, DType::F32)?;
    let logits = vec![0.0, 2.0, -1.0, 0.5, 1.0, -3.0, -2.0, 4.0, 0.0, 1.0];
    let mut expected_tokens = vec![0_u32; spec.rows()];
    let mut expected_logprobs = vec![0.0_f32; spec.rows()];
    greedy_sample_logprobs_f32_reference(
        &logits,
        &mut expected_tokens,
        &mut expected_logprobs,
        spec,
    )?;

    let device_logits = DeviceBuffer::from_slice(&logits)?;
    let mut device_tokens = DeviceBuffer::<i32>::uninitialized(spec.rows())?;
    let mut device_logprobs = DeviceBuffer::<f32>::uninitialized(spec.rows())?;
    let mut device_ranks = DeviceBuffer::<i64>::uninitialized(spec.rows())?;
    backend.greedy_sample_logprobs_f32(
        &device_logits,
        &mut device_tokens,
        &mut device_logprobs,
        &mut device_ranks,
        spec,
        RowStridedLayout::contiguous(spec.vocab_size()),
    )?;
    backend.stream().synchronize()?;

    let actual_tokens = device_tokens.copy_to_vec()?;
    let expected_tokens: Vec<i32> = expected_tokens
        .into_iter()
        .map(|token| token as i32)
        .collect();
    if actual_tokens != expected_tokens {
        return Err(io::Error::other(format!(
            "greedy token mismatch: actual={actual_tokens:?}, expected={expected_tokens:?}"
        ))
        .into());
    }
    let actual_ranks = device_ranks.copy_to_vec()?;
    if actual_ranks != vec![1_i64; spec.rows()] {
        return Err(io::Error::other(format!(
            "greedy rank mismatch: actual={actual_ranks:?}, expected all ones"
        ))
        .into());
    }
    assert_close(
        "greedy sampled logprobs",
        &device_logprobs.copy_to_vec()?,
        &expected_logprobs,
        2.0e-5,
    )?;
    Ok(())
}

fn assert_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(io::Error::other(format!(
            "{label} length mismatch: actual={}, expected={}",
            actual.len(),
            expected.len()
        ))
        .into());
    }
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if (actual - expected).abs() > tolerance {
            return Err(io::Error::other(format!(
                "{label}[{index}] mismatch: actual={actual}, expected={expected}, tolerance={tolerance}"
            ))
            .into());
        }
    }
    Ok(())
}
