//! Safe CUDA dispatch for normalization contracts.

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{AddRmsNormSpec, DType, RmsNormDynamicFp8Spec, RmsNormSpec};

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Launches F32 RMSNorm asynchronously on this backend's stream.
    pub fn rms_norm_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        weight: &impl CudaDeviceRead<f32>,
        output: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_f32(
                input.as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Launches pair-vectorized FP16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        weight: &impl CudaDeviceRead<f16>,
        output: &mut impl CudaDeviceWrite<f16>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_f16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Launches pair-vectorized BF16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        output: &mut impl CudaDeviceWrite<bf16>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_buffers(input, weight, output, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_bf16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        weight: &impl CudaDeviceRead<f32>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let (rows, hidden_size) =
            validate_rms_norm_dynamic_fp8_buffers(input, weight, output, scales, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_f32(
                input.as_ptr(),
                weight.as_ptr(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        weight: &impl CudaDeviceRead<f16>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let (rows, hidden_size) =
            validate_rms_norm_dynamic_fp8_buffers(input, weight, output, scales, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_f16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError> {
        if spec.input_dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 RMSNorm+FP8 cannot execute {:?}",
                spec.input_dtype()
            )));
        }
        let (rows, hidden_size) =
            validate_rms_norm_dynamic_fp8_buffers(input, weight, output, scales, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_rms_norm_dynamic_fp8_bf16(
                input.as_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses F32 residual addition and RMSNorm, updating both buffers in place.
    pub fn add_rms_norm_f32(
        &self,
        input: &mut impl CudaDeviceWrite<f32>,
        residual: &mut impl CudaDeviceWrite<f32>,
        weight: &impl CudaDeviceRead<f32>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F32 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "F32 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_f32(
                input.as_mut_ptr(),
                residual.as_mut_ptr(),
                weight.as_ptr(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses pair-vectorized FP16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_f16(
        &self,
        input: &mut impl CudaDeviceWrite<f16>,
        residual: &mut impl CudaDeviceWrite<f16>,
        weight: &impl CudaDeviceRead<f16>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::F16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "FP16 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_f16(
                input.as_mut_ptr().cast::<u16>(),
                residual.as_mut_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }

    /// Fuses pair-vectorized BF16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_bf16(
        &self,
        input: &mut impl CudaDeviceWrite<bf16>,
        residual: &mut impl CudaDeviceWrite<bf16>,
        weight: &impl CudaDeviceRead<bf16>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError> {
        if spec.dtype() != DType::Bf16 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "BF16 Add+RMSNorm cannot execute {:?}",
                spec.dtype()
            )));
        }
        let (rows, hidden_size) = validate_add_rms_norm_buffers(input, residual, weight, spec)?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_add_rms_norm_bf16(
                input.as_mut_ptr().cast::<u16>(),
                residual.as_mut_ptr().cast::<u16>(),
                weight.as_ptr().cast::<u16>(),
                rows,
                hidden_size,
                spec.epsilon(),
                self.raw_stream(),
            )
        })
    }
}

fn validate_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    output: &impl CudaDeviceRead<T>,
    spec: RmsNormSpec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "RMSNorm input")?;
    weight.require_len(spec.hidden_size(), "RMSNorm weight")?;
    output.require_len(spec.numel(), "RMSNorm output")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

fn validate_add_rms_norm_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    residual: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    spec: AddRmsNormSpec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "Add+RMSNorm input")?;
    residual.require_len(spec.numel(), "Add+RMSNorm residual")?;
    weight.require_len(spec.hidden_size(), "Add+RMSNorm weight")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("Add+RMSNorm rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("Add+RMSNorm hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}

fn validate_rms_norm_dynamic_fp8_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    weight: &impl CudaDeviceRead<T>,
    output: &impl CudaDeviceRead<u8>,
    scales: &impl CudaDeviceRead<f32>,
    spec: RmsNormDynamicFp8Spec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.numel(), "RMSNorm+FP8 input")?;
    weight.require_len(spec.hidden_size(), "RMSNorm+FP8 weight")?;
    output.require_len(spec.numel(), "RMSNorm+FP8 output")?;
    scales.require_len(spec.scale_count(), "RMSNorm+FP8 scales")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+FP8 rows exceed the CUDA ABI".into())
    })?;
    let hidden_size = u32::try_from(spec.hidden_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("RMSNorm+FP8 hidden size exceeds the CUDA ABI".into())
    })?;
    Ok((rows, hidden_size))
}
