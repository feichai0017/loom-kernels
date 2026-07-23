use crate::rms_norm::CudaBackend;
use crate::runtime::{
    loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle, DeviceSlice,
};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{DType, SiluAndMulDynamicFp8Spec, SiluAndMulSpec};

/// Physical ordering of per-block FP8 scales.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Fp8ScaleLayout {
    #[default]
    RowMajor,
    GroupMajor,
}

/// Execution options for fused SiLU-and-Mul plus dynamic FP8.
#[derive(Clone, Copy, Debug, Default)]
pub struct SiluAndMulDynamicFp8Options<'a> {
    pub scale_upper_bound: Option<DeviceSlice<'a, f32>>,
    pub scale_layout: Fp8ScaleLayout,
}

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Launches vectorized F32 SiLU-and-Mul asynchronously on this stream.
    pub fn silu_and_mul_f32(
        &self,
        input: &impl CudaDeviceRead<f32>,
        output: &mut impl CudaDeviceWrite<f32>,
        spec: SiluAndMulSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let (rows, width) = validate_buffers(input, output, spec)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_silu_and_mul_f32(
                input.as_ptr(),
                output.as_mut_ptr(),
                rows,
                width,
                self.raw_stream(),
            )
        })
    }

    /// Launches vectorized FP16 SiLU-and-Mul asynchronously on this stream.
    pub fn silu_and_mul_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        output: &mut impl CudaDeviceWrite<f16>,
        spec: SiluAndMulSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let (rows, width) = validate_buffers(input, output, spec)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_silu_and_mul_f16(
                input.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                width,
                self.raw_stream(),
            )
        })
    }

    /// Launches vectorized BF16 SiLU-and-Mul asynchronously on this stream.
    pub fn silu_and_mul_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        output: &mut impl CudaDeviceWrite<bf16>,
        spec: SiluAndMulSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let (rows, width) = validate_buffers(input, output, spec)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_silu_and_mul_bf16(
                input.as_ptr().cast::<u16>(),
                output.as_mut_ptr().cast::<u16>(),
                rows,
                width,
                self.raw_stream(),
            )
        })
    }

    /// Launches fused FP16 SwiGLU and dynamic per-block FP8 asynchronously.
    pub fn silu_and_mul_dynamic_fp8_f16(
        &self,
        input: &impl CudaDeviceRead<f16>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: SiluAndMulDynamicFp8Spec,
        options: SiluAndMulDynamicFp8Options<'_>,
    ) -> Result<(), CudaExecutorError> {
        require_quant_dtype(spec, DType::F16)?;
        let (rows, width, group_size) = validate_quant_buffers(input, output, scales, spec)?;
        let scale_upper_bound = validate_options(options)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_silu_and_mul_dynamic_fp8_f16(
                input.as_ptr().cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                width,
                group_size,
                scale_upper_bound,
                u32::from(options.scale_layout == Fp8ScaleLayout::GroupMajor),
                self.raw_stream(),
            )
        })
    }

    /// Launches fused BF16 SwiGLU and dynamic per-block FP8 asynchronously.
    pub fn silu_and_mul_dynamic_fp8_bf16(
        &self,
        input: &impl CudaDeviceRead<bf16>,
        output: &mut impl CudaDeviceWrite<u8>,
        scales: &mut impl CudaDeviceWrite<f32>,
        spec: SiluAndMulDynamicFp8Spec,
        options: SiluAndMulDynamicFp8Options<'_>,
    ) -> Result<(), CudaExecutorError> {
        require_quant_dtype(spec, DType::Bf16)?;
        let (rows, width, group_size) = validate_quant_buffers(input, output, scales, spec)?;
        let scale_upper_bound = validate_options(options)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_silu_and_mul_dynamic_fp8_bf16(
                input.as_ptr().cast::<u16>(),
                output.as_mut_ptr(),
                scales.as_mut_ptr(),
                rows,
                width,
                group_size,
                scale_upper_bound,
                u32::from(options.scale_layout == Fp8ScaleLayout::GroupMajor),
                self.raw_stream(),
            )
        })
    }
}

fn validate_options(
    options: SiluAndMulDynamicFp8Options<'_>,
) -> Result<*const f32, CudaExecutorError> {
    match options.scale_upper_bound {
        Some(scale_upper_bound) => {
            scale_upper_bound.require_len(1, "SiLU-and-Mul+FP8 scale upper bound")?;
            Ok(scale_upper_bound.as_ptr())
        }
        None => Ok(std::ptr::null()),
    }
}

fn require_dtype(spec: SiluAndMulSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "{expected:?} SiLU-and-Mul cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    output: &impl CudaDeviceRead<T>,
    spec: SiluAndMulSpec,
) -> Result<(u32, u32), CudaExecutorError> {
    input.require_len(spec.input_numel(), "SiLU-and-Mul input")?;
    output.require_len(spec.output_numel(), "SiLU-and-Mul output")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("SiLU-and-Mul rows exceed the CUDA ABI".into())
    })?;
    let width = u32::try_from(spec.width()).map_err(|_| {
        CudaExecutorError::InvalidContract("SiLU-and-Mul width exceeds the CUDA ABI".into())
    })?;
    Ok((rows, width))
}

fn require_quant_dtype(
    spec: SiluAndMulDynamicFp8Spec,
    expected: DType,
) -> Result<(), CudaExecutorError> {
    if spec.input_dtype() == expected && spec.output_dtype() == DType::Fp8E4M3Fn {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "{expected:?} SiLU-and-Mul+FP8 cannot execute {:?} -> {:?}",
            spec.input_dtype(),
            spec.output_dtype()
        )))
    }
}

fn validate_quant_buffers<T: Copy>(
    input: &impl CudaDeviceRead<T>,
    output: &impl CudaDeviceRead<u8>,
    scales: &impl CudaDeviceRead<f32>,
    spec: SiluAndMulDynamicFp8Spec,
) -> Result<(u32, u32, u32), CudaExecutorError> {
    input.require_len(spec.input_numel(), "SiLU-and-Mul+FP8 input")?;
    output.require_len(spec.output_numel(), "SiLU-and-Mul+FP8 output")?;
    scales.require_len(spec.scale_count(), "SiLU-and-Mul+FP8 scales")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("SiLU-and-Mul+FP8 rows exceed the CUDA ABI".into())
    })?;
    let width = u32::try_from(spec.width()).map_err(|_| {
        CudaExecutorError::InvalidContract("SiLU-and-Mul+FP8 width exceeds the CUDA ABI".into())
    })?;
    let group_size = u32::try_from(spec.group_size()).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "SiLU-and-Mul+FP8 group size exceeds the CUDA ABI".into(),
        )
    })?;
    Ok((rows, width, group_size))
}
