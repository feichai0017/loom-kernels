use crate::runtime::{loom_status_result, CudaStream, DeviceBuffer};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{
    AddRmsNormSpec, Backend, DType, OperatorSpec, RmsNormDynamicFp8Spec, RmsNormSpec, Support,
};

/// CUDA backend bound to one owned execution stream.
#[derive(Debug)]
pub struct CudaBackend {
    stream: CudaStream,
}

impl CudaBackend {
    pub fn new() -> Result<Self, CudaExecutorError> {
        Ok(Self {
            stream: CudaStream::new()?,
        })
    }

    pub const fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Launches F32 RMSNorm asynchronously on this backend's stream.
    pub fn rms_norm_f32(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
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
                self.stream.raw(),
            )
        })
    }

    /// Launches pair-vectorized FP16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_f16(
        &self,
        input: &DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<f16>,
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
                self.stream.raw(),
            )
        })
    }

    /// Launches pair-vectorized BF16 RMSNorm asynchronously on this stream.
    pub fn rms_norm_bf16(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses F32 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f32(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses FP16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_f16(
        &self,
        input: &DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses BF16 RMSNorm with dynamic per-token FP8 E4M3FN quantization.
    pub fn rms_norm_dynamic_fp8_bf16(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<f32>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses F32 residual addition and RMSNorm, updating both buffers in place.
    pub fn add_rms_norm_f32(
        &self,
        input: &mut DeviceBuffer<f32>,
        residual: &mut DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses pair-vectorized FP16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_f16(
        &self,
        input: &mut DeviceBuffer<f16>,
        residual: &mut DeviceBuffer<f16>,
        weight: &DeviceBuffer<f16>,
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
                self.stream.raw(),
            )
        })
    }

    /// Fuses pair-vectorized BF16 residual addition and RMSNorm in place.
    pub fn add_rms_norm_bf16(
        &self,
        input: &mut DeviceBuffer<bf16>,
        residual: &mut DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
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
                self.stream.raw(),
            )
        })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &'static str {
        "loom-cuda"
    }

    fn supports(&self, operation: &OperatorSpec) -> Support {
        match operation {
            OperatorSpec::RmsNorm(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::RmsNorm(_) => {
                Support::Unsupported("CUDA RMSNorm supports F32, FP16, and BF16")
            }
            OperatorSpec::AddRmsNorm(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::AddRmsNorm(_) => {
                Support::Unsupported("CUDA Add+RMSNorm supports F32, FP16, and BF16")
            }
            OperatorSpec::RmsNormDynamicFp8(spec)
                if matches!(spec.input_dtype(), DType::F32 | DType::F16 | DType::Bf16)
                    && spec.output_dtype() == DType::Fp8E4M3Fn =>
            {
                Support::Supported
            }
            OperatorSpec::RmsNormDynamicFp8(_) => {
                Support::Unsupported("CUDA dynamic FP8 RMSNorm supports F32, FP16, and BF16 inputs")
            }
            OperatorSpec::SiluAndMul(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::SiluAndMul(_) => {
                Support::Unsupported("CUDA SiLU-and-Mul supports F32, FP16, and BF16")
            }
            OperatorSpec::SiluAndMulDynamicFp8(spec)
                if matches!(spec.input_dtype(), DType::F16 | DType::Bf16)
                    && spec.output_dtype() == DType::Fp8E4M3Fn =>
            {
                Support::Supported
            }
            OperatorSpec::SiluAndMulDynamicFp8(_) => {
                Support::Unsupported("CUDA SiLU-and-Mul+FP8 supports FP16 and BF16 inputs")
            }
            OperatorSpec::GreedySampleLogprobs(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::GreedySampleLogprobs(_) => {
                Support::Unsupported("CUDA greedy sampling supports F32, FP16, and BF16 logits")
            }
            OperatorSpec::SelectedTokenLogprobs(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::SelectedTokenLogprobs(_) => Support::Unsupported(
                "CUDA selected-token logprobs support F32, FP16, and BF16 logits",
            ),
            OperatorSpec::MinPFilter(spec)
                if matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::MinPFilter(_) => {
                Support::Unsupported("CUDA min-p filtering supports F32, FP16, and BF16 logits")
            }
            OperatorSpec::PagedDecodeAttention(_) => Support::Unsupported(
                "CUDA paged decode attention is not implemented yet; only the Rust contract and CPU oracle are available",
            ),
            OperatorSpec::RotaryEmbedding(_) => Support::Unsupported(
                "standalone CUDA RoPE is not exposed yet; use the fused RoPE+paged-KV contract",
            ),
            OperatorSpec::RopePagedKvWrite(spec)
                if matches!(spec.rotary().dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Supported
            }
            OperatorSpec::RopePagedKvWrite(_) => Support::Unsupported(
                "CUDA RoPE+paged-KV supports F32, FP16, and BF16 native caches",
            ),
        }
    }
}

fn validate_buffers<T: Copy>(
    input: &DeviceBuffer<T>,
    weight: &DeviceBuffer<T>,
    output: &DeviceBuffer<T>,
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
    input: &DeviceBuffer<T>,
    residual: &DeviceBuffer<T>,
    weight: &DeviceBuffer<T>,
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
    input: &DeviceBuffer<T>,
    weight: &DeviceBuffer<T>,
    output: &DeviceBuffer<u8>,
    scales: &DeviceBuffer<f32>,
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
