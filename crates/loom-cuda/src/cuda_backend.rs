//! Concrete CUDA backend and capability admission for Loom operator contracts.

use crate::runtime::{CudaStream, CudaStreamHandle};
use crate::CudaExecutorError;
use loom_kernels::{Backend, DType, OperatorSpec, Support};

/// CUDA backend bound to an owned stream by default or a borrowed stream when
/// constructed with [`CudaBackend::from_stream`].
#[derive(Debug)]
pub struct CudaBackend<S = CudaStream> {
    stream: S,
}

impl CudaBackend<CudaStream> {
    pub fn new() -> Result<Self, CudaExecutorError> {
        Ok(Self {
            stream: CudaStream::new()?,
        })
    }
}

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Uses an existing owned or borrowed stream handle without allocating a
    /// second execution stream.
    pub const fn from_stream(stream: S) -> Self {
        Self { stream }
    }

    pub const fn stream(&self) -> &S {
        &self.stream
    }

    pub(crate) fn raw_stream(&self) -> *mut std::ffi::c_void {
        self.stream.raw()
    }
}

impl<S: CudaStreamHandle> Backend for CudaBackend<S> {
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
            OperatorSpec::GreedySpeculativeVerify(_) => Support::Supported,
            OperatorSpec::PagedDecodeAttention(spec)
                if crate::attention_dispatch::supports_spec(*spec) =>
            {
                Support::Supported
            }
            OperatorSpec::PagedDecodeAttention(spec)
                if !matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16) =>
            {
                Support::Unsupported(
                    "CUDA paged decode attention supports F32, FP16, and BF16 native caches",
                )
            }
            OperatorSpec::PagedDecodeAttention(spec)
                if spec.max_sequence_length()
                    > crate::attention_dispatch::PAGED_DECODE_MAX_CONTEXT =>
            {
                Support::Unsupported(
                    "CUDA paged decode attention supports at most 1024 context tokens",
                )
            }
            OperatorSpec::PagedDecodeAttention(_) => {
                Support::Unsupported("paged decode attention shape exceeds the CUDA ABI")
            }
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
