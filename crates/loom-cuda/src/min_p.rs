use crate::rms_norm::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::{CudaExecutorError, RowStridedLayout};
use half::{bf16, f16};
use loom_kernels::{DType, MinPFilterSpec};

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Applies in-place F32 min-p filtering without materializing softmax.
    pub fn min_p_filter_f32(
        &self,
        logits: &mut impl CudaDeviceWrite<f32>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_f32(
                logits.as_mut_ptr(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies in-place FP16 min-p filtering without materializing softmax.
    pub fn min_p_filter_f16(
        &self,
        logits: &mut impl CudaDeviceWrite<f16>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_f16(
                logits.as_mut_ptr().cast::<u16>(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Applies in-place BF16 min-p filtering without materializing softmax.
    pub fn min_p_filter_bf16(
        &self,
        logits: &mut impl CudaDeviceWrite<bf16>,
        min_p: &impl CudaDeviceRead<f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) = validate_buffers(logits, min_p, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_min_p_filter_bf16(
                logits.as_mut_ptr().cast::<u16>(),
                min_p.as_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }
}

fn require_dtype(spec: MinPFilterSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "min-p filtering for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    min_p: &impl CudaDeviceRead<f32>,
    spec: MinPFilterSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "min-p logits",
    )?;
    min_p.require_len(spec.rows(), "min-p probabilities")?;
    let rows = u32::try_from(spec.rows())
        .map_err(|_| CudaExecutorError::InvalidContract("min-p rows exceed the CUDA ABI".into()))?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("min-p vocabulary exceeds the CUDA ABI".into())
    })?;
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("min-p row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::min_p_filter_f32_reference;

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = MinPFilterSpec::new(3, 5, DType::F32).unwrap();
        let original = [
            1.0_f32, 3.0, 2.0, -1.0, 0.5, //
            -2.0, -1.0, 2.0, 0.0, 1.0, //
            4.0, 4.0, 3.0, -8.0, 0.0,
        ];
        let probabilities = [0.0_f32, 0.2, 1.0];
        let mut expected = original;
        min_p_filter_f32_reference(&mut expected, &probabilities, spec).unwrap();

        let backend = CudaBackend::new().unwrap();
        let mut logits = DeviceBuffer::from_slice(&original).unwrap();
        let min_p = DeviceBuffer::from_slice(&probabilities).unwrap();
        backend
            .min_p_filter_f32(
                &mut logits,
                &min_p,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(logits.copy_to_vec().unwrap(), expected);
    }
}
