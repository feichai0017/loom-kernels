use crate::rms_norm::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::{CudaExecutorError, RowStridedLayout};
use half::{bf16, f16};
use loom_kernels::{DType, GreedySampleLogprobsSpec, SelectedTokenLogprobsSpec};

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Fuses F32 greedy selection with the sampled token's logprob and rank.
    pub fn greedy_sample_logprobs_f32(
        &self,
        logits: &impl CudaDeviceRead<f32>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_f32(
                logits.as_ptr(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Fuses FP16 greedy selection with an F32 sampled-token logprob.
    pub fn greedy_sample_logprobs_f16(
        &self,
        logits: &impl CudaDeviceRead<f16>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_f16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Fuses BF16 greedy selection with an F32 sampled-token logprob.
    pub fn greedy_sample_logprobs_bf16(
        &self,
        logits: &impl CudaDeviceRead<bf16>,
        token_ids: &mut impl CudaDeviceWrite<i32>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) =
            validate_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_sample_logprobs_bf16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_mut_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from F32 logits.
    pub fn selected_token_logprobs_f32(
        &self,
        logits: &impl CudaDeviceRead<f32>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::F32)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_f32(
                logits.as_ptr(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from FP16 logits.
    pub fn selected_token_logprobs_f16(
        &self,
        logits: &impl CudaDeviceRead<f16>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::F16)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_f16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }

    /// Computes selected-token F32 logprobs and ranks from BF16 logits.
    pub fn selected_token_logprobs_bf16(
        &self,
        logits: &impl CudaDeviceRead<bf16>,
        token_ids: &impl CudaDeviceRead<i64>,
        logprobs: &mut impl CudaDeviceWrite<f32>,
        ranks: &mut impl CudaDeviceWrite<i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError> {
        require_selected_dtype(spec, DType::Bf16)?;
        let (rows, vocab_size, row_stride) =
            validate_selected_buffers(logits, token_ids, logprobs, ranks, spec, layout)?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_selected_token_logprobs_bf16(
                logits.as_ptr().cast::<u16>(),
                token_ids.as_ptr(),
                logprobs.as_mut_ptr(),
                ranks.as_mut_ptr(),
                rows,
                vocab_size,
                row_stride,
                self.raw_stream(),
            )
        })
    }
}

fn require_dtype(spec: GreedySampleLogprobsSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "greedy sampling for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    token_ids: &impl CudaDeviceRead<i32>,
    logprobs: &impl CudaDeviceRead<f32>,
    ranks: &impl CudaDeviceRead<i64>,
    spec: GreedySampleLogprobsSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "greedy-sampling logits",
    )?;
    token_ids.require_len(spec.rows(), "greedy-sampling token IDs")?;
    logprobs.require_len(spec.rows(), "greedy-sampling logprobs")?;
    ranks.require_len(spec.rows(), "greedy-sampling ranks")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling rows exceed the CUDA ABI".into())
    })?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling vocabulary exceeds the CUDA ABI".into())
    })?;
    if vocab_size > i32::MAX as u32 {
        return Err(CudaExecutorError::InvalidContract(
            "greedy-sampling vocabulary exceeds int32 token IDs".into(),
        ));
    }
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("greedy-sampling row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

fn require_selected_dtype(
    spec: SelectedTokenLogprobsSpec,
    expected: DType,
) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "selected-token logprobs for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

fn validate_selected_buffers<T: Copy>(
    logits: &impl CudaDeviceRead<T>,
    token_ids: &impl CudaDeviceRead<i64>,
    logprobs: &impl CudaDeviceRead<f32>,
    ranks: &impl CudaDeviceRead<i64>,
    spec: SelectedTokenLogprobsSpec,
    layout: RowStridedLayout,
) -> Result<(u32, u32, u64), CudaExecutorError> {
    logits.require_len(
        layout.storage_elements(spec.rows(), spec.vocab_size())?,
        "selected-token logits",
    )?;
    token_ids.require_len(spec.rows(), "selected token IDs")?;
    logprobs.require_len(spec.rows(), "selected-token logprobs")?;
    ranks.require_len(spec.rows(), "selected-token ranks")?;
    let rows = u32::try_from(spec.rows()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token rows exceed the CUDA ABI".into())
    })?;
    let vocab_size = u32::try_from(spec.vocab_size()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token vocabulary exceeds the CUDA ABI".into())
    })?;
    let row_stride = u64::try_from(layout.row_stride()).map_err(|_| {
        CudaExecutorError::InvalidContract("selected-token row stride exceeds the CUDA ABI".into())
    })?;
    Ok((rows, vocab_size, row_stride))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::{
        greedy_sample_logprobs_f32_reference, selected_token_logprobs_f32_reference,
    };

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = GreedySampleLogprobsSpec::new(2, 5, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, 0.5, -2.0, -1.0, 2.0, 0.0, 1.0];
        let mut expected_ids = [u32::MAX; 2];
        let mut expected_logprobs = [0.0_f32; 2];
        greedy_sample_logprobs_f32_reference(
            &logits,
            &mut expected_ids,
            &mut expected_logprobs,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let logits_device = DeviceBuffer::from_slice(&logits).unwrap();
        let mut ids_device = DeviceBuffer::from_slice(&[-1_i32; 2]).unwrap();
        let mut logprobs_device = DeviceBuffer::from_slice(&[0.0_f32; 2]).unwrap();
        let mut ranks_device = DeviceBuffer::from_slice(&[0_i64; 2]).unwrap();
        backend
            .greedy_sample_logprobs_f32(
                &logits_device,
                &mut ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        let actual_ids = ids_device.copy_to_vec().unwrap();
        assert_eq!(actual_ids, expected_ids.map(|value| value as i32));
        for (actual, expected) in logprobs_device
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_logprobs)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(ranks_device.copy_to_vec().unwrap(), vec![2_i64, 1_i64]);
    }

    #[test]
    fn selected_token_wrapper_matches_the_cpu_oracle() {
        let spec = SelectedTokenLogprobsSpec::new(2, 5, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, 0.5, -2.0, -1.0, 2.0, 0.0, 1.0];
        let token_ids = [0_i64, 4_i64];
        let mut expected_logprobs = [0.0_f32; 2];
        let mut expected_ranks = [0_i64; 2];
        selected_token_logprobs_f32_reference(
            &logits,
            &token_ids,
            &mut expected_logprobs,
            &mut expected_ranks,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let logits_device = DeviceBuffer::from_slice(&logits).unwrap();
        let ids_device = DeviceBuffer::from_slice(&token_ids).unwrap();
        let mut logprobs_device = DeviceBuffer::from_slice(&[0.0_f32; 2]).unwrap();
        let mut ranks_device = DeviceBuffer::from_slice(&[0_i64; 2]).unwrap();
        backend
            .selected_token_logprobs_f32(
                &logits_device,
                &ids_device,
                &mut logprobs_device,
                &mut ranks_device,
                spec,
                RowStridedLayout::contiguous(spec.vocab_size()),
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        for (actual, expected) in logprobs_device
            .copy_to_vec()
            .unwrap()
            .iter()
            .zip(expected_logprobs)
        {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        assert_eq!(ranks_device.copy_to_vec().unwrap(), expected_ranks);
    }
}
