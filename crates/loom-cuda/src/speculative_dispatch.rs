//! Safe CUDA dispatch for speculative-verification contracts.

use crate::cuda_backend::CudaBackend;
use crate::runtime::{loom_status_result, CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle};
use crate::CudaExecutorError;
use loom_kernels::GreedySpeculativeVerifySpec;

impl<S: CudaStreamHandle> CudaBackend<S> {
    /// Verifies flattened ragged greedy draft tokens and compacts output rows.
    #[allow(clippy::too_many_arguments)]
    pub fn greedy_speculative_verify(
        &self,
        draft_token_ids: &impl CudaDeviceRead<i32>,
        target_token_ids: &impl CudaDeviceRead<i64>,
        bonus_token_ids: &impl CudaDeviceRead<i32>,
        cumulative_draft_lengths: &impl CudaDeviceRead<i32>,
        output_token_ids: &mut impl CudaDeviceWrite<i32>,
        accepted_lengths: &mut impl CudaDeviceWrite<i32>,
        emitted_lengths: &mut impl CudaDeviceWrite<i32>,
        spec: GreedySpeculativeVerifySpec,
    ) -> Result<(), CudaExecutorError> {
        draft_token_ids.require_len(spec.draft_tokens(), "speculative draft token IDs")?;
        target_token_ids.require_len(spec.draft_tokens(), "speculative target token IDs")?;
        bonus_token_ids.require_len(spec.requests(), "speculative bonus token IDs")?;
        cumulative_draft_lengths.require_len(spec.requests(), "cumulative draft lengths")?;
        output_token_ids.require_len(spec.output_numel(), "speculative output token IDs")?;
        accepted_lengths.require_len(spec.requests(), "speculative accepted lengths")?;
        emitted_lengths.require_len(spec.requests(), "speculative emitted lengths")?;

        let requests = u32::try_from(spec.requests()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "speculative request count exceeds the CUDA ABI".into(),
            )
        })?;
        let draft_tokens = u32::try_from(spec.draft_tokens()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "speculative draft token count exceeds the CUDA ABI".into(),
            )
        })?;
        let max_draft_tokens = u32::try_from(spec.max_draft_tokens()).map_err(|_| {
            CudaExecutorError::InvalidContract(
                "speculative draft width exceeds the CUDA ABI".into(),
            )
        })?;

        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_greedy_speculative_verify(
                draft_token_ids.as_ptr(),
                target_token_ids.as_ptr(),
                bonus_token_ids.as_ptr(),
                cumulative_draft_lengths.as_ptr(),
                output_token_ids.as_mut_ptr(),
                accepted_lengths.as_mut_ptr(),
                emitted_lengths.as_mut_ptr(),
                requests,
                draft_tokens,
                max_draft_tokens,
                self.raw_stream(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DeviceBuffer;
    use loom_kernels::greedy_speculative_verify_reference;

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = GreedySpeculativeVerifySpec::new(3, 7, 4).unwrap();
        let draft = [10_i32, 11, 12, 20, 21, 22, 23];
        let target = [10_i64, 99, 12, 20, 21, 22, 23];
        let bonus = [100_i32, 200, 300];
        let cumulative = [3_i32, 3, 7];
        let mut expected_output = [0_i32; 15];
        let mut expected_accepted = [0_i32; 3];
        let mut expected_emitted = [0_i32; 3];
        greedy_speculative_verify_reference(
            &draft,
            &target,
            &bonus,
            &cumulative,
            &mut expected_output,
            &mut expected_accepted,
            &mut expected_emitted,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let draft = DeviceBuffer::from_slice(&draft).unwrap();
        let target = DeviceBuffer::from_slice(&target).unwrap();
        let bonus = DeviceBuffer::from_slice(&bonus).unwrap();
        let cumulative = DeviceBuffer::from_slice(&cumulative).unwrap();
        let mut output = DeviceBuffer::from_slice(&[0_i32; 15]).unwrap();
        let mut accepted = DeviceBuffer::from_slice(&[0_i32; 3]).unwrap();
        let mut emitted = DeviceBuffer::from_slice(&[0_i32; 3]).unwrap();
        backend
            .greedy_speculative_verify(
                &draft,
                &target,
                &bonus,
                &cumulative,
                &mut output,
                &mut accepted,
                &mut emitted,
                spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();

        assert_eq!(output.copy_to_vec().unwrap(), expected_output);
        assert_eq!(accepted.copy_to_vec().unwrap(), expected_accepted);
        assert_eq!(emitted.copy_to_vec().unwrap(), expected_emitted);
    }
}
