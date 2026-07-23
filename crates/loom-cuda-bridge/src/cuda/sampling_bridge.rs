//! Checked C bridge entrypoints for token selection and logprobs.

use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn launch_greedy_sample_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    token_ids: *mut i32,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) = unsafe { read_slice(logits, logits_elements, "greedy logits") }?;
    let (mut token_ids, token_ids_range) =
        unsafe { write_slice(token_ids, token_id_elements, "greedy token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "greedy logprobs") }?;
    let (mut ranks, ranks_range) = unsafe { write_slice(ranks, rank_elements, "greedy ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "greedy sampling",
    )?;
    let spec = GreedySampleLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "greedy row stride")?,
    )?;
    T::greedy_sample_logprobs(
        &stream_backend(stream),
        &logits,
        &mut token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_GREEDY_SAMPLE_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_selected_token_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) =
        unsafe { read_slice(logits, logits_elements, "selected-token logits") }?;
    let (token_ids, token_ids_range) =
        unsafe { read_slice(token_ids, token_id_elements, "selected-token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "selected-token logprobs") }?;
    let (mut ranks, ranks_range) =
        unsafe { write_slice(ranks, rank_elements, "selected-token ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "selected-token logprobs",
    )?;
    let spec = SelectedTokenLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "selected-token row stride")?,
    )?;
    T::selected_token_logprobs(
        &stream_backend(stream),
        &logits,
        &token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_SELECTED_TOKEN_LOGPROBS);
    Ok(())
}

/// Checked greedy selection, sampled-token logprob, and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *mut i32,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_greedy_sample_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked selected-token logprob and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_selected_token_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_selected_token_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

#[cfg(test)]
mod tests {
    use super::launch_greedy_sample_logprobs;
    use loom_cuda::CudaExecutorError;

    #[test]
    fn greedy_rejects_bad_storage_before_submission() {
        let result = unsafe {
            launch_greedy_sample_logprobs::<f32>(
                0x1000_usize as *const f32,
                7,
                0x2000_usize as *mut i32,
                2,
                0x3000_usize as *mut f32,
                2,
                0x4000_usize as *mut i64,
                2,
                2,
                4,
                4,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(result, Err(CudaExecutorError::InvalidContract(_))));
    }
}
