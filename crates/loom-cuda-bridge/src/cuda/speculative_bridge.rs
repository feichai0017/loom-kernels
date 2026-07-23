//! Checked C bridge entrypoints for speculative verification.

use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn launch_greedy_speculative_verify(
    draft_token_ids: *const i32,
    draft_token_id_elements: u64,
    target_token_ids: *const i64,
    target_token_id_elements: u64,
    bonus_token_ids: *const i32,
    bonus_token_id_elements: u64,
    cumulative_draft_lengths: *const i32,
    cumulative_draft_length_elements: u64,
    output_token_ids: *mut i32,
    output_token_id_elements: u64,
    accepted_lengths: *mut i32,
    accepted_length_elements: u64,
    emitted_lengths: *mut i32,
    emitted_length_elements: u64,
    requests: u32,
    draft_tokens: u32,
    max_draft_tokens: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (draft_token_ids, draft_range) = unsafe {
        read_slice(
            draft_token_ids,
            draft_token_id_elements,
            "speculative draft token IDs",
        )
    }?;
    let (target_token_ids, target_range) = unsafe {
        read_slice(
            target_token_ids,
            target_token_id_elements,
            "speculative target token IDs",
        )
    }?;
    let (bonus_token_ids, bonus_range) = unsafe {
        read_slice(
            bonus_token_ids,
            bonus_token_id_elements,
            "speculative bonus token IDs",
        )
    }?;
    let (cumulative_draft_lengths, cumulative_range) = unsafe {
        read_slice(
            cumulative_draft_lengths,
            cumulative_draft_length_elements,
            "cumulative draft lengths",
        )
    }?;
    let (mut output_token_ids, output_range) = unsafe {
        write_slice(
            output_token_ids,
            output_token_id_elements,
            "speculative output token IDs",
        )
    }?;
    let (mut accepted_lengths, accepted_range) = unsafe {
        write_slice(
            accepted_lengths,
            accepted_length_elements,
            "speculative accepted lengths",
        )
    }?;
    let (mut emitted_lengths, emitted_range) = unsafe {
        write_slice(
            emitted_lengths,
            emitted_length_elements,
            "speculative emitted lengths",
        )
    }?;
    require_disjoint(
        &[
            ("draft token IDs", draft_range),
            ("target token IDs", target_range),
            ("bonus token IDs", bonus_range),
            ("cumulative draft lengths", cumulative_range),
            ("output token IDs", output_range),
            ("accepted lengths", accepted_range),
            ("emitted lengths", emitted_range),
        ],
        "greedy speculative verification",
    )?;
    let spec = GreedySpeculativeVerifySpec::new(
        requests as usize,
        draft_tokens as usize,
        max_draft_tokens as usize,
    )
    .map_err(invalid_contract)?;
    stream_backend(stream).greedy_speculative_verify(
        &draft_token_ids,
        &target_token_ids,
        &bonus_token_ids,
        &cumulative_draft_lengths,
        &mut output_token_ids,
        &mut accepted_lengths,
        &mut emitted_lengths,
        spec,
    )?;
    record_launch(OP_GREEDY_SPECULATIVE_VERIFY);
    Ok(())
}

/// Checked deterministic greedy speculative verification.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_greedy_speculative_verify(
    draft_token_ids: *const i32,
    draft_token_id_elements: u64,
    target_token_ids: *const i64,
    target_token_id_elements: u64,
    bonus_token_ids: *const i32,
    bonus_token_id_elements: u64,
    cumulative_draft_lengths: *const i32,
    cumulative_draft_length_elements: u64,
    output_token_ids: *mut i32,
    output_token_id_elements: u64,
    accepted_lengths: *mut i32,
    accepted_length_elements: u64,
    emitted_lengths: *mut i32,
    emitted_length_elements: u64,
    requests: u32,
    draft_tokens: u32,
    max_draft_tokens: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| unsafe {
        launch_greedy_speculative_verify(
            draft_token_ids,
            draft_token_id_elements,
            target_token_ids,
            target_token_id_elements,
            bonus_token_ids,
            bonus_token_id_elements,
            cumulative_draft_lengths,
            cumulative_draft_length_elements,
            output_token_ids,
            output_token_id_elements,
            accepted_lengths,
            accepted_length_elements,
            emitted_lengths,
            emitted_length_elements,
            requests,
            draft_tokens,
            max_draft_tokens,
            stream,
        )
    })
}
