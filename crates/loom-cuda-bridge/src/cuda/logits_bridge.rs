//! Checked C bridge entrypoints for logits processing.

use super::*;

#[allow(clippy::too_many_arguments)]
unsafe fn launch_min_p_filter<T: Scalar>(
    logits: *mut T,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut logits, logits_range) =
        unsafe { write_slice(logits, logits_elements, "min-p logits") }?;
    let (min_p, min_p_range) = unsafe { read_slice(min_p, min_p_elements, "min-p values") }?;
    require_disjoint(
        &[("logits", logits_range), ("min-p", min_p_range)],
        "min-p filtering",
    )?;
    let spec = MinPFilterSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "min-p row stride")?,
    )?;
    T::min_p_filter(&stream_backend(stream), &mut logits, &min_p, spec, layout)?;
    record_launch(OP_MIN_P_FILTER);
    Ok(())
}

/// Checked in-place min-p filtering.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_min_p_filter(
    dtype: u32,
    logits: *mut c_void,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_min_p_filter(
                logits.cast(),
                logits_elements,
                min_p,
                min_p_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}
