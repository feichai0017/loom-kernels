//! Checked C bridge entrypoints for paged decode attention.

use super::*;

#[allow(clippy::too_many_arguments)]
fn paged_decode_spec(
    dtype: DType,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
) -> Result<PagedDecodeAttentionSpec, CudaExecutorError> {
    PagedDecodeAttentionSpec::new(
        sequences as usize,
        query_heads as usize,
        kv_heads as usize,
        head_size as usize,
        value_head_size as usize,
        num_blocks as usize,
        block_size as usize,
        max_blocks_per_sequence as usize,
        max_sequence_length as usize,
        scale,
        dtype,
    )
    .map_err(invalid_contract)
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_workspace_elements(
    dtype: DType,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
) -> Result<u64, CudaExecutorError> {
    let spec = paged_decode_spec(
        dtype,
        sequences,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        num_blocks,
        block_size,
        max_blocks_per_sequence,
        max_sequence_length,
        scale,
    )?;
    let elements = paged_decode_attention_split_k_workspace_elements(spec)?.unwrap_or(0);
    u64::try_from(elements).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "paged decode split-K workspace exceeds the bridge ABI".into(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_paged_decode_attention<T: Scalar>(
    query: *const T,
    query_elements: u64,
    key_cache: *const T,
    key_cache_elements: u64,
    value_cache: *const T,
    value_cache_elements: u64,
    block_tables: *const i32,
    block_table_elements: u64,
    sequence_lengths: *const i32,
    sequence_length_elements: u64,
    output: *mut T,
    output_elements: u64,
    workspace: *mut f32,
    workspace_elements: u64,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    key_block_stride: u64,
    value_block_stride: u64,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let spec = paged_decode_spec(
        T::DTYPE,
        sequences,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        num_blocks,
        block_size,
        max_blocks_per_sequence,
        max_sequence_length,
        scale,
    )?;
    let layout = PagedDecodeLayout::new(
        spec,
        element_count(key_block_stride, "paged key block stride")?,
        element_count(value_block_stride, "paged value block stride")?,
    )?;
    let expected_workspace = paged_decode_attention_split_k_workspace_elements(spec)?.unwrap_or(0);
    if element_count(workspace_elements, "paged decode workspace")? != expected_workspace {
        return Err(CudaExecutorError::InvalidContract(format!(
            "paged decode workspace has {workspace_elements} F32 elements, expected {expected_workspace}"
        )));
    }
    if expected_workspace == 0 && !workspace.is_null() {
        return Err(CudaExecutorError::InvalidContract(
            "base paged decode requires a null workspace pointer".into(),
        ));
    }
    if expected_workspace != 0 && workspace.is_null() {
        return Err(CudaExecutorError::InvalidContract(
            "split-K paged decode requires a workspace pointer".into(),
        ));
    }

    let (query, query_range) = unsafe { read_slice(query, query_elements, "paged decode query") }?;
    let (key_cache, key_cache_range) =
        unsafe { read_slice(key_cache, key_cache_elements, "paged decode key cache") }?;
    let (value_cache, value_cache_range) = unsafe {
        read_slice(
            value_cache,
            value_cache_elements,
            "paged decode value cache",
        )
    }?;
    let (block_tables, block_tables_range) = unsafe {
        read_slice(
            block_tables,
            block_table_elements,
            "paged decode block tables",
        )
    }?;
    let (sequence_lengths, sequence_lengths_range) = unsafe {
        read_slice(
            sequence_lengths,
            sequence_length_elements,
            "paged decode sequence lengths",
        )
    }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "paged decode output") }?;
    let inputs = [
        ("query", query_range),
        ("key cache", key_cache_range),
        ("value cache", value_cache_range),
        ("block tables", block_tables_range),
        ("sequence lengths", sequence_lengths_range),
    ];
    require_disjoint_from("output", output_range, &inputs, "paged decode")?;

    if expected_workspace == 0 {
        T::paged_decode_attention(
            &stream_backend(stream),
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &mut output,
            spec,
            layout,
        )?;
    } else {
        let (mut workspace, workspace_range) = unsafe {
            write_slice(
                workspace,
                workspace_elements,
                "paged decode split-K workspace",
            )
        }?;
        require_disjoint_from(
            "workspace",
            workspace_range,
            &[
                ("query", query_range),
                ("key cache", key_cache_range),
                ("value cache", value_cache_range),
                ("block tables", block_tables_range),
                ("sequence lengths", sequence_lengths_range),
                ("output", output_range),
            ],
            "paged decode",
        )?;
        T::paged_decode_attention_split_k(
            &stream_backend(stream),
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &mut output,
            &mut workspace,
            spec,
            layout,
        )?;
    }
    record_launch(OP_PAGED_DECODE_ATTENTION);
    Ok(())
}
/// Return the exact caller-owned F32 workspace required by paged decode.
///
/// # Safety
///
/// `workspace_elements` must be a valid writable host pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_paged_decode_workspace_elements(
    dtype: u32,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    workspace_elements: *mut u64,
) -> c_int {
    bridge_call(|| {
        if workspace_elements.is_null()
            || !(workspace_elements as usize).is_multiple_of(align_of::<u64>())
        {
            return Err(CudaExecutorError::InvalidContract(
                "workspace-size output pointer is null or misaligned".into(),
            ));
        }
        let dtype = match scalar_kind(dtype)? {
            ScalarKind::F32 => DType::F32,
            ScalarKind::F16 => DType::F16,
            ScalarKind::Bf16 => DType::Bf16,
        };
        let elements = paged_decode_workspace_elements(
            dtype,
            sequences,
            query_heads,
            kv_heads,
            head_size,
            value_head_size,
            num_blocks,
            block_size,
            max_blocks_per_sequence,
            max_sequence_length,
            scale,
        )?;
        unsafe {
            *workspace_elements = elements;
        }
        Ok(())
    })
}

/// Checked paged MQA/GQA decode. A null, zero-length workspace selects the
/// base path; the exact non-zero workspace selects split-K.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_paged_decode_attention(
    dtype: u32,
    query: *const c_void,
    query_elements: u64,
    key_cache: *const c_void,
    key_cache_elements: u64,
    value_cache: *const c_void,
    value_cache_elements: u64,
    block_tables: *const i32,
    block_table_elements: u64,
    sequence_lengths: *const i32,
    sequence_length_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    workspace: *mut f32,
    workspace_elements: u64,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    key_block_stride: u64,
    value_block_stride: u64,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_paged_decode_attention(
                query.cast(),
                query_elements,
                key_cache.cast(),
                key_cache_elements,
                value_cache.cast(),
                value_cache_elements,
                block_tables,
                block_table_elements,
                sequence_lengths,
                sequence_length_elements,
                output.cast(),
                output_elements,
                workspace,
                workspace_elements,
                sequences,
                query_heads,
                kv_heads,
                head_size,
                value_head_size,
                num_blocks,
                block_size,
                key_block_stride,
                value_block_stride,
                max_blocks_per_sequence,
                max_sequence_length,
                scale,
                stream,
            )
        )
    })
}
