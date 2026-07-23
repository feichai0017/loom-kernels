//! Checked C bridge entrypoints for rotary embedding and paged-KV writes.

use super::*;

unsafe fn launch_rope_paged_kv_write<T: Scalar>(
    query: *mut T,
    query_elements: u64,
    key: *mut T,
    key_elements: u64,
    value: *const T,
    value_elements: u64,
    positions: *const i64,
    position_elements: u64,
    cos_sin_cache: *const T,
    cos_sin_cache_elements: u64,
    key_cache: *mut T,
    key_cache_elements: u64,
    value_cache: *mut T,
    value_cache_elements: u64,
    slot_mapping: *const i64,
    slot_mapping_elements: u64,
    tokens: u32,
    cache_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    rotary_dim: u32,
    max_position: u32,
    num_blocks: u32,
    block_size: u32,
    query_token_stride: u64,
    query_head_stride: u64,
    key_token_stride: u64,
    source_key_head_stride: u64,
    value_token_stride: u64,
    source_value_head_stride: u64,
    key_block_stride: u64,
    key_page_stride: u64,
    key_head_stride: u64,
    value_block_stride: u64,
    value_page_stride: u64,
    value_cache_head_stride: u64,
    is_neox: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let style = match is_neox {
        0 => RotaryStyle::Interleaved,
        1 => RotaryStyle::NeoX,
        _ => {
            return Err(CudaExecutorError::InvalidContract(
                "RoPE style flag must be 0 or 1".into(),
            ))
        }
    };
    let rotary = RotaryEmbeddingSpec::new(
        tokens as usize,
        query_heads as usize,
        kv_heads as usize,
        head_size as usize,
        rotary_dim as usize,
        max_position as usize,
        T::DTYPE,
        style,
    )
    .map_err(invalid_contract)?;
    let spec = RopePagedKvWriteSpec::new(
        rotary,
        value_head_size as usize,
        num_blocks as usize,
        block_size as usize,
    )
    .map_err(invalid_contract)?;
    let layout = RopePagedKvLayout::new(
        spec,
        cache_tokens as usize,
        element_count(query_token_stride, "query token stride")?,
        element_count(query_head_stride, "query head stride")?,
        element_count(key_token_stride, "key token stride")?,
        element_count(source_key_head_stride, "source key head stride")?,
        element_count(value_token_stride, "value token stride")?,
        element_count(source_value_head_stride, "source value head stride")?,
        element_count(key_block_stride, "key block stride")?,
        element_count(key_page_stride, "key page stride")?,
        element_count(key_head_stride, "key cache head stride")?,
        element_count(value_block_stride, "value block stride")?,
        element_count(value_page_stride, "value page stride")?,
        element_count(value_cache_head_stride, "value cache head stride")?,
    )?;

    let (mut query, query_range) = unsafe { write_slice(query, query_elements, "RoPE query") }?;
    let (mut key, key_range) = unsafe { write_slice(key, key_elements, "RoPE key") }?;
    let (value, value_range) = unsafe { read_slice(value, value_elements, "RoPE value") }?;
    let (positions, positions_range) =
        unsafe { read_slice(positions, position_elements, "RoPE positions") }?;
    let (cos_sin_cache, cos_sin_cache_range) =
        unsafe { read_slice(cos_sin_cache, cos_sin_cache_elements, "RoPE cos/sin cache") }?;
    let (mut key_cache, key_cache_range) =
        unsafe { write_slice(key_cache, key_cache_elements, "paged key cache") }?;
    let (mut value_cache, value_cache_range) =
        unsafe { write_slice(value_cache, value_cache_elements, "paged value cache") }?;
    let (slot_mapping, slot_mapping_range) =
        unsafe { read_slice(slot_mapping, slot_mapping_elements, "paged slot mapping") }?;

    let dense_query_token = (layout.query_head_stride() == spec.rotary().head_size())
        .then(|| {
            spec.rotary()
                .query_heads()
                .checked_mul(spec.rotary().head_size())
        })
        .flatten();
    let dense_key_token = (layout.source_key_head_stride() == spec.rotary().head_size())
        .then(|| {
            spec.rotary()
                .key_heads()
                .checked_mul(spec.rotary().head_size())
        })
        .flatten();
    let dense_value_token = (layout.source_value_head_stride() == spec.value_head_size())
        .then(|| {
            spec.rotary()
                .key_heads()
                .checked_mul(spec.value_head_size())
        })
        .flatten();
    require_disjoint_or_dense_packed_axis::<T>(
        "query",
        query_range,
        layout.query_token_stride(),
        dense_query_token,
        "key",
        key_range,
        layout.key_token_stride(),
        dense_key_token,
        "RoPE+paged-KV",
    )?;
    require_disjoint_or_dense_packed_axis::<T>(
        "query",
        query_range,
        layout.query_token_stride(),
        dense_query_token,
        "value",
        value_range,
        layout.value_token_stride(),
        dense_value_token,
        "RoPE+paged-KV",
    )?;
    require_disjoint_or_dense_packed_axis::<T>(
        "key",
        key_range,
        layout.key_token_stride(),
        dense_key_token,
        "value",
        value_range,
        layout.value_token_stride(),
        dense_value_token,
        "RoPE+paged-KV",
    )?;

    let key_cache_block_elements = spec
        .block_size()
        .checked_mul(spec.rotary().key_heads())
        .and_then(|value| value.checked_mul(spec.rotary().head_size()))
        .ok_or_else(|| invalid_contract("paged key cache block size overflows usize"))?;
    let value_cache_block_elements = spec
        .block_size()
        .checked_mul(spec.rotary().key_heads())
        .and_then(|value| value.checked_mul(spec.value_head_size()))
        .ok_or_else(|| invalid_contract("paged value cache block size overflows usize"))?;
    let dense_key_cache_block = (layout.key_block_storage_elements(spec)?
        == key_cache_block_elements)
        .then_some(key_cache_block_elements);
    let dense_value_cache_block = (layout.value_block_storage_elements(spec)?
        == value_cache_block_elements)
        .then_some(value_cache_block_elements);
    require_disjoint_or_dense_packed_axis::<T>(
        "key cache",
        key_cache_range,
        layout.key_block_stride(),
        dense_key_cache_block,
        "value cache",
        value_cache_range,
        layout.value_block_stride(),
        dense_value_cache_block,
        "RoPE+paged-KV",
    )?;

    let metadata_and_caches = [
        ("positions", positions_range),
        ("cos/sin cache", cos_sin_cache_range),
        ("key cache", key_cache_range),
        ("value cache", value_cache_range),
        ("slot mapping", slot_mapping_range),
    ];
    require_disjoint_from("query", query_range, &metadata_and_caches, "RoPE+paged-KV")?;
    require_disjoint_from("key", key_range, &metadata_and_caches, "RoPE+paged-KV")?;
    require_disjoint_from(
        "key cache",
        key_cache_range,
        &[
            ("value", value_range),
            ("positions", positions_range),
            ("cos/sin cache", cos_sin_cache_range),
            ("slot mapping", slot_mapping_range),
        ],
        "RoPE+paged-KV",
    )?;
    require_disjoint_from(
        "value cache",
        value_cache_range,
        &[
            ("value", value_range),
            ("positions", positions_range),
            ("cos/sin cache", cos_sin_cache_range),
            ("slot mapping", slot_mapping_range),
        ],
        "RoPE+paged-KV",
    )?;

    T::rope_paged_kv_write(
        &stream_backend(stream),
        &mut query,
        &mut key,
        &value,
        &positions,
        &cos_sin_cache,
        &mut key_cache,
        &mut value_cache,
        &slot_mapping,
        spec,
        layout,
    )?;
    record_launch(OP_ROPE_PAGED_KV_WRITE);
    Ok(())
}

/// Checked fused RoPE plus paged-KV write over explicit framework layouts.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes. Logical mutable
/// tensor elements must not alias, including when packed views have
/// overlapping bounding spans.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rope_paged_kv_write(
    dtype: u32,
    query: *mut c_void,
    query_elements: u64,
    key: *mut c_void,
    key_elements: u64,
    value: *const c_void,
    value_elements: u64,
    positions: *const i64,
    position_elements: u64,
    cos_sin_cache: *const c_void,
    cos_sin_cache_elements: u64,
    key_cache: *mut c_void,
    key_cache_elements: u64,
    value_cache: *mut c_void,
    value_cache_elements: u64,
    slot_mapping: *const i64,
    slot_mapping_elements: u64,
    tokens: u32,
    cache_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    rotary_dim: u32,
    max_position: u32,
    num_blocks: u32,
    block_size: u32,
    query_token_stride: u64,
    query_head_stride: u64,
    key_token_stride: u64,
    source_key_head_stride: u64,
    value_token_stride: u64,
    source_value_head_stride: u64,
    key_block_stride: u64,
    key_page_stride: u64,
    key_head_stride: u64,
    value_block_stride: u64,
    value_page_stride: u64,
    value_cache_head_stride: u64,
    is_neox: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rope_paged_kv_write(
                query.cast(),
                query_elements,
                key.cast(),
                key_elements,
                value.cast(),
                value_elements,
                positions,
                position_elements,
                cos_sin_cache.cast(),
                cos_sin_cache_elements,
                key_cache.cast(),
                key_cache_elements,
                value_cache.cast(),
                value_cache_elements,
                slot_mapping,
                slot_mapping_elements,
                tokens,
                cache_tokens,
                query_heads,
                kv_heads,
                head_size,
                value_head_size,
                rotary_dim,
                max_position,
                num_blocks,
                block_size,
                query_token_stride,
                query_head_stride,
                key_token_stride,
                source_key_head_stride,
                value_token_stride,
                source_value_head_stride,
                key_block_stride,
                key_page_stride,
                key_head_stride,
                value_block_stride,
                value_page_stride,
                value_cache_head_stride,
                is_neox,
                stream,
            )
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{checked_byte_range, require_disjoint_or_dense_packed_axis};

    #[test]
    fn packed_token_aliasing_accepts_only_disjoint_logical_views() {
        let query = checked_byte_range(0x1000_usize as *const f32, 16, "packed query").unwrap();
        let key = checked_byte_range(0x1010_usize as *const f32, 16, "packed key").unwrap();
        let overlapping =
            checked_byte_range(0x1008_usize as *const f32, 16, "overlapping key").unwrap();

        require_disjoint_or_dense_packed_axis::<f32>(
            "query",
            query,
            12,
            Some(4),
            "key",
            key,
            12,
            Some(4),
            "RoPE+paged-KV",
        )
        .unwrap();
        assert!(require_disjoint_or_dense_packed_axis::<f32>(
            "query",
            query,
            12,
            Some(4),
            "key",
            overlapping,
            12,
            Some(4),
            "RoPE+paged-KV",
        )
        .is_err());
    }
}
