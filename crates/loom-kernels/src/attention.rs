//! Paged decode-attention contracts and CPU reference implementations.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Minimal engine-owned paged MQA/GQA decode contract.
///
/// Each sequence contributes exactly one query token. Query has logical shape
/// `[sequences, query_heads, head_size]`; K and V caches have logical NHD
/// shapes `[num_blocks, block_size, kv_heads, head_size]` and
/// `[num_blocks, block_size, kv_heads, value_head_size]`. `block_tables` maps
/// each logical sequence block to a physical cache block. Sequence lengths
/// include the current token already written to the cache.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedDecodeAttentionSpec {
    sequences: usize,
    query_heads: usize,
    kv_heads: usize,
    head_size: usize,
    value_head_size: usize,
    num_blocks: usize,
    block_size: usize,
    max_blocks_per_sequence: usize,
    max_sequence_length: usize,
    scale: f32,
    dtype: DType,
}

impl PagedDecodeAttentionSpec {
    /// Creates the base causal decode contract without ALiBi, sliding windows,
    /// soft caps, quantized KV, or speculative multi-token queries.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequences: usize,
        query_heads: usize,
        kv_heads: usize,
        head_size: usize,
        value_head_size: usize,
        num_blocks: usize,
        block_size: usize,
        max_blocks_per_sequence: usize,
        max_sequence_length: usize,
        scale: f32,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if sequences == 0
            || query_heads == 0
            || kv_heads == 0
            || head_size == 0
            || value_head_size == 0
            || num_blocks == 0
            || block_size == 0
            || max_blocks_per_sequence == 0
            || max_sequence_length == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        if !query_heads.is_multiple_of(kv_heads) {
            return Err(ContractError::HeadCountNotDivisible {
                query_heads,
                kv_heads,
            });
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ContractError::InvalidScale(scale));
        }

        checked_product(&[sequences, query_heads, head_size])?;
        checked_product(&[sequences, query_heads, value_head_size])?;
        checked_product(&[num_blocks, block_size, kv_heads, head_size])?;
        checked_product(&[num_blocks, block_size, kv_heads, value_head_size])?;
        checked_product(&[sequences, max_blocks_per_sequence])?;
        let table_capacity = max_blocks_per_sequence
            .checked_mul(block_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        if max_sequence_length > table_capacity {
            return Err(ContractError::MaxSequenceLengthOutOfBounds {
                length: max_sequence_length,
                capacity: table_capacity,
            });
        }

        Ok(Self {
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
            dtype,
        })
    }

    pub const fn sequences(self) -> usize {
        self.sequences
    }

    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    pub const fn kv_heads(self) -> usize {
        self.kv_heads
    }

    pub const fn queries_per_kv(self) -> usize {
        self.query_heads / self.kv_heads
    }

    pub const fn head_size(self) -> usize {
        self.head_size
    }

    pub const fn value_head_size(self) -> usize {
        self.value_head_size
    }

    pub const fn num_blocks(self) -> usize {
        self.num_blocks
    }

    pub const fn block_size(self) -> usize {
        self.block_size
    }

    pub const fn max_blocks_per_sequence(self) -> usize {
        self.max_blocks_per_sequence
    }

    pub const fn max_sequence_length(self) -> usize {
        self.max_sequence_length
    }

    pub const fn scale(self) -> f32 {
        self.scale
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn query_numel(self) -> usize {
        self.sequences * self.query_heads * self.head_size
    }

    pub const fn output_numel(self) -> usize {
        self.sequences * self.query_heads * self.value_head_size
    }

    pub const fn key_cache_numel(self) -> usize {
        self.num_blocks * self.block_size * self.kv_heads * self.head_size
    }

    pub const fn value_cache_numel(self) -> usize {
        self.num_blocks * self.block_size * self.kv_heads * self.value_head_size
    }

    pub const fn block_table_numel(self) -> usize {
        self.sequences * self.max_blocks_per_sequence
    }
}

/// Computes F32 paged MQA/GQA attention for one decode query per sequence.
pub fn paged_decode_attention_f32_reference(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    block_tables: &[i64],
    sequence_lengths: &[i64],
    output: &mut [f32],
    spec: PagedDecodeAttentionSpec,
) -> Result<(), ContractError> {
    paged_decode_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        spec,
        DType::F32,
    )
}

/// Computes FP16 paged MQA/GQA attention with F64 reference accumulation.
pub fn paged_decode_attention_f16_reference(
    query: &[f16],
    key_cache: &[f16],
    value_cache: &[f16],
    block_tables: &[i64],
    sequence_lengths: &[i64],
    output: &mut [f16],
    spec: PagedDecodeAttentionSpec,
) -> Result<(), ContractError> {
    paged_decode_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        spec,
        DType::F16,
    )
}

/// Computes BF16 paged MQA/GQA attention with F64 reference accumulation.
pub fn paged_decode_attention_bf16_reference(
    query: &[bf16],
    key_cache: &[bf16],
    value_cache: &[bf16],
    block_tables: &[i64],
    sequence_lengths: &[i64],
    output: &mut [bf16],
    spec: PagedDecodeAttentionSpec,
) -> Result<(), ContractError> {
    paged_decode_attention_reference(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        spec,
        DType::Bf16,
    )
}

trait AttentionElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl AttentionElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl AttentionElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl AttentionElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_attention_reference<T: AttentionElement>(
    query: &[T],
    key_cache: &[T],
    value_cache: &[T],
    block_tables: &[i64],
    sequence_lengths: &[i64],
    output: &mut [T],
    spec: PagedDecodeAttentionSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    validate_buffers(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        output,
        spec,
        expected_dtype,
    )?;

    let queries_per_kv = spec.queries_per_kv();
    for (sequence, &sequence_length) in sequence_lengths.iter().enumerate() {
        let sequence_length = sequence_length as usize;
        for query_head in 0..spec.query_heads() {
            let kv_head = query_head / queries_per_kv;
            let query_offset = (sequence * spec.query_heads() + query_head) * spec.head_size();
            let query_head_values = &query[query_offset..query_offset + spec.head_size()];

            let mut maximum = f64::NEG_INFINITY;
            for position in 0..sequence_length {
                let key_offset = cache_offset(
                    sequence,
                    position,
                    kv_head,
                    spec.head_size(),
                    block_tables,
                    spec,
                );
                let key = &key_cache[key_offset..key_offset + spec.head_size()];
                maximum = maximum.max(attention_score(query_head_values, key, spec.scale()));
            }

            let mut denominator = 0.0_f64;
            let mut accumulator = vec![0.0_f64; spec.value_head_size()];
            for position in 0..sequence_length {
                let key_offset = cache_offset(
                    sequence,
                    position,
                    kv_head,
                    spec.head_size(),
                    block_tables,
                    spec,
                );
                let key = &key_cache[key_offset..key_offset + spec.head_size()];
                let weight =
                    (attention_score(query_head_values, key, spec.scale()) - maximum).exp();
                denominator += weight;

                let value_offset = cache_offset(
                    sequence,
                    position,
                    kv_head,
                    spec.value_head_size(),
                    block_tables,
                    spec,
                );
                for (sum, &value) in accumulator
                    .iter_mut()
                    .zip(&value_cache[value_offset..value_offset + spec.value_head_size()])
                {
                    *sum += weight * f64::from(value.to_f32());
                }
            }

            let output_offset =
                (sequence * spec.query_heads() + query_head) * spec.value_head_size();
            for (destination, sum) in output[output_offset..output_offset + spec.value_head_size()]
                .iter_mut()
                .zip(accumulator)
            {
                *destination = T::from_f32((sum / denominator) as f32);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_buffers<T>(
    query: &[T],
    key_cache: &[T],
    value_cache: &[T],
    block_tables: &[i64],
    sequence_lengths: &[i64],
    output: &[T],
    spec: PagedDecodeAttentionSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("query", query.len(), spec.query_numel())?;
    require_len("key_cache", key_cache.len(), spec.key_cache_numel())?;
    require_len("value_cache", value_cache.len(), spec.value_cache_numel())?;
    require_len("block_tables", block_tables.len(), spec.block_table_numel())?;
    require_len("sequence_lengths", sequence_lengths.len(), spec.sequences())?;
    require_len("output", output.len(), spec.output_numel())?;

    for (sequence, &length) in sequence_lengths.iter().enumerate() {
        if length <= 0 || length as u128 > spec.max_sequence_length() as u128 {
            return Err(ContractError::SequenceLengthOutOfBounds {
                sequence,
                length,
                capacity: spec.max_sequence_length(),
            });
        }
        let active_blocks = (length as usize).div_ceil(spec.block_size());
        let table_offset = sequence * spec.max_blocks_per_sequence();
        for logical_block in 0..active_blocks {
            let block_id = block_tables[table_offset + logical_block];
            if block_id < 0 || block_id as u128 >= spec.num_blocks() as u128 {
                return Err(ContractError::BlockIdOutOfBounds {
                    sequence,
                    logical_block,
                    block_id,
                    num_blocks: spec.num_blocks(),
                });
            }
        }
    }
    Ok(())
}

fn attention_score<T: AttentionElement>(query: &[T], key: &[T], scale: f32) -> f64 {
    query
        .iter()
        .zip(key)
        .map(|(&query, &key)| f64::from(query.to_f32()) * f64::from(key.to_f32()))
        .sum::<f64>()
        * f64::from(scale)
}

fn cache_offset(
    sequence: usize,
    position: usize,
    kv_head: usize,
    head_size: usize,
    block_tables: &[i64],
    spec: PagedDecodeAttentionSpec,
) -> usize {
    let logical_block = position / spec.block_size();
    let block_offset = position % spec.block_size();
    let table_offset = sequence * spec.max_blocks_per_sequence();
    let physical_block = block_tables[table_offset + logical_block] as usize;
    ((physical_block * spec.block_size() + block_offset) * spec.kv_heads() + kv_head) * head_size
}

fn checked_product(dimensions: &[usize]) -> Result<usize, ContractError> {
    dimensions
        .iter()
        .try_fold(1_usize, |product, &dimension| {
            product.checked_mul(dimension)
        })
        .ok_or(ContractError::ElementCountOverflow)
}
