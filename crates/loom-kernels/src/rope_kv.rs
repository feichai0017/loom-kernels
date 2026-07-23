//! Rotary embedding and paged K/V cache contracts with CPU references.

use std::collections::HashMap;

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Pairing convention used by rotary positional embeddings.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RotaryStyle {
    /// Pairs the first and second halves of the rotary prefix.
    NeoX,
    /// Pairs adjacent even and odd elements (GPT-J style).
    Interleaved,
}

/// Contract for in-place rotary embedding of contiguous Q and K tensors.
///
/// Query and key have logical shapes `[tokens, query_heads, head_size]` and
/// `[tokens, key_heads, head_size]`. `cos_sin_cache` has shape
/// `[max_position, rotary_dim]`, with cosine values in its first half and sine
/// values in its second half, matching vLLM's `_C::rotary_embedding` contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RotaryEmbeddingSpec {
    tokens: usize,
    query_heads: usize,
    key_heads: usize,
    head_size: usize,
    rotary_dim: usize,
    max_position: usize,
    dtype: DType,
    style: RotaryStyle,
}

impl RotaryEmbeddingSpec {
    /// Creates a validated contiguous rotary-embedding contract.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokens: usize,
        query_heads: usize,
        key_heads: usize,
        head_size: usize,
        rotary_dim: usize,
        max_position: usize,
        dtype: DType,
        style: RotaryStyle,
    ) -> Result<Self, ContractError> {
        if tokens == 0 || query_heads == 0 || key_heads == 0 || head_size == 0 || max_position == 0
        {
            return Err(ContractError::ZeroDimension);
        }
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_size {
            return Err(ContractError::InvalidRotaryDimension {
                rotary_dim,
                head_size,
            });
        }

        tokens
            .checked_mul(query_heads)
            .and_then(|elements| elements.checked_mul(head_size))
            .ok_or(ContractError::ElementCountOverflow)?;
        tokens
            .checked_mul(key_heads)
            .and_then(|elements| elements.checked_mul(head_size))
            .ok_or(ContractError::ElementCountOverflow)?;
        max_position
            .checked_mul(rotary_dim)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            tokens,
            query_heads,
            key_heads,
            head_size,
            rotary_dim,
            max_position,
            dtype,
            style,
        })
    }

    pub const fn tokens(self) -> usize {
        self.tokens
    }

    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    pub const fn key_heads(self) -> usize {
        self.key_heads
    }

    pub const fn head_size(self) -> usize {
        self.head_size
    }

    pub const fn rotary_dim(self) -> usize {
        self.rotary_dim
    }

    pub const fn max_position(self) -> usize {
        self.max_position
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn style(self) -> RotaryStyle {
        self.style
    }

    pub const fn query_numel(self) -> usize {
        self.tokens * self.query_heads * self.head_size
    }

    pub const fn key_numel(self) -> usize {
        self.tokens * self.key_heads * self.head_size
    }

    pub const fn cos_sin_cache_numel(self) -> usize {
        self.max_position * self.rotary_dim
    }
}

/// Fused in-place RoPE plus paged K/V cache-write contract.
///
/// The source value tensor is `[tokens, key_heads, value_head_size]`. Separate
/// key and value caches use the logical NHD shapes
/// `[num_blocks, block_size, key_heads, head_size]` and
/// `[num_blocks, block_size, key_heads, value_head_size]`. Accelerator adapters
/// may preserve those logical dimensions with non-contiguous physical strides,
/// as vLLM does for HND and interleaved K/V storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopePagedKvWriteSpec {
    rotary: RotaryEmbeddingSpec,
    value_head_size: usize,
    num_blocks: usize,
    block_size: usize,
}

impl RopePagedKvWriteSpec {
    pub fn new(
        rotary: RotaryEmbeddingSpec,
        value_head_size: usize,
        num_blocks: usize,
        block_size: usize,
    ) -> Result<Self, ContractError> {
        if value_head_size == 0 || num_blocks == 0 || block_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rotary
            .tokens()
            .checked_mul(rotary.key_heads())
            .and_then(|elements| elements.checked_mul(value_head_size))
            .ok_or(ContractError::ElementCountOverflow)?;
        num_blocks
            .checked_mul(block_size)
            .and_then(|slots| slots.checked_mul(rotary.key_heads()))
            .and_then(|elements| elements.checked_mul(rotary.head_size()))
            .ok_or(ContractError::ElementCountOverflow)?;
        num_blocks
            .checked_mul(block_size)
            .and_then(|slots| slots.checked_mul(rotary.key_heads()))
            .and_then(|elements| elements.checked_mul(value_head_size))
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rotary,
            value_head_size,
            num_blocks,
            block_size,
        })
    }

    pub const fn rotary(self) -> RotaryEmbeddingSpec {
        self.rotary
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

    pub const fn slot_capacity(self) -> usize {
        self.num_blocks * self.block_size
    }

    pub const fn value_numel(self) -> usize {
        self.rotary.tokens * self.rotary.key_heads * self.value_head_size
    }

    pub const fn key_cache_numel(self) -> usize {
        self.slot_capacity() * self.rotary.key_heads * self.rotary.head_size
    }

    pub const fn value_cache_numel(self) -> usize {
        self.slot_capacity() * self.rotary.key_heads * self.value_head_size
    }
}

/// Applies vLLM-compatible F32 rotary embedding to Q and K in place.
pub fn rotary_embedding_f32_reference(
    query: &mut [f32],
    key: &mut [f32],
    positions: &[i64],
    cos_sin_cache: &[f32],
    spec: RotaryEmbeddingSpec,
) -> Result<(), ContractError> {
    rotary_embedding_reference(query, key, positions, cos_sin_cache, spec, DType::F32)
}

/// Applies vLLM-compatible FP16 rotary embedding to Q and K in place.
pub fn rotary_embedding_f16_reference(
    query: &mut [f16],
    key: &mut [f16],
    positions: &[i64],
    cos_sin_cache: &[f16],
    spec: RotaryEmbeddingSpec,
) -> Result<(), ContractError> {
    rotary_embedding_reference(query, key, positions, cos_sin_cache, spec, DType::F16)
}

/// Applies vLLM-compatible BF16 rotary embedding to Q and K in place.
pub fn rotary_embedding_bf16_reference(
    query: &mut [bf16],
    key: &mut [bf16],
    positions: &[i64],
    cos_sin_cache: &[bf16],
    spec: RotaryEmbeddingSpec,
) -> Result<(), ContractError> {
    rotary_embedding_reference(query, key, positions, cos_sin_cache, spec, DType::Bf16)
}

/// Applies F32 RoPE and writes rotated K plus V into contiguous paged caches.
///
/// Any negative slot is a padding entry: Q and K are still rotated, while its
/// cache write is skipped. Non-negative slots must be unique so the result is
/// deterministic across CPU and massively parallel accelerator backends.
#[allow(clippy::too_many_arguments)]
pub fn rope_paged_kv_write_f32_reference(
    query: &mut [f32],
    key: &mut [f32],
    value: &[f32],
    positions: &[i64],
    cos_sin_cache: &[f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    slot_mapping: &[i64],
    spec: RopePagedKvWriteSpec,
) -> Result<(), ContractError> {
    rope_paged_kv_write_reference(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        spec,
        DType::F32,
    )
}

/// Applies FP16 RoPE and writes rotated K plus V into paged caches.
#[allow(clippy::too_many_arguments)]
pub fn rope_paged_kv_write_f16_reference(
    query: &mut [f16],
    key: &mut [f16],
    value: &[f16],
    positions: &[i64],
    cos_sin_cache: &[f16],
    key_cache: &mut [f16],
    value_cache: &mut [f16],
    slot_mapping: &[i64],
    spec: RopePagedKvWriteSpec,
) -> Result<(), ContractError> {
    rope_paged_kv_write_reference(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        spec,
        DType::F16,
    )
}

/// Applies BF16 RoPE and writes rotated K plus V into paged caches.
#[allow(clippy::too_many_arguments)]
pub fn rope_paged_kv_write_bf16_reference(
    query: &mut [bf16],
    key: &mut [bf16],
    value: &[bf16],
    positions: &[i64],
    cos_sin_cache: &[bf16],
    key_cache: &mut [bf16],
    value_cache: &mut [bf16],
    slot_mapping: &[i64],
    spec: RopePagedKvWriteSpec,
) -> Result<(), ContractError> {
    rope_paged_kv_write_reference(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        spec,
        DType::Bf16,
    )
}

trait RotaryElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl RotaryElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl RotaryElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl RotaryElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

fn rotary_embedding_reference<T: RotaryElement>(
    query: &mut [T],
    key: &mut [T],
    positions: &[i64],
    cos_sin_cache: &[T],
    spec: RotaryEmbeddingSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    validate_rotary_buffers(query, key, positions, cos_sin_cache, spec, expected_dtype)?;
    apply_validated_rotary_embedding(query, key, positions, cos_sin_cache, spec);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rope_paged_kv_write_reference<T: RotaryElement>(
    query: &mut [T],
    key: &mut [T],
    value: &[T],
    positions: &[i64],
    cos_sin_cache: &[T],
    key_cache: &mut [T],
    value_cache: &mut [T],
    slot_mapping: &[i64],
    spec: RopePagedKvWriteSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    let rotary = spec.rotary();
    validate_rotary_buffers(query, key, positions, cos_sin_cache, rotary, expected_dtype)?;
    require_len("value", value.len(), spec.value_numel())?;
    require_len("key_cache", key_cache.len(), spec.key_cache_numel())?;
    require_len("value_cache", value_cache.len(), spec.value_cache_numel())?;
    require_len("slot_mapping", slot_mapping.len(), rotary.tokens())?;
    validate_slot_mapping(slot_mapping, spec.slot_capacity())?;

    apply_validated_rotary_embedding(query, key, positions, cos_sin_cache, rotary);

    for (token, &slot) in slot_mapping.iter().enumerate() {
        if slot < 0 {
            continue;
        }
        let slot = slot as usize;
        for head in 0..rotary.key_heads() {
            let source_key = (token * rotary.key_heads() + head) * rotary.head_size();
            let target_key = (slot * rotary.key_heads() + head) * rotary.head_size();
            key_cache[target_key..target_key + rotary.head_size()]
                .copy_from_slice(&key[source_key..source_key + rotary.head_size()]);

            let source_value = (token * rotary.key_heads() + head) * spec.value_head_size();
            let target_value = (slot * rotary.key_heads() + head) * spec.value_head_size();
            value_cache[target_value..target_value + spec.value_head_size()]
                .copy_from_slice(&value[source_value..source_value + spec.value_head_size()]);
        }
    }
    Ok(())
}

fn validate_rotary_buffers<T>(
    query: &[T],
    key: &[T],
    positions: &[i64],
    cos_sin_cache: &[T],
    spec: RotaryEmbeddingSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("query", query.len(), spec.query_numel())?;
    require_len("key", key.len(), spec.key_numel())?;
    require_len("positions", positions.len(), spec.tokens())?;
    require_len(
        "cos_sin_cache",
        cos_sin_cache.len(),
        spec.cos_sin_cache_numel(),
    )?;
    for (token, &position) in positions.iter().enumerate() {
        if position < 0 || position as u128 >= spec.max_position() as u128 {
            return Err(ContractError::PositionOutOfBounds {
                token,
                position,
                max_position: spec.max_position(),
            });
        }
    }
    Ok(())
}

fn validate_slot_mapping(slot_mapping: &[i64], slot_capacity: usize) -> Result<(), ContractError> {
    let mut owners = HashMap::with_capacity(slot_mapping.len());
    for (token, &slot) in slot_mapping.iter().enumerate() {
        if slot < 0 {
            continue;
        }
        if slot as u128 >= slot_capacity as u128 {
            return Err(ContractError::SlotOutOfBounds {
                token,
                slot,
                slot_capacity,
            });
        }
        let slot = slot as usize;
        if let Some(&first_token) = owners.get(&slot) {
            return Err(ContractError::DuplicateSlot {
                first_token,
                second_token: token,
                slot,
            });
        }
        owners.insert(slot, token);
    }
    Ok(())
}

fn apply_validated_rotary_embedding<T: RotaryElement>(
    query: &mut [T],
    key: &mut [T],
    positions: &[i64],
    cos_sin_cache: &[T],
    spec: RotaryEmbeddingSpec,
) {
    for (token, &position) in positions.iter().enumerate() {
        let cache_offset = position as usize * spec.rotary_dim();
        let cache_row = &cos_sin_cache[cache_offset..cache_offset + spec.rotary_dim()];

        let query_token_offset = token * spec.query_heads() * spec.head_size();
        for head in 0..spec.query_heads() {
            let head_offset = query_token_offset + head * spec.head_size();
            apply_rotary_to_head(
                &mut query[head_offset..head_offset + spec.head_size()],
                cache_row,
                spec,
            );
        }

        let key_token_offset = token * spec.key_heads() * spec.head_size();
        for head in 0..spec.key_heads() {
            let head_offset = key_token_offset + head * spec.head_size();
            apply_rotary_to_head(
                &mut key[head_offset..head_offset + spec.head_size()],
                cache_row,
                spec,
            );
        }
    }
}

fn apply_rotary_to_head<T: RotaryElement>(
    head: &mut [T],
    cos_sin: &[T],
    spec: RotaryEmbeddingSpec,
) {
    let half = spec.rotary_dim() / 2;
    for pair in 0..half {
        let (first_index, second_index) = match spec.style() {
            RotaryStyle::NeoX => (pair, pair + half),
            RotaryStyle::Interleaved => (pair * 2, pair * 2 + 1),
        };
        let first = head[first_index].to_f32();
        let second = head[second_index].to_f32();
        let cosine = cos_sin[pair].to_f32();
        let sine = cos_sin[half + pair].to_f32();
        head[first_index] = T::from_f32(first * cosine - second * sine);
        head[second_index] = T::from_f32(second * cosine + first * sine);
    }
}
