//! Shared tensor metadata and contract errors.

use std::fmt;

/// Element type stored by a tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 half precision.
    F16,
    /// Brain floating point with an eight-bit exponent.
    Bf16,
    /// FP8 E4M3 finite-numbers encoding.
    Fp8E4M3Fn,
}

impl DType {
    /// Returns the number of bytes occupied by one element.
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Fp8E4M3Fn => 1,
        }
    }
}

/// A shape and stride contract without a data pointer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    dtype: DType,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorSpec {
    /// Builds a row-major contiguous tensor specification.
    pub fn contiguous(dtype: DType, shape: impl Into<Vec<usize>>) -> Result<Self, ContractError> {
        let shape = shape.into();
        validate_shape(&shape)?;

        let mut strides = vec![1_usize; shape.len()];
        for index in (0..shape.len().saturating_sub(1)).rev() {
            strides[index] = strides[index + 1]
                .checked_mul(shape[index + 1])
                .ok_or(ContractError::ElementCountOverflow)?;
        }

        Ok(Self {
            dtype,
            shape,
            strides,
        })
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_in_bytes(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }
}

/// Operator contract or host-buffer validation failure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContractError {
    EmptyShape,
    ZeroDimension,
    ElementCountOverflow,
    InvalidEpsilon(f32),
    InvalidGroupSize(usize),
    WidthNotDivisible {
        width: usize,
        group_size: usize,
    },
    InvalidRotaryDimension {
        rotary_dim: usize,
        head_size: usize,
    },
    PositionOutOfBounds {
        token: usize,
        position: i64,
        max_position: usize,
    },
    SlotOutOfBounds {
        token: usize,
        slot: i64,
        slot_capacity: usize,
    },
    DuplicateSlot {
        first_token: usize,
        second_token: usize,
        slot: usize,
    },
    TokenIdOutOfBounds {
        row: usize,
        token_id: i64,
        vocab_size: usize,
    },
    InvalidProbability {
        parameter: &'static str,
        row: usize,
        value: f32,
    },
    InvalidScale(f32),
    HeadCountNotDivisible {
        query_heads: usize,
        kv_heads: usize,
    },
    SequenceLengthOutOfBounds {
        sequence: usize,
        length: i64,
        capacity: usize,
    },
    BlockIdOutOfBounds {
        sequence: usize,
        logical_block: usize,
        block_id: i64,
        num_blocks: usize,
    },
    LengthMismatch {
        buffer: &'static str,
        expected: usize,
        actual: usize,
    },
    UnsupportedDType(DType),
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyShape => write!(formatter, "tensor shape must not be empty"),
            Self::ZeroDimension => write!(formatter, "tensor dimensions must be non-zero"),
            Self::ElementCountOverflow => write!(formatter, "tensor element count overflowed"),
            Self::InvalidEpsilon(value) => write!(
                formatter,
                "RMSNorm epsilon must be finite and positive, got {value}"
            ),
            Self::InvalidGroupSize(value) => write!(
                formatter,
                "FP8 block group size must be 64 or 128, got {value}"
            ),
            Self::WidthNotDivisible { width, group_size } => write!(
                formatter,
                "output width {width} is not divisible by FP8 group size {group_size}"
            ),
            Self::InvalidRotaryDimension {
                rotary_dim,
                head_size,
            } => write!(
                formatter,
                "rotary dimension must be non-zero, even, and no larger than head size; got rotary_dim={rotary_dim}, head_size={head_size}"
            ),
            Self::PositionOutOfBounds {
                token,
                position,
                max_position,
            } => write!(
                formatter,
                "position {position} for token {token} is outside [0, {max_position})"
            ),
            Self::SlotOutOfBounds {
                token,
                slot,
                slot_capacity,
            } => write!(
                formatter,
                "cache slot {slot} for token {token} is outside [0, {slot_capacity})"
            ),
            Self::DuplicateSlot {
                first_token,
                second_token,
                slot,
            } => write!(
                formatter,
                "cache slot {slot} is assigned to both token {first_token} and token {second_token}"
            ),
            Self::TokenIdOutOfBounds {
                row,
                token_id,
                vocab_size,
            } => write!(
                formatter,
                "selected token ID {token_id} for row {row} is outside [0, {vocab_size})"
            ),
            Self::InvalidProbability {
                parameter,
                row,
                value,
            } => write!(
                formatter,
                "{parameter} for row {row} must be finite and in [0, 1], got {value}"
            ),
            Self::InvalidScale(value) => write!(
                formatter,
                "attention scale must be finite and positive, got {value}"
            ),
            Self::HeadCountNotDivisible {
                query_heads,
                kv_heads,
            } => write!(
                formatter,
                "query head count {query_heads} must be divisible by KV head count {kv_heads}"
            ),
            Self::SequenceLengthOutOfBounds {
                sequence,
                length,
                capacity,
            } => write!(
                formatter,
                "sequence length {length} for sequence {sequence} is outside [1, {capacity}]"
            ),
            Self::BlockIdOutOfBounds {
                sequence,
                logical_block,
                block_id,
                num_blocks,
            } => write!(
                formatter,
                "physical block ID {block_id} for sequence {sequence}, logical block {logical_block} is outside [0, {num_blocks})"
            ),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => write!(
                formatter,
                "{buffer} length mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedDType(dtype) => {
                write!(formatter, "CPU reference does not support dtype {dtype:?}")
            }
        }
    }
}

impl std::error::Error for ContractError {}

fn validate_shape(shape: &[usize]) -> Result<(), ContractError> {
    if shape.is_empty() {
        return Err(ContractError::EmptyShape);
    }
    if shape.contains(&0) {
        return Err(ContractError::ZeroDimension);
    }
    shape
        .iter()
        .try_fold(1_usize, |elements, &dimension| {
            elements.checked_mul(dimension)
        })
        .ok_or(ContractError::ElementCountOverflow)?;
    Ok(())
}

pub(crate) fn require_len(
    buffer: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ContractError> {
    if actual == expected {
        Ok(())
    } else {
        Err(ContractError::LengthMismatch {
            buffer,
            expected,
            actual,
        })
    }
}
