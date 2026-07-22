//! Backend-independent contracts and CPU references for LLM inference operators.
//!
//! This crate deliberately contains no FFI or accelerator dependency. CUDA,
//! ROCm, CPU SIMD, and other providers implement these contracts in separate
//! crates and must report unsupported shapes instead of silently falling back.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::fmt;

use half::{bf16, f16};

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

/// Contract for a two-dimensional RMSNorm operation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormSpec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    dtype: DType,
}

/// Maximum finite magnitude of the OCP FP8 E4M3FN encoding.
pub const FP8_E4M3FN_MAX: f32 = 448.0;

/// vLLM-compatible lower bound for a dynamic per-token FP8 scale.
///
/// The non-zero floor keeps a zero row quantizable and avoids division by
/// zero. It matches `1 / (FP8_E4M3FN_MAX * 512)`.
pub const DYNAMIC_FP8_MIN_SCALE: f32 = 1.0 / (FP8_E4M3FN_MAX * 512.0);

/// Contract for RMSNorm followed by dynamic per-token FP8 quantization.
///
/// Inputs and weights are contiguous `[rows, hidden_size]` and
/// `[hidden_size]` tensors. The output contains FP8 E4M3FN storage bytes with
/// the same logical shape, and `rows` F32 scales satisfy approximately
/// `normalized = fp8(output) * scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormDynamicFp8Spec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    input_dtype: DType,
    output_dtype: DType,
}

impl RmsNormDynamicFp8Spec {
    /// Creates a validated shape and dtype contract.
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        input_dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || hidden_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ContractError::InvalidEpsilon(epsilon));
        }
        rows.checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rows,
            hidden_size,
            epsilon,
            input_dtype,
            output_dtype: DType::Fp8E4M3Fn,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub const fn input_dtype(self) -> DType {
        self.input_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }

    pub const fn scale_count(self) -> usize {
        self.rows
    }
}

/// Contract for fused residual addition followed by RMSNorm.
///
/// Backends implementing this contract update both operands in place:
/// `residual = input + residual`, then
/// `input = RMSNorm(residual, weight, epsilon)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddRmsNormSpec {
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
    dtype: DType,
}

/// Contract for the fused SwiGLU activation `silu(gate) * up`.
///
/// Input rows have shape `[2 * width]`, with the gate in the first half and
/// the up projection in the second half. Output rows have shape `[width]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiluAndMulSpec {
    rows: usize,
    width: usize,
    dtype: DType,
}

/// Contract for SwiGLU followed by dynamic per-block FP8 quantization.
///
/// Input rows use the same split-half `[gate, up]` layout as
/// [`SiluAndMulSpec`]. Output contains FP8 E4M3FN bytes with shape
/// `[rows, width]`; F32 scales are row-major `[rows, width / group_size]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiluAndMulDynamicFp8Spec {
    rows: usize,
    width: usize,
    group_size: usize,
    input_dtype: DType,
    output_dtype: DType,
}

/// Contract for fused greedy token selection and its normalized logprob.
///
/// Logits are contiguous `[rows, vocab_size]`. Each output row contains the
/// lowest token index attaining the maximum logit, its log-softmax value, and
/// an integration-defined sampled-token rank. The CUDA and Python adapters
/// match vLLM's tie-aware rank by counting all logits equal to the maximum.
/// This deterministic boundary is useful for greedy decode requests that ask
/// only for the sampled token's logprob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedySampleLogprobsSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

/// Contract for normalizing and ranking one caller-selected token per row.
///
/// Logits are contiguous `[rows, vocab_size]`; token IDs are one int64 value
/// per row and must be in `[0, vocab_size)`. Outputs are F32 logprobs and
/// int64 ranks. Rank uses vLLM's tie-aware definition: the number of logits
/// greater than or equal to the selected logit. This boundary lets an engine
/// keep its own greedy, top-k/top-p, penalty, and random-sampling policy while
/// avoiding a materialized full-vocabulary F32 log-softmax tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectedTokenLogprobsSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

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

impl SiluAndMulDynamicFp8Spec {
    /// Creates a vLLM-compatible 64- or 128-element block-quant contract.
    pub fn new(
        rows: usize,
        width: usize,
        group_size: usize,
        input_dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || width == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !matches!(group_size, 64 | 128) {
            return Err(ContractError::InvalidGroupSize(group_size));
        }
        if !width.is_multiple_of(group_size) {
            return Err(ContractError::WidthNotDivisible { width, group_size });
        }
        let output_elements = rows
            .checked_mul(width)
            .ok_or(ContractError::ElementCountOverflow)?;
        output_elements
            .checked_mul(2)
            .ok_or(ContractError::ElementCountOverflow)?;
        rows.checked_mul(width / group_size)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rows,
            width,
            group_size,
            input_dtype,
            output_dtype: DType::Fp8E4M3Fn,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn group_size(self) -> usize {
        self.group_size
    }

    pub const fn group_count(self) -> usize {
        self.width / self.group_size
    }

    pub const fn input_dtype(self) -> DType {
        self.input_dtype
    }

    pub const fn output_dtype(self) -> DType {
        self.output_dtype
    }

    pub const fn input_numel(self) -> usize {
        self.rows * self.width * 2
    }

    pub const fn output_numel(self) -> usize {
        self.rows * self.width
    }

    pub const fn scale_count(self) -> usize {
        self.rows * self.group_count()
    }
}

impl GreedySampleLogprobsSpec {
    /// Creates a validated contiguous logits contract.
    pub fn new(rows: usize, vocab_size: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            rows,
            vocab_size,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }
}

impl SelectedTokenLogprobsSpec {
    /// Creates a validated contiguous logits contract.
    pub fn new(rows: usize, vocab_size: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || vocab_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        rows.checked_mul(vocab_size)
            .ok_or(ContractError::ElementCountOverflow)?;
        Ok(Self {
            rows,
            vocab_size,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn logits_numel(self) -> usize {
        self.rows * self.vocab_size
    }
}

impl SiluAndMulSpec {
    /// Creates a validated contiguous split-half SwiGLU contract.
    pub fn new(rows: usize, width: usize, dtype: DType) -> Result<Self, ContractError> {
        if rows == 0 || width == 0 {
            return Err(ContractError::ZeroDimension);
        }
        let output_elements = rows
            .checked_mul(width)
            .ok_or(ContractError::ElementCountOverflow)?;
        output_elements
            .checked_mul(2)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self { rows, width, dtype })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn width(self) -> usize {
        self.width
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn input_numel(self) -> usize {
        self.rows * self.width * 2
    }

    pub const fn output_numel(self) -> usize {
        self.rows * self.width
    }
}

impl AddRmsNormSpec {
    /// Creates a validated fused Add+RMSNorm contract.
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || hidden_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ContractError::InvalidEpsilon(epsilon));
        }
        rows.checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rows,
            hidden_size,
            epsilon,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }
}

impl RmsNormSpec {
    /// Creates a validated RMSNorm contract.
    pub fn new(
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
        dtype: DType,
    ) -> Result<Self, ContractError> {
        if rows == 0 || hidden_size == 0 {
            return Err(ContractError::ZeroDimension);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ContractError::InvalidEpsilon(epsilon));
        }
        rows.checked_mul(hidden_size)
            .ok_or(ContractError::ElementCountOverflow)?;

        Ok(Self {
            rows,
            hidden_size,
            epsilon,
            dtype,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows
    }

    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub const fn epsilon(self) -> f32 {
        self.epsilon
    }

    pub const fn dtype(self) -> DType {
        self.dtype
    }

    pub const fn numel(self) -> usize {
        self.rows * self.hidden_size
    }
}

/// Backend-independent operator description.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatorSpec {
    RmsNorm(RmsNormSpec),
    AddRmsNorm(AddRmsNormSpec),
    RmsNormDynamicFp8(RmsNormDynamicFp8Spec),
    SiluAndMul(SiluAndMulSpec),
    SiluAndMulDynamicFp8(SiluAndMulDynamicFp8Spec),
    GreedySampleLogprobs(GreedySampleLogprobsSpec),
    SelectedTokenLogprobs(SelectedTokenLogprobsSpec),
    RotaryEmbedding(RotaryEmbeddingSpec),
    RopePagedKvWrite(RopePagedKvWriteSpec),
}

/// Whether a backend can execute an operator contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Support {
    Supported,
    Unsupported(&'static str),
}

/// Capability interface shared by accelerator backends.
pub trait Backend {
    /// Stable identifier used in logs and benchmark artifacts.
    fn name(&self) -> &'static str;

    /// Reports support without launching work or silently falling back.
    fn supports(&self, operation: &OperatorSpec) -> Support;
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

/// Computes an F32 RMSNorm reference with F64 accumulation.
pub fn rms_norm_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [f32],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;

    for (input_row, output_row) in input
        .chunks_exact(spec.hidden_size())
        .zip(output.chunks_exact_mut(spec.hidden_size()))
    {
        let mean_square = input_row
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>()
            / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();

        for ((destination, &value), &scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            *destination = (f64::from(value) * inverse_rms * f64::from(scale)) as f32;
        }
    }

    Ok(())
}

/// Computes an F16 RMSNorm reference with F64 accumulation over quantized inputs.
pub fn rms_norm_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [f16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_low_precision_reference(input, weight, output, spec, DType::F16)
}

/// Computes a BF16 RMSNorm reference with F64 accumulation over quantized inputs.
pub fn rms_norm_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [bf16],
    spec: RmsNormSpec,
) -> Result<(), ContractError> {
    rms_norm_low_precision_reference(input, weight, output, spec, DType::Bf16)
}

/// Computes F32 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_f32_reference(
    input: &[f32],
    weight: &[f32],
    output: &mut [u8],
    scales: &mut [f32],
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, spec, DType::F32)
}

/// Computes FP16 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_f16_reference(
    input: &[f16],
    weight: &[f16],
    output: &mut [u8],
    scales: &mut [f32],
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, spec, DType::F16)
}

/// Computes BF16 RMSNorm followed by dynamic per-token FP8 E4M3FN quantization.
pub fn rms_norm_dynamic_fp8_bf16_reference(
    input: &[bf16],
    weight: &[bf16],
    output: &mut [u8],
    scales: &mut [f32],
    spec: RmsNormDynamicFp8Spec,
) -> Result<(), ContractError> {
    rms_norm_dynamic_fp8_reference(input, weight, output, scales, spec, DType::Bf16)
}

/// Decodes one OCP FP8 E4M3FN storage byte into F32.
pub fn fp8_e4m3fn_to_f32(bits: u8) -> f32 {
    let magnitude = bits & 0x7f;
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    if magnitude == 0x7f {
        return f32::NAN.copysign(sign);
    }

    let exponent = magnitude >> 3;
    let mantissa = magnitude & 0x07;
    let value = if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    sign * value
}

/// Encodes F32 as OCP FP8 E4M3FN using round-to-nearest-even and finite
/// saturation, matching CUDA's `__NV_SATFINITE` conversion behavior.
pub fn fp8_e4m3fn_from_f32(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0x00 };
    if value.is_nan() {
        return sign | 0x7f;
    }

    let magnitude = value.abs();
    if magnitude >= FP8_E4M3FN_MAX {
        return sign | 0x7e;
    }

    let mut best_bits = 0_u8;
    let mut best_distance = f32::INFINITY;
    for candidate in 0_u8..=0x7e {
        let decoded = fp8_e4m3fn_to_f32(candidate);
        let distance = (decoded - magnitude).abs();
        if distance < best_distance
            || (distance == best_distance && candidate & 1 == 0 && best_bits & 1 != 0)
        {
            best_bits = candidate;
            best_distance = distance;
        }
    }
    sign | best_bits
}

/// Computes fused in-place F32 Add+RMSNorm with F64 accumulation.
///
/// On success `residual` contains the elementwise sum and `input` contains its
/// normalized, weighted value. The two mutable slices must not alias.
pub fn add_rms_norm_f32_reference(
    input: &mut [f32],
    residual: &mut [f32],
    weight: &[f32],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("residual", residual.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;

    for (input_row, residual_row) in input
        .chunks_exact_mut(spec.hidden_size())
        .zip(residual.chunks_exact_mut(spec.hidden_size()))
    {
        let mut square_sum = 0.0_f64;
        for (input_value, residual_value) in input_row.iter().zip(residual_row.iter_mut()) {
            let sum = *input_value + *residual_value;
            *residual_value = sum;
            let sum = f64::from(sum);
            square_sum += sum * sum;
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();
        for ((destination, &sum), &scale) in
            input_row.iter_mut().zip(residual_row.iter()).zip(weight)
        {
            *destination = (f64::from(sum) * inverse_rms * f64::from(scale)) as f32;
        }
    }

    Ok(())
}

/// Computes fused in-place FP16 Add+RMSNorm.
///
/// The elementwise sum is rounded to FP16 before the RMS statistic is
/// accumulated, matching a materialized FP16 residual tensor.
pub fn add_rms_norm_f16_reference(
    input: &mut [f16],
    residual: &mut [f16],
    weight: &[f16],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    add_rms_norm_low_precision_reference(input, residual, weight, spec, DType::F16)
}

/// Computes fused in-place BF16 Add+RMSNorm.
///
/// The elementwise sum is rounded to BF16 before the RMS statistic is
/// accumulated, matching a materialized BF16 residual tensor.
pub fn add_rms_norm_bf16_reference(
    input: &mut [bf16],
    residual: &mut [bf16],
    weight: &[bf16],
    spec: AddRmsNormSpec,
) -> Result<(), ContractError> {
    add_rms_norm_low_precision_reference(input, residual, weight, spec, DType::Bf16)
}

/// Computes F32 `silu(gate) * up` over contiguous split-half rows.
pub fn silu_and_mul_f32_reference(
    input: &[f32],
    output: &mut [f32],
    spec: SiluAndMulSpec,
) -> Result<(), ContractError> {
    if spec.dtype() != DType::F32 {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.input_numel())?;
    require_len("output", output.len(), spec.output_numel())?;

    for (input_row, output_row) in input
        .chunks_exact(spec.width() * 2)
        .zip(output.chunks_exact_mut(spec.width()))
    {
        let (gate, up) = input_row.split_at(spec.width());
        for ((destination, &gate_value), &up_value) in output_row.iter_mut().zip(gate).zip(up) {
            let activated = gate_value / (1.0 + (-gate_value).exp());
            *destination = activated * up_value;
        }
    }
    Ok(())
}

/// Computes FP16 `silu(gate) * up` with vLLM-compatible storage rounding.
pub fn silu_and_mul_f16_reference(
    input: &[f16],
    output: &mut [f16],
    spec: SiluAndMulSpec,
) -> Result<(), ContractError> {
    silu_and_mul_low_precision_reference(input, output, spec, DType::F16)
}

/// Computes BF16 `silu(gate) * up` with vLLM-compatible storage rounding.
pub fn silu_and_mul_bf16_reference(
    input: &[bf16],
    output: &mut [bf16],
    spec: SiluAndMulSpec,
) -> Result<(), ContractError> {
    silu_and_mul_low_precision_reference(input, output, spec, DType::Bf16)
}

/// Computes FP16 SwiGLU followed by row-major dynamic per-block FP8.
pub fn silu_and_mul_dynamic_fp8_f16_reference(
    input: &[f16],
    output: &mut [u8],
    scales: &mut [f32],
    spec: SiluAndMulDynamicFp8Spec,
) -> Result<(), ContractError> {
    silu_and_mul_dynamic_fp8_reference(input, output, scales, spec, DType::F16)
}

/// Computes BF16 SwiGLU followed by row-major dynamic per-block FP8.
pub fn silu_and_mul_dynamic_fp8_bf16_reference(
    input: &[bf16],
    output: &mut [u8],
    scales: &mut [f32],
    spec: SiluAndMulDynamicFp8Spec,
) -> Result<(), ContractError> {
    silu_and_mul_dynamic_fp8_reference(input, output, scales, spec, DType::Bf16)
}

/// Selects the first maximum F32 logit per row and returns its log-softmax.
pub fn greedy_sample_logprobs_f32_reference(
    logits: &[f32],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::F32)
}

/// Selects the first maximum FP16 logit per row and returns its F32 log-softmax.
pub fn greedy_sample_logprobs_f16_reference(
    logits: &[f16],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::F16)
}

/// Selects the first maximum BF16 logit per row and returns its F32 log-softmax.
pub fn greedy_sample_logprobs_bf16_reference(
    logits: &[bf16],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
) -> Result<(), ContractError> {
    greedy_sample_logprobs_reference(logits, token_ids, logprobs, spec, DType::Bf16)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected F32 tokens.
pub fn selected_token_logprobs_f32_reference(
    logits: &[f32],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::F32)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected FP16 tokens.
pub fn selected_token_logprobs_f16_reference(
    logits: &[f16],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::F16)
}

/// Returns F32 logprobs and tie-aware ranks for caller-selected BF16 tokens.
pub fn selected_token_logprobs_bf16_reference(
    logits: &[bf16],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
) -> Result<(), ContractError> {
    selected_token_logprobs_reference(logits, token_ids, logprobs, ranks, spec, DType::Bf16)
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

trait LogitElement: Copy {
    fn to_f32(self) -> f32;
}

impl LogitElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl LogitElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl LogitElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

fn greedy_sample_logprobs_reference<T: LogitElement>(
    logits: &[T],
    token_ids: &mut [u32],
    logprobs: &mut [f32],
    spec: GreedySampleLogprobsSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("token_ids", token_ids.len(), spec.rows())?;
    require_len("logprobs", logprobs.len(), spec.rows())?;

    for ((row, token_id), logprob) in logits
        .chunks_exact(spec.vocab_size())
        .zip(token_ids.iter_mut())
        .zip(logprobs.iter_mut())
    {
        let mut maximum = row[0].to_f32();
        let mut maximum_index = 0_usize;
        for (index, &value) in row.iter().enumerate().skip(1) {
            let value = value.to_f32();
            if value > maximum {
                maximum = value;
                maximum_index = index;
            }
        }

        let exponential_sum = row
            .iter()
            .map(|&value| f64::from(value.to_f32() - maximum).exp())
            .sum::<f64>();
        *token_id = maximum_index as u32;
        *logprob = -(exponential_sum.ln() as f32);
    }
    Ok(())
}

fn selected_token_logprobs_reference<T: LogitElement>(
    logits: &[T],
    token_ids: &[i64],
    logprobs: &mut [f32],
    ranks: &mut [i64],
    spec: SelectedTokenLogprobsSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("token_ids", token_ids.len(), spec.rows())?;
    require_len("logprobs", logprobs.len(), spec.rows())?;
    require_len("ranks", ranks.len(), spec.rows())?;

    for (row_index, (((row, &token_id), logprob), rank)) in logits
        .chunks_exact(spec.vocab_size())
        .zip(token_ids.iter())
        .zip(logprobs.iter_mut())
        .zip(ranks.iter_mut())
        .enumerate()
    {
        let selected_index =
            usize::try_from(token_id).map_err(|_| ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            })?;
        if selected_index >= spec.vocab_size() {
            return Err(ContractError::TokenIdOutOfBounds {
                row: row_index,
                token_id,
                vocab_size: spec.vocab_size(),
            });
        }

        let selected = row[selected_index].to_f32();
        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let exponential_sum = row
            .iter()
            .map(|&value| f64::from(value.to_f32() - maximum).exp())
            .sum::<f64>();
        *logprob = selected - maximum - exponential_sum.ln() as f32;
        *rank = row
            .iter()
            .filter(|&&value| value.to_f32() >= selected)
            .count() as i64;
    }
    Ok(())
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
    apply_rotary_embedding_unchecked(query, key, positions, cos_sin_cache, spec);
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

    apply_rotary_embedding_unchecked(query, key, positions, cos_sin_cache, rotary);

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

fn apply_rotary_embedding_unchecked<T: RotaryElement>(
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

trait LowPrecisionElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

trait DynamicFp8Input: Copy {
    fn to_f32(self) -> f32;
    fn round_to_storage(value: f32) -> f32;
}

impl DynamicFp8Input for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn round_to_storage(value: f32) -> f32 {
        value
    }
}

impl DynamicFp8Input for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn round_to_storage(value: f32) -> f32 {
        Self::from_f32(value).to_f32()
    }
}

impl DynamicFp8Input for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn round_to_storage(value: f32) -> f32 {
        Self::from_f32(value).to_f32()
    }
}

impl LowPrecisionElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl LowPrecisionElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

fn rms_norm_low_precision_reference<T: LowPrecisionElement>(
    input: &[T],
    weight: &[T],
    output: &mut [T],
    spec: RmsNormSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;

    for (input_row, output_row) in input
        .chunks_exact(spec.hidden_size())
        .zip(output.chunks_exact_mut(spec.hidden_size()))
    {
        let mean_square = input_row
            .iter()
            .map(|&value| {
                let value = f64::from(value.to_f32());
                value * value
            })
            .sum::<f64>()
            / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();

        for ((destination, &value), &scale) in output_row.iter_mut().zip(input_row).zip(weight) {
            let normalized = f64::from(value.to_f32()) * inverse_rms * f64::from(scale.to_f32());
            *destination = T::from_f32(normalized as f32);
        }
    }

    Ok(())
}

fn rms_norm_dynamic_fp8_reference<T: DynamicFp8Input>(
    input: &[T],
    weight: &[T],
    output: &mut [u8],
    scales: &mut [f32],
    spec: RmsNormDynamicFp8Spec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.input_dtype() != expected_dtype || spec.output_dtype() != DType::Fp8E4M3Fn {
        return Err(ContractError::UnsupportedDType(spec.input_dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;
    require_len("output", output.len(), spec.numel())?;
    require_len("scales", scales.len(), spec.scale_count())?;

    let mut normalized = vec![0.0_f32; spec.hidden_size()];
    for ((input_row, output_row), scale) in input
        .chunks_exact(spec.hidden_size())
        .zip(output.chunks_exact_mut(spec.hidden_size()))
        .zip(scales.iter_mut())
    {
        let mean_square = input_row
            .iter()
            .map(|&value| {
                let value = f64::from(value.to_f32());
                value * value
            })
            .sum::<f64>()
            / spec.hidden_size() as f64;
        let inverse_rms = (1.0 / (mean_square + f64::from(spec.epsilon())).sqrt()) as f32;

        let mut absolute_maximum = 0.0_f32;
        for (column, (&value, &weight_value)) in input_row.iter().zip(weight).enumerate() {
            let rounded_normalized = T::round_to_storage(value.to_f32() * inverse_rms);
            let weighted = T::round_to_storage(rounded_normalized * weight_value.to_f32());
            normalized[column] = weighted;
            absolute_maximum = absolute_maximum.max(weighted.abs());
        }

        *scale = (absolute_maximum / FP8_E4M3FN_MAX).max(DYNAMIC_FP8_MIN_SCALE);
        for (destination, &value) in output_row.iter_mut().zip(&normalized) {
            *destination = fp8_e4m3fn_from_f32(value / *scale);
        }
    }

    Ok(())
}

fn add_rms_norm_low_precision_reference<T: LowPrecisionElement>(
    input: &mut [T],
    residual: &mut [T],
    weight: &[T],
    spec: AddRmsNormSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.numel())?;
    require_len("residual", residual.len(), spec.numel())?;
    require_len("weight", weight.len(), spec.hidden_size())?;

    for (input_row, residual_row) in input
        .chunks_exact_mut(spec.hidden_size())
        .zip(residual.chunks_exact_mut(spec.hidden_size()))
    {
        let mut square_sum = 0.0_f64;
        for (input_value, residual_value) in input_row.iter().zip(residual_row.iter_mut()) {
            let quantized_sum = T::from_f32(input_value.to_f32() + residual_value.to_f32());
            *residual_value = quantized_sum;
            let sum = f64::from(quantized_sum.to_f32());
            square_sum += sum * sum;
        }

        let mean_square = square_sum / spec.hidden_size() as f64;
        let inverse_rms = 1.0 / (mean_square + f64::from(spec.epsilon())).sqrt();
        for ((destination, &sum), &scale) in
            input_row.iter_mut().zip(residual_row.iter()).zip(weight)
        {
            let normalized = f64::from(sum.to_f32()) * inverse_rms * f64::from(scale.to_f32());
            *destination = T::from_f32(normalized as f32);
        }
    }

    Ok(())
}

fn silu_and_mul_low_precision_reference<T: LowPrecisionElement>(
    input: &[T],
    output: &mut [T],
    spec: SiluAndMulSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("input", input.len(), spec.input_numel())?;
    require_len("output", output.len(), spec.output_numel())?;

    for (input_row, output_row) in input
        .chunks_exact(spec.width() * 2)
        .zip(output.chunks_exact_mut(spec.width()))
    {
        let (gate, up) = input_row.split_at(spec.width());
        for ((destination, &gate_value), &up_value) in output_row.iter_mut().zip(gate).zip(up) {
            let gate_value = gate_value.to_f32();
            let activated = T::from_f32(gate_value / (1.0 + (-gate_value).exp()));
            *destination = T::from_f32(activated.to_f32() * up_value.to_f32());
        }
    }
    Ok(())
}

fn silu_and_mul_dynamic_fp8_reference<T: LowPrecisionElement>(
    input: &[T],
    output: &mut [u8],
    scales: &mut [f32],
    spec: SiluAndMulDynamicFp8Spec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.input_dtype() != expected_dtype || spec.output_dtype() != DType::Fp8E4M3Fn {
        return Err(ContractError::UnsupportedDType(spec.input_dtype()));
    }
    require_len("input", input.len(), spec.input_numel())?;
    require_len("output", output.len(), spec.output_numel())?;
    require_len("scales", scales.len(), spec.scale_count())?;

    for row in 0..spec.rows() {
        let input_offset = row * spec.width() * 2;
        let output_offset = row * spec.width();
        let gate = &input[input_offset..input_offset + spec.width()];
        let up = &input[input_offset + spec.width()..input_offset + spec.width() * 2];
        let output_row = &mut output[output_offset..output_offset + spec.width()];
        let scale_offset = row * spec.group_count();
        let scale_row = &mut scales[scale_offset..scale_offset + spec.group_count()];

        for (group_index, ((gate_group, up_group), output_group)) in gate
            .chunks_exact(spec.group_size())
            .zip(up.chunks_exact(spec.group_size()))
            .zip(output_row.chunks_exact_mut(spec.group_size()))
            .enumerate()
        {
            let absolute_maximum = gate_group
                .iter()
                .zip(up_group)
                .map(|(&gate_value, &up_value)| {
                    let gate_value = gate_value.to_f32();
                    let sigmoid_gate = 1.0 / (1.0 + (-gate_value).exp());
                    let activated = gate_value * sigmoid_gate;
                    (activated * up_value.to_f32()).abs()
                })
                .fold(0.0_f32, f32::max);
            let scale = (absolute_maximum / FP8_E4M3FN_MAX).max(DYNAMIC_FP8_MIN_SCALE);
            scale_row[group_index] = scale;

            for ((destination, &gate_value), &up_value) in
                output_group.iter_mut().zip(gate_group).zip(up_group)
            {
                let gate_value = gate_value.to_f32();
                let sigmoid_gate = 1.0 / (1.0 + (-gate_value).exp());
                let activated = gate_value * sigmoid_gate;
                *destination = fp8_e4m3fn_from_f32(activated * up_value.to_f32() / scale);
            }
        }
    }
    Ok(())
}

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

fn require_len(buffer: &'static str, actual: usize, expected: usize) -> Result<(), ContractError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_tensor_has_expected_strides() {
        let tensor = TensorSpec::contiguous(DType::Bf16, vec![2, 3, 5]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3, 5]);
        assert_eq!(tensor.strides(), &[15, 5, 1]);
        assert_eq!(tensor.numel(), 30);
        assert_eq!(tensor.size_in_bytes(), 60);
    }

    #[test]
    fn invalid_shapes_are_rejected() {
        assert_eq!(
            TensorSpec::contiguous(DType::F32, vec![]),
            Err(ContractError::EmptyShape)
        );
        assert_eq!(
            TensorSpec::contiguous(DType::F32, vec![2, 0]),
            Err(ContractError::ZeroDimension)
        );
    }

    #[test]
    fn rms_norm_matches_hand_computed_result() {
        let spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::F32).unwrap();
        let input = [3.0, 4.0];
        let weight = [1.0, 0.5];
        let mut output = [0.0; 2];

        rms_norm_f32_reference(&input, &weight, &mut output, spec).unwrap();

        let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
        assert!((output[0] - 3.0 * inverse_rms).abs() < 1.0e-6);
        assert!((output[1] - 2.0 * inverse_rms).abs() < 1.0e-6);
    }

    #[test]
    fn rms_norm_validates_every_buffer() {
        let spec = RmsNormSpec::new(2, 4, 1.0e-5, DType::F32).unwrap();
        let error = rms_norm_f32_reference(&[0.0; 7], &[1.0; 4], &mut [0.0; 8], spec).unwrap_err();
        assert_eq!(
            error,
            ContractError::LengthMismatch {
                buffer: "input",
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn low_precision_references_quantize_the_f32_result() {
        let input_f32 = [3.0_f32, 4.0];
        let weight_f32 = [1.0_f32, 0.5];

        let f16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::F16).unwrap();
        let input_f16 = input_f32.map(f16::from_f32);
        let weight_f16 = weight_f32.map(f16::from_f32);
        let mut output_f16 = [f16::ZERO; 2];
        rms_norm_f16_reference(&input_f16, &weight_f16, &mut output_f16, f16_spec).unwrap();

        let bf16_spec = RmsNormSpec::new(1, 2, 1.0e-6, DType::Bf16).unwrap();
        let input_bf16 = input_f32.map(bf16::from_f32);
        let weight_bf16 = weight_f32.map(bf16::from_f32);
        let mut output_bf16 = [bf16::ZERO; 2];
        rms_norm_bf16_reference(&input_bf16, &weight_bf16, &mut output_bf16, bf16_spec).unwrap();

        let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
        let expected = [3.0 * inverse_rms, 2.0 * inverse_rms];
        for (actual, expected) in output_f16.iter().map(|value| value.to_f32()).zip(expected) {
            assert!((actual - expected).abs() < 1.0e-3);
        }
        for (actual, expected) in output_bf16.iter().map(|value| value.to_f32()).zip(expected) {
            assert!((actual - expected).abs() < 1.0e-2);
        }
    }

    #[test]
    fn add_rms_norm_updates_both_f32_buffers() {
        let spec = AddRmsNormSpec::new(1, 2, 1.0e-6, DType::F32).unwrap();
        let mut input = [1.0_f32, 2.0];
        let mut residual = [2.0_f32, 2.0];
        let weight = [1.0_f32, 0.5];

        add_rms_norm_f32_reference(&mut input, &mut residual, &weight, spec).unwrap();

        assert_eq!(residual, [3.0, 4.0]);
        let inverse_rms = 1.0_f32 / (12.5_f32 + 1.0e-6).sqrt();
        assert!((input[0] - 3.0 * inverse_rms).abs() < 1.0e-6);
        assert!((input[1] - 2.0 * inverse_rms).abs() < 1.0e-6);
    }

    #[test]
    fn add_rms_norm_low_precision_materializes_quantized_residual() {
        let mut input = [f16::from_f32(0.3333), f16::from_f32(-0.7777)];
        let mut residual = [f16::from_f32(0.1111), f16::from_f32(0.2222)];
        let original_input = input;
        let original_residual = residual;
        let weight = [f16::ONE; 2];
        let spec = AddRmsNormSpec::new(1, 2, 1.0e-5, DType::F16).unwrap();

        add_rms_norm_f16_reference(&mut input, &mut residual, &weight, spec).unwrap();

        for index in 0..2 {
            assert_eq!(
                residual[index],
                f16::from_f32(original_input[index].to_f32() + original_residual[index].to_f32())
            );
        }
        assert!(input.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn add_rms_norm_validates_residual_length_and_dtype() {
        let spec = AddRmsNormSpec::new(2, 4, 1.0e-5, DType::F32).unwrap();
        let error =
            add_rms_norm_f32_reference(&mut [0.0; 8], &mut [0.0; 7], &[1.0; 4], spec).unwrap_err();
        assert_eq!(
            error,
            ContractError::LengthMismatch {
                buffer: "residual",
                expected: 8,
                actual: 7,
            }
        );

        let wrong_dtype = AddRmsNormSpec::new(1, 2, 1.0e-5, DType::Bf16).unwrap();
        assert_eq!(
            add_rms_norm_f16_reference(
                &mut [f16::ZERO; 2],
                &mut [f16::ZERO; 2],
                &[f16::ONE; 2],
                wrong_dtype,
            ),
            Err(ContractError::UnsupportedDType(DType::Bf16))
        );
    }

    #[test]
    fn fp8_e4m3fn_encoding_matches_known_values_and_ties() {
        let fixtures = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (1.0, 0x38),
            (-1.0, 0xb8),
            (448.0, 0x7e),
            (500.0, 0x7e),
            (2.0_f32.powi(-9), 0x01),
            (1.0625, 0x38),
            (1.1875, 0x3a),
        ];
        for (value, expected) in fixtures {
            assert_eq!(fp8_e4m3fn_from_f32(value), expected, "value={value}");
        }
        assert_eq!(fp8_e4m3fn_from_f32(f32::NAN), 0x7f);
        assert_eq!(fp8_e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x7e), 448.0);
    }

    #[test]
    fn dynamic_fp8_reference_emits_per_row_scale_and_zero_floor() {
        let spec = RmsNormDynamicFp8Spec::new(2, 2, 1.0e-6, DType::Bf16).unwrap();
        let input = [
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::ZERO,
            bf16::ZERO,
        ];
        let weight = [bf16::ONE, bf16::ONE];
        let mut output = [0_u8; 4];
        let mut scales = [0.0_f32; 2];

        rms_norm_dynamic_fp8_bf16_reference(&input, &weight, &mut output, &mut scales, spec)
            .unwrap();

        assert_eq!(output[1], 0x7e);
        assert_eq!(&output[2..], &[0x00, 0x00]);
        assert!(scales[0] > DYNAMIC_FP8_MIN_SCALE);
        assert_eq!(scales[1], DYNAMIC_FP8_MIN_SCALE);
    }

    #[test]
    fn dynamic_fp8_reference_validates_output_and_scale_lengths() {
        let spec = RmsNormDynamicFp8Spec::new(2, 4, 1.0e-5, DType::F16).unwrap();
        let error = rms_norm_dynamic_fp8_f16_reference(
            &[f16::ZERO; 8],
            &[f16::ONE; 4],
            &mut [0_u8; 7],
            &mut [0.0; 2],
            spec,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ContractError::LengthMismatch {
                buffer: "output",
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn silu_and_mul_matches_split_half_definition() {
        let spec = SiluAndMulSpec::new(1, 3, DType::F32).unwrap();
        let input = [0.0_f32, 1.0, -1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 3];

        silu_and_mul_f32_reference(&input, &mut output, spec).unwrap();

        assert_eq!(output[0], 0.0);
        assert!((output[1] - 3.0 / (1.0 + (-1.0_f32).exp())).abs() < 1.0e-6);
        assert!((output[2] - (-4.0 / (1.0 + 1.0_f32.exp()))).abs() < 1.0e-6);
    }

    #[test]
    fn silu_and_mul_low_precision_rounds_activation_before_multiply() {
        let spec = SiluAndMulSpec::new(1, 1, DType::F16).unwrap();
        let gate = f16::from_f32(0.3333);
        let up = f16::from_f32(1.7777);
        let mut output = [f16::ZERO];

        silu_and_mul_f16_reference(&[gate, up], &mut output, spec).unwrap();

        let gate_f32 = gate.to_f32();
        let activated = f16::from_f32(gate_f32 / (1.0 + (-gate_f32).exp()));
        let expected = f16::from_f32(activated.to_f32() * up.to_f32());
        assert_eq!(output[0], expected);
    }

    #[test]
    fn silu_and_mul_validates_buffer_lengths_and_dtype() {
        let spec = SiluAndMulSpec::new(2, 4, DType::Bf16).unwrap();
        let error =
            silu_and_mul_bf16_reference(&[bf16::ZERO; 15], &mut [bf16::ZERO; 8], spec).unwrap_err();
        assert_eq!(
            error,
            ContractError::LengthMismatch {
                buffer: "input",
                expected: 16,
                actual: 15,
            }
        );

        let wrong_dtype = SiluAndMulSpec::new(1, 2, DType::F32).unwrap();
        assert_eq!(
            silu_and_mul_f16_reference(&[f16::ZERO; 4], &mut [f16::ZERO; 2], wrong_dtype,),
            Err(ContractError::UnsupportedDType(DType::F32))
        );
    }

    #[test]
    fn silu_and_mul_dynamic_fp8_validates_group_contract() {
        assert_eq!(
            SiluAndMulDynamicFp8Spec::new(1, 128, 32, DType::F16),
            Err(ContractError::InvalidGroupSize(32))
        );
        assert_eq!(
            SiluAndMulDynamicFp8Spec::new(1, 192, 128, DType::Bf16),
            Err(ContractError::WidthNotDivisible {
                width: 192,
                group_size: 128,
            })
        );
    }

    #[test]
    fn silu_and_mul_dynamic_fp8_uses_f32_activation_and_per_group_scales() {
        let spec = SiluAndMulDynamicFp8Spec::new(1, 128, 64, DType::F16).unwrap();
        let gate_value = f16::from_f32(0.3333);
        let up_value = f16::from_f32(1.7777);
        let mut input = [f16::ZERO; 256];
        input[0] = gate_value;
        input[128] = up_value;
        let mut output = [0_u8; 128];
        let mut scales = [0.0_f32; 2];

        silu_and_mul_dynamic_fp8_f16_reference(&input, &mut output, &mut scales, spec).unwrap();

        let gate_f32 = gate_value.to_f32();
        let sigmoid_gate = 1.0 / (1.0 + (-gate_f32).exp());
        let full_precision = gate_f32 * sigmoid_gate * up_value.to_f32();
        assert_eq!(scales[0], full_precision.abs() / FP8_E4M3FN_MAX);
        assert_eq!(scales[1], DYNAMIC_FP8_MIN_SCALE);
        assert_eq!(output[0], 0x7e);
        assert!(output[1..].iter().all(|&value| value == 0));
    }

    #[test]
    fn silu_and_mul_dynamic_fp8_validates_buffers_and_dtype() {
        let spec = SiluAndMulDynamicFp8Spec::new(2, 64, 64, DType::Bf16).unwrap();
        let error = silu_and_mul_dynamic_fp8_bf16_reference(
            &[bf16::ZERO; 256],
            &mut [0_u8; 127],
            &mut [0.0_f32; 2],
            spec,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ContractError::LengthMismatch {
                buffer: "output",
                expected: 128,
                actual: 127,
            }
        );

        let wrong_dtype = SiluAndMulDynamicFp8Spec::new(1, 64, 64, DType::F32).unwrap();
        assert_eq!(
            silu_and_mul_dynamic_fp8_f16_reference(
                &[f16::ZERO; 128],
                &mut [0_u8; 64],
                &mut [0.0_f32; 1],
                wrong_dtype,
            ),
            Err(ContractError::UnsupportedDType(DType::F32))
        );
    }

    #[test]
    fn greedy_sample_logprobs_selects_first_tie_and_normalizes() {
        let spec = GreedySampleLogprobsSpec::new(2, 4, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, -2.0, -1.0, 2.0, 0.0];
        let mut token_ids = [u32::MAX; 2];
        let mut logprobs = [0.0_f32; 2];

        greedy_sample_logprobs_f32_reference(&logits, &mut token_ids, &mut logprobs, spec).unwrap();

        assert_eq!(token_ids, [1, 2]);
        let first_sum = (-2.0_f64).exp() + 1.0 + 1.0 + (-4.0_f64).exp();
        let second_sum = (-4.0_f64).exp() + (-3.0_f64).exp() + 1.0 + (-2.0_f64).exp();
        assert!((logprobs[0] + first_sum.ln() as f32).abs() < 1.0e-6);
        assert!((logprobs[1] + second_sum.ln() as f32).abs() < 1.0e-6);
    }

    #[test]
    fn greedy_sample_logprobs_supports_low_precision_and_validates_buffers() {
        let spec = GreedySampleLogprobsSpec::new(1, 3, DType::Bf16).unwrap();
        let logits = [
            bf16::from_f32(-1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(0.5),
        ];
        let mut token_ids = [u32::MAX];
        let mut logprobs = [0.0_f32];
        greedy_sample_logprobs_bf16_reference(&logits, &mut token_ids, &mut logprobs, spec)
            .unwrap();
        assert_eq!(token_ids, [1]);
        assert!(logprobs[0].is_finite() && logprobs[0] < 0.0);

        assert_eq!(
            greedy_sample_logprobs_bf16_reference(&logits, &mut [u32::MAX; 2], &mut logprobs, spec,),
            Err(ContractError::LengthMismatch {
                buffer: "token_ids",
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn selected_token_logprobs_normalizes_and_counts_tie_aware_ranks() {
        let spec = SelectedTokenLogprobsSpec::new(2, 4, DType::F32).unwrap();
        let logits = [1.0_f32, 3.0, 3.0, -1.0, -2.0, -1.0, 2.0, 0.0];
        let token_ids = [0_i64, 1_i64];
        let mut logprobs = [0.0_f32; 2];
        let mut ranks = [0_i64; 2];

        selected_token_logprobs_f32_reference(&logits, &token_ids, &mut logprobs, &mut ranks, spec)
            .unwrap();

        let first_sum = (-2.0_f64).exp() + 1.0 + 1.0 + (-4.0_f64).exp();
        let second_sum = (-4.0_f64).exp() + (-3.0_f64).exp() + 1.0 + (-2.0_f64).exp();
        assert!((logprobs[0] - (-2.0 - first_sum.ln() as f32)).abs() < 1.0e-6);
        assert!((logprobs[1] - (-3.0 - second_sum.ln() as f32)).abs() < 1.0e-6);
        assert_eq!(ranks, [3, 3]);
    }

    #[test]
    fn selected_token_logprobs_validates_ids_and_low_precision_buffers() {
        let spec = SelectedTokenLogprobsSpec::new(1, 3, DType::Bf16).unwrap();
        let logits = [
            bf16::from_f32(-1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(0.5),
        ];
        let mut logprobs = [0.0_f32];
        let mut ranks = [0_i64];
        selected_token_logprobs_bf16_reference(&logits, &[2_i64], &mut logprobs, &mut ranks, spec)
            .unwrap();
        assert!(logprobs[0].is_finite() && logprobs[0] < 0.0);
        assert_eq!(ranks, [2]);

        assert_eq!(
            selected_token_logprobs_bf16_reference(
                &logits,
                &[-1_i64],
                &mut logprobs,
                &mut ranks,
                spec,
            ),
            Err(ContractError::TokenIdOutOfBounds {
                row: 0,
                token_id: -1,
                vocab_size: 3,
            })
        );
        assert_eq!(
            selected_token_logprobs_bf16_reference(
                &logits,
                &[3_i64],
                &mut logprobs,
                &mut ranks,
                spec,
            ),
            Err(ContractError::TokenIdOutOfBounds {
                row: 0,
                token_id: 3,
                vocab_size: 3,
            })
        );
    }

    #[test]
    fn rotary_contract_rejects_invalid_partial_dimensions() {
        assert_eq!(
            RotaryEmbeddingSpec::new(1, 2, 1, 8, 3, 16, DType::F16, RotaryStyle::NeoX,),
            Err(ContractError::InvalidRotaryDimension {
                rotary_dim: 3,
                head_size: 8,
            })
        );
        assert_eq!(
            RotaryEmbeddingSpec::new(1, 2, 1, 8, 10, 16, DType::F16, RotaryStyle::NeoX,),
            Err(ContractError::InvalidRotaryDimension {
                rotary_dim: 10,
                head_size: 8,
            })
        );
    }

    #[test]
    fn rotary_reference_supports_both_pairing_styles_and_partial_rope() {
        let cos_sin_cache = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let positions = [1_i64];

        let neox =
            RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
        let mut neox_query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut neox_key = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        rotary_embedding_f32_reference(
            &mut neox_query,
            &mut neox_key,
            &positions,
            &cos_sin_cache,
            neox,
        )
        .unwrap();
        assert_eq!(neox_query, [-3.0, -4.0, 1.0, 2.0, 5.0, 6.0]);
        assert_eq!(neox_key, [-9.0, -10.0, 7.0, 8.0, 11.0, 12.0]);

        let interleaved =
            RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::Interleaved)
                .unwrap();
        let mut interleaved_query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut interleaved_key = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        rotary_embedding_f32_reference(
            &mut interleaved_query,
            &mut interleaved_key,
            &positions,
            &cos_sin_cache,
            interleaved,
        )
        .unwrap();
        assert_eq!(interleaved_query, [-2.0, 1.0, -4.0, 3.0, 5.0, 6.0]);
        assert_eq!(interleaved_key, [-8.0, 7.0, -10.0, 9.0, 11.0, 12.0]);
    }

    #[test]
    fn fused_rope_paged_write_rotates_padding_but_skips_its_cache_slot() {
        let rotary =
            RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
        let spec = RopePagedKvWriteSpec::new(rotary, 2, 2, 2).unwrap();
        let mut query = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut key = [9.0_f32, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let value = [17.0_f32, 18.0, 19.0, 20.0];
        let positions = [0_i64, 1];
        let cos_sin_cache = [1.0_f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let slots = [3_i64, -1];
        let mut key_cache = [-99.0_f32; 16];
        let mut value_cache = [-99.0_f32; 8];

        rope_paged_kv_write_f32_reference(
            &mut query,
            &mut key,
            &value,
            &positions,
            &cos_sin_cache,
            &mut key_cache,
            &mut value_cache,
            &slots,
            spec,
        )
        .unwrap();

        assert_eq!(&query[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&query[4..], &[-7.0, -8.0, 5.0, 6.0]);
        assert_eq!(&key[4..], &[-15.0, -16.0, 13.0, 14.0]);
        assert!(key_cache[..12].iter().all(|&value| value == -99.0));
        assert_eq!(&key_cache[12..], &[9.0, 10.0, 11.0, 12.0]);
        assert!(value_cache[..6].iter().all(|&value| value == -99.0));
        assert_eq!(&value_cache[6..], &[17.0, 18.0]);
    }

    #[test]
    fn fused_rope_paged_write_rejects_bad_metadata_before_mutation() {
        let rotary =
            RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
        let spec = RopePagedKvWriteSpec::new(rotary, 4, 1, 2).unwrap();
        let original = [1.0_f32; 8];
        let mut query = original;
        let mut key = original;
        let value = [2.0_f32; 8];
        let cache = [1.0_f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let mut key_cache = [0.0_f32; 8];
        let mut value_cache = [0.0_f32; 8];

        let error = rope_paged_kv_write_f32_reference(
            &mut query,
            &mut key,
            &value,
            &[0, 1],
            &cache,
            &mut key_cache,
            &mut value_cache,
            &[1, 1],
            spec,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ContractError::DuplicateSlot {
                first_token: 0,
                second_token: 1,
                slot: 1,
            }
        );
        assert_eq!(query, original);
        assert_eq!(key, original);

        let error = rotary_embedding_f32_reference(&mut query, &mut key, &[0, 2], &cache, rotary)
            .unwrap_err();
        assert_eq!(
            error,
            ContractError::PositionOutOfBounds {
                token: 1,
                position: 2,
                max_position: 2,
            }
        );
        assert_eq!(query, original);
        assert_eq!(key, original);
    }
}
