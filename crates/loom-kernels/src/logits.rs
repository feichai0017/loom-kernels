//! Logits-processing contracts and CPU reference implementations.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};

/// Contract for in-place min-p filtering over rank-2 logits.
///
/// Each row keeps tokens whose probability is at least `min_p[row]` times
/// the row's maximum probability. The softmax denominator cancels, so
/// backends can apply the equivalent threshold
/// `logit >= max(logits) + log(min_p)` without materializing probabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinPFilterSpec {
    rows: usize,
    vocab_size: usize,
    dtype: DType,
}

impl MinPFilterSpec {
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

/// Applies min-p filtering to F32 logits in place.
pub fn min_p_filter_f32_reference(
    logits: &mut [f32],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::F32)
}

/// Applies min-p filtering to FP16 logits in place.
pub fn min_p_filter_f16_reference(
    logits: &mut [f16],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::F16)
}

/// Applies min-p filtering to BF16 logits in place.
pub fn min_p_filter_bf16_reference(
    logits: &mut [bf16],
    min_p: &[f32],
    spec: MinPFilterSpec,
) -> Result<(), ContractError> {
    min_p_filter_reference(logits, min_p, spec, DType::Bf16)
}

trait MinPElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl MinPElement for f32 {
    fn to_f32(self) -> f32 {
        self
    }

    fn from_f32(value: f32) -> Self {
        value
    }
}

impl MinPElement for f16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl MinPElement for bf16 {
    fn to_f32(self) -> f32 {
        self.to_f32()
    }

    fn from_f32(value: f32) -> Self {
        Self::from_f32(value)
    }
}

fn min_p_filter_reference<T: MinPElement>(
    logits: &mut [T],
    min_p: &[f32],
    spec: MinPFilterSpec,
    expected_dtype: DType,
) -> Result<(), ContractError> {
    if spec.dtype() != expected_dtype {
        return Err(ContractError::UnsupportedDType(spec.dtype()));
    }
    require_len("logits", logits.len(), spec.logits_numel())?;
    require_len("min_p", min_p.len(), spec.rows())?;
    for (row, &probability) in min_p.iter().enumerate() {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(ContractError::InvalidProbability {
                parameter: "min_p",
                row,
                value: probability,
            });
        }
    }

    for (row, &probability) in logits.chunks_exact_mut(spec.vocab_size()).zip(min_p) {
        if probability == 0.0 {
            continue;
        }
        let maximum = row
            .iter()
            .map(|&value| value.to_f32())
            .fold(f32::NEG_INFINITY, f32::max);
        let threshold = maximum + probability.ln();
        for value in row {
            if value.to_f32() < threshold {
                *value = T::from_f32(f32::NEG_INFINITY);
            }
        }
    }
    Ok(())
}
