//! Activation contracts and CPU reference implementations.

use half::{bf16, f16};

use crate::contract::{require_len, ContractError, DType};
use crate::element::LowPrecisionElement;
use crate::quantization::{fp8_e4m3fn_from_f32, DYNAMIC_FP8_MIN_SCALE, FP8_E4M3FN_MAX};

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
