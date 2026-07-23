use super::*;
use half::{bf16, f16};

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

    rms_norm_dynamic_fp8_bf16_reference(&input, &weight, &mut output, &mut scales, spec).unwrap();

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
    greedy_sample_logprobs_bf16_reference(&logits, &mut token_ids, &mut logprobs, spec).unwrap();
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
        selected_token_logprobs_bf16_reference(&logits, &[-1_i64], &mut logprobs, &mut ranks, spec,),
        Err(ContractError::TokenIdOutOfBounds {
            row: 0,
            token_id: -1,
            vocab_size: 3,
        })
    );
    assert_eq!(
        selected_token_logprobs_bf16_reference(&logits, &[3_i64], &mut logprobs, &mut ranks, spec,),
        Err(ContractError::TokenIdOutOfBounds {
            row: 0,
            token_id: 3,
            vocab_size: 3,
        })
    );
}

#[test]
fn min_p_filter_matches_the_softmax_ratio_definition() {
    let spec = MinPFilterSpec::new(3, 4, DType::F32).unwrap();
    let original = [
        1.0_f32, 3.0, 2.0, -1.0, //
        -2.0, -1.0, 2.0, 0.0, //
        4.0, 4.0, 3.0, -8.0,
    ];
    let mut logits = original;

    min_p_filter_f32_reference(&mut logits, &[0.0, 0.2, 1.0], spec).unwrap();

    assert_eq!(&logits[..4], &original[..4]);
    let threshold = 2.0 + 0.2_f32.ln();
    for (actual, &input) in logits[4..8].iter().zip(&original[4..8]) {
        if input < threshold {
            assert_eq!(*actual, f32::NEG_INFINITY);
        } else {
            assert_eq!(*actual, input);
        }
    }
    assert_eq!(
        &logits[8..],
        &[4.0, 4.0, f32::NEG_INFINITY, f32::NEG_INFINITY]
    );
}

#[test]
fn min_p_filter_validates_metadata_before_mutating_logits() {
    let spec = MinPFilterSpec::new(2, 2, DType::F16).unwrap();
    let original = [
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
        f16::from_f32(4.0),
    ];
    let mut logits = original;

    let error = min_p_filter_f16_reference(&mut logits, &[0.5, 1.1], spec).unwrap_err();

    assert_eq!(
        error,
        ContractError::InvalidProbability {
            parameter: "min_p",
            row: 1,
            value: 1.1,
        }
    );
    assert_eq!(logits, original);
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

    let neox = RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
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
        RotaryEmbeddingSpec::new(1, 1, 1, 6, 4, 2, DType::F32, RotaryStyle::Interleaved).unwrap();
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
    let rotary = RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
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
    let rotary = RotaryEmbeddingSpec::new(2, 1, 1, 4, 4, 2, DType::F32, RotaryStyle::NeoX).unwrap();
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

    let error =
        rotary_embedding_f32_reference(&mut query, &mut key, &[0, 2], &cache, rotary).unwrap_err();
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

#[test]
fn paged_decode_attention_follows_block_indirection_and_gqa_mapping() {
    let spec = PagedDecodeAttentionSpec::new(1, 2, 1, 1, 1, 2, 2, 2, 4, 1.0, DType::F32).unwrap();
    let query = [1.0_f32, 1.0];
    // Physical block 0 is the second logical block. Its second slot is
    // outside the active sequence and must not affect the result.
    let key_cache = [2.0_f32, 99.0, 0.0, 1.0];
    let value_cache = [20.0_f32, 99.0, 1.0, 10.0];
    let block_tables = [1_i64, 0];
    let mut output = [-1.0_f32; 2];

    paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &block_tables,
        &[3],
        &mut output,
        spec,
    )
    .unwrap();

    let expected =
        (1.0 + 10.0 * 1.0_f32.exp() + 20.0 * 2.0_f32.exp()) / (1.0 + 1.0_f32.exp() + 2.0_f32.exp());
    assert!((output[0] - expected).abs() < 1.0e-6);
    assert!((output[1] - expected).abs() < 1.0e-6);
}

#[test]
fn paged_decode_attention_supports_low_precision_and_distinct_value_width() {
    let spec = PagedDecodeAttentionSpec::new(1, 1, 1, 2, 3, 1, 1, 1, 1, 0.5, DType::Bf16).unwrap();
    let query = [bf16::from_f32(2.0), bf16::from_f32(-1.0)];
    let key_cache = [bf16::from_f32(4.0), bf16::from_f32(3.0)];
    let value_cache = [
        bf16::from_f32(1.25),
        bf16::from_f32(-2.5),
        bf16::from_f32(7.0),
    ];
    let mut output = [bf16::ZERO; 3];

    paged_decode_attention_bf16_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0],
        &[1],
        &mut output,
        spec,
    )
    .unwrap();

    assert_eq!(output, value_cache);
}

#[test]
fn paged_decode_attention_validates_metadata_before_mutating_output() {
    assert_eq!(
        PagedDecodeAttentionSpec::new(1, 3, 2, 4, 4, 1, 16, 1, 16, 0.5, DType::F16),
        Err(ContractError::HeadCountNotDivisible {
            query_heads: 3,
            kv_heads: 2,
        })
    );
    assert_eq!(
        PagedDecodeAttentionSpec::new(1, 2, 1, 4, 4, 1, 16, 1, 16, 0.0, DType::F16),
        Err(ContractError::InvalidScale(0.0))
    );

    let spec = PagedDecodeAttentionSpec::new(2, 1, 1, 1, 1, 2, 2, 2, 4, 1.0, DType::F32).unwrap();
    let query = [1.0_f32; 2];
    let key_cache = [1.0_f32; 4];
    let value_cache = [2.0_f32; 4];
    let mut output = [-7.0_f32; 2];

    let error = paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0, 1, 1, -1],
        &[3, 5],
        &mut output,
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::SequenceLengthOutOfBounds {
            sequence: 1,
            length: 5,
            capacity: 4,
        }
    );
    assert_eq!(output, [-7.0; 2]);

    let error = paged_decode_attention_f32_reference(
        &query,
        &key_cache,
        &value_cache,
        &[0, 2, 1, -1],
        &[3, 1],
        &mut output,
        spec,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ContractError::BlockIdOutOfBounds {
            sequence: 0,
            logical_block: 1,
            block_id: 2,
            num_blocks: 2,
        }
    );
    assert_eq!(output, [-7.0; 2]);
}
