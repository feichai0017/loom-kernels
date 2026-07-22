//! Backend-independent FP8 constants and encoding helpers.

/// Maximum finite magnitude of the OCP FP8 E4M3FN encoding.
pub const FP8_E4M3FN_MAX: f32 = 448.0;

/// vLLM-compatible lower bound for a dynamic per-token FP8 scale.
///
/// The non-zero floor keeps a zero row quantizable and avoids division by
/// zero. It matches `1 / (FP8_E4M3FN_MAX * 512)`.
pub const DYNAMIC_FP8_MIN_SCALE: f32 = 1.0 / (FP8_E4M3FN_MAX * 512.0);

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
