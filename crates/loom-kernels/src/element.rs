//! Internal element conversions shared by CPU reference implementations.

use half::{bf16, f16};

pub(crate) trait LowPrecisionElement: Copy {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

pub(crate) trait DynamicFp8Input: Copy {
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
