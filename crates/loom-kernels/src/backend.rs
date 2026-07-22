//! Backend capability discovery shared by accelerator providers.

use crate::activation::{SiluAndMulDynamicFp8Spec, SiluAndMulSpec};
use crate::attention::PagedDecodeAttentionSpec;
use crate::logits::MinPFilterSpec;
use crate::norm::{AddRmsNormSpec, RmsNormDynamicFp8Spec, RmsNormSpec};
use crate::rope_kv::{RopePagedKvWriteSpec, RotaryEmbeddingSpec};
use crate::sampling::{GreedySampleLogprobsSpec, SelectedTokenLogprobsSpec};

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
    MinPFilter(MinPFilterSpec),
    PagedDecodeAttention(PagedDecodeAttentionSpec),
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
