//! Engine-neutral distributed core-attention contract.
//!
//! Executors consume registered tensor handles. The pure Rust reference math is
//! intentionally small and exists to prove exact split-KV merge semantics.

use crate::types::{ComputeCapabilities, KvBlockId, KvLayout, TensorHandle, WorkerId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum AttentionError {
    #[error("attention shape is invalid: {0}")]
    InvalidShape(String),
    #[error("attention executor does not support the operation: {0}")]
    Unsupported(String),
    #[error("attention execution failed: {0}")]
    Execution(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionExecutionMode {
    Local,
    RouteQuery,
    StageKv,
    Sharded,
}

/// Generation-pinned view of the historical KV consumed by one operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvView {
    /// Logical order is significant and must match the engine block table.
    pub blocks: Vec<KvBlockId>,
    pub page_table_generation: u64,
    /// Pool read leases covering every object resolved from `blocks`.
    pub lease_ids: Vec<String>,
}

impl KvView {
    pub fn token_count(&self) -> u64 {
        self.blocks
            .iter()
            .map(|block| u64::from(block.token_count))
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionMask {
    Causal,
    SlidingWindow { window_tokens: u32 },
    Bidirectional,
    Custom { tensor: TensorHandle },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvAppend {
    pub key: TensorHandle,
    pub value: TensorHandle,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionOp {
    pub request_id: String,
    pub layer_id: u32,
    pub mode: AttentionExecutionMode,
    pub layout: KvLayout,
    pub query: TensorHandle,
    /// Present only on the worker that owns the mutable tail for this step.
    pub append: Option<KvAppend>,
    pub output: TensorHandle,
    pub kv: KvView,
    pub mask: AttentionMask,
    pub scale: f32,
    pub deadline_unix_us: u64,
}

impl AttentionOp {
    pub fn validate(&self) -> Result<(), AttentionError> {
        if self.request_id.is_empty() {
            return Err(AttentionError::InvalidShape(
                "request_id must be non-empty".into(),
            ));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(AttentionError::InvalidShape(
                "attention scale must be finite and positive".into(),
            ));
        }
        for (name, tensor) in [("query", &self.query), ("output", &self.output)] {
            if tensor.bytes == 0 {
                return Err(AttentionError::InvalidShape(format!(
                    "{name} tensor must be non-empty"
                )));
            }
        }
        if let Some(append) = &self.append {
            for (name, tensor) in [("append.key", &append.key), ("append.value", &append.value)] {
                if tensor.bytes == 0 {
                    return Err(AttentionError::InvalidShape(format!(
                        "{name} tensor must be non-empty"
                    )));
                }
            }
        }
        if self
            .kv
            .blocks
            .iter()
            .any(|block| block.layer_id != self.layer_id)
        {
            return Err(AttentionError::InvalidShape(
                "KV view contains a block from another layer".into(),
            ));
        }
        if !self.kv.blocks.is_empty()
            && (self.kv.page_table_generation == 0 || self.kv.lease_ids.is_empty())
        {
            return Err(AttentionError::InvalidShape(
                "historical KV requires a generation-pinned leased view".into(),
            ));
        }
        if matches!(self.mask, AttentionMask::SlidingWindow { window_tokens: 0 }) {
            return Err(AttentionError::InvalidShape(
                "sliding-window attention requires a non-zero window".into(),
            ));
        }
        if let AttentionMask::Custom { tensor } = &self.mask {
            if tensor.bytes == 0 {
                return Err(AttentionError::InvalidShape(
                    "custom attention mask must be non-empty".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftmaxStats {
    /// Number of independent query rows represented by this partial result.
    pub rows: usize,
    pub value_dim: usize,
    pub max_logits: Vec<f32>,
    pub exp_sums: Vec<f32>,
    /// Unnormalized weighted values, row-major `[rows, value_dim]`.
    pub weighted_values: Vec<f32>,
}

impl SoftmaxStats {
    pub fn validate(&self) -> Result<(), AttentionError> {
        if self.rows == 0 || self.value_dim == 0 {
            return Err(AttentionError::InvalidShape(
                "rows and value_dim must be non-zero".into(),
            ));
        }
        if self.max_logits.len() != self.rows || self.exp_sums.len() != self.rows {
            return Err(AttentionError::InvalidShape(
                "one max and exp sum are required per row".into(),
            ));
        }
        if self.weighted_values.len() != self.rows * self.value_dim {
            return Err(AttentionError::InvalidShape(
                "weighted value shape does not match rows * value_dim".into(),
            ));
        }
        if self.exp_sums.iter().any(|sum| *sum <= 0.0) {
            return Err(AttentionError::InvalidShape(
                "partial exp sums must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionCompletion {
    pub completion_id: String,
    pub executor: WorkerId,
    pub output: TensorHandle,
    pub stats: Option<TensorHandle>,
}

pub trait AttentionExecutor: std::fmt::Debug + Send + Sync {
    fn capabilities(&self) -> &ComputeCapabilities;

    /// Queue work on the caller-provided stream and return without synchronizing.
    fn submit(
        &self,
        operation: &AttentionOp,
        cuda_stream: Option<u64>,
    ) -> Result<AttentionCompletion, AttentionError>;
}

/// Merge exact online-softmax statistics from disjoint KV shards.
pub fn merge_softmax_partials(partials: &[SoftmaxStats]) -> Result<Vec<f32>, AttentionError> {
    let first = partials
        .first()
        .ok_or_else(|| AttentionError::InvalidShape("at least one shard is required".into()))?;
    first.validate()?;
    for partial in &partials[1..] {
        partial.validate()?;
        if partial.rows != first.rows || partial.value_dim != first.value_dim {
            return Err(AttentionError::InvalidShape(
                "all shards must have the same row and value dimensions".into(),
            ));
        }
    }

    let mut output = vec![0.0_f32; first.rows * first.value_dim];
    for row in 0..first.rows {
        let global_max = partials
            .iter()
            .map(|partial| partial.max_logits[row])
            .fold(f32::NEG_INFINITY, f32::max);
        let denominator: f32 = partials
            .iter()
            .map(|partial| (partial.max_logits[row] - global_max).exp() * partial.exp_sums[row])
            .sum();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(AttentionError::Execution(
                "merged softmax denominator is not finite and positive".into(),
            ));
        }
        for value_index in 0..first.value_dim {
            let numerator: f32 = partials
                .iter()
                .map(|partial| {
                    let weight = (partial.max_logits[row] - global_max).exp();
                    weight * partial.weighted_values[row * first.value_dim + value_index]
                })
                .sum();
            output[row * first.value_dim + value_index] = numerator / denominator;
        }
    }
    Ok(output)
}

/// One-query reference implementation for a single KV shard.
pub fn reference_partial_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    tokens: usize,
    value_dim: usize,
    scale: f32,
) -> Result<SoftmaxStats, AttentionError> {
    if tokens == 0 || query.is_empty() || value_dim == 0 {
        return Err(AttentionError::InvalidShape(
            "query, tokens, and value_dim must be non-zero".into(),
        ));
    }
    if keys.len() != tokens * query.len() || values.len() != tokens * value_dim {
        return Err(AttentionError::InvalidShape(
            "flattened key/value shapes do not match tokens".into(),
        ));
    }

    let logits: Vec<f32> = keys
        .chunks_exact(query.len())
        .map(|key| query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * scale)
        .collect();
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits
        .iter()
        .map(|logit| (*logit - max_logit).exp())
        .collect();
    let exp_sum: f32 = weights.iter().sum();
    let mut weighted_values = vec![0.0_f32; value_dim];
    for (token, value) in values.chunks_exact(value_dim).enumerate() {
        for (out, element) in weighted_values.iter_mut().zip(value) {
            *out += weights[token] * element;
        }
    }

    Ok(SoftmaxStats {
        rows: 1,
        value_dim,
        max_logits: vec![max_logit],
        exp_sums: vec![exp_sum],
        weighted_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AttentionKind, DType, DeviceKind, IdentityScope, KvLayout, WorkerId};

    fn tensor(bytes: u64) -> TensorHandle {
        TensorHandle {
            owner: WorkerId("engine".into()),
            device_kind: DeviceKind::Cuda,
            device_index: 0,
            address: 0x1000,
            bytes,
            registration_key: None,
            generation: 1,
        }
    }

    fn operation() -> AttentionOp {
        AttentionOp {
            request_id: "request".into(),
            layer_id: 0,
            mode: AttentionExecutionMode::RouteQuery,
            layout: KvLayout {
                attention_kind: AttentionKind::Gqa,
                dtype: DType::Bf16,
                num_attention_heads: 32,
                num_kv_heads: 8,
                head_dim: 128,
                block_tokens: 16,
                tensor_parallel_rank: 0,
                tensor_parallel_size: 1,
                layout_digest: "layout".into(),
            },
            query: tensor(8_192),
            append: Some(KvAppend {
                key: tensor(2_048),
                value: tensor(2_048),
            }),
            output: tensor(8_192),
            kv: KvView {
                blocks: vec![KvBlockId {
                    scope: IdentityScope {
                        tenant_id: "tenant".into(),
                        model_id: "model".into(),
                        tokenizer_id: "tokenizer".into(),
                        adapter_id: None,
                    },
                    prefix_hash: "prefix".into(),
                    block_hash: "block".into(),
                    layer_id: 0,
                    block_index: 0,
                    token_count: 16,
                }],
                page_table_generation: 1,
                lease_ids: vec!["lease".into()],
            },
            mask: AttentionMask::Causal,
            scale: 0.088_388_35,
            deadline_unix_us: 1_000,
        }
    }

    fn close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(right) {
            assert!((a - b).abs() < 1e-5, "{a} != {b}");
        }
    }

    #[test]
    fn split_kv_merge_matches_full_attention() {
        let query = [1.0, -0.5];
        let keys = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.5];
        let values = [1.0, 2.0, 3.0, 4.0, -2.0, 1.0, 0.5, -1.0];

        let full = reference_partial_attention(&query, &keys, &values, 4, 2, 1.0).unwrap();
        let expected = merge_softmax_partials(&[full]).unwrap();

        let shard_a =
            reference_partial_attention(&query, &keys[..4], &values[..4], 2, 2, 1.0).unwrap();
        let shard_b =
            reference_partial_attention(&query, &keys[4..], &values[4..], 2, 2, 1.0).unwrap();
        let distributed = merge_softmax_partials(&[shard_a, shard_b]).unwrap();
        close(&expected, &distributed);
    }

    #[test]
    fn operation_requires_leased_generation_pinned_historical_kv() {
        let mut operation = operation();
        assert_eq!(operation.kv.token_count(), 16);
        operation.validate().unwrap();

        operation.kv.lease_ids.clear();
        assert!(matches!(
            operation.validate(),
            Err(AttentionError::InvalidShape(message))
                if message.contains("generation-pinned leased view")
        ));
    }

    #[test]
    fn merge_rejects_incompatible_shards() {
        let good = SoftmaxStats {
            rows: 1,
            value_dim: 2,
            max_logits: vec![0.0],
            exp_sums: vec![1.0],
            weighted_values: vec![1.0, 2.0],
        };
        let bad = SoftmaxStats {
            rows: 1,
            value_dim: 1,
            max_logits: vec![0.0],
            exp_sums: vec![1.0],
            weighted_values: vec![1.0],
        };
        assert!(merge_softmax_partials(&[good, bad]).is_err());
    }
}
