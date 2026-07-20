//! Rust-side contracts and references for Loom CUDA attention kernels.
//!
//! CUDA is opt-in. The default build keeps CPU CI and macOS development free
//! of a CUDA toolkit dependency, while `--features cuda` compiles and links the
//! native kernels in `../../cuda`.

use loom_attention::types::{DeviceKind, TensorHandle};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
use thiserror::Error;

const MAX_HEAD_DIM: u32 = 256;
const MAX_TAIL_TOKENS: u32 = 64;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CudaExecutorError {
    #[error("invalid fused-attention contract: {0}")]
    InvalidContract(String),
    #[error("Loom was built without the CUDA feature")]
    BackendUnavailable,
    #[error("CUDA kernel submission failed with status {status}: {message}")]
    KernelSubmission { status: i32, message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaAttentionDType {
    Fp16,
    Bf16,
}

impl CudaAttentionDType {
    pub const fn element_bytes(self) -> u64 {
        2
    }

    #[cfg(feature = "cuda")]
    const fn raw(self) -> loom_cuda_sys::LoomCudaDType {
        match self {
            Self::Fp16 => loom_cuda_sys::LoomCudaDType::Fp16,
            Self::Bf16 => loom_cuda_sys::LoomCudaDType::Bf16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusedTailShape {
    pub rows: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub tail_tokens: u32,
    pub scale: f32,
    pub dtype: CudaAttentionDType,
}

impl FusedTailShape {
    pub fn validate(self) -> Result<(), CudaExecutorError> {
        if self.rows == 0 || self.query_heads == 0 || self.kv_heads == 0 {
            return Err(CudaExecutorError::InvalidContract(
                "rows and head counts must be positive".into(),
            ));
        }
        if !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(CudaExecutorError::InvalidContract(
                "kv_heads must divide query_heads".into(),
            ));
        }
        if self.head_dim == 0 || self.head_dim > MAX_HEAD_DIM {
            return Err(CudaExecutorError::InvalidContract(format!(
                "head_dim must be in 1..={MAX_HEAD_DIM}"
            )));
        }
        if self.tail_tokens == 0 || self.tail_tokens > MAX_TAIL_TOKENS {
            return Err(CudaExecutorError::InvalidContract(format!(
                "tail_tokens must be in 1..={MAX_TAIL_TOKENS}"
            )));
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(CudaExecutorError::InvalidContract(
                "attention scale must be finite and positive".into(),
            ));
        }
        Ok(())
    }

    pub fn state_elements(self) -> Result<u64, CudaExecutorError> {
        product(&[self.rows, self.query_heads, self.head_dim])
    }

    pub fn lse_elements(self) -> Result<u64, CudaExecutorError> {
        product(&[self.rows, self.query_heads])
    }

    pub fn tail_elements(self) -> Result<u64, CudaExecutorError> {
        product(&[self.tail_tokens, self.kv_heads, self.head_dim])
    }
}

#[derive(Debug)]
pub struct FusedTailAttentionMerge<'a> {
    pub shape: FusedTailShape,
    pub query: &'a TensorHandle,
    pub tail_key: &'a TensorHandle,
    pub tail_value: &'a TensorHandle,
    pub remote_output: &'a TensorHandle,
    pub remote_lse: &'a TensorHandle,
    pub merged_output: &'a TensorHandle,
    pub merged_lse: &'a TensorHandle,
}

impl FusedTailAttentionMerge<'_> {
    pub fn validate(&self) -> Result<(), CudaExecutorError> {
        self.shape.validate()?;
        let tensors = [
            ("query", self.query),
            ("tail_key", self.tail_key),
            ("tail_value", self.tail_value),
            ("remote_output", self.remote_output),
            ("remote_lse", self.remote_lse),
            ("merged_output", self.merged_output),
            ("merged_lse", self.merged_lse),
        ];
        let first = self.query;
        for (name, tensor) in tensors {
            if tensor.device_kind != DeviceKind::Cuda {
                return Err(CudaExecutorError::InvalidContract(format!(
                    "{name} must be a CUDA tensor"
                )));
            }
            if tensor.owner != first.owner || tensor.device_index != first.device_index {
                return Err(CudaExecutorError::InvalidContract(format!(
                    "{name} must share the query owner and CUDA device"
                )));
            }
            if tensor.address == 0 || tensor.generation == 0 {
                return Err(CudaExecutorError::InvalidContract(format!(
                    "{name} must carry a live generation-pinned address"
                )));
            }
        }

        let element_bytes = self.shape.dtype.element_bytes();
        let state_bytes = checked_bytes(self.shape.state_elements()?, element_bytes)?;
        let tail_bytes = checked_bytes(self.shape.tail_elements()?, element_bytes)?;
        let lse_bytes = checked_bytes(self.shape.lse_elements()?, 4)?;
        for (name, tensor, required) in [
            ("query", self.query, state_bytes),
            ("tail_key", self.tail_key, tail_bytes),
            ("tail_value", self.tail_value, tail_bytes),
            ("remote_output", self.remote_output, state_bytes),
            ("remote_lse", self.remote_lse, lse_bytes),
            ("merged_output", self.merged_output, state_bytes),
            ("merged_lse", self.merged_lse, lse_bytes),
        ] {
            if tensor.bytes < required {
                return Err(CudaExecutorError::InvalidContract(format!(
                    "{name} has {} bytes but requires {required}",
                    tensor.bytes
                )));
            }
        }
        for (output_name, output) in [
            ("merged_output", self.merged_output),
            ("merged_lse", self.merged_lse),
        ] {
            for (input_name, input) in [
                ("query", self.query),
                ("tail_key", self.tail_key),
                ("tail_value", self.tail_value),
                ("remote_output", self.remote_output),
                ("remote_lse", self.remote_lse),
            ] {
                if ranges_overlap(output, input)? {
                    return Err(CudaExecutorError::InvalidContract(format!(
                        "{output_name} must not overlap {input_name}"
                    )));
                }
            }
        }
        if ranges_overlap(self.merged_output, self.merged_lse)? {
            return Err(CudaExecutorError::InvalidContract(
                "merged_output and merged_lse must not overlap".into(),
            ));
        }
        Ok(())
    }

    /// Queue the fused kernel on a caller-owned CUDA stream.
    ///
    /// # Safety
    ///
    /// Every tensor address must remain valid and resident on the declared
    /// device until the caller observes completion on `cuda_stream`. The
    /// caller must also make the tensor device current and pass a stream owned
    /// by that CUDA context.
    pub unsafe fn submit(&self, cuda_stream: u64) -> Result<(), CudaExecutorError> {
        self.validate()?;
        #[cfg(feature = "cuda")]
        {
            // SAFETY: the caller owns pointer lifetime and stream ordering; the
            // validated byte bounds cover every range consumed by the C ABI.
            let status = unsafe {
                loom_cuda_sys::loom_cuda_fused_tail_attention_merge(
                    self.query.address as *const c_void,
                    self.tail_key.address as *const c_void,
                    self.tail_value.address as *const c_void,
                    self.remote_output.address as *const c_void,
                    self.remote_lse.address as *const f32,
                    self.merged_output.address as *mut c_void,
                    self.merged_lse.address as *mut f32,
                    self.shape.rows,
                    self.shape.query_heads,
                    self.shape.kv_heads,
                    self.shape.head_dim,
                    self.shape.tail_tokens,
                    self.shape.scale,
                    self.shape.dtype.raw(),
                    cuda_stream as *mut c_void,
                )
            };
            status_result(status)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = cuda_stream;
            Err(CudaExecutorError::BackendUnavailable)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CpuAttentionState {
    pub output: Vec<f32>,
    pub logsumexp: Vec<f32>,
}

/// CPU oracle for the fused local-tail attention plus remote-state merge.
pub fn reference_fused_tail_attention_merge(
    shape: FusedTailShape,
    query: &[f32],
    tail_key: &[f32],
    tail_value: &[f32],
    remote: &CpuAttentionState,
) -> Result<CpuAttentionState, CudaExecutorError> {
    shape.validate()?;
    let state_elements = usize_from(shape.state_elements()?)?;
    let tail_elements = usize_from(shape.tail_elements()?)?;
    let lse_elements = usize_from(shape.lse_elements()?)?;
    require_len("query", query.len(), state_elements)?;
    require_len("tail_key", tail_key.len(), tail_elements)?;
    require_len("tail_value", tail_value.len(), tail_elements)?;
    require_len("remote.output", remote.output.len(), state_elements)?;
    require_len("remote.logsumexp", remote.logsumexp.len(), lse_elements)?;

    let mut output = vec![0.0; state_elements];
    let mut logsumexp = vec![0.0; lse_elements];
    let group_size = shape.query_heads / shape.kv_heads;
    for row in 0..shape.rows as usize {
        for query_head in 0..shape.query_heads as usize {
            let row_head = row * shape.query_heads as usize + query_head;
            let kv_head = query_head / group_size as usize;
            let query_base = row_head * shape.head_dim as usize;
            let mut logits = Vec::with_capacity(shape.tail_tokens as usize);
            for token in 0..shape.tail_tokens as usize {
                let key_base =
                    (token * shape.kv_heads as usize + kv_head) * shape.head_dim as usize;
                let dot: f32 = (0..shape.head_dim as usize)
                    .map(|dimension| query[query_base + dimension] * tail_key[key_base + dimension])
                    .sum();
                logits.push(dot * shape.scale);
            }
            let local_lse = logsumexp_values(&logits);
            let merged_lse = logaddexp(remote.logsumexp[row_head], local_lse);
            logsumexp[row_head] = merged_lse;
            let remote_weight = (remote.logsumexp[row_head] - merged_lse).exp();
            for dimension in 0..shape.head_dim as usize {
                let mut value = remote_weight * remote.output[query_base + dimension];
                for (token, logit) in logits.iter().copied().enumerate() {
                    let value_index = (token * shape.kv_heads as usize + kv_head)
                        * shape.head_dim as usize
                        + dimension;
                    value += (logit - merged_lse).exp() * tail_value[value_index];
                }
                output[query_base + dimension] = value;
            }
        }
    }
    Ok(CpuAttentionState { output, logsumexp })
}

#[cfg(feature = "cuda")]
fn status_result(status: i32) -> Result<(), CudaExecutorError> {
    if status == loom_cuda_sys::LOOM_CUDA_SUCCESS {
        return Ok(());
    }
    // SAFETY: the CUDA library returns a process-lifetime static C string for
    // every integer status.
    let message = unsafe {
        let pointer = loom_cuda_sys::loom_cuda_status_string(status);
        if pointer.is_null() {
            "unknown status".to_owned()
        } else {
            std::ffi::CStr::from_ptr(pointer)
                .to_string_lossy()
                .into_owned()
        }
    };
    Err(CudaExecutorError::KernelSubmission { status, message })
}

fn product(values: &[u32]) -> Result<u64, CudaExecutorError> {
    values.iter().try_fold(1_u64, |product, value| {
        product.checked_mul(u64::from(*value)).ok_or_else(|| {
            CudaExecutorError::InvalidContract("tensor element count overflow".into())
        })
    })
}

fn checked_bytes(elements: u64, element_bytes: u64) -> Result<u64, CudaExecutorError> {
    elements
        .checked_mul(element_bytes)
        .ok_or_else(|| CudaExecutorError::InvalidContract("tensor byte count overflow".into()))
}

fn ranges_overlap(left: &TensorHandle, right: &TensorHandle) -> Result<bool, CudaExecutorError> {
    let left_end = left.address.checked_add(left.bytes).ok_or_else(|| {
        CudaExecutorError::InvalidContract("tensor address range overflow".into())
    })?;
    let right_end = right.address.checked_add(right.bytes).ok_or_else(|| {
        CudaExecutorError::InvalidContract("tensor address range overflow".into())
    })?;
    Ok(left.address < right_end && right.address < left_end)
}

fn usize_from(value: u64) -> Result<usize, CudaExecutorError> {
    usize::try_from(value)
        .map_err(|_| CudaExecutorError::InvalidContract("tensor is too large for this host".into()))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), CudaExecutorError> {
    if actual != expected {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{name} has {actual} elements, expected {expected}"
        )));
    }
    Ok(())
}

fn logsumexp_values(values: &[f32]) -> f32 {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    maximum
        + values
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f32>()
            .ln()
}

fn logaddexp(left: f32, right: f32) -> f32 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use loom_attention::types::WorkerId;

    fn shape() -> FusedTailShape {
        FusedTailShape {
            rows: 2,
            query_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            tail_tokens: 3,
            scale: 8.0_f32.sqrt().recip(),
            dtype: CudaAttentionDType::Fp16,
        }
    }

    fn tensor(address: u64, bytes: u64) -> TensorHandle {
        TensorHandle {
            owner: WorkerId("engine-0".into()),
            device_kind: DeviceKind::Cuda,
            device_index: 0,
            address,
            bytes,
            registration_key: None,
            generation: 7,
        }
    }

    #[test]
    fn validates_generation_pinned_device_ranges() {
        let shape = shape();
        let state_bytes = shape.state_elements().unwrap() * 2;
        let tail_bytes = shape.tail_elements().unwrap() * 2;
        let lse_bytes = shape.lse_elements().unwrap() * 4;
        let query = tensor(0x1000, state_bytes);
        let tail_key = tensor(0x2000, tail_bytes);
        let tail_value = tensor(0x3000, tail_bytes);
        let remote_output = tensor(0x4000, state_bytes);
        let remote_lse = tensor(0x5000, lse_bytes);
        let merged_output = tensor(0x6000, state_bytes);
        let merged_lse = tensor(0x7000, lse_bytes);
        FusedTailAttentionMerge {
            shape,
            query: &query,
            tail_key: &tail_key,
            tail_value: &tail_value,
            remote_output: &remote_output,
            remote_lse: &remote_lse,
            merged_output: &merged_output,
            merged_lse: &merged_lse,
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn rejects_undersized_output() {
        let shape = shape();
        let state_bytes = shape.state_elements().unwrap() * 2;
        let tail_bytes = shape.tail_elements().unwrap() * 2;
        let lse_bytes = shape.lse_elements().unwrap() * 4;
        let query = tensor(0x1000, state_bytes);
        let tail_key = tensor(0x2000, tail_bytes);
        let tail_value = tensor(0x3000, tail_bytes);
        let remote_output = tensor(0x4000, state_bytes);
        let remote_lse = tensor(0x5000, lse_bytes);
        let merged_output = tensor(0x6000, state_bytes - 1);
        let merged_lse = tensor(0x7000, lse_bytes);
        let error = FusedTailAttentionMerge {
            shape,
            query: &query,
            tail_key: &tail_key,
            tail_value: &tail_value,
            remote_output: &remote_output,
            remote_lse: &remote_lse,
            merged_output: &merged_output,
            merged_lse: &merged_lse,
        }
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("merged_output"));
    }

    #[test]
    fn rejects_output_aliasing_an_input() {
        let shape = shape();
        let state_bytes = shape.state_elements().unwrap() * 2;
        let tail_bytes = shape.tail_elements().unwrap() * 2;
        let lse_bytes = shape.lse_elements().unwrap() * 4;
        let query = tensor(0x1000, state_bytes);
        let tail_key = tensor(0x2000, tail_bytes);
        let tail_value = tensor(0x3000, tail_bytes);
        let remote_output = tensor(0x4000, state_bytes);
        let remote_lse = tensor(0x5000, lse_bytes);
        let merged_output = tensor(0x4000, state_bytes);
        let merged_lse = tensor(0x7000, lse_bytes);
        let error = FusedTailAttentionMerge {
            shape,
            query: &query,
            tail_key: &tail_key,
            tail_value: &tail_value,
            remote_output: &remote_output,
            remote_lse: &remote_lse,
            merged_output: &merged_output,
            merged_lse: &merged_lse,
        }
        .validate()
        .unwrap_err();
        assert!(error.to_string().contains("must not overlap remote_output"));
    }

    #[test]
    fn fused_reference_matches_attention_over_concatenated_kv() {
        let shape = shape();
        let prefix_tokens = 5;
        let query = deterministic(shape.state_elements().unwrap() as usize, 0.07);
        let prefix_key = deterministic(
            prefix_tokens * shape.kv_heads as usize * shape.head_dim as usize,
            0.03,
        );
        let prefix_value = deterministic(prefix_key.len(), 0.05);
        let tail_key = deterministic(shape.tail_elements().unwrap() as usize, 0.02);
        let tail_value = deterministic(tail_key.len(), 0.04);
        let remote = reference_segment(shape, &query, &prefix_key, &prefix_value, prefix_tokens);
        let fused =
            reference_fused_tail_attention_merge(shape, &query, &tail_key, &tail_value, &remote)
                .unwrap();

        let mut full_key = prefix_key;
        full_key.extend_from_slice(&tail_key);
        let mut full_value = prefix_value;
        full_value.extend_from_slice(&tail_value);
        let full = reference_segment(
            shape,
            &query,
            &full_key,
            &full_value,
            prefix_tokens + shape.tail_tokens as usize,
        );
        close(&fused.output, &full.output, 2e-6);
        close(&fused.logsumexp, &full.logsumexp, 2e-6);
    }

    fn deterministic(length: usize, scale: f32) -> Vec<f32> {
        (0..length)
            .map(|index| (((index * 17 + 11) % 31) as f32 - 15.0) * scale)
            .collect()
    }

    fn reference_segment(
        shape: FusedTailShape,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        tokens: usize,
    ) -> CpuAttentionState {
        let mut output = vec![0.0; shape.state_elements().unwrap() as usize];
        let mut logsumexp = vec![0.0; shape.lse_elements().unwrap() as usize];
        let group_size = shape.query_heads / shape.kv_heads;
        for row in 0..shape.rows as usize {
            for query_head in 0..shape.query_heads as usize {
                let row_head = row * shape.query_heads as usize + query_head;
                let kv_head = query_head / group_size as usize;
                let query_base = row_head * shape.head_dim as usize;
                let logits: Vec<f32> = (0..tokens)
                    .map(|token| {
                        let key_base =
                            (token * shape.kv_heads as usize + kv_head) * shape.head_dim as usize;
                        (0..shape.head_dim as usize)
                            .map(|dimension| {
                                query[query_base + dimension] * key[key_base + dimension]
                            })
                            .sum::<f32>()
                            * shape.scale
                    })
                    .collect();
                let lse = logsumexp_values(&logits);
                logsumexp[row_head] = lse;
                for dimension in 0..shape.head_dim as usize {
                    output[query_base + dimension] = logits
                        .iter()
                        .enumerate()
                        .map(|(token, logit)| {
                            let value_index = (token * shape.kv_heads as usize + kv_head)
                                * shape.head_dim as usize
                                + dimension;
                            (*logit - lse).exp() * value[value_index]
                        })
                        .sum();
                }
            }
        }
        CpuAttentionState { output, logsumexp }
    }

    fn close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left, right)) in left.iter().zip(right).enumerate() {
            assert!(
                (left - right).abs() <= tolerance,
                "mismatch at {index}: {left} vs {right}"
            );
        }
    }
}
