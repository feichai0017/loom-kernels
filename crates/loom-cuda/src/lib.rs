//! Safe Rust CUDA backend for Loom Kernels.
//!
//! CUDA is opt-in so the default workspace remains buildable on machines
//! without an NVIDIA toolkit. Enabling `cuda` compiles the handwritten kernels
//! and exposes owned or borrowed streams and device memory, events, and checked
//! operator entrypoints.

use thiserror::Error;

#[cfg(feature = "cuda")]
mod activation_dispatch;
#[cfg(feature = "cuda")]
mod attention_dispatch;
#[cfg(feature = "cuda")]
mod cuda_backend;
#[cfg(feature = "cuda")]
mod layout;
#[cfg(feature = "cuda")]
pub use layout::{PagedDecodeLayout, RopePagedKvLayout, RowStridedLayout};
#[cfg(feature = "cuda")]
mod logits_dispatch;
#[cfg(feature = "cuda")]
mod norm_dispatch;
#[cfg(feature = "cuda")]
mod rope_kv_dispatch;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
mod sampling_dispatch;
#[cfg(feature = "cuda")]
mod speculative_dispatch;
#[cfg(feature = "cuda")]
pub use activation_dispatch::{Fp8ScaleLayout, SiluAndMulDynamicFp8Options};
#[cfg(feature = "cuda")]
pub use attention_dispatch::paged_decode_attention_split_k_workspace_elements;
#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;

/// Validation, availability, or CUDA launch failure.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CudaExecutorError {
    #[error("invalid operator contract: {0}")]
    InvalidContract(String),
    #[error("Loom Kernels was built without the CUDA feature")]
    BackendUnavailable,
    #[error("CUDA kernel submission failed with status {status}: {message}")]
    KernelSubmission { status: i32, message: String },
}

/// Whether this build contains the native CUDA backend.
pub const fn compiled_with_cuda() -> bool {
    loom_cuda_sys::compiled_with_cuda()
}
