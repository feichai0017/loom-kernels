//! Safe Rust CUDA backend for Loom Kernels.
//!
//! CUDA is opt-in so the default workspace remains buildable on machines
//! without an NVIDIA toolkit. Enabling `cuda` compiles the handwritten kernels
//! and exposes owned streams, buffers, events, and checked operator entrypoints.

use thiserror::Error;

#[cfg(feature = "cuda")]
mod greedy_sample;
#[cfg(feature = "cuda")]
mod min_p;
#[cfg(feature = "cuda")]
mod paged_decode;
#[cfg(feature = "cuda")]
pub use paged_decode::paged_decode_attention_split_k_workspace_elements;
#[cfg(feature = "cuda")]
mod rms_norm;
#[cfg(feature = "cuda")]
mod rope_paged_kv;
#[cfg(feature = "cuda")]
pub mod runtime;
#[cfg(feature = "cuda")]
mod silu_and_mul;
#[cfg(feature = "cuda")]
pub use rms_norm::CudaBackend;

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
