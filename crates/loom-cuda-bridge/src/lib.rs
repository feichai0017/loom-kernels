//! Checked C entrypoints into Loom Kernels' Rust CUDA runtime.
//!
//! Framework adapters pass borrowed tensor storage and their current CUDA
//! stream through this single ABI. Rust validates the semantic contract,
//! physical layout, buffer spans, and mutable aliasing before submission.

#![deny(unsafe_op_in_unsafe_fn)]

/// Whether this build contains the checked CUDA bridge.
pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;
