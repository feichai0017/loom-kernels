//! Backend-independent contracts and CPU references for LLM inference operators.
//!
//! This crate deliberately contains no FFI or accelerator dependency. CUDA,
//! ROCm, CPU SIMD, and other providers implement these contracts in separate
//! crates and must report unsupported shapes instead of silently falling back.

#![forbid(unsafe_code)]

mod backend;
mod contract;
mod element;

pub mod activation;
pub mod attention;
pub mod logits;
pub mod norm;
pub mod quantization;
pub mod rope_kv;
pub mod sampling;

pub use activation::*;
pub use attention::*;
pub use backend::*;
pub use contract::*;
pub use logits::*;
pub use norm::*;
pub use quantization::*;
pub use rope_kv::*;
pub use sampling::*;

#[cfg(test)]
mod tests;
