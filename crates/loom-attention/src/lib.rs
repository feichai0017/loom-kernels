//! Contracts and node-local state for the Loom attention runtime.
//!
//! The Rust control endpoint, worker endpoint, catalog, planner, and runtime
//! share one release. Engine adapters and native GPU kernels retain their own
//! language and toolchain boundaries.

#![forbid(unsafe_code)]

pub mod attention;
pub mod catalog;
pub mod pool;
pub mod runtime;
pub mod scheduler;
pub mod transport;
pub mod types;
