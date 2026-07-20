//! Raw bindings to Loom's dependency-light CUDA C ABI.

use std::ffi::c_int;
#[cfg(feature = "cuda")]
use std::ffi::{c_char, c_void};

pub const LOOM_CUDA_SUCCESS: c_int = 0;
pub const LOOM_CUDA_INVALID_ARGUMENT: c_int = 1;
pub const LOOM_CUDA_UNSUPPORTED: c_int = 2;
pub const LOOM_CUDA_LAUNCH_ERROR: c_int = 3;
pub const LOOM_CUDA_UNAVAILABLE: c_int = 4;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoomCudaDType {
    Fp16 = 0,
    Bf16 = 1,
}

pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const CUDA_STREAM_NON_BLOCKING: u32 = 1;

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn loom_cuda_status_string(status: c_int) -> *const c_char;

    pub fn loom_cuda_tail_attention_state(
        query: *const c_void,
        tail_key: *const c_void,
        tail_value: *const c_void,
        tail_output: *mut c_void,
        tail_lse: *mut f32,
        rows: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tail_tokens: u32,
        scale: f32,
        dtype: LoomCudaDType,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_merge_two_states(
        left_output: *const c_void,
        left_lse: *const f32,
        right_output: *const c_void,
        right_lse: *const f32,
        merged_output: *mut c_void,
        merged_lse: *mut f32,
        rows: u32,
        query_heads: u32,
        head_dim: u32,
        dtype: LoomCudaDType,
        stream: *mut c_void,
    ) -> c_int;

    pub fn loom_cuda_fused_tail_attention_merge(
        query: *const c_void,
        tail_key: *const c_void,
        tail_value: *const c_void,
        remote_output: *const c_void,
        remote_lse: *const f32,
        merged_output: *mut c_void,
        merged_lse: *mut f32,
        rows: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        tail_tokens: u32,
        scale: f32,
        dtype: LoomCudaDType,
        stream: *mut c_void,
    ) -> c_int;

    pub fn cudaMalloc(pointer: *mut *mut c_void, bytes: usize) -> c_int;
    pub fn cudaFree(pointer: *mut c_void) -> c_int;
    pub fn cudaMemcpy(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: c_int,
    ) -> c_int;
    pub fn cudaDeviceSynchronize() -> c_int;
    pub fn cudaGetErrorString(error: c_int) -> *const c_char;
    pub fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> c_int;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    pub fn cudaEventCreate(event: *mut *mut c_void) -> c_int;
    pub fn cudaEventDestroy(event: *mut c_void) -> c_int;
    pub fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    pub fn cudaEventSynchronize(event: *mut c_void) -> c_int;
    pub fn cudaEventElapsedTime(
        milliseconds: *mut f32,
        start: *mut c_void,
        end: *mut c_void,
    ) -> c_int;
}

pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}
