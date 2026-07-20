use clap::{Parser, ValueEnum};
use half::{bf16, f16};
use loom_attention::types::{DeviceKind, TensorHandle, WorkerId};
use loom_cuda::{
    reference_fused_tail_attention_merge, CpuAttentionState, CudaAttentionDType,
    FusedTailAttentionMerge, FusedTailShape,
};
use loom_cuda_sys as sys;
use serde::Serialize;
use std::error::Error;
use std::ffi::{c_void, CStr};
use std::ptr;

type BenchResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DTypeArg {
    Fp16,
    Bf16,
}

impl DTypeArg {
    const fn kernel(self) -> CudaAttentionDType {
        match self {
            Self::Fp16 => CudaAttentionDType::Fp16,
            Self::Bf16 => CudaAttentionDType::Bf16,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Benchmark Loom's fused local-tail attention and state merge")]
struct Args {
    #[arg(long, default_value_t = 1)]
    rows: u32,
    #[arg(long, default_value_t = 32)]
    query_heads: u32,
    #[arg(long, default_value_t = 8)]
    kv_heads: u32,
    #[arg(long, default_value_t = 128)]
    head_dim: u32,
    #[arg(long, default_value_t = 16)]
    tail_tokens: u32,
    #[arg(long, value_enum, default_value_t = DTypeArg::Fp16)]
    dtype: DTypeArg,
    #[arg(long, default_value_t = 100)]
    warmup: usize,
    #[arg(long, default_value_t = 1_000)]
    iterations: usize,
    #[arg(long, default_value_t = 20)]
    samples: usize,
}

#[derive(Debug, Serialize)]
struct LatencySummary {
    minimum_us: f64,
    median_us: f64,
    maximum_us: f64,
}

#[derive(Debug, Serialize)]
struct AccuracySummary {
    fused_output_max_abs: f32,
    fused_lse_max_abs: f32,
    baseline_output_max_abs: f32,
    baseline_lse_max_abs: f32,
}

#[derive(Debug, Serialize)]
struct Report {
    rows: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    tail_tokens: u32,
    dtype: &'static str,
    warmup: usize,
    iterations_per_sample: usize,
    samples: usize,
    baseline_two_kernel: LatencySummary,
    fused_one_kernel: LatencySummary,
    median_speedup: f64,
    accuracy: AccuracySummary,
}

fn main() -> BenchResult<()> {
    let args = Args::parse();
    if args.iterations == 0 || args.samples == 0 {
        return Err("iterations and samples must be positive".into());
    }
    let shape = FusedTailShape {
        rows: args.rows,
        query_heads: args.query_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        tail_tokens: args.tail_tokens,
        scale: (args.head_dim as f32).sqrt().recip(),
        dtype: args.dtype.kernel(),
    };
    shape.validate()?;

    let state_elements = usize::try_from(shape.state_elements()?)?;
    let lse_elements = usize::try_from(shape.lse_elements()?)?;
    let tail_elements = usize::try_from(shape.tail_elements()?)?;
    let query_f32 = deterministic(state_elements, 0.031);
    let tail_key_f32 = deterministic(tail_elements, 0.019);
    let tail_value_f32 = deterministic(tail_elements, 0.027);
    let remote_output_f32 = deterministic(state_elements, 0.041);
    let remote_lse: Vec<f32> = (0..lse_elements)
        .map(|index| 3.5 + (index % 13) as f32 * 0.017)
        .collect();

    let query_encoded = encode(&query_f32, args.dtype);
    let tail_key_encoded = encode(&tail_key_f32, args.dtype);
    let tail_value_encoded = encode(&tail_value_f32, args.dtype);
    let remote_output_encoded = encode(&remote_output_f32, args.dtype);
    let query_quantized = decode(&query_encoded, args.dtype);
    let tail_key_quantized = decode(&tail_key_encoded, args.dtype);
    let tail_value_quantized = decode(&tail_value_encoded, args.dtype);
    let remote_output_quantized = decode(&remote_output_encoded, args.dtype);
    let reference = reference_fused_tail_attention_merge(
        shape,
        &query_quantized,
        &tail_key_quantized,
        &tail_value_quantized,
        &CpuAttentionState {
            output: remote_output_quantized,
            logsumexp: remote_lse.clone(),
        },
    )?;

    let query = DeviceAllocation::from_slice(&query_encoded)?;
    let tail_key = DeviceAllocation::from_slice(&tail_key_encoded)?;
    let tail_value = DeviceAllocation::from_slice(&tail_value_encoded)?;
    let remote_output = DeviceAllocation::from_slice(&remote_output_encoded)?;
    let remote_lse_device = DeviceAllocation::from_slice(&remote_lse)?;
    let tail_output = DeviceAllocation::new(state_elements * 2)?;
    let tail_lse = DeviceAllocation::new(lse_elements * 4)?;
    let baseline_output = DeviceAllocation::new(state_elements * 2)?;
    let baseline_lse = DeviceAllocation::new(lse_elements * 4)?;
    let fused_output = DeviceAllocation::new(state_elements * 2)?;
    let fused_lse = DeviceAllocation::new(lse_elements * 4)?;
    let stream = CudaStream::new()?;

    let owner = WorkerId("h20-benchmark".into());
    let query_handle = query.tensor_handle(&owner);
    let tail_key_handle = tail_key.tensor_handle(&owner);
    let tail_value_handle = tail_value.tensor_handle(&owner);
    let remote_output_handle = remote_output.tensor_handle(&owner);
    let remote_lse_handle = remote_lse_device.tensor_handle(&owner);
    let fused_output_handle = fused_output.tensor_handle(&owner);
    let fused_lse_handle = fused_lse.tensor_handle(&owner);
    let fused_operation = FusedTailAttentionMerge {
        shape,
        query: &query_handle,
        tail_key: &tail_key_handle,
        tail_value: &tail_value_handle,
        remote_output: &remote_output_handle,
        remote_lse: &remote_lse_handle,
        merged_output: &fused_output_handle,
        merged_lse: &fused_lse_handle,
    };
    // Exercise the safe Rust contract once before timing the raw kernel path.
    unsafe { fused_operation.submit(stream.raw() as u64)? };
    cuda_check(unsafe { sys::cudaDeviceSynchronize() })?;

    for _ in 0..args.warmup {
        launch_baseline(
            &query,
            &tail_key,
            &tail_value,
            &remote_output,
            &remote_lse_device,
            &tail_output,
            &tail_lse,
            &baseline_output,
            &baseline_lse,
            shape,
            stream.raw(),
        )?;
        launch_fused(
            &query,
            &tail_key,
            &tail_value,
            &remote_output,
            &remote_lse_device,
            &fused_output,
            &fused_lse,
            shape,
            stream.raw(),
        )?;
    }
    cuda_check(unsafe { sys::cudaDeviceSynchronize() })?;

    let baseline_encoded: Vec<u16> = baseline_output.copy_to_vec(state_elements)?;
    let baseline_lse_host: Vec<f32> = baseline_lse.copy_to_vec(lse_elements)?;
    let fused_encoded: Vec<u16> = fused_output.copy_to_vec(state_elements)?;
    let fused_lse_host: Vec<f32> = fused_lse.copy_to_vec(lse_elements)?;
    let baseline_decoded = decode(&baseline_encoded, args.dtype);
    let fused_decoded = decode(&fused_encoded, args.dtype);
    let accuracy = AccuracySummary {
        fused_output_max_abs: max_abs(&fused_decoded, &reference.output),
        fused_lse_max_abs: max_abs(&fused_lse_host, &reference.logsumexp),
        baseline_output_max_abs: max_abs(&baseline_decoded, &reference.output),
        baseline_lse_max_abs: max_abs(&baseline_lse_host, &reference.logsumexp),
    };
    let output_tolerance = match args.dtype {
        DTypeArg::Fp16 => 3e-3,
        DTypeArg::Bf16 => 2e-2,
    };
    if accuracy.fused_output_max_abs > output_tolerance
        || accuracy.baseline_output_max_abs > output_tolerance * 2.0
        || accuracy.fused_lse_max_abs > 2e-4
        || accuracy.baseline_lse_max_abs > 2e-4
    {
        return Err(format!("CUDA correctness gate failed: {accuracy:?}").into());
    }

    let baseline_samples = time_samples(&stream, args.samples, args.iterations, || {
        launch_baseline(
            &query,
            &tail_key,
            &tail_value,
            &remote_output,
            &remote_lse_device,
            &tail_output,
            &tail_lse,
            &baseline_output,
            &baseline_lse,
            shape,
            stream.raw(),
        )
    })?;
    let fused_samples = time_samples(&stream, args.samples, args.iterations, || {
        launch_fused(
            &query,
            &tail_key,
            &tail_value,
            &remote_output,
            &remote_lse_device,
            &fused_output,
            &fused_lse,
            shape,
            stream.raw(),
        )
    })?;
    let baseline_summary = summarize(baseline_samples);
    let fused_summary = summarize(fused_samples);
    let median_speedup = baseline_summary.median_us / fused_summary.median_us;
    let report = Report {
        rows: args.rows,
        query_heads: args.query_heads,
        kv_heads: args.kv_heads,
        head_dim: args.head_dim,
        tail_tokens: args.tail_tokens,
        dtype: match args.dtype {
            DTypeArg::Fp16 => "fp16",
            DTypeArg::Bf16 => "bf16",
        },
        warmup: args.warmup,
        iterations_per_sample: args.iterations,
        samples: args.samples,
        baseline_two_kernel: baseline_summary,
        fused_one_kernel: fused_summary,
        median_speedup,
        accuracy,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_baseline(
    query: &DeviceAllocation,
    tail_key: &DeviceAllocation,
    tail_value: &DeviceAllocation,
    remote_output: &DeviceAllocation,
    remote_lse: &DeviceAllocation,
    tail_output: &DeviceAllocation,
    tail_lse: &DeviceAllocation,
    merged_output: &DeviceAllocation,
    merged_lse: &DeviceAllocation,
    shape: FusedTailShape,
    stream: *mut c_void,
) -> BenchResult<()> {
    let tail_status = unsafe {
        sys::loom_cuda_tail_attention_state(
            query.pointer,
            tail_key.pointer,
            tail_value.pointer,
            tail_output.pointer,
            tail_lse.pointer.cast(),
            shape.rows,
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.tail_tokens,
            shape.scale,
            raw_dtype(shape.dtype),
            stream,
        )
    };
    loom_check(tail_status)?;
    let merge_status = unsafe {
        sys::loom_cuda_merge_two_states(
            remote_output.pointer,
            remote_lse.pointer.cast(),
            tail_output.pointer,
            tail_lse.pointer.cast(),
            merged_output.pointer,
            merged_lse.pointer.cast(),
            shape.rows,
            shape.query_heads,
            shape.head_dim,
            raw_dtype(shape.dtype),
            stream,
        )
    };
    loom_check(merge_status)
}

#[allow(clippy::too_many_arguments)]
fn launch_fused(
    query: &DeviceAllocation,
    tail_key: &DeviceAllocation,
    tail_value: &DeviceAllocation,
    remote_output: &DeviceAllocation,
    remote_lse: &DeviceAllocation,
    merged_output: &DeviceAllocation,
    merged_lse: &DeviceAllocation,
    shape: FusedTailShape,
    stream: *mut c_void,
) -> BenchResult<()> {
    let status = unsafe {
        sys::loom_cuda_fused_tail_attention_merge(
            query.pointer,
            tail_key.pointer,
            tail_value.pointer,
            remote_output.pointer,
            remote_lse.pointer.cast(),
            merged_output.pointer,
            merged_lse.pointer.cast(),
            shape.rows,
            shape.query_heads,
            shape.kv_heads,
            shape.head_dim,
            shape.tail_tokens,
            shape.scale,
            raw_dtype(shape.dtype),
            stream,
        )
    };
    loom_check(status)
}

fn raw_dtype(dtype: CudaAttentionDType) -> sys::LoomCudaDType {
    match dtype {
        CudaAttentionDType::Fp16 => sys::LoomCudaDType::Fp16,
        CudaAttentionDType::Bf16 => sys::LoomCudaDType::Bf16,
    }
}

fn time_samples(
    stream: &CudaStream,
    samples: usize,
    iterations: usize,
    mut operation: impl FnMut() -> BenchResult<()>,
) -> BenchResult<Vec<f64>> {
    let mut values = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = CudaEvent::new()?;
        let end = CudaEvent::new()?;
        start.record(stream.raw())?;
        for _ in 0..iterations {
            operation()?;
        }
        end.record(stream.raw())?;
        end.synchronize()?;
        let milliseconds = start.elapsed_ms(&end)?;
        values.push(f64::from(milliseconds) * 1_000.0 / iterations as f64);
    }
    Ok(values)
}

fn summarize(mut values: Vec<f64>) -> LatencySummary {
    values.sort_by(f64::total_cmp);
    LatencySummary {
        minimum_us: values[0],
        median_us: values[values.len() / 2],
        maximum_us: values[values.len() - 1],
    }
}

fn deterministic(length: usize, scale: f32) -> Vec<f32> {
    (0..length)
        .map(|index| (((index * 29 + 7) % 47) as f32 - 23.0) * scale)
        .collect()
}

fn encode(values: &[f32], dtype: DTypeArg) -> Vec<u16> {
    values
        .iter()
        .map(|value| match dtype {
            DTypeArg::Fp16 => f16::from_f32(*value).to_bits(),
            DTypeArg::Bf16 => bf16::from_f32(*value).to_bits(),
        })
        .collect()
}

fn decode(values: &[u16], dtype: DTypeArg) -> Vec<f32> {
    values
        .iter()
        .map(|value| match dtype {
            DTypeArg::Fp16 => f16::from_bits(*value).to_f32(),
            DTypeArg::Bf16 => bf16::from_bits(*value).to_f32(),
        })
        .collect()
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f32::max)
}

struct DeviceAllocation {
    pointer: *mut c_void,
    bytes: usize,
}

impl DeviceAllocation {
    fn new(bytes: usize) -> BenchResult<Self> {
        let mut pointer = ptr::null_mut();
        cuda_check(unsafe { sys::cudaMalloc(&mut pointer, bytes) })?;
        Ok(Self { pointer, bytes })
    }

    fn from_slice<T: Copy>(values: &[T]) -> BenchResult<Self> {
        let bytes = std::mem::size_of_val(values);
        let allocation = Self::new(bytes)?;
        cuda_check(unsafe {
            sys::cudaMemcpy(
                allocation.pointer,
                values.as_ptr().cast(),
                bytes,
                sys::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })?;
        Ok(allocation)
    }

    fn copy_to_vec<T: Copy + Default>(&self, elements: usize) -> BenchResult<Vec<T>> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<T>())
            .ok_or("host copy byte count overflow")?;
        if bytes > self.bytes {
            return Err("host copy exceeds device allocation".into());
        }
        let mut values = vec![T::default(); elements];
        cuda_check(unsafe {
            sys::cudaMemcpy(
                values.as_mut_ptr().cast(),
                self.pointer,
                bytes,
                sys::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        })?;
        Ok(values)
    }

    fn tensor_handle(&self, owner: &WorkerId) -> TensorHandle {
        TensorHandle {
            owner: owner.clone(),
            device_kind: DeviceKind::Cuda,
            device_index: 0,
            address: self.pointer as u64,
            bytes: self.bytes as u64,
            registration_key: None,
            generation: 1,
        }
    }
}

impl Drop for DeviceAllocation {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe {
                sys::cudaFree(self.pointer);
            }
        }
    }
}

struct CudaStream(*mut c_void);

impl CudaStream {
    fn new() -> BenchResult<Self> {
        let mut stream = ptr::null_mut();
        cuda_check(unsafe {
            sys::cudaStreamCreateWithFlags(&mut stream, sys::CUDA_STREAM_NON_BLOCKING)
        })?;
        Ok(Self(stream))
    }

    const fn raw(&self) -> *mut c_void {
        self.0
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            sys::cudaStreamDestroy(self.0);
        }
    }
}

struct CudaEvent(*mut c_void);

impl CudaEvent {
    fn new() -> BenchResult<Self> {
        let mut event = ptr::null_mut();
        cuda_check(unsafe { sys::cudaEventCreate(&mut event) })?;
        Ok(Self(event))
    }

    fn record(&self, stream: *mut c_void) -> BenchResult<()> {
        cuda_check(unsafe { sys::cudaEventRecord(self.0, stream) })
    }

    fn synchronize(&self) -> BenchResult<()> {
        cuda_check(unsafe { sys::cudaEventSynchronize(self.0) })
    }

    fn elapsed_ms(&self, end: &Self) -> BenchResult<f32> {
        let mut milliseconds = 0.0;
        cuda_check(unsafe { sys::cudaEventElapsedTime(&mut milliseconds, self.0, end.0) })?;
        Ok(milliseconds)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            sys::cudaEventDestroy(self.0);
        }
    }
}

fn loom_check(status: i32) -> BenchResult<()> {
    if status == sys::LOOM_CUDA_SUCCESS {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(sys::loom_cuda_status_string(status)) }.to_string_lossy();
    Err(format!("Loom CUDA status {status}: {message}").into())
}

fn cuda_check(status: i32) -> BenchResult<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { CStr::from_ptr(sys::cudaGetErrorString(status)) }.to_string_lossy();
    Err(format!("CUDA runtime status {status}: {message}").into())
}
