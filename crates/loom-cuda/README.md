# loom-cuda

Safe Rust execution for Loom Kernels' handwritten CUDA operators.

The crate validates every public operator contract before launch and reports
unsupported inputs explicitly. Standalone Rust programs can use Loom-owned
streams, allocations, and events. Inference-engine adapters can instead lend
Loom their existing CUDA stream and tensor storage, without a copy, allocation,
implicit synchronization, or ownership transfer.

CUDA is opt-in so ordinary documentation and CPU-only dependency builds do not
require an NVIDIA toolkit.

```toml
[dependencies]
loom-cuda = { version = "1.0.0-alpha.1", features = ["cuda"] }
loom-kernels = "1.0.0-alpha.1"
```

```bash
CUDA_HOME=/usr/local/cuda LOOM_CUDA_ARCHS=90 \
  cargo run -p loom-cuda --features cuda --release \
  --example rust_cuda_smoke
```

## Framework-owned resources

The `v1.0.0-alpha.1` repository API separates the unsafe framework boundary
from safe checked dispatch:

```rust
use loom_cuda::{
    runtime::{CudaStreamRef, DeviceSlice, DeviceSliceMut},
    CudaBackend, CudaExecutorError,
};
use loom_kernels::AddRmsNormSpec;
use std::ffi::c_void;

unsafe fn launch_on_framework_stream(
    stream: *mut c_void,
    input: *mut f32,
    residual: *mut f32,
    weight: *const f32,
    spec: AddRmsNormSpec,
) -> Result<(), CudaExecutorError> {
    let backend = CudaBackend::from_stream(CudaStreamRef::from_raw(stream));
    let mut input = DeviceSliceMut::from_raw_parts(input, spec.numel())?;
    let mut residual = DeviceSliceMut::from_raw_parts(residual, spec.numel())?;
    let weight = DeviceSlice::from_raw_parts(weight, spec.hidden_size())?;

    backend.add_rms_norm_f32(&mut input, &mut residual, &weight, spec)
}
```

`CudaStreamRef`, `DeviceSlice`, and `DeviceSliceMut` never free external
resources. Their raw constructors are `unsafe` because the adapter must prove
device, context, dtype, lifetime, and aliasing. After construction, the same
safe operator methods validate element counts and contracts for both owned and
borrowed memory. Launches remain asynchronous, so the adapter must keep storage
alive and preserve stream ordering until the work completes.

The alpha API supports normalization and quantization, SwiGLU, RoPE plus
paged-KV writes, decode-tail sampling/logprob operations, Min-P, and paged
MQA/GQA decode attention. See the
[project documentation](https://feichai0017.github.io/loom-kernels/) for exact
shape gates and H20 evidence.

Licensed under MIT.
