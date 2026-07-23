//! Checked C entrypoints into Loom Kernels' safe Rust CUDA runtime.
//!
//! Framework adapters own tensor allocations and stream lifetime. This crate
//! converts that raw boundary into [`loom_cuda`] borrowed resources and keeps
//! panics, validation errors, and CUDA submission failures behind a stable C
//! status ABI.

#![deny(unsafe_op_in_unsafe_fn)]

/// Whether this build contains the checked CUDA bridge.
pub const fn compiled_with_cuda() -> bool {
    cfg!(feature = "cuda")
}

#[cfg(feature = "cuda")]
mod cuda {
    use half::{bf16, f16};
    use loom_cuda::runtime::{
        CudaDeviceRead, CudaDeviceWrite, CudaStreamHandle, CudaStreamRef, DeviceSlice,
        DeviceSliceMut,
    };
    use loom_cuda::{CudaBackend, CudaExecutorError};
    use loom_kernels::{AddRmsNormSpec, DType};
    use std::cell::RefCell;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::mem::{align_of, size_of};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicU64, Ordering};

    const SUCCESS: c_int = 0;
    const INVALID_ARGUMENT: c_int = 1;
    const UNSUPPORTED: c_int = 2;
    const LAUNCH_ERROR: c_int = 3;
    const UNAVAILABLE: c_int = 4;

    static ADD_RMS_NORM_LAUNCHES: AtomicU64 = AtomicU64::new(0);

    thread_local! {
        static LAST_ERROR: RefCell<CString> = RefCell::new(
            CString::new("no bridge error has been recorded")
                .expect("static bridge message contains no NUL")
        );
    }

    trait AddRmsNormScalar: Copy {
        const DTYPE: DType;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>;
    }

    impl AddRmsNormScalar for f32 {
        const DTYPE: DType = DType::F32;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_f32(input, residual, weight, spec)
        }
    }

    impl AddRmsNormScalar for f16 {
        const DTYPE: DType = DType::F16;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_f16(input, residual, weight, spec)
        }
    }

    impl AddRmsNormScalar for bf16 {
        const DTYPE: DType = DType::Bf16;

        fn launch<S, I, R, W>(
            backend: &CudaBackend<S>,
            input: &mut I,
            residual: &mut R,
            weight: &W,
            spec: AddRmsNormSpec,
        ) -> Result<(), CudaExecutorError>
        where
            S: CudaStreamHandle,
            I: CudaDeviceWrite<Self>,
            R: CudaDeviceWrite<Self>,
            W: CudaDeviceRead<Self>,
        {
            backend.add_rms_norm_bf16(input, residual, weight, spec)
        }
    }

    #[derive(Clone, Copy)]
    struct ByteRange {
        start: usize,
        end: usize,
    }

    fn element_count(value: u64, name: &str) -> Result<usize, CudaExecutorError> {
        usize::try_from(value).map_err(|_| {
            CudaExecutorError::InvalidContract(format!("{name} element count exceeds the host ABI"))
        })
    }

    fn checked_byte_range<T>(
        pointer: *const T,
        elements: usize,
        name: &str,
    ) -> Result<ByteRange, CudaExecutorError> {
        if pointer.is_null() {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} pointer is null"
            )));
        }
        if elements == 0 {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} region is empty"
            )));
        }
        if !(pointer as usize).is_multiple_of(align_of::<T>()) {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{name} pointer is not aligned to {} bytes",
                align_of::<T>()
            )));
        }
        let bytes = elements.checked_mul(size_of::<T>()).ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!("{name} byte size overflows usize"))
        })?;
        let start = pointer as usize;
        let end = start.checked_add(bytes).ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!("{name} address range overflows usize"))
        })?;
        Ok(ByteRange { start, end })
    }

    fn ranges_overlap(left: ByteRange, right: ByteRange) -> bool {
        left.start < right.end && right.start < left.end
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn launch_add_rms_norm<T: AddRmsNormScalar>(
        input: *mut T,
        input_elements: u64,
        residual: *mut T,
        residual_elements: u64,
        weight: *const T,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> Result<(), CudaExecutorError> {
        let input_elements = element_count(input_elements, "Add+RMSNorm input")?;
        let residual_elements = element_count(residual_elements, "Add+RMSNorm residual")?;
        let weight_elements = element_count(weight_elements, "Add+RMSNorm weight")?;

        let input_range =
            checked_byte_range(input.cast_const(), input_elements, "Add+RMSNorm input")?;
        let residual_range = checked_byte_range(
            residual.cast_const(),
            residual_elements,
            "Add+RMSNorm residual",
        )?;
        let weight_range = checked_byte_range(weight, weight_elements, "Add+RMSNorm weight")?;
        if ranges_overlap(input_range, residual_range)
            || ranges_overlap(input_range, weight_range)
            || ranges_overlap(residual_range, weight_range)
        {
            return Err(CudaExecutorError::InvalidContract(
                "Add+RMSNorm input, residual, and weight regions must not overlap".into(),
            ));
        }

        let spec = AddRmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
            .map_err(|error| CudaExecutorError::InvalidContract(error.to_string()))?;
        let stream = unsafe { CudaStreamRef::from_raw(stream) };
        let backend = CudaBackend::from_stream(stream);
        let mut input = unsafe { DeviceSliceMut::from_raw_parts(input, input_elements) }?;
        let mut residual = unsafe { DeviceSliceMut::from_raw_parts(residual, residual_elements) }?;
        let weight = unsafe { DeviceSlice::from_raw_parts(weight, weight_elements) }?;

        T::launch(&backend, &mut input, &mut residual, &weight, spec)?;
        ADD_RMS_NORM_LAUNCHES.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn status_for_error(error: &CudaExecutorError) -> c_int {
        match error {
            CudaExecutorError::InvalidContract(_) => INVALID_ARGUMENT,
            CudaExecutorError::BackendUnavailable => UNAVAILABLE,
            CudaExecutorError::KernelSubmission { status, .. }
                if matches!(
                    *status,
                    INVALID_ARGUMENT | UNSUPPORTED | LAUNCH_ERROR | UNAVAILABLE
                ) =>
            {
                *status
            }
            CudaExecutorError::KernelSubmission { .. } => LAUNCH_ERROR,
        }
    }

    fn set_last_error(message: impl AsRef<str>) {
        let sanitized = message.as_ref().replace('\0', "\\0");
        let message = CString::new(sanitized).unwrap_or_else(|_| {
            CString::new("bridge error contained an invalid NUL")
                .expect("static bridge message contains no NUL")
        });
        LAST_ERROR.with(|slot| {
            *slot.borrow_mut() = message;
        });
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_owned()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else {
            "non-string panic payload".to_owned()
        }
    }

    fn run_ffi(operation: impl FnOnce() -> Result<(), CudaExecutorError>) -> c_int {
        match catch_unwind(AssertUnwindSafe(operation)) {
            Ok(Ok(())) => SUCCESS,
            Ok(Err(error)) => {
                let status = status_for_error(&error);
                set_last_error(error.to_string());
                status
            }
            Err(payload) => {
                set_last_error(format!(
                    "panic inside Loom Rust CUDA bridge: {}",
                    panic_message(payload)
                ));
                LAUNCH_ERROR
            }
        }
    }

    /// Return the detailed error for the most recent failed bridge call on
    /// this host thread.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_last_error_message() -> *const c_char {
        LAST_ERROR.with(|message| message.borrow().as_ptr())
    }

    /// Launch checked F32 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_f32(
        input: *mut f32,
        input_elements: u64,
        residual: *mut f32,
        residual_elements: u64,
        weight: *const f32,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input,
                input_elements,
                residual,
                residual_elements,
                weight,
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked FP16 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_f16(
        input: *mut u16,
        input_elements: u64,
        residual: *mut u16,
        residual_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input.cast::<f16>(),
                input_elements,
                residual.cast::<f16>(),
                residual_elements,
                weight.cast::<f16>(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Launch checked BF16 Add+RMSNorm through borrowed Rust CUDA resources.
    ///
    /// # Safety
    ///
    /// Device pointers, element counts, active CUDA context, stream lifetime,
    /// and asynchronous allocation lifetime must satisfy the bridge header.
    #[no_mangle]
    pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm_bf16(
        input: *mut u16,
        input_elements: u64,
        residual: *mut u16,
        residual_elements: u64,
        weight: *const u16,
        weight_elements: u64,
        rows: u32,
        hidden_size: u32,
        epsilon: f32,
        stream: *mut c_void,
    ) -> c_int {
        run_ffi(|| unsafe {
            launch_add_rms_norm(
                input.cast::<bf16>(),
                input_elements,
                residual.cast::<bf16>(),
                residual_elements,
                weight.cast::<bf16>(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        })
    }

    /// Return successful Add+RMSNorm submissions through the Rust bridge.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_add_rms_norm_launch_count() -> u64 {
        ADD_RMS_NORM_LAUNCHES.load(Ordering::Relaxed)
    }

    /// Reset Add+RMSNorm bridge launch telemetry.
    #[no_mangle]
    pub extern "C" fn loom_cuda_bridge_reset_add_rms_norm_launch_count() {
        ADD_RMS_NORM_LAUNCHES.store(0, Ordering::Relaxed);
    }

    #[cfg(test)]
    mod tests {
        use super::{checked_byte_range, ranges_overlap};

        #[test]
        fn byte_ranges_reject_overflow_and_detect_overlap() {
            let first = checked_byte_range(0x1000_usize as *const f32, 8, "first").unwrap();
            let overlapping = checked_byte_range(0x1010_usize as *const f32, 8, "overlap").unwrap();
            let disjoint = checked_byte_range(0x1020_usize as *const f32, 8, "disjoint").unwrap();
            assert!(ranges_overlap(first, overlapping));
            assert!(!ranges_overlap(first, disjoint));

            assert!(checked_byte_range::<f32>(std::ptr::null(), 8, "null").is_err());
            assert!(checked_byte_range(0x1000_usize as *const f32, 0, "empty").is_err());
            assert!(checked_byte_range(
                0x1000_usize as *const f32,
                usize::MAX / std::mem::size_of::<f32>() + 1,
                "overflow",
            )
            .is_err());
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda::*;
