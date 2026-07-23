//! Owned and borrowed CUDA resources used by operator implementations.

use crate::CudaExecutorError;
use loom_cuda_sys as sys;
use std::ffi::{c_void, CStr};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;

mod private {
    pub trait SealedStream {}
    pub trait SealedRead<T> {}
    pub trait SealedWrite<T> {}
}

/// CUDA stream handle accepted by [`crate::CudaBackend`].
///
/// This trait is sealed: Loom owns the implementations so safe operator code
/// can rely on the handle having been constructed under the documented CUDA
/// lifetime contract.
pub trait CudaStreamHandle: private::SealedStream {
    /// Returns the raw `cudaStream_t` value. A null value denotes CUDA's
    /// legacy default stream and is valid for a borrowed handle.
    fn raw(&self) -> *mut c_void;

    /// Waits until all preceding work on this stream is complete.
    fn synchronize(&self) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaStreamSynchronize(self.raw()) })
    }
}

/// An owned non-blocking CUDA stream.
#[derive(Debug)]
pub struct CudaStream(NonNull<c_void>);

impl CudaStream {
    pub fn new() -> Result<Self, CudaExecutorError> {
        let mut stream = std::ptr::null_mut();
        cuda_runtime_result(unsafe {
            sys::cudaStreamCreateWithFlags(&mut stream, sys::CUDA_STREAM_NON_BLOCKING)
        })?;
        let stream = NonNull::new(stream).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null owned stream".into())
        })?;
        Ok(Self(stream))
    }

    pub fn synchronize(&self) -> Result<(), CudaExecutorError> {
        CudaStreamHandle::synchronize(self)
    }

    pub const fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl private::SealedStream for CudaStream {}

impl CudaStreamHandle for CudaStream {
    fn raw(&self) -> *mut c_void {
        self.raw()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaStreamDestroy(self.raw()) };
    }
}

/// A non-owning CUDA stream whose lifetime is controlled by another runtime.
///
/// Dropping this value never destroys or synchronizes the stream. This is the
/// handle used to launch Loom kernels on a framework's current stream.
#[derive(Clone, Copy, Debug)]
pub struct CudaStreamRef<'stream> {
    raw: *mut c_void,
    marker: PhantomData<&'stream c_void>,
}

impl<'stream> CudaStreamRef<'stream> {
    /// Borrows a Loom-owned stream without transferring ownership.
    pub const fn from_stream(stream: &'stream CudaStream) -> Self {
        Self {
            raw: stream.raw(),
            marker: PhantomData,
        }
    }

    /// Borrows an external `cudaStream_t`.
    ///
    /// A null pointer is accepted because it represents CUDA's legacy default
    /// stream.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `raw` is either a valid CUDA stream for the
    /// active device/context or the null default-stream handle, and that the
    /// stream remains valid for `'stream`. The caller also remains responsible
    /// for any framework-level ordering and destruction.
    pub const unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self {
            raw,
            marker: PhantomData,
        }
    }

    pub fn synchronize(&self) -> Result<(), CudaExecutorError> {
        CudaStreamHandle::synchronize(self)
    }

    pub const fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl private::SealedStream for CudaStreamRef<'_> {}

impl CudaStreamHandle for CudaStreamRef<'_> {
    fn raw(&self) -> *mut c_void {
        self.raw()
    }
}

/// Read-only contiguous CUDA device memory accepted by Loom operators.
///
/// Implementations are sealed to owned [`DeviceBuffer`] values and borrowed
/// [`DeviceSlice`] / [`DeviceSliceMut`] views.
pub trait CudaDeviceRead<T: Copy>: private::SealedRead<T> {
    fn len(&self) -> usize;
    fn as_ptr(&self) -> *const T;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[doc(hidden)]
    fn require_len(&self, expected: usize, name: &str) -> Result<(), CudaExecutorError> {
        if self.len() == expected {
            Ok(())
        } else {
            Err(CudaExecutorError::InvalidContract(format!(
                "{name} has {} elements, expected {expected}",
                self.len()
            )))
        }
    }
}

/// Mutable contiguous CUDA device memory accepted by Loom operators.
pub trait CudaDeviceWrite<T: Copy>: CudaDeviceRead<T> + private::SealedWrite<T> {
    fn as_mut_ptr(&mut self) -> *mut T;
}

/// An owned CUDA allocation with an element count known to Rust.
#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pointer: NonNull<T>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    pub fn uninitialized(len: usize) -> Result<Self, CudaExecutorError> {
        let bytes = checked_region_bytes::<T>(len, "device allocation")?;

        let mut pointer = std::ptr::null_mut();
        cuda_runtime_result(unsafe { sys::cudaMalloc(&mut pointer, bytes) })?;
        let pointer = NonNull::new(pointer.cast::<T>()).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null allocation".into())
        })?;
        Ok(Self {
            pointer,
            len,
            marker: PhantomData,
        })
    }

    pub fn from_slice(values: &[T]) -> Result<Self, CudaExecutorError> {
        let mut allocation = Self::uninitialized(values.len())?;
        allocation.copy_from_slice(values)?;
        Ok(allocation)
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_device_slice(&self) -> DeviceSlice<'_, T> {
        DeviceSlice {
            pointer: self.pointer,
            len: self.len,
            marker: PhantomData,
        }
    }

    pub fn as_device_slice_mut(&mut self) -> DeviceSliceMut<'_, T> {
        DeviceSliceMut {
            pointer: self.pointer,
            len: self.len,
            marker: PhantomData,
        }
    }

    pub fn copy_from_slice(&mut self, values: &[T]) -> Result<(), CudaExecutorError> {
        CudaDeviceRead::require_len(self, values.len(), "host-to-device copy")?;
        cuda_runtime_result(unsafe {
            sys::cudaMemcpy(
                self.as_mut_ptr().cast::<c_void>(),
                values.as_ptr().cast::<c_void>(),
                self.len * size_of::<T>(),
                sys::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }

    pub fn copy_to_vec(&self) -> Result<Vec<T>, CudaExecutorError>
    where
        T: Default,
    {
        let mut values = vec![T::default(); self.len];
        cuda_runtime_result(unsafe {
            sys::cudaMemcpy(
                values.as_mut_ptr().cast::<c_void>(),
                self.as_ptr().cast::<c_void>(),
                self.len * size_of::<T>(),
                sys::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        })?;
        Ok(values)
    }

    pub(crate) const fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }

    pub(crate) const fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }
}

impl<T: Copy> private::SealedRead<T> for DeviceBuffer<T> {}
impl<T: Copy> private::SealedWrite<T> for DeviceBuffer<T> {}

impl<T: Copy> CudaDeviceRead<T> for DeviceBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }
}

impl<T: Copy> CudaDeviceWrite<T> for DeviceBuffer<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaFree(self.pointer.as_ptr().cast::<c_void>()) };
    }
}

/// A read-only borrowed view over addressable CUDA device storage.
#[derive(Clone, Copy, Debug)]
pub struct DeviceSlice<'memory, T: Copy> {
    pointer: NonNull<T>,
    len: usize,
    marker: PhantomData<&'memory T>,
}

impl<'memory, T: Copy> DeviceSlice<'memory, T> {
    /// Creates a borrowed view from a framework-owned CUDA allocation.
    ///
    /// # Safety
    ///
    /// `pointer` must be aligned for `T` and identify an addressable span of at
    /// least `len` contiguous `T` values in device memory for `'memory`.
    /// The allocation must belong to the CUDA context used for execution, and
    /// no mutable access may race with the logical elements selected by the
    /// validated operator layout. Disjoint strided views may share one
    /// allocation even when their addressable bounding spans overlap.
    pub unsafe fn from_raw_parts(pointer: *const T, len: usize) -> Result<Self, CudaExecutorError> {
        Ok(Self {
            pointer: checked_device_pointer(pointer, len, "borrowed device slice")?,
            len,
            marker: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }
}

impl<T: Copy> private::SealedRead<T> for DeviceSlice<'_, T> {}

impl<T: Copy> CudaDeviceRead<T> for DeviceSlice<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }
}

/// A writable borrowed view over addressable CUDA device storage.
#[derive(Debug)]
pub struct DeviceSliceMut<'memory, T: Copy> {
    pointer: NonNull<T>,
    len: usize,
    marker: PhantomData<&'memory mut T>,
}

impl<'memory, T: Copy> DeviceSliceMut<'memory, T> {
    /// Creates an exclusive borrowed view from framework-owned CUDA memory.
    ///
    /// # Safety
    ///
    /// `pointer` must be aligned for `T` and identify an addressable span of at
    /// least `len` writable, contiguous `T` elements in device memory for
    /// `'memory`. The caller must hold exclusive access to every logical
    /// element selected by the validated operator layout until all
    /// asynchronous Loom work using the view is ordered complete. Disjoint
    /// strided views may share one allocation even when their addressable
    /// bounding spans overlap.
    pub unsafe fn from_raw_parts(pointer: *mut T, len: usize) -> Result<Self, CudaExecutorError> {
        Ok(Self {
            pointer: checked_device_pointer(pointer.cast_const(), len, "mutable device slice")?,
            len,
            marker: PhantomData,
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }

    pub fn as_device_slice(&self) -> DeviceSlice<'_, T> {
        DeviceSlice {
            pointer: self.pointer,
            len: self.len,
            marker: PhantomData,
        }
    }
}

impl<T: Copy> private::SealedRead<T> for DeviceSliceMut<'_, T> {}
impl<T: Copy> private::SealedWrite<T> for DeviceSliceMut<'_, T> {}

impl<T: Copy> CudaDeviceRead<T> for DeviceSliceMut<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const T {
        self.pointer.as_ptr()
    }
}

impl<T: Copy> CudaDeviceWrite<T> for DeviceSliceMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.pointer.as_ptr()
    }
}

/// CUDA event used for device-side elapsed-time measurements.
#[derive(Debug)]
pub struct CudaEvent(NonNull<c_void>);

impl CudaEvent {
    pub fn new() -> Result<Self, CudaExecutorError> {
        let mut event = std::ptr::null_mut();
        cuda_runtime_result(unsafe { sys::cudaEventCreate(&mut event) })?;
        let event = NonNull::new(event).ok_or_else(|| {
            CudaExecutorError::InvalidContract("CUDA returned a null event".into())
        })?;
        Ok(Self(event))
    }

    pub fn record(
        &self,
        stream: &(impl CudaStreamHandle + ?Sized),
    ) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaEventRecord(self.raw(), stream.raw()) })
    }

    pub fn synchronize(&self) -> Result<(), CudaExecutorError> {
        cuda_runtime_result(unsafe { sys::cudaEventSynchronize(self.raw()) })
    }

    pub fn elapsed_ms(&self, end: &Self) -> Result<f32, CudaExecutorError> {
        let mut milliseconds = 0.0;
        cuda_runtime_result(unsafe {
            sys::cudaEventElapsedTime(&mut milliseconds, self.raw(), end.raw())
        })?;
        Ok(milliseconds)
    }

    const fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaEventDestroy(self.raw()) };
    }
}

fn checked_region_bytes<T>(len: usize, name: &str) -> Result<usize, CudaExecutorError> {
    if size_of::<T>() == 0 {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{name} cannot contain zero-sized elements"
        )));
    }
    let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| {
        CudaExecutorError::InvalidContract(format!("{name} size overflows usize"))
    })?;
    if bytes == 0 {
        return Err(CudaExecutorError::InvalidContract(format!(
            "zero-sized {name} is not supported"
        )));
    }
    Ok(bytes)
}

fn checked_device_pointer<T>(
    pointer: *const T,
    len: usize,
    name: &str,
) -> Result<NonNull<T>, CudaExecutorError> {
    checked_region_bytes::<T>(len, name)?;
    if pointer.is_null() {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{name} has a null pointer"
        )));
    }
    if !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{name} pointer is not aligned to {} bytes",
            align_of::<T>()
        )));
    }
    NonNull::new(pointer.cast_mut())
        .ok_or_else(|| CudaExecutorError::InvalidContract(format!("{name} has a null pointer")))
}

pub(crate) fn loom_status_result(status: i32) -> Result<(), CudaExecutorError> {
    if status == sys::LOOM_CUDA_SUCCESS {
        return Ok(());
    }
    let message = unsafe {
        let pointer = sys::loom_cuda_status_string(status);
        if pointer.is_null() {
            "unknown Loom CUDA status".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaExecutorError::KernelSubmission { status, message })
}

fn cuda_runtime_result(status: i32) -> Result<(), CudaExecutorError> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe {
        let pointer = sys::cudaGetErrorString(status);
        if pointer.is_null() {
            "unknown CUDA runtime status".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Err(CudaExecutorError::KernelSubmission { status, message })
}

#[cfg(test)]
mod tests {
    use super::{CudaDeviceRead, DeviceSlice, DeviceSliceMut};
    use crate::CudaExecutorError;

    #[test]
    fn borrowed_device_regions_validate_pointer_length_and_alignment() {
        let error = unsafe { DeviceSlice::<f32>::from_raw_parts(std::ptr::null(), 4) }
            .expect_err("null device pointers must be rejected");
        assert!(matches!(error, CudaExecutorError::InvalidContract(_)));

        let aligned = 0x1000_usize as *mut f32;
        let error = unsafe { DeviceSlice::from_raw_parts(aligned.cast_const(), 0) }
            .expect_err("empty borrowed regions must be rejected");
        assert!(matches!(error, CudaExecutorError::InvalidContract(_)));

        let read = unsafe { DeviceSlice::from_raw_parts(aligned.cast_const(), 4) }.unwrap();
        assert_eq!(read.len(), 4);
        assert_eq!(CudaDeviceRead::as_ptr(&read), aligned.cast_const());

        let mut write = unsafe { DeviceSliceMut::from_raw_parts(aligned, 4) }.unwrap();
        assert_eq!(write.len(), 4);
        assert_eq!(write.as_mut_ptr(), aligned);

        let unaligned = 0x1001_usize as *const f32;
        let error = unsafe { DeviceSlice::from_raw_parts(unaligned, 4) }
            .expect_err("misaligned pointers must be rejected");
        assert!(matches!(error, CudaExecutorError::InvalidContract(_)));

        let overflowing_len = usize::MAX / std::mem::size_of::<f32>() + 1;
        let error = unsafe { DeviceSlice::from_raw_parts(aligned, overflowing_len) }
            .expect_err("overflowing borrowed regions must be rejected");
        assert!(matches!(error, CudaExecutorError::InvalidContract(_)));
    }
}
