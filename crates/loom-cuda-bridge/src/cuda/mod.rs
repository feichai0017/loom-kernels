use half::{bf16, f16};
use loom_cuda::runtime::{CudaStreamHandle, CudaStreamRef, DeviceSlice, DeviceSliceMut};
use loom_cuda::{
    paged_decode_attention_split_k_workspace_elements, CudaBackend, CudaExecutorError,
    Fp8ScaleLayout, PagedDecodeLayout, RopePagedKvLayout, RowStridedLayout,
    SiluAndMulDynamicFp8Options,
};
use loom_kernels::{
    AddRmsNormSpec, DType, GreedySampleLogprobsSpec, GreedySpeculativeVerifySpec, MinPFilterSpec,
    PagedDecodeAttentionSpec, RmsNormDynamicFp8Spec, RmsNormSpec, RopePagedKvWriteSpec,
    RotaryEmbeddingSpec, RotaryStyle, SelectedTokenLogprobsSpec, SiluAndMulDynamicFp8Spec,
    SiluAndMulSpec,
};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CString};
use std::mem::{align_of, size_of};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};

const SUCCESS: c_int = 0;
const INVALID_ARGUMENT: c_int = 1;
const LAUNCH_ERROR: c_int = 2;
const UNAVAILABLE: c_int = 3;

const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;
const DTYPE_BF16: u32 = 2;

const OP_RMS_NORM: usize = 0;
const OP_ADD_RMS_NORM: usize = 1;
const OP_RMS_NORM_DYNAMIC_FP8: usize = 2;
const OP_SILU_AND_MUL: usize = 3;
const OP_SILU_AND_MUL_DYNAMIC_FP8: usize = 4;
const OP_ROPE_PAGED_KV_WRITE: usize = 5;
const OP_GREEDY_SAMPLE_LOGPROBS: usize = 6;
const OP_SELECTED_TOKEN_LOGPROBS: usize = 7;
const OP_MIN_P_FILTER: usize = 8;
const OP_PAGED_DECODE_ATTENTION: usize = 9;
const OP_GREEDY_SPECULATIVE_VERIFY: usize = 10;
const OPERATOR_COUNT: usize = 11;

static LAUNCH_COUNTS: [AtomicU64; OPERATOR_COUNT] = [const { AtomicU64::new(0) }; OPERATOR_COUNT];

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(
        CString::new("no bridge error has been recorded")
            .expect("static bridge message contains no NUL")
    );
}

#[derive(Clone, Copy)]
struct ByteRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Copy)]
enum ScalarKind {
    F32,
    F16,
    Bf16,
}

fn scalar_kind(dtype: u32) -> Result<ScalarKind, CudaExecutorError> {
    match dtype {
        DTYPE_F32 => Ok(ScalarKind::F32),
        DTYPE_F16 => Ok(ScalarKind::F16),
        DTYPE_BF16 => Ok(ScalarKind::Bf16),
        _ => Err(CudaExecutorError::InvalidContract(format!(
            "unknown bridge dtype code {dtype}"
        ))),
    }
}

fn record_launch(operation: usize) {
    LAUNCH_COUNTS[operation].fetch_add(1, Ordering::Relaxed);
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

fn require_disjoint(
    regions: &[(&str, ByteRange)],
    operation: &str,
) -> Result<(), CudaExecutorError> {
    for (index, &(left_name, left)) in regions.iter().enumerate() {
        for &(right_name, right) in &regions[index + 1..] {
            if ranges_overlap(left, right) {
                return Err(CudaExecutorError::InvalidContract(format!(
                    "{operation} regions {left_name} and {right_name} must not overlap"
                )));
            }
        }
    }
    Ok(())
}

fn require_disjoint_from(
    target_name: &str,
    target: ByteRange,
    others: &[(&str, ByteRange)],
    operation: &str,
) -> Result<(), CudaExecutorError> {
    for &(other_name, other) in others {
        if ranges_overlap(target, other) {
            return Err(CudaExecutorError::InvalidContract(format!(
                "{operation} regions {target_name} and {other_name} must not overlap"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_disjoint_or_dense_packed_axis<T>(
    left_name: &str,
    left: ByteRange,
    left_outer_stride: usize,
    left_dense_inner_elements: Option<usize>,
    right_name: &str,
    right: ByteRange,
    right_outer_stride: usize,
    right_dense_inner_elements: Option<usize>,
    operation: &str,
) -> Result<(), CudaExecutorError> {
    if !ranges_overlap(left, right) {
        return Ok(());
    }
    let (Some(left_inner_elements), Some(right_inner_elements)) =
        (left_dense_inner_elements, right_dense_inner_elements)
    else {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{operation} regions {left_name} and {right_name} overlap without a dense packed layout"
        )));
    };
    if left_outer_stride != right_outer_stride {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{operation} packed regions {left_name} and {right_name} must share an outer stride"
        )));
    }

    let element_size = size_of::<T>();
    let outer_stride = left_outer_stride.checked_mul(element_size).ok_or_else(|| {
        CudaExecutorError::InvalidContract(format!(
            "{operation} packed outer stride overflows usize"
        ))
    })?;
    let left_inner_bytes = left_inner_elements
        .checked_mul(element_size)
        .ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!(
                "{operation} {left_name} packed inner span overflows usize"
            ))
        })?;
    let right_inner_bytes = right_inner_elements
        .checked_mul(element_size)
        .ok_or_else(|| {
            CudaExecutorError::InvalidContract(format!(
                "{operation} {right_name} packed inner span overflows usize"
            ))
        })?;
    let (earlier_name, earlier, earlier_inner_bytes, later_name, later, later_inner_bytes) =
        if left.start <= right.start {
            (
                left_name,
                left,
                left_inner_bytes,
                right_name,
                right,
                right_inner_bytes,
            )
        } else {
            (
                right_name,
                right,
                right_inner_bytes,
                left_name,
                left,
                left_inner_bytes,
            )
        };
    let offset = later.start - earlier.start;
    let later_end = offset.checked_add(later_inner_bytes).ok_or_else(|| {
        CudaExecutorError::InvalidContract(format!("{operation} packed inner span overflows usize"))
    })?;
    if offset >= outer_stride || earlier_inner_bytes > offset || later_end > outer_stride {
        return Err(CudaExecutorError::InvalidContract(format!(
            "{operation} logical elements in packed regions {earlier_name} and {later_name} overlap"
        )));
    }
    Ok(())
}

unsafe fn read_slice<'a, T: Copy>(
    pointer: *const T,
    elements: u64,
    name: &str,
) -> Result<(DeviceSlice<'a, T>, ByteRange), CudaExecutorError> {
    let elements = element_count(elements, name)?;
    let range = checked_byte_range(pointer, elements, name)?;
    let slice = unsafe { DeviceSlice::from_raw_parts(pointer, elements) }?;
    Ok((slice, range))
}

unsafe fn write_slice<'a, T: Copy>(
    pointer: *mut T,
    elements: u64,
    name: &str,
) -> Result<(DeviceSliceMut<'a, T>, ByteRange), CudaExecutorError> {
    let elements = element_count(elements, name)?;
    let range = checked_byte_range(pointer.cast_const(), elements, name)?;
    let slice = unsafe { DeviceSliceMut::from_raw_parts(pointer, elements) }?;
    Ok((slice, range))
}

fn stream_backend(stream: *mut c_void) -> CudaBackend<CudaStreamRef<'static>> {
    let stream = unsafe { CudaStreamRef::from_raw(stream) };
    CudaBackend::from_stream(stream)
}

trait Scalar: Copy {
    const DTYPE: DType;

    fn rms_norm<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        weight: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, Self>,
        spec: RmsNormSpec,
    ) -> Result<(), CudaExecutorError>;

    fn add_rms_norm<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &mut DeviceSliceMut<'_, Self>,
        residual: &mut DeviceSliceMut<'_, Self>,
        weight: &DeviceSlice<'_, Self>,
        spec: AddRmsNormSpec,
    ) -> Result<(), CudaExecutorError>;

    fn rms_norm_dynamic_fp8<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        weight: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, u8>,
        scales: &mut DeviceSliceMut<'_, f32>,
        spec: RmsNormDynamicFp8Spec,
    ) -> Result<(), CudaExecutorError>;

    fn silu_and_mul<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, Self>,
        spec: SiluAndMulSpec,
    ) -> Result<(), CudaExecutorError>;

    fn greedy_sample_logprobs<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        logits: &DeviceSlice<'_, Self>,
        token_ids: &mut DeviceSliceMut<'_, i32>,
        logprobs: &mut DeviceSliceMut<'_, f32>,
        ranks: &mut DeviceSliceMut<'_, i64>,
        spec: GreedySampleLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError>;

    fn selected_token_logprobs<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        logits: &DeviceSlice<'_, Self>,
        token_ids: &DeviceSlice<'_, i64>,
        logprobs: &mut DeviceSliceMut<'_, f32>,
        ranks: &mut DeviceSliceMut<'_, i64>,
        spec: SelectedTokenLogprobsSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError>;

    fn min_p_filter<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        logits: &mut DeviceSliceMut<'_, Self>,
        min_p: &DeviceSlice<'_, f32>,
        spec: MinPFilterSpec,
        layout: RowStridedLayout,
    ) -> Result<(), CudaExecutorError>;

    #[allow(clippy::too_many_arguments)]
    fn rope_paged_kv_write<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        query: &mut DeviceSliceMut<'_, Self>,
        key: &mut DeviceSliceMut<'_, Self>,
        value: &DeviceSlice<'_, Self>,
        positions: &DeviceSlice<'_, i64>,
        cos_sin_cache: &DeviceSlice<'_, Self>,
        key_cache: &mut DeviceSliceMut<'_, Self>,
        value_cache: &mut DeviceSliceMut<'_, Self>,
        slot_mapping: &DeviceSlice<'_, i64>,
        spec: RopePagedKvWriteSpec,
        layout: RopePagedKvLayout,
    ) -> Result<(), CudaExecutorError>;

    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        query: &DeviceSlice<'_, Self>,
        key_cache: &DeviceSlice<'_, Self>,
        value_cache: &DeviceSlice<'_, Self>,
        block_tables: &DeviceSlice<'_, i32>,
        sequence_lengths: &DeviceSlice<'_, i32>,
        output: &mut DeviceSliceMut<'_, Self>,
        spec: PagedDecodeAttentionSpec,
        layout: PagedDecodeLayout,
    ) -> Result<(), CudaExecutorError>;

    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention_split_k<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        query: &DeviceSlice<'_, Self>,
        key_cache: &DeviceSlice<'_, Self>,
        value_cache: &DeviceSlice<'_, Self>,
        block_tables: &DeviceSlice<'_, i32>,
        sequence_lengths: &DeviceSlice<'_, i32>,
        output: &mut DeviceSliceMut<'_, Self>,
        workspace: &mut DeviceSliceMut<'_, f32>,
        spec: PagedDecodeAttentionSpec,
        layout: PagedDecodeLayout,
    ) -> Result<(), CudaExecutorError>;
}

macro_rules! impl_scalar {
    (
        $scalar:ty,
        $dtype:expr,
        $rms_norm:ident,
        $add_rms_norm:ident,
        $rms_norm_dynamic_fp8:ident,
        $silu_and_mul:ident,
        $greedy:ident,
        $selected:ident,
        $min_p:ident,
        $rope:ident,
        $paged:ident,
        $paged_split_k:ident
    ) => {
        impl Scalar for $scalar {
            const DTYPE: DType = $dtype;

            fn rms_norm<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                input: &DeviceSlice<'_, Self>,
                weight: &DeviceSlice<'_, Self>,
                output: &mut DeviceSliceMut<'_, Self>,
                spec: RmsNormSpec,
            ) -> Result<(), CudaExecutorError> {
                backend.$rms_norm(input, weight, output, spec)
            }

            fn add_rms_norm<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                input: &mut DeviceSliceMut<'_, Self>,
                residual: &mut DeviceSliceMut<'_, Self>,
                weight: &DeviceSlice<'_, Self>,
                spec: AddRmsNormSpec,
            ) -> Result<(), CudaExecutorError> {
                backend.$add_rms_norm(input, residual, weight, spec)
            }

            fn rms_norm_dynamic_fp8<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                input: &DeviceSlice<'_, Self>,
                weight: &DeviceSlice<'_, Self>,
                output: &mut DeviceSliceMut<'_, u8>,
                scales: &mut DeviceSliceMut<'_, f32>,
                spec: RmsNormDynamicFp8Spec,
            ) -> Result<(), CudaExecutorError> {
                backend.$rms_norm_dynamic_fp8(input, weight, output, scales, spec)
            }

            fn silu_and_mul<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                input: &DeviceSlice<'_, Self>,
                output: &mut DeviceSliceMut<'_, Self>,
                spec: SiluAndMulSpec,
            ) -> Result<(), CudaExecutorError> {
                backend.$silu_and_mul(input, output, spec)
            }

            fn greedy_sample_logprobs<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                logits: &DeviceSlice<'_, Self>,
                token_ids: &mut DeviceSliceMut<'_, i32>,
                logprobs: &mut DeviceSliceMut<'_, f32>,
                ranks: &mut DeviceSliceMut<'_, i64>,
                spec: GreedySampleLogprobsSpec,
                layout: RowStridedLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$greedy(logits, token_ids, logprobs, ranks, spec, layout)
            }

            fn selected_token_logprobs<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                logits: &DeviceSlice<'_, Self>,
                token_ids: &DeviceSlice<'_, i64>,
                logprobs: &mut DeviceSliceMut<'_, f32>,
                ranks: &mut DeviceSliceMut<'_, i64>,
                spec: SelectedTokenLogprobsSpec,
                layout: RowStridedLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$selected(logits, token_ids, logprobs, ranks, spec, layout)
            }

            fn min_p_filter<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                logits: &mut DeviceSliceMut<'_, Self>,
                min_p: &DeviceSlice<'_, f32>,
                spec: MinPFilterSpec,
                layout: RowStridedLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$min_p(logits, min_p, spec, layout)
            }

            fn rope_paged_kv_write<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                query: &mut DeviceSliceMut<'_, Self>,
                key: &mut DeviceSliceMut<'_, Self>,
                value: &DeviceSlice<'_, Self>,
                positions: &DeviceSlice<'_, i64>,
                cos_sin_cache: &DeviceSlice<'_, Self>,
                key_cache: &mut DeviceSliceMut<'_, Self>,
                value_cache: &mut DeviceSliceMut<'_, Self>,
                slot_mapping: &DeviceSlice<'_, i64>,
                spec: RopePagedKvWriteSpec,
                layout: RopePagedKvLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$rope(
                    query,
                    key,
                    value,
                    positions,
                    cos_sin_cache,
                    key_cache,
                    value_cache,
                    slot_mapping,
                    spec,
                    layout,
                )
            }

            fn paged_decode_attention<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                query: &DeviceSlice<'_, Self>,
                key_cache: &DeviceSlice<'_, Self>,
                value_cache: &DeviceSlice<'_, Self>,
                block_tables: &DeviceSlice<'_, i32>,
                sequence_lengths: &DeviceSlice<'_, i32>,
                output: &mut DeviceSliceMut<'_, Self>,
                spec: PagedDecodeAttentionSpec,
                layout: PagedDecodeLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$paged(
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    sequence_lengths,
                    output,
                    spec,
                    layout,
                )
            }

            fn paged_decode_attention_split_k<S: CudaStreamHandle>(
                backend: &CudaBackend<S>,
                query: &DeviceSlice<'_, Self>,
                key_cache: &DeviceSlice<'_, Self>,
                value_cache: &DeviceSlice<'_, Self>,
                block_tables: &DeviceSlice<'_, i32>,
                sequence_lengths: &DeviceSlice<'_, i32>,
                output: &mut DeviceSliceMut<'_, Self>,
                workspace: &mut DeviceSliceMut<'_, f32>,
                spec: PagedDecodeAttentionSpec,
                layout: PagedDecodeLayout,
            ) -> Result<(), CudaExecutorError> {
                backend.$paged_split_k(
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    sequence_lengths,
                    output,
                    workspace,
                    spec,
                    layout,
                )
            }
        }
    };
}

impl_scalar!(
    f32,
    DType::F32,
    rms_norm_f32,
    add_rms_norm_f32,
    rms_norm_dynamic_fp8_f32,
    silu_and_mul_f32,
    greedy_sample_logprobs_f32,
    selected_token_logprobs_f32,
    min_p_filter_f32,
    rope_paged_kv_write_f32,
    paged_decode_attention_f32,
    paged_decode_attention_split_k_f32
);
impl_scalar!(
    f16,
    DType::F16,
    rms_norm_f16,
    add_rms_norm_f16,
    rms_norm_dynamic_fp8_f16,
    silu_and_mul_f16,
    greedy_sample_logprobs_f16,
    selected_token_logprobs_f16,
    min_p_filter_f16,
    rope_paged_kv_write_f16,
    paged_decode_attention_f16,
    paged_decode_attention_split_k_f16
);
impl_scalar!(
    bf16,
    DType::Bf16,
    rms_norm_bf16,
    add_rms_norm_bf16,
    rms_norm_dynamic_fp8_bf16,
    silu_and_mul_bf16,
    greedy_sample_logprobs_bf16,
    selected_token_logprobs_bf16,
    min_p_filter_bf16,
    rope_paged_kv_write_bf16,
    paged_decode_attention_bf16,
    paged_decode_attention_split_k_bf16
);

trait Fp8Scalar: Copy {
    const DTYPE: DType;

    fn silu_and_mul_dynamic_fp8<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, u8>,
        scales: &mut DeviceSliceMut<'_, f32>,
        spec: SiluAndMulDynamicFp8Spec,
        options: SiluAndMulDynamicFp8Options<'_>,
    ) -> Result<(), CudaExecutorError>;
}

impl Fp8Scalar for f16 {
    const DTYPE: DType = DType::F16;

    fn silu_and_mul_dynamic_fp8<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, u8>,
        scales: &mut DeviceSliceMut<'_, f32>,
        spec: SiluAndMulDynamicFp8Spec,
        options: SiluAndMulDynamicFp8Options<'_>,
    ) -> Result<(), CudaExecutorError> {
        backend.silu_and_mul_dynamic_fp8_f16(input, output, scales, spec, options)
    }
}

impl Fp8Scalar for bf16 {
    const DTYPE: DType = DType::Bf16;

    fn silu_and_mul_dynamic_fp8<S: CudaStreamHandle>(
        backend: &CudaBackend<S>,
        input: &DeviceSlice<'_, Self>,
        output: &mut DeviceSliceMut<'_, u8>,
        scales: &mut DeviceSliceMut<'_, f32>,
        spec: SiluAndMulDynamicFp8Spec,
        options: SiluAndMulDynamicFp8Options<'_>,
    ) -> Result<(), CudaExecutorError> {
        backend.silu_and_mul_dynamic_fp8_bf16(input, output, scales, spec, options)
    }
}

fn invalid_contract(error: impl std::fmt::Display) -> CudaExecutorError {
    CudaExecutorError::InvalidContract(error.to_string())
}

#[allow(clippy::too_many_arguments)]

fn set_last_error(message: &str) {
    let sanitized = message.replace('\0', "\\0");
    LAST_ERROR.with(|last_error| {
        *last_error.borrow_mut() =
            CString::new(sanitized).expect("sanitized bridge error contains no NUL");
    });
}

fn error_status(error: &CudaExecutorError) -> c_int {
    match error {
        CudaExecutorError::InvalidContract(_) => INVALID_ARGUMENT,
        CudaExecutorError::BackendUnavailable => UNAVAILABLE,
        CudaExecutorError::KernelSubmission { .. } => LAUNCH_ERROR,
    }
}

fn bridge_call(operation: impl FnOnce() -> Result<(), CudaExecutorError>) -> c_int {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => SUCCESS,
        Ok(Err(error)) => {
            set_last_error(&error.to_string());
            error_status(&error)
        }
        Err(_) => {
            set_last_error("Rust panic was contained at the Loom CUDA bridge");
            LAUNCH_ERROR
        }
    }
}

macro_rules! dispatch_scalar {
    ($kind:expr, $function:ident ( $($argument:expr),* $(,)? )) => {
        match $kind {
            ScalarKind::F32 => unsafe { $function::<f32>($($argument),*) },
            ScalarKind::F16 => unsafe { $function::<f16>($($argument),*) },
            ScalarKind::Bf16 => unsafe { $function::<bf16>($($argument),*) },
        }
    };
}

/// Return the bridge ABI version.
#[no_mangle]
pub extern "C" fn loom_cuda_bridge_abi_version() -> u32 {
    1
}

/// Return the detailed error recorded by the most recent failed bridge call
/// on this host thread.
#[no_mangle]
pub extern "C" fn loom_cuda_bridge_last_error_message() -> *const c_char {
    LAST_ERROR.with(|last_error| last_error.borrow().as_ptr())
}

/// Read successful launch telemetry for one bridge operator.
///
/// # Safety
///
/// `count` must be a valid writable host pointer.
#[no_mangle]
pub unsafe extern "C" fn loom_cuda_bridge_launch_count(operation: u32, count: *mut u64) -> c_int {
    bridge_call(|| {
        let operation = usize::try_from(operation)
            .map_err(|_| CudaExecutorError::InvalidContract("invalid operator code".into()))?;
        if operation >= OPERATOR_COUNT {
            return Err(CudaExecutorError::InvalidContract(format!(
                "unknown bridge operator code {operation}"
            )));
        }
        if count.is_null() || !(count as usize).is_multiple_of(align_of::<u64>()) {
            return Err(CudaExecutorError::InvalidContract(
                "launch-count output pointer is null or misaligned".into(),
            ));
        }
        unsafe {
            *count = LAUNCH_COUNTS[operation].load(Ordering::Relaxed);
        }
        Ok(())
    })
}

/// Reset successful launch telemetry for one bridge operator.
#[no_mangle]
pub extern "C" fn loom_cuda_bridge_reset_launch_count(operation: u32) -> c_int {
    bridge_call(|| {
        let operation = usize::try_from(operation)
            .map_err(|_| CudaExecutorError::InvalidContract("invalid operator code".into()))?;
        if operation >= OPERATOR_COUNT {
            return Err(CudaExecutorError::InvalidContract(format!(
                "unknown bridge operator code {operation}"
            )));
        }
        LAUNCH_COUNTS[operation].store(0, Ordering::Relaxed);
        Ok(())
    })
}

mod activation_bridge;
mod attention_bridge;
mod logits_bridge;
mod norm_bridge;
mod rope_kv_bridge;
mod sampling_bridge;
mod speculative_bridge;
#[cfg(test)]
mod tests;

pub use activation_bridge::*;
pub use attention_bridge::*;
pub use logits_bridge::*;
pub use norm_bridge::*;
pub use rope_kv_bridge::*;
pub use sampling_bridge::*;
pub use speculative_bridge::*;
