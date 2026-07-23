use half::{bf16, f16};
use loom_cuda::runtime::{CudaStreamHandle, CudaStreamRef, DeviceSlice, DeviceSliceMut};
use loom_cuda::{
    paged_decode_attention_split_k_workspace_elements, CudaBackend, CudaExecutorError,
    Fp8ScaleLayout, PagedDecodeLayout, RopePagedKvLayout, RowStridedLayout,
    SiluAndMulDynamicFp8Options,
};
use loom_kernels::{
    AddRmsNormSpec, DType, GreedySampleLogprobsSpec, MinPFilterSpec, PagedDecodeAttentionSpec,
    RmsNormDynamicFp8Spec, RmsNormSpec, RopePagedKvWriteSpec, RotaryEmbeddingSpec, RotaryStyle,
    SelectedTokenLogprobsSpec, SiluAndMulDynamicFp8Spec, SiluAndMulSpec,
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
const OPERATOR_COUNT: usize = 10;

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
unsafe fn launch_rms_norm<T: Scalar>(
    input: *const T,
    input_elements: u64,
    weight: *const T,
    weight_elements: u64,
    output: *mut T,
    output_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "RMSNorm input") }?;
    let (weight, weight_range) = unsafe { read_slice(weight, weight_elements, "RMSNorm weight") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "RMSNorm output") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("weight", weight_range),
            ("output", output_range),
        ],
        "RMSNorm",
    )?;
    let spec = RmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::rms_norm(&stream_backend(stream), &input, &weight, &mut output, spec)?;
    record_launch(OP_RMS_NORM);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_add_rms_norm<T: Scalar>(
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
    let (mut input, input_range) =
        unsafe { write_slice(input, input_elements, "Add+RMSNorm input") }?;
    let (mut residual, residual_range) =
        unsafe { write_slice(residual, residual_elements, "Add+RMSNorm residual") }?;
    let (weight, weight_range) =
        unsafe { read_slice(weight, weight_elements, "Add+RMSNorm weight") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("residual", residual_range),
            ("weight", weight_range),
        ],
        "Add+RMSNorm",
    )?;
    let spec = AddRmsNormSpec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::add_rms_norm(
        &stream_backend(stream),
        &mut input,
        &mut residual,
        &weight,
        spec,
    )?;
    record_launch(OP_ADD_RMS_NORM);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_rms_norm_dynamic_fp8<T: Scalar>(
    input: *const T,
    input_elements: u64,
    weight: *const T,
    weight_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "RMSNorm+FP8 input") }?;
    let (weight, weight_range) =
        unsafe { read_slice(weight, weight_elements, "RMSNorm+FP8 weight") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "RMSNorm+FP8 output") }?;
    let (mut scales, scales_range) =
        unsafe { write_slice(scales, scale_elements, "RMSNorm+FP8 scales") }?;
    require_disjoint(
        &[
            ("input", input_range),
            ("weight", weight_range),
            ("output", output_range),
            ("scales", scales_range),
        ],
        "RMSNorm+FP8",
    )?;
    let spec = RmsNormDynamicFp8Spec::new(rows as usize, hidden_size as usize, epsilon, T::DTYPE)
        .map_err(invalid_contract)?;
    T::rms_norm_dynamic_fp8(
        &stream_backend(stream),
        &input,
        &weight,
        &mut output,
        &mut scales,
        spec,
    )?;
    record_launch(OP_RMS_NORM_DYNAMIC_FP8);
    Ok(())
}

unsafe fn launch_silu_and_mul<T: Scalar>(
    input: *const T,
    input_elements: u64,
    output: *mut T,
    output_elements: u64,
    rows: u32,
    width: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) = unsafe { read_slice(input, input_elements, "SiLU-and-Mul input") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "SiLU-and-Mul output") }?;
    require_disjoint(
        &[("input", input_range), ("output", output_range)],
        "SiLU-and-Mul",
    )?;
    let spec =
        SiluAndMulSpec::new(rows as usize, width as usize, T::DTYPE).map_err(invalid_contract)?;
    T::silu_and_mul(&stream_backend(stream), &input, &mut output, spec)?;
    record_launch(OP_SILU_AND_MUL);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_silu_and_mul_dynamic_fp8<T: Fp8Scalar>(
    input: *const T,
    input_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    scale_upper_bound: *const f32,
    scale_upper_bound_elements: u64,
    rows: u32,
    width: u32,
    group_size: u32,
    scales_transposed: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (input, input_range) =
        unsafe { read_slice(input, input_elements, "SiLU-and-Mul+FP8 input") }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "SiLU-and-Mul+FP8 output") }?;
    let (mut scales, scales_range) =
        unsafe { write_slice(scales, scale_elements, "SiLU-and-Mul+FP8 scales") }?;
    let scale_upper_bound = if scale_upper_bound.is_null() {
        if scale_upper_bound_elements != 0 {
            return Err(CudaExecutorError::InvalidContract(
                "null FP8 scale upper bound must have zero elements".into(),
            ));
        }
        None
    } else {
        if scale_upper_bound_elements != 1 {
            return Err(CudaExecutorError::InvalidContract(
                "FP8 scale upper bound must contain exactly one F32 element".into(),
            ));
        }
        let (value, range) = unsafe {
            read_slice(
                scale_upper_bound,
                scale_upper_bound_elements,
                "SiLU-and-Mul+FP8 scale upper bound",
            )
        }?;
        require_disjoint(
            &[
                ("input", input_range),
                ("output", output_range),
                ("scales", scales_range),
                ("scale upper bound", range),
            ],
            "SiLU-and-Mul+FP8",
        )?;
        Some(value)
    };
    if scale_upper_bound.is_none() {
        require_disjoint(
            &[
                ("input", input_range),
                ("output", output_range),
                ("scales", scales_range),
            ],
            "SiLU-and-Mul+FP8",
        )?;
    }
    let scale_layout = match scales_transposed {
        0 => Fp8ScaleLayout::RowMajor,
        1 => Fp8ScaleLayout::GroupMajor,
        _ => {
            return Err(CudaExecutorError::InvalidContract(
                "FP8 scale layout flag must be 0 or 1".into(),
            ))
        }
    };
    let spec =
        SiluAndMulDynamicFp8Spec::new(rows as usize, width as usize, group_size as usize, T::DTYPE)
            .map_err(invalid_contract)?;
    T::silu_and_mul_dynamic_fp8(
        &stream_backend(stream),
        &input,
        &mut output,
        &mut scales,
        spec,
        SiluAndMulDynamicFp8Options {
            scale_upper_bound,
            scale_layout,
        },
    )?;
    record_launch(OP_SILU_AND_MUL_DYNAMIC_FP8);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_greedy_sample_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    token_ids: *mut i32,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) = unsafe { read_slice(logits, logits_elements, "greedy logits") }?;
    let (mut token_ids, token_ids_range) =
        unsafe { write_slice(token_ids, token_id_elements, "greedy token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "greedy logprobs") }?;
    let (mut ranks, ranks_range) = unsafe { write_slice(ranks, rank_elements, "greedy ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "greedy sampling",
    )?;
    let spec = GreedySampleLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "greedy row stride")?,
    )?;
    T::greedy_sample_logprobs(
        &stream_backend(stream),
        &logits,
        &mut token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_GREEDY_SAMPLE_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_selected_token_logprobs<T: Scalar>(
    logits: *const T,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (logits, logits_range) =
        unsafe { read_slice(logits, logits_elements, "selected-token logits") }?;
    let (token_ids, token_ids_range) =
        unsafe { read_slice(token_ids, token_id_elements, "selected-token IDs") }?;
    let (mut logprobs, logprobs_range) =
        unsafe { write_slice(logprobs, logprob_elements, "selected-token logprobs") }?;
    let (mut ranks, ranks_range) =
        unsafe { write_slice(ranks, rank_elements, "selected-token ranks") }?;
    require_disjoint(
        &[
            ("logits", logits_range),
            ("token IDs", token_ids_range),
            ("logprobs", logprobs_range),
            ("ranks", ranks_range),
        ],
        "selected-token logprobs",
    )?;
    let spec = SelectedTokenLogprobsSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "selected-token row stride")?,
    )?;
    T::selected_token_logprobs(
        &stream_backend(stream),
        &logits,
        &token_ids,
        &mut logprobs,
        &mut ranks,
        spec,
        layout,
    )?;
    record_launch(OP_SELECTED_TOKEN_LOGPROBS);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_min_p_filter<T: Scalar>(
    logits: *mut T,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let (mut logits, logits_range) =
        unsafe { write_slice(logits, logits_elements, "min-p logits") }?;
    let (min_p, min_p_range) = unsafe { read_slice(min_p, min_p_elements, "min-p values") }?;
    require_disjoint(
        &[("logits", logits_range), ("min-p", min_p_range)],
        "min-p filtering",
    )?;
    let spec = MinPFilterSpec::new(rows as usize, vocab_size as usize, T::DTYPE)
        .map_err(invalid_contract)?;
    let layout = RowStridedLayout::new(
        vocab_size as usize,
        element_count(row_stride, "min-p row stride")?,
    )?;
    T::min_p_filter(&stream_backend(stream), &mut logits, &min_p, spec, layout)?;
    record_launch(OP_MIN_P_FILTER);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_rope_paged_kv_write<T: Scalar>(
    query: *mut T,
    query_elements: u64,
    key: *mut T,
    key_elements: u64,
    value: *const T,
    value_elements: u64,
    positions: *const i64,
    position_elements: u64,
    cos_sin_cache: *const T,
    cos_sin_cache_elements: u64,
    key_cache: *mut T,
    key_cache_elements: u64,
    value_cache: *mut T,
    value_cache_elements: u64,
    slot_mapping: *const i64,
    slot_mapping_elements: u64,
    tokens: u32,
    cache_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    rotary_dim: u32,
    max_position: u32,
    num_blocks: u32,
    block_size: u32,
    query_token_stride: u64,
    query_head_stride: u64,
    key_token_stride: u64,
    source_key_head_stride: u64,
    value_token_stride: u64,
    source_value_head_stride: u64,
    key_block_stride: u64,
    key_page_stride: u64,
    key_head_stride: u64,
    value_block_stride: u64,
    value_page_stride: u64,
    value_cache_head_stride: u64,
    is_neox: u32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let style = match is_neox {
        0 => RotaryStyle::Interleaved,
        1 => RotaryStyle::NeoX,
        _ => {
            return Err(CudaExecutorError::InvalidContract(
                "RoPE style flag must be 0 or 1".into(),
            ))
        }
    };
    let rotary = RotaryEmbeddingSpec::new(
        tokens as usize,
        query_heads as usize,
        kv_heads as usize,
        head_size as usize,
        rotary_dim as usize,
        max_position as usize,
        T::DTYPE,
        style,
    )
    .map_err(invalid_contract)?;
    let spec = RopePagedKvWriteSpec::new(
        rotary,
        value_head_size as usize,
        num_blocks as usize,
        block_size as usize,
    )
    .map_err(invalid_contract)?;
    let layout = RopePagedKvLayout::new(
        spec,
        cache_tokens as usize,
        element_count(query_token_stride, "query token stride")?,
        element_count(query_head_stride, "query head stride")?,
        element_count(key_token_stride, "key token stride")?,
        element_count(source_key_head_stride, "source key head stride")?,
        element_count(value_token_stride, "value token stride")?,
        element_count(source_value_head_stride, "source value head stride")?,
        element_count(key_block_stride, "key block stride")?,
        element_count(key_page_stride, "key page stride")?,
        element_count(key_head_stride, "key cache head stride")?,
        element_count(value_block_stride, "value block stride")?,
        element_count(value_page_stride, "value page stride")?,
        element_count(value_cache_head_stride, "value cache head stride")?,
    )?;

    let (mut query, query_range) = unsafe { write_slice(query, query_elements, "RoPE query") }?;
    let (mut key, key_range) = unsafe { write_slice(key, key_elements, "RoPE key") }?;
    let (value, value_range) = unsafe { read_slice(value, value_elements, "RoPE value") }?;
    let (positions, positions_range) =
        unsafe { read_slice(positions, position_elements, "RoPE positions") }?;
    let (cos_sin_cache, cos_sin_cache_range) =
        unsafe { read_slice(cos_sin_cache, cos_sin_cache_elements, "RoPE cos/sin cache") }?;
    let (mut key_cache, key_cache_range) =
        unsafe { write_slice(key_cache, key_cache_elements, "paged key cache") }?;
    let (mut value_cache, value_cache_range) =
        unsafe { write_slice(value_cache, value_cache_elements, "paged value cache") }?;
    let (slot_mapping, slot_mapping_range) =
        unsafe { read_slice(slot_mapping, slot_mapping_elements, "paged slot mapping") }?;

    let dense_query_token = (layout.query_head_stride() == spec.rotary().head_size())
        .then(|| {
            spec.rotary()
                .query_heads()
                .checked_mul(spec.rotary().head_size())
        })
        .flatten();
    let dense_key_token = (layout.source_key_head_stride() == spec.rotary().head_size())
        .then(|| {
            spec.rotary()
                .key_heads()
                .checked_mul(spec.rotary().head_size())
        })
        .flatten();
    let dense_value_token = (layout.source_value_head_stride() == spec.value_head_size())
        .then(|| {
            spec.rotary()
                .key_heads()
                .checked_mul(spec.value_head_size())
        })
        .flatten();
    require_disjoint_or_dense_packed_axis::<T>(
        "query",
        query_range,
        layout.query_token_stride(),
        dense_query_token,
        "key",
        key_range,
        layout.key_token_stride(),
        dense_key_token,
        "RoPE+paged-KV",
    )?;
    require_disjoint_or_dense_packed_axis::<T>(
        "query",
        query_range,
        layout.query_token_stride(),
        dense_query_token,
        "value",
        value_range,
        layout.value_token_stride(),
        dense_value_token,
        "RoPE+paged-KV",
    )?;
    require_disjoint_or_dense_packed_axis::<T>(
        "key",
        key_range,
        layout.key_token_stride(),
        dense_key_token,
        "value",
        value_range,
        layout.value_token_stride(),
        dense_value_token,
        "RoPE+paged-KV",
    )?;

    let key_cache_block_elements = spec
        .block_size()
        .checked_mul(spec.rotary().key_heads())
        .and_then(|value| value.checked_mul(spec.rotary().head_size()))
        .ok_or_else(|| invalid_contract("paged key cache block size overflows usize"))?;
    let value_cache_block_elements = spec
        .block_size()
        .checked_mul(spec.rotary().key_heads())
        .and_then(|value| value.checked_mul(spec.value_head_size()))
        .ok_or_else(|| invalid_contract("paged value cache block size overflows usize"))?;
    let dense_key_cache_block = (layout.key_block_storage_elements(spec)?
        == key_cache_block_elements)
        .then_some(key_cache_block_elements);
    let dense_value_cache_block = (layout.value_block_storage_elements(spec)?
        == value_cache_block_elements)
        .then_some(value_cache_block_elements);
    require_disjoint_or_dense_packed_axis::<T>(
        "key cache",
        key_cache_range,
        layout.key_block_stride(),
        dense_key_cache_block,
        "value cache",
        value_cache_range,
        layout.value_block_stride(),
        dense_value_cache_block,
        "RoPE+paged-KV",
    )?;

    let metadata_and_caches = [
        ("positions", positions_range),
        ("cos/sin cache", cos_sin_cache_range),
        ("key cache", key_cache_range),
        ("value cache", value_cache_range),
        ("slot mapping", slot_mapping_range),
    ];
    require_disjoint_from("query", query_range, &metadata_and_caches, "RoPE+paged-KV")?;
    require_disjoint_from("key", key_range, &metadata_and_caches, "RoPE+paged-KV")?;
    require_disjoint_from(
        "key cache",
        key_cache_range,
        &[
            ("value", value_range),
            ("positions", positions_range),
            ("cos/sin cache", cos_sin_cache_range),
            ("slot mapping", slot_mapping_range),
        ],
        "RoPE+paged-KV",
    )?;
    require_disjoint_from(
        "value cache",
        value_cache_range,
        &[
            ("value", value_range),
            ("positions", positions_range),
            ("cos/sin cache", cos_sin_cache_range),
            ("slot mapping", slot_mapping_range),
        ],
        "RoPE+paged-KV",
    )?;

    T::rope_paged_kv_write(
        &stream_backend(stream),
        &mut query,
        &mut key,
        &value,
        &positions,
        &cos_sin_cache,
        &mut key_cache,
        &mut value_cache,
        &slot_mapping,
        spec,
        layout,
    )?;
    record_launch(OP_ROPE_PAGED_KV_WRITE);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_spec(
    dtype: DType,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
) -> Result<PagedDecodeAttentionSpec, CudaExecutorError> {
    PagedDecodeAttentionSpec::new(
        sequences as usize,
        query_heads as usize,
        kv_heads as usize,
        head_size as usize,
        value_head_size as usize,
        num_blocks as usize,
        block_size as usize,
        max_blocks_per_sequence as usize,
        max_sequence_length as usize,
        scale,
        dtype,
    )
    .map_err(invalid_contract)
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_workspace_elements(
    dtype: DType,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
) -> Result<u64, CudaExecutorError> {
    let spec = paged_decode_spec(
        dtype,
        sequences,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        num_blocks,
        block_size,
        max_blocks_per_sequence,
        max_sequence_length,
        scale,
    )?;
    let elements = paged_decode_attention_split_k_workspace_elements(spec)?.unwrap_or(0);
    u64::try_from(elements).map_err(|_| {
        CudaExecutorError::InvalidContract(
            "paged decode split-K workspace exceeds the bridge ABI".into(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_paged_decode_attention<T: Scalar>(
    query: *const T,
    query_elements: u64,
    key_cache: *const T,
    key_cache_elements: u64,
    value_cache: *const T,
    value_cache_elements: u64,
    block_tables: *const i32,
    block_table_elements: u64,
    sequence_lengths: *const i32,
    sequence_length_elements: u64,
    output: *mut T,
    output_elements: u64,
    workspace: *mut f32,
    workspace_elements: u64,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    key_block_stride: u64,
    value_block_stride: u64,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    stream: *mut c_void,
) -> Result<(), CudaExecutorError> {
    let spec = paged_decode_spec(
        T::DTYPE,
        sequences,
        query_heads,
        kv_heads,
        head_size,
        value_head_size,
        num_blocks,
        block_size,
        max_blocks_per_sequence,
        max_sequence_length,
        scale,
    )?;
    let layout = PagedDecodeLayout::new(
        spec,
        element_count(key_block_stride, "paged key block stride")?,
        element_count(value_block_stride, "paged value block stride")?,
    )?;
    let expected_workspace = paged_decode_attention_split_k_workspace_elements(spec)?.unwrap_or(0);
    if element_count(workspace_elements, "paged decode workspace")? != expected_workspace {
        return Err(CudaExecutorError::InvalidContract(format!(
            "paged decode workspace has {workspace_elements} F32 elements, expected {expected_workspace}"
        )));
    }
    if expected_workspace == 0 && !workspace.is_null() {
        return Err(CudaExecutorError::InvalidContract(
            "base paged decode requires a null workspace pointer".into(),
        ));
    }
    if expected_workspace != 0 && workspace.is_null() {
        return Err(CudaExecutorError::InvalidContract(
            "split-K paged decode requires a workspace pointer".into(),
        ));
    }

    let (query, query_range) = unsafe { read_slice(query, query_elements, "paged decode query") }?;
    let (key_cache, key_cache_range) =
        unsafe { read_slice(key_cache, key_cache_elements, "paged decode key cache") }?;
    let (value_cache, value_cache_range) = unsafe {
        read_slice(
            value_cache,
            value_cache_elements,
            "paged decode value cache",
        )
    }?;
    let (block_tables, block_tables_range) = unsafe {
        read_slice(
            block_tables,
            block_table_elements,
            "paged decode block tables",
        )
    }?;
    let (sequence_lengths, sequence_lengths_range) = unsafe {
        read_slice(
            sequence_lengths,
            sequence_length_elements,
            "paged decode sequence lengths",
        )
    }?;
    let (mut output, output_range) =
        unsafe { write_slice(output, output_elements, "paged decode output") }?;
    let inputs = [
        ("query", query_range),
        ("key cache", key_cache_range),
        ("value cache", value_cache_range),
        ("block tables", block_tables_range),
        ("sequence lengths", sequence_lengths_range),
    ];
    require_disjoint_from("output", output_range, &inputs, "paged decode")?;

    if expected_workspace == 0 {
        T::paged_decode_attention(
            &stream_backend(stream),
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &mut output,
            spec,
            layout,
        )?;
    } else {
        let (mut workspace, workspace_range) = unsafe {
            write_slice(
                workspace,
                workspace_elements,
                "paged decode split-K workspace",
            )
        }?;
        require_disjoint_from(
            "workspace",
            workspace_range,
            &[
                ("query", query_range),
                ("key cache", key_cache_range),
                ("value cache", value_cache_range),
                ("block tables", block_tables_range),
                ("sequence lengths", sequence_lengths_range),
                ("output", output_range),
            ],
            "paged decode",
        )?;
        T::paged_decode_attention_split_k(
            &stream_backend(stream),
            &query,
            &key_cache,
            &value_cache,
            &block_tables,
            &sequence_lengths,
            &mut output,
            &mut workspace,
            spec,
            layout,
        )?;
    }
    record_launch(OP_PAGED_DECODE_ATTENTION);
    Ok(())
}

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

/// Checked standalone RMSNorm.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rms_norm(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rms_norm(
                input.cast(),
                input_elements,
                weight.cast(),
                weight_elements,
                output.cast(),
                output_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}

/// Checked in-place residual Add+RMSNorm.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_add_rms_norm(
    dtype: u32,
    input: *mut c_void,
    input_elements: u64,
    residual: *mut c_void,
    residual_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_add_rms_norm(
                input.cast(),
                input_elements,
                residual.cast(),
                residual_elements,
                weight.cast(),
                weight_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}

/// Checked RMSNorm followed by dynamic per-token FP8.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rms_norm_dynamic_fp8(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    weight: *const c_void,
    weight_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    rows: u32,
    hidden_size: u32,
    epsilon: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rms_norm_dynamic_fp8(
                input.cast(),
                input_elements,
                weight.cast(),
                weight_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                rows,
                hidden_size,
                epsilon,
                stream,
            )
        )
    })
}

/// Checked split-half SiLU-and-Mul.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_silu_and_mul(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    rows: u32,
    width: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_silu_and_mul(
                input.cast(),
                input_elements,
                output.cast(),
                output_elements,
                rows,
                width,
                stream,
            )
        )
    })
}

/// Checked SiLU-and-Mul followed by dynamic per-block FP8.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_silu_and_mul_dynamic_fp8(
    dtype: u32,
    input: *const c_void,
    input_elements: u64,
    output: *mut u8,
    output_elements: u64,
    scales: *mut f32,
    scale_elements: u64,
    scale_upper_bound: *const f32,
    scale_upper_bound_elements: u64,
    rows: u32,
    width: u32,
    group_size: u32,
    scales_transposed: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| match scalar_kind(dtype)? {
        ScalarKind::F16 => unsafe {
            launch_silu_and_mul_dynamic_fp8::<f16>(
                input.cast(),
                input_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                scale_upper_bound,
                scale_upper_bound_elements,
                rows,
                width,
                group_size,
                scales_transposed,
                stream,
            )
        },
        ScalarKind::Bf16 => unsafe {
            launch_silu_and_mul_dynamic_fp8::<bf16>(
                input.cast(),
                input_elements,
                output,
                output_elements,
                scales,
                scale_elements,
                scale_upper_bound,
                scale_upper_bound_elements,
                rows,
                width,
                group_size,
                scales_transposed,
                stream,
            )
        },
        ScalarKind::F32 => Err(CudaExecutorError::InvalidContract(
            "SiLU-and-Mul+FP8 supports FP16 and BF16 input".into(),
        )),
    })
}

/// Checked greedy selection, sampled-token logprob, and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_greedy_sample_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *mut i32,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_greedy_sample_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked selected-token logprob and rank.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_selected_token_logprobs(
    dtype: u32,
    logits: *const c_void,
    logits_elements: u64,
    token_ids: *const i64,
    token_id_elements: u64,
    logprobs: *mut f32,
    logprob_elements: u64,
    ranks: *mut i64,
    rank_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_selected_token_logprobs(
                logits.cast(),
                logits_elements,
                token_ids,
                token_id_elements,
                logprobs,
                logprob_elements,
                ranks,
                rank_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked in-place min-p filtering.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_min_p_filter(
    dtype: u32,
    logits: *mut c_void,
    logits_elements: u64,
    min_p: *const f32,
    min_p_elements: u64,
    rows: u32,
    vocab_size: u32,
    row_stride: u64,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_min_p_filter(
                logits.cast(),
                logits_elements,
                min_p,
                min_p_elements,
                rows,
                vocab_size,
                row_stride,
                stream,
            )
        )
    })
}

/// Checked fused RoPE plus paged-KV write over explicit framework layouts.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes. Logical mutable
/// tensor elements must not alias, including when packed views have
/// overlapping bounding spans.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_rope_paged_kv_write(
    dtype: u32,
    query: *mut c_void,
    query_elements: u64,
    key: *mut c_void,
    key_elements: u64,
    value: *const c_void,
    value_elements: u64,
    positions: *const i64,
    position_elements: u64,
    cos_sin_cache: *const c_void,
    cos_sin_cache_elements: u64,
    key_cache: *mut c_void,
    key_cache_elements: u64,
    value_cache: *mut c_void,
    value_cache_elements: u64,
    slot_mapping: *const i64,
    slot_mapping_elements: u64,
    tokens: u32,
    cache_tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    rotary_dim: u32,
    max_position: u32,
    num_blocks: u32,
    block_size: u32,
    query_token_stride: u64,
    query_head_stride: u64,
    key_token_stride: u64,
    source_key_head_stride: u64,
    value_token_stride: u64,
    source_value_head_stride: u64,
    key_block_stride: u64,
    key_page_stride: u64,
    key_head_stride: u64,
    value_block_stride: u64,
    value_page_stride: u64,
    value_cache_head_stride: u64,
    is_neox: u32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_rope_paged_kv_write(
                query.cast(),
                query_elements,
                key.cast(),
                key_elements,
                value.cast(),
                value_elements,
                positions,
                position_elements,
                cos_sin_cache.cast(),
                cos_sin_cache_elements,
                key_cache.cast(),
                key_cache_elements,
                value_cache.cast(),
                value_cache_elements,
                slot_mapping,
                slot_mapping_elements,
                tokens,
                cache_tokens,
                query_heads,
                kv_heads,
                head_size,
                value_head_size,
                rotary_dim,
                max_position,
                num_blocks,
                block_size,
                query_token_stride,
                query_head_stride,
                key_token_stride,
                source_key_head_stride,
                value_token_stride,
                source_value_head_stride,
                key_block_stride,
                key_page_stride,
                key_head_stride,
                value_block_stride,
                value_page_stride,
                value_cache_head_stride,
                is_neox,
                stream,
            )
        )
    })
}

/// Return the exact caller-owned F32 workspace required by paged decode.
///
/// # Safety
///
/// `workspace_elements` must be a valid writable host pointer.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_paged_decode_workspace_elements(
    dtype: u32,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    workspace_elements: *mut u64,
) -> c_int {
    bridge_call(|| {
        if workspace_elements.is_null()
            || !(workspace_elements as usize).is_multiple_of(align_of::<u64>())
        {
            return Err(CudaExecutorError::InvalidContract(
                "workspace-size output pointer is null or misaligned".into(),
            ));
        }
        let dtype = match scalar_kind(dtype)? {
            ScalarKind::F32 => DType::F32,
            ScalarKind::F16 => DType::F16,
            ScalarKind::Bf16 => DType::Bf16,
        };
        let elements = paged_decode_workspace_elements(
            dtype,
            sequences,
            query_heads,
            kv_heads,
            head_size,
            value_head_size,
            num_blocks,
            block_size,
            max_blocks_per_sequence,
            max_sequence_length,
            scale,
        )?;
        unsafe {
            *workspace_elements = elements;
        }
        Ok(())
    })
}

/// Checked paged MQA/GQA decode. A null, zero-length workspace selects the
/// base path; the exact non-zero workspace selects split-K.
///
/// # Safety
///
/// Every pointer must identify the declared CUDA storage on the active
/// context and remain alive until work on `stream` completes.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn loom_cuda_bridge_paged_decode_attention(
    dtype: u32,
    query: *const c_void,
    query_elements: u64,
    key_cache: *const c_void,
    key_cache_elements: u64,
    value_cache: *const c_void,
    value_cache_elements: u64,
    block_tables: *const i32,
    block_table_elements: u64,
    sequence_lengths: *const i32,
    sequence_length_elements: u64,
    output: *mut c_void,
    output_elements: u64,
    workspace: *mut f32,
    workspace_elements: u64,
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    key_block_stride: u64,
    value_block_stride: u64,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
    scale: f32,
    stream: *mut c_void,
) -> c_int {
    bridge_call(|| {
        let kind = scalar_kind(dtype)?;
        dispatch_scalar!(
            kind,
            launch_paged_decode_attention(
                query.cast(),
                query_elements,
                key_cache.cast(),
                key_cache_elements,
                value_cache.cast(),
                value_cache_elements,
                block_tables,
                block_table_elements,
                sequence_lengths,
                sequence_length_elements,
                output.cast(),
                output_elements,
                workspace,
                workspace_elements,
                sequences,
                query_heads,
                kv_heads,
                head_size,
                value_head_size,
                num_blocks,
                block_size,
                key_block_stride,
                value_block_stride,
                max_blocks_per_sequence,
                max_sequence_length,
                scale,
                stream,
            )
        )
    })
}

#[cfg(test)]
mod tests {
    use super::{
        checked_byte_range, launch_greedy_sample_logprobs, ranges_overlap,
        require_disjoint_or_dense_packed_axis,
    };
    use loom_cuda::CudaExecutorError;

    #[test]
    fn byte_ranges_reject_overflow_and_detect_overlap() {
        let first = checked_byte_range(0x1000_usize as *const f32, 8, "first").unwrap();
        let overlapping = checked_byte_range(0x1010_usize as *const f32, 8, "overlap").unwrap();
        let disjoint = checked_byte_range(0x1020_usize as *const f32, 8, "disjoint").unwrap();
        assert!(ranges_overlap(first, overlapping));
        assert!(!ranges_overlap(first, disjoint));
        assert!(checked_byte_range::<f32>(std::ptr::null(), 8, "null").is_err());
        assert!(checked_byte_range(0x1000_usize as *const f32, 0, "empty").is_err());
    }

    #[test]
    fn greedy_rejects_bad_storage_before_submission() {
        let result = unsafe {
            launch_greedy_sample_logprobs::<f32>(
                0x1000_usize as *const f32,
                7,
                0x2000_usize as *mut i32,
                2,
                0x3000_usize as *mut f32,
                2,
                0x4000_usize as *mut i64,
                2,
                2,
                4,
                4,
                std::ptr::null_mut(),
            )
        };
        assert!(matches!(result, Err(CudaExecutorError::InvalidContract(_))));
    }

    #[test]
    fn packed_token_aliasing_accepts_only_disjoint_logical_views() {
        let query = checked_byte_range(0x1000_usize as *const f32, 16, "packed query").unwrap();
        let key = checked_byte_range(0x1010_usize as *const f32, 16, "packed key").unwrap();
        let overlapping =
            checked_byte_range(0x1008_usize as *const f32, 16, "overlapping key").unwrap();

        require_disjoint_or_dense_packed_axis::<f32>(
            "query",
            query,
            12,
            Some(4),
            "key",
            key,
            12,
            Some(4),
            "RoPE+paged-KV",
        )
        .unwrap();
        assert!(require_disjoint_or_dense_packed_axis::<f32>(
            "query",
            query,
            12,
            Some(4),
            "key",
            overlapping,
            12,
            Some(4),
            "RoPE+paged-KV",
        )
        .is_err());
    }
}
