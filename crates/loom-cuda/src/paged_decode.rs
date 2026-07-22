use crate::rms_norm::CudaBackend;
use crate::runtime::{loom_status_result, DeviceBuffer};
use crate::CudaExecutorError;
use half::{bf16, f16};
use loom_kernels::{DType, PagedDecodeAttentionSpec};

pub(crate) const PAGED_DECODE_MAX_CONTEXT: usize = 1024;

impl CudaBackend {
    /// Executes base F32 paged MQA/GQA decode attention asynchronously.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_attention_f32(
        &self,
        query: &DeviceBuffer<f32>,
        key_cache: &DeviceBuffer<f32>,
        value_cache: &DeviceBuffer<f32>,
        block_tables: &DeviceBuffer<i32>,
        sequence_lengths: &DeviceBuffer<i32>,
        output: &mut DeviceBuffer<f32>,
        spec: PagedDecodeAttentionSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F32)?;
        let shape = validate_buffers(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            output,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_paged_decode_attention_f32(
                query.as_ptr(),
                key_cache.as_ptr(),
                value_cache.as_ptr(),
                block_tables.as_ptr(),
                sequence_lengths.as_ptr(),
                output.as_mut_ptr(),
                shape.sequences,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.num_blocks,
                shape.block_size,
                shape.max_blocks_per_sequence,
                shape.max_sequence_length,
                spec.scale(),
                self.stream().raw(),
            )
        })
    }

    /// Executes base FP16 paged MQA/GQA decode attention asynchronously.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_attention_f16(
        &self,
        query: &DeviceBuffer<f16>,
        key_cache: &DeviceBuffer<f16>,
        value_cache: &DeviceBuffer<f16>,
        block_tables: &DeviceBuffer<i32>,
        sequence_lengths: &DeviceBuffer<i32>,
        output: &mut DeviceBuffer<f16>,
        spec: PagedDecodeAttentionSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::F16)?;
        let shape = validate_buffers(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            output,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_paged_decode_attention_f16(
                query.as_ptr().cast::<u16>(),
                key_cache.as_ptr().cast::<u16>(),
                value_cache.as_ptr().cast::<u16>(),
                block_tables.as_ptr(),
                sequence_lengths.as_ptr(),
                output.as_mut_ptr().cast::<u16>(),
                shape.sequences,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.num_blocks,
                shape.block_size,
                shape.max_blocks_per_sequence,
                shape.max_sequence_length,
                spec.scale(),
                self.stream().raw(),
            )
        })
    }

    /// Executes base BF16 paged MQA/GQA decode attention asynchronously.
    #[allow(clippy::too_many_arguments)]
    pub fn paged_decode_attention_bf16(
        &self,
        query: &DeviceBuffer<bf16>,
        key_cache: &DeviceBuffer<bf16>,
        value_cache: &DeviceBuffer<bf16>,
        block_tables: &DeviceBuffer<i32>,
        sequence_lengths: &DeviceBuffer<i32>,
        output: &mut DeviceBuffer<bf16>,
        spec: PagedDecodeAttentionSpec,
    ) -> Result<(), CudaExecutorError> {
        require_dtype(spec, DType::Bf16)?;
        let shape = validate_buffers(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            output,
            spec,
        )?;
        loom_status_result(unsafe {
            loom_cuda_sys::loom_cuda_paged_decode_attention_bf16(
                query.as_ptr().cast::<u16>(),
                key_cache.as_ptr().cast::<u16>(),
                value_cache.as_ptr().cast::<u16>(),
                block_tables.as_ptr(),
                sequence_lengths.as_ptr(),
                output.as_mut_ptr().cast::<u16>(),
                shape.sequences,
                shape.query_heads,
                shape.kv_heads,
                shape.head_size,
                shape.value_head_size,
                shape.num_blocks,
                shape.block_size,
                shape.max_blocks_per_sequence,
                shape.max_sequence_length,
                spec.scale(),
                self.stream().raw(),
            )
        })
    }
}

pub(crate) fn supports_spec(spec: PagedDecodeAttentionSpec) -> bool {
    matches!(spec.dtype(), DType::F32 | DType::F16 | DType::Bf16)
        && spec.max_sequence_length() <= PAGED_DECODE_MAX_CONTEXT
        && u32::try_from(spec.sequences()).is_ok()
        && u32::try_from(spec.query_heads()).is_ok()
        && u32::try_from(spec.kv_heads()).is_ok()
        && u32::try_from(spec.head_size()).is_ok()
        && u32::try_from(spec.value_head_size()).is_ok()
        && u32::try_from(spec.num_blocks()).is_ok()
        && u32::try_from(spec.block_size()).is_ok()
        && u32::try_from(spec.max_blocks_per_sequence()).is_ok()
        && spec
            .sequences()
            .checked_mul(spec.query_heads())
            .is_some_and(|blocks| blocks <= i32::MAX as usize)
}

fn require_dtype(spec: PagedDecodeAttentionSpec, expected: DType) -> Result<(), CudaExecutorError> {
    if spec.dtype() == expected {
        Ok(())
    } else {
        Err(CudaExecutorError::InvalidContract(format!(
            "paged decode attention for {expected:?} cannot execute {:?}",
            spec.dtype()
        )))
    }
}

#[derive(Clone, Copy)]
struct AbiShape {
    sequences: u32,
    query_heads: u32,
    kv_heads: u32,
    head_size: u32,
    value_head_size: u32,
    num_blocks: u32,
    block_size: u32,
    max_blocks_per_sequence: u32,
    max_sequence_length: u32,
}

#[allow(clippy::too_many_arguments)]
fn validate_buffers<T: Copy>(
    query: &DeviceBuffer<T>,
    key_cache: &DeviceBuffer<T>,
    value_cache: &DeviceBuffer<T>,
    block_tables: &DeviceBuffer<i32>,
    sequence_lengths: &DeviceBuffer<i32>,
    output: &DeviceBuffer<T>,
    spec: PagedDecodeAttentionSpec,
) -> Result<AbiShape, CudaExecutorError> {
    query.require_len(spec.query_numel(), "paged decode query")?;
    key_cache.require_len(spec.key_cache_numel(), "paged decode key cache")?;
    value_cache.require_len(spec.value_cache_numel(), "paged decode value cache")?;
    block_tables.require_len(spec.block_table_numel(), "paged decode block tables")?;
    sequence_lengths.require_len(spec.sequences(), "paged decode sequence lengths")?;
    output.require_len(spec.output_numel(), "paged decode output")?;
    if spec.max_sequence_length() > PAGED_DECODE_MAX_CONTEXT {
        return Err(CudaExecutorError::InvalidContract(format!(
            "paged decode maximum context {} exceeds the first CUDA kernel limit {PAGED_DECODE_MAX_CONTEXT}",
            spec.max_sequence_length()
        )));
    }
    if !supports_spec(spec) {
        return Err(CudaExecutorError::InvalidContract(
            "paged decode shape exceeds the CUDA ABI".into(),
        ));
    }
    let u32_value = |value: usize, name: &str| {
        u32::try_from(value)
            .map_err(|_| CudaExecutorError::InvalidContract(format!("{name} exceeds the CUDA ABI")))
    };
    Ok(AbiShape {
        sequences: u32_value(spec.sequences(), "sequence count")?,
        query_heads: u32_value(spec.query_heads(), "query head count")?,
        kv_heads: u32_value(spec.kv_heads(), "KV head count")?,
        head_size: u32_value(spec.head_size(), "head size")?,
        value_head_size: u32_value(spec.value_head_size(), "value head size")?,
        num_blocks: u32_value(spec.num_blocks(), "cache block count")?,
        block_size: u32_value(spec.block_size(), "cache block size")?,
        max_blocks_per_sequence: u32_value(
            spec.max_blocks_per_sequence(),
            "maximum blocks per sequence",
        )?,
        max_sequence_length: u32_value(spec.max_sequence_length(), "maximum context")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use loom_kernels::paged_decode_attention_f32_reference;

    #[test]
    fn safe_rust_wrapper_matches_the_cpu_oracle() {
        let spec = PagedDecodeAttentionSpec::new(2, 4, 2, 4, 3, 4, 2, 2, 0.5, DType::F32).unwrap();
        let query: Vec<f32> = (0..spec.query_numel())
            .map(|index| (index as f32 - 9.0) * 0.07)
            .collect();
        let key_cache: Vec<f32> = (0..spec.key_cache_numel())
            .map(|index| (index as f32 % 13.0 - 6.0) * 0.11)
            .collect();
        let value_cache: Vec<f32> = (0..spec.value_cache_numel())
            .map(|index| (index as f32 % 17.0 - 8.0) * 0.09)
            .collect();
        let block_tables = [2_i32, 0, 1, 3];
        let sequence_lengths = [3_i32, 2];
        let mut expected = vec![0.0_f32; spec.output_numel()];
        paged_decode_attention_f32_reference(
            &query,
            &key_cache,
            &value_cache,
            &block_tables.map(i64::from),
            &sequence_lengths.map(i64::from),
            &mut expected,
            spec,
        )
        .unwrap();

        let backend = CudaBackend::new().unwrap();
        let query = DeviceBuffer::from_slice(&query).unwrap();
        let key_cache = DeviceBuffer::from_slice(&key_cache).unwrap();
        let value_cache = DeviceBuffer::from_slice(&value_cache).unwrap();
        let block_tables = DeviceBuffer::from_slice(&block_tables).unwrap();
        let sequence_lengths = DeviceBuffer::from_slice(&sequence_lengths).unwrap();
        let mut output = DeviceBuffer::<f32>::uninitialized(spec.output_numel()).unwrap();
        backend
            .paged_decode_attention_f32(
                &query,
                &key_cache,
                &value_cache,
                &block_tables,
                &sequence_lengths,
                &mut output,
                spec,
            )
            .unwrap();
        backend.stream().synchronize().unwrap();
        let actual = output.copy_to_vec().unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "{actual} != {expected}"
            );
        }
    }
}
