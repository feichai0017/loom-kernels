//! Explicit physical layouts accepted by the CUDA backend.
//!
//! Semantic operator specs live in `loom-kernels`. These types describe the
//! framework-owned storage that backs those logical tensors.

use crate::CudaExecutorError;
use loom_kernels::{PagedDecodeAttentionSpec, RopePagedKvWriteSpec};

fn invalid(message: impl Into<String>) -> CudaExecutorError {
    CudaExecutorError::InvalidContract(message.into())
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, CudaExecutorError> {
    left.checked_add(right)
        .ok_or_else(|| invalid(format!("{name} storage span overflows usize")))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, CudaExecutorError> {
    left.checked_mul(right)
        .ok_or_else(|| invalid(format!("{name} storage span overflows usize")))
}

fn strided_span(
    dimensions: &[(usize, usize)],
    contiguous_width: usize,
    name: &str,
) -> Result<usize, CudaExecutorError> {
    if contiguous_width == 0 {
        return Err(invalid(format!("{name} contiguous width must be positive")));
    }
    let mut span = contiguous_width;
    for &(size, stride) in dimensions {
        if size == 0 || stride == 0 {
            return Err(invalid(format!(
                "{name} dimensions and strides must be positive"
            )));
        }
        span = checked_add(span, checked_mul(size - 1, stride, name)?, name)?;
    }
    Ok(span)
}

fn validate_non_overlapping_layout(
    dimensions: &[(usize, usize)],
    contiguous_width: usize,
    name: &str,
) -> Result<(), CudaExecutorError> {
    let mut ordered = Vec::with_capacity(dimensions.len() + 1);
    ordered.push((contiguous_width, 1_usize));
    ordered.extend_from_slice(dimensions);
    ordered.sort_unstable_by_key(|&(_, stride)| stride);

    let mut occupied_span = 1_usize;
    for &(size, stride) in &ordered {
        if size == 0 || stride == 0 {
            return Err(invalid(format!(
                "{name} dimensions and strides must be positive"
            )));
        }
        if size == 1 {
            continue;
        }
        if stride < occupied_span {
            return Err(invalid(format!("{name} layout overlaps itself")));
        }
        occupied_span = checked_add(occupied_span, checked_mul(size - 1, stride, name)?, name)?;
    }
    Ok(())
}

/// Row-major logical matrix with an explicit physical row stride.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowStridedLayout {
    row_stride: usize,
}

impl RowStridedLayout {
    pub fn new(width: usize, row_stride: usize) -> Result<Self, CudaExecutorError> {
        if width == 0 || row_stride < width {
            return Err(invalid(
                "row stride must be at least the positive logical row width",
            ));
        }
        Ok(Self { row_stride })
    }

    pub const fn contiguous(width: usize) -> Self {
        Self { row_stride: width }
    }

    pub const fn row_stride(self) -> usize {
        self.row_stride
    }

    pub fn storage_elements(self, rows: usize, width: usize) -> Result<usize, CudaExecutorError> {
        if rows == 0 || width == 0 {
            return Err(invalid("row-strided matrix dimensions must be positive"));
        }
        checked_add(
            checked_mul(rows - 1, self.row_stride, "row-strided matrix")?,
            width,
            "row-strided matrix",
        )
    }
}

/// Physical block strides for dense-inner NHD paged K/V cache views.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PagedDecodeLayout {
    key_block_stride: usize,
    value_block_stride: usize,
}

impl PagedDecodeLayout {
    pub fn new(
        spec: PagedDecodeAttentionSpec,
        key_block_stride: usize,
        value_block_stride: usize,
    ) -> Result<Self, CudaExecutorError> {
        let key_block_elements = spec
            .block_size()
            .checked_mul(spec.kv_heads())
            .and_then(|value| value.checked_mul(spec.head_size()))
            .ok_or_else(|| invalid("paged key block size overflows usize"))?;
        let value_block_elements = spec
            .block_size()
            .checked_mul(spec.kv_heads())
            .and_then(|value| value.checked_mul(spec.value_head_size()))
            .ok_or_else(|| invalid("paged value block size overflows usize"))?;
        if key_block_stride < key_block_elements {
            return Err(invalid(
                "paged key block stride is smaller than one dense NHD block",
            ));
        }
        if value_block_stride < value_block_elements {
            return Err(invalid(
                "paged value block stride is smaller than one dense NHD block",
            ));
        }
        Ok(Self {
            key_block_stride,
            value_block_stride,
        })
    }

    pub fn contiguous(spec: PagedDecodeAttentionSpec) -> Result<Self, CudaExecutorError> {
        let key_block_stride = spec
            .block_size()
            .checked_mul(spec.kv_heads())
            .and_then(|value| value.checked_mul(spec.head_size()))
            .ok_or_else(|| invalid("paged key block size overflows usize"))?;
        let value_block_stride = spec
            .block_size()
            .checked_mul(spec.kv_heads())
            .and_then(|value| value.checked_mul(spec.value_head_size()))
            .ok_or_else(|| invalid("paged value block size overflows usize"))?;
        Self::new(spec, key_block_stride, value_block_stride)
    }

    pub const fn key_block_stride(self) -> usize {
        self.key_block_stride
    }

    pub const fn value_block_stride(self) -> usize {
        self.value_block_stride
    }

    pub fn key_storage_elements(
        self,
        spec: PagedDecodeAttentionSpec,
    ) -> Result<usize, CudaExecutorError> {
        let block_elements = spec.block_size() * spec.kv_heads() * spec.head_size();
        checked_add(
            checked_mul(
                spec.num_blocks() - 1,
                self.key_block_stride,
                "paged key cache",
            )?,
            block_elements,
            "paged key cache",
        )
    }

    pub fn value_storage_elements(
        self,
        spec: PagedDecodeAttentionSpec,
    ) -> Result<usize, CudaExecutorError> {
        let block_elements = spec.block_size() * spec.kv_heads() * spec.value_head_size();
        checked_add(
            checked_mul(
                spec.num_blocks() - 1,
                self.value_block_stride,
                "paged value cache",
            )?,
            block_elements,
            "paged value cache",
        )
    }
}

/// Physical strides for fused RoPE over framework-owned Q/K/V and paged cache
/// views. The innermost head dimension always has unit stride.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RopePagedKvLayout {
    cache_tokens: usize,
    query_token_stride: usize,
    query_head_stride: usize,
    key_token_stride: usize,
    source_key_head_stride: usize,
    value_token_stride: usize,
    source_value_head_stride: usize,
    key_block_stride: usize,
    key_page_stride: usize,
    key_head_stride: usize,
    value_block_stride: usize,
    value_page_stride: usize,
    value_head_stride: usize,
}

impl RopePagedKvLayout {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spec: RopePagedKvWriteSpec,
        cache_tokens: usize,
        query_token_stride: usize,
        query_head_stride: usize,
        key_token_stride: usize,
        source_key_head_stride: usize,
        value_token_stride: usize,
        source_value_head_stride: usize,
        key_block_stride: usize,
        key_page_stride: usize,
        key_head_stride: usize,
        value_block_stride: usize,
        value_page_stride: usize,
        value_head_stride: usize,
    ) -> Result<Self, CudaExecutorError> {
        let rotary = spec.rotary();
        if cache_tokens == 0 || cache_tokens > rotary.tokens() {
            return Err(invalid(
                "RoPE+paged-KV cache token count must be in [1, tokens]",
            ));
        }
        validate_non_overlapping_layout(
            &[
                (rotary.tokens(), query_token_stride),
                (rotary.query_heads(), query_head_stride),
            ],
            rotary.head_size(),
            "RoPE query",
        )?;
        validate_non_overlapping_layout(
            &[
                (rotary.tokens(), key_token_stride),
                (rotary.key_heads(), source_key_head_stride),
            ],
            rotary.head_size(),
            "RoPE key",
        )?;
        validate_non_overlapping_layout(
            &[
                (rotary.tokens(), value_token_stride),
                (rotary.key_heads(), source_value_head_stride),
            ],
            spec.value_head_size(),
            "RoPE value",
        )?;
        validate_non_overlapping_layout(
            &[
                (spec.num_blocks(), key_block_stride),
                (spec.block_size(), key_page_stride),
                (rotary.key_heads(), key_head_stride),
            ],
            rotary.head_size(),
            "paged key cache",
        )?;
        validate_non_overlapping_layout(
            &[
                (spec.num_blocks(), value_block_stride),
                (spec.block_size(), value_page_stride),
                (rotary.key_heads(), value_head_stride),
            ],
            spec.value_head_size(),
            "paged value cache",
        )?;
        Ok(Self {
            cache_tokens,
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
            value_head_stride,
        })
    }

    pub fn contiguous(spec: RopePagedKvWriteSpec) -> Result<Self, CudaExecutorError> {
        let rotary = spec.rotary();
        let query_head_stride = rotary.head_size();
        let query_token_stride = rotary.query_heads() * query_head_stride;
        let source_key_head_stride = rotary.head_size();
        let key_token_stride = rotary.key_heads() * source_key_head_stride;
        let source_value_head_stride = spec.value_head_size();
        let value_token_stride = rotary.key_heads() * source_value_head_stride;
        let key_head_stride = rotary.head_size();
        let key_page_stride = rotary.key_heads() * key_head_stride;
        let key_block_stride = spec.block_size() * key_page_stride;
        let value_head_stride = spec.value_head_size();
        let value_page_stride = rotary.key_heads() * value_head_stride;
        let value_block_stride = spec.block_size() * value_page_stride;
        Self::new(
            spec,
            rotary.tokens(),
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
            value_head_stride,
        )
    }

    pub const fn cache_tokens(self) -> usize {
        self.cache_tokens
    }

    pub const fn query_token_stride(self) -> usize {
        self.query_token_stride
    }

    pub const fn query_head_stride(self) -> usize {
        self.query_head_stride
    }

    pub const fn key_token_stride(self) -> usize {
        self.key_token_stride
    }

    pub const fn source_key_head_stride(self) -> usize {
        self.source_key_head_stride
    }

    pub const fn value_token_stride(self) -> usize {
        self.value_token_stride
    }

    pub const fn source_value_head_stride(self) -> usize {
        self.source_value_head_stride
    }

    pub const fn key_block_stride(self) -> usize {
        self.key_block_stride
    }

    pub const fn key_page_stride(self) -> usize {
        self.key_page_stride
    }

    pub const fn key_head_stride(self) -> usize {
        self.key_head_stride
    }

    pub const fn value_block_stride(self) -> usize {
        self.value_block_stride
    }

    pub const fn value_page_stride(self) -> usize {
        self.value_page_stride
    }

    pub const fn value_head_stride(self) -> usize {
        self.value_head_stride
    }

    pub fn key_block_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        let rotary = spec.rotary();
        strided_span(
            &[
                (spec.block_size(), self.key_page_stride),
                (rotary.key_heads(), self.key_head_stride),
            ],
            rotary.head_size(),
            "paged key cache block",
        )
    }

    pub fn value_block_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        let rotary = spec.rotary();
        strided_span(
            &[
                (spec.block_size(), self.value_page_stride),
                (rotary.key_heads(), self.value_head_stride),
            ],
            spec.value_head_size(),
            "paged value cache block",
        )
    }

    pub fn query_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        let rotary = spec.rotary();
        strided_span(
            &[
                (rotary.tokens(), self.query_token_stride),
                (rotary.query_heads(), self.query_head_stride),
            ],
            rotary.head_size(),
            "RoPE query",
        )
    }

    pub fn key_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        let rotary = spec.rotary();
        strided_span(
            &[
                (rotary.tokens(), self.key_token_stride),
                (rotary.key_heads(), self.source_key_head_stride),
            ],
            rotary.head_size(),
            "RoPE key",
        )
    }

    pub fn value_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        let rotary = spec.rotary();
        strided_span(
            &[
                (rotary.tokens(), self.value_token_stride),
                (rotary.key_heads(), self.source_value_head_stride),
            ],
            spec.value_head_size(),
            "RoPE value",
        )
    }

    pub fn key_cache_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        checked_add(
            checked_mul(
                spec.num_blocks() - 1,
                self.key_block_stride,
                "paged key cache",
            )?,
            self.key_block_storage_elements(spec)?,
            "paged key cache",
        )
    }

    pub fn value_cache_storage_elements(
        self,
        spec: RopePagedKvWriteSpec,
    ) -> Result<usize, CudaExecutorError> {
        checked_add(
            checked_mul(
                spec.num_blocks() - 1,
                self.value_block_stride,
                "paged value cache",
            )?,
            self.value_block_storage_elements(spec)?,
            "paged value cache",
        )
    }
}
