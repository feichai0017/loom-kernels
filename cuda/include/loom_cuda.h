#ifndef LOOM_CUDA_H_
#define LOOM_CUDA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum LoomCudaStatus {
  LOOM_CUDA_SUCCESS = 0,
  LOOM_CUDA_INVALID_ARGUMENT = 1,
  LOOM_CUDA_UNSUPPORTED = 2,
  LOOM_CUDA_LAUNCH_ERROR = 3,
  LOOM_CUDA_UNAVAILABLE = 4,
};

const char* loom_cuda_status_string(int status);

// F32 bring-up implementation of RMSNorm over contiguous [rows, hidden_size]
// input/output tensors and one contiguous [hidden_size] weight vector. The
// launch is asynchronous with respect to the supplied CUDA stream.
int loom_cuda_rms_norm_f32(const float* input, const float* weight,
                           float* output, uint32_t rows,
                           uint32_t hidden_size, float epsilon, void* stream);

// Pair-vectorized FP16 and BF16 implementations. Odd hidden sizes use a
// scalar fallback so row starts never violate four-byte pair alignment.
int loom_cuda_rms_norm_f16(const uint16_t* input, const uint16_t* weight,
                           uint16_t* output, uint32_t rows,
                           uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_rms_norm_bf16(const uint16_t* input, const uint16_t* weight,
                            uint16_t* output, uint32_t rows,
                            uint32_t hidden_size, float epsilon, void* stream);

// RMSNorm followed by dynamic per-token FP8 E4M3FN quantization. Output holds
// raw FP8 storage bytes and scales has one F32 value per row, with
// approximately normalized_value = fp8(output) * scale. Low-precision inputs
// follow the input scalar arithmetic boundaries for both normalization and
// weight multiplication, matching vLLM's fused quantization contract.
int loom_cuda_rms_norm_dynamic_fp8_f32(
    const float* input, const float* weight, uint8_t* output, float* scales,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_rms_norm_dynamic_fp8_f16(
    const uint16_t* input, const uint16_t* weight, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream);

int loom_cuda_rms_norm_dynamic_fp8_bf16(
    const uint16_t* input, const uint16_t* weight, uint8_t* output,
    float* scales, uint32_t rows, uint32_t hidden_size, float epsilon,
    void* stream);

// Fused residual addition and RMSNorm over contiguous tensors. Both input and
// residual are updated in place:
//   residual = input + residual
//   input = RMSNorm(residual, weight, epsilon)
// input, residual, and weight must point to non-overlapping allocations.
int loom_cuda_add_rms_norm_f32(float* input, float* residual,
                               const float* weight, uint32_t rows,
                               uint32_t hidden_size, float epsilon,
                               void* stream);

// FP16 and BF16 use 128-bit/eight-element vectors when pointer/row alignment
// permits, then two-element vectors, then a scalar fallback. The materialized
// sum is rounded to the storage dtype before computing the RMS statistic.
int loom_cuda_add_rms_norm_f16(uint16_t* input, uint16_t* residual,
                               const uint16_t* weight, uint32_t rows,
                               uint32_t hidden_size, float epsilon,
                               void* stream);

int loom_cuda_add_rms_norm_bf16(uint16_t* input, uint16_t* residual,
                                const uint16_t* weight, uint32_t rows,
                                uint32_t hidden_size, float epsilon,
                                void* stream);

// Fused split-half SwiGLU activation over input [rows, 2 * width] and output
// [rows, width]: output = silu(input[:, :width]) * input[:, width:]. Low-
// precision activation values are rounded to their storage dtype before the
// multiply, matching vLLM. Input/output storage ranges must not overlap.
int loom_cuda_silu_and_mul_f32(const float* input, float* output,
                               uint32_t rows, uint32_t width, void* stream);

int loom_cuda_silu_and_mul_f16(const uint16_t* input, uint16_t* output,
                               uint32_t rows, uint32_t width, void* stream);

int loom_cuda_silu_and_mul_bf16(const uint16_t* input, uint16_t* output,
                                uint32_t rows, uint32_t width, void* stream);

// Fused SwiGLU and dynamic per-block FP8 E4M3FN quantization. FP16/BF16
// inputs use [rows, 2 * width]; output holds [rows, width] raw FP8 bytes.
// Scales have logical shape [rows, width / group_size] and may use row-major
// or group-major storage. group_size must be 64 or 128. scale_ub may be null;
// activation and multiplication remain in F32 until direct FP8 conversion.
int loom_cuda_silu_and_mul_dynamic_fp8_f16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream);

int loom_cuda_silu_and_mul_dynamic_fp8_bf16(
    const uint16_t* input, uint8_t* output, float* scales, uint32_t rows,
    uint32_t width, uint32_t group_size, const float* scale_ub,
    uint32_t scales_transposed, void* stream);

// Fused greedy argmax and sampled-token logprob over logical
// [rows, vocab_size] logits with a unit vocabulary stride and explicit row
// stride. Token IDs use first-index tie breaking, logprobs are F32, and ranks
// are int64 counts of tokens tied at the maximum, matching vLLM's greater-than
// or-equal rank semantics. Logits must be finite.
int loom_cuda_greedy_sample_logprobs_f32(
    const float* logits, int32_t* token_ids, float* logprobs, int64_t* ranks,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_greedy_sample_logprobs_f16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_greedy_sample_logprobs_bf16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

// Compute only the normalized logprob and tie-aware rank of one caller-selected
// token per row. This preserves engine-owned sampling policies while avoiding
// a full [rows, vocab_size] F32 log-softmax output. token_ids are int64 engine
// metadata and must be in [0, vocab_size); logits must be finite.
int loom_cuda_selected_token_logprobs_f32(
    const float* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_selected_token_logprobs_f16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_selected_token_logprobs_bf16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream);

int loom_cuda_min_p_filter_f32(float* logits, const float* min_p,
                               uint32_t rows, uint32_t vocab_size,
                               uint64_t row_stride, void* stream);
int loom_cuda_min_p_filter_f16(uint16_t* logits, const float* min_p,
                               uint32_t rows, uint32_t vocab_size,
                               uint64_t row_stride, void* stream);
int loom_cuda_min_p_filter_bf16(uint16_t* logits, const float* min_p,
                                uint32_t rows, uint32_t vocab_size,
                                uint64_t row_stride, void* stream);

// Base paged MQA/GQA decode attention for one query token per sequence.
// Query/output are contiguous [sequences, query_heads, dim]; native K/V
// caches have dense inner NHD [block_size, kv_heads, dim] dimensions and an
// explicit element stride between blocks. This accepts both separate caches
// and K/V views of vLLM's interleaved [blocks, 2, block_size, kv_heads, dim]
// storage. Block tables and sequence lengths are contiguous int32 engine
// metadata. Sequence lengths include the current token and are trusted to be
// in [1, max_sequence_length]; active block IDs are trusted to be in
// [0, num_blocks). This kernel family is intentionally limited to
// max_sequence_length <= 1024 and does not implement ALiBi, sliding windows,
// soft caps, quantized KV, or multi-token queries.
int loom_cuda_paged_decode_attention_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

int loom_cuda_paged_decode_attention_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

int loom_cuda_paged_decode_attention_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint64_t key_block_stride, uint64_t value_block_stride,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream);

// Optional long-context split-K path. The sizing function returns zero when
// the shape should use the base ABI above. Otherwise the caller owns an F32
// workspace with at least the returned element count for the complete pair of
// partial and stable log-sum-exp merge kernels. The original entry points stay
// allocation-free and ABI-compatible.
uint64_t loom_cuda_paged_decode_attention_split_k_workspace_elements(
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size,
    uint32_t max_sequence_length);

int loom_cuda_paged_decode_attention_split_k_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, float* workspace, uint64_t workspace_elements,
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t num_blocks,
    uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

int loom_cuda_paged_decode_attention_split_k_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, float* workspace,
    uint64_t workspace_elements, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

int loom_cuda_paged_decode_attention_split_k_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, float* workspace,
    uint64_t workspace_elements, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

// Fused in-place RoPE and paged K/V cache write. Query, key, and value have
// logical [tokens, heads, dim] dimensions, a unit dim stride, and explicit
// token/head element strides so packed-QKV views do not need materialization.
// The cosine/sine cache is contiguous [max_position, rotary_dim], with cosine
// then sine halves. Key/value cache tensors have logical
// [blocks, block_size, kv_heads, dim] dimensions; their element strides make
// both vLLM NHD and HND physical layouts expressible. cache_tokens may be less
// than tokens when the engine pads Q/K/V but not slot_mapping. Negative slots
// skip the cache write while RoPE still updates Q/K. Positions and non-negative
// slots are trusted engine metadata and must be in range.
int loom_cuda_rope_paged_kv_write_f32(
    float* query, float* key, const float* value, const int64_t* positions,
    const float* cos_sin_cache, float* key_cache, float* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

int loom_cuda_rope_paged_kv_write_f16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    uint16_t* key_cache, uint16_t* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

int loom_cuda_rope_paged_kv_write_bf16(
    uint16_t* query, uint16_t* key, const uint16_t* value,
    const int64_t* positions, const uint16_t* cos_sin_cache,
    uint16_t* key_cache, uint16_t* value_cache,
    const int64_t* slot_mapping, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t key_head_stride, uint64_t value_token_stride,
    uint64_t value_head_stride, uint64_t key_cache_block_stride,
    uint64_t key_cache_page_stride, uint64_t key_cache_head_stride,
    uint64_t value_cache_block_stride, uint64_t value_cache_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CUDA_H_
