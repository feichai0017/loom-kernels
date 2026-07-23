#ifndef LOOM_CUDA_BRIDGE_H_
#define LOOM_CUDA_BRIDGE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum loom_cuda_bridge_status {
  LOOM_CUDA_BRIDGE_SUCCESS = 0,
  LOOM_CUDA_BRIDGE_INVALID_ARGUMENT = 1,
  LOOM_CUDA_BRIDGE_LAUNCH_ERROR = 2,
  LOOM_CUDA_BRIDGE_UNAVAILABLE = 3,
};

enum loom_cuda_bridge_dtype {
  LOOM_CUDA_BRIDGE_F32 = 0,
  LOOM_CUDA_BRIDGE_F16 = 1,
  LOOM_CUDA_BRIDGE_BF16 = 2,
};

enum loom_cuda_bridge_operator {
  LOOM_CUDA_BRIDGE_RMS_NORM = 0,
  LOOM_CUDA_BRIDGE_ADD_RMS_NORM = 1,
  LOOM_CUDA_BRIDGE_RMS_NORM_DYNAMIC_FP8 = 2,
  LOOM_CUDA_BRIDGE_SILU_AND_MUL = 3,
  LOOM_CUDA_BRIDGE_SILU_AND_MUL_DYNAMIC_FP8 = 4,
  LOOM_CUDA_BRIDGE_ROPE_PAGED_KV_WRITE = 5,
  LOOM_CUDA_BRIDGE_GREEDY_SAMPLE_LOGPROBS = 6,
  LOOM_CUDA_BRIDGE_SELECTED_TOKEN_LOGPROBS = 7,
  LOOM_CUDA_BRIDGE_MIN_P_FILTER = 8,
  LOOM_CUDA_BRIDGE_PAGED_DECODE_ATTENTION = 9,
};

uint32_t loom_cuda_bridge_abi_version(void);
const char* loom_cuda_bridge_last_error_message(void);
int loom_cuda_bridge_launch_count(uint32_t operation, uint64_t* count);
int loom_cuda_bridge_reset_launch_count(uint32_t operation);

int loom_cuda_bridge_rms_norm(
    uint32_t dtype, const void* input, uint64_t input_elements,
    const void* weight, uint64_t weight_elements, void* output,
    uint64_t output_elements, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

int loom_cuda_bridge_add_rms_norm(
    uint32_t dtype, void* input, uint64_t input_elements, void* residual,
    uint64_t residual_elements, const void* weight, uint64_t weight_elements,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_bridge_rms_norm_dynamic_fp8(
    uint32_t dtype, const void* input, uint64_t input_elements,
    const void* weight, uint64_t weight_elements, uint8_t* output,
    uint64_t output_elements, float* scales, uint64_t scale_elements,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_bridge_silu_and_mul(
    uint32_t dtype, const void* input, uint64_t input_elements, void* output,
    uint64_t output_elements, uint32_t rows, uint32_t width, void* stream);

int loom_cuda_bridge_silu_and_mul_dynamic_fp8(
    uint32_t dtype, const void* input, uint64_t input_elements,
    uint8_t* output, uint64_t output_elements, float* scales,
    uint64_t scale_elements, const float* scale_upper_bound,
    uint64_t scale_upper_bound_elements, uint32_t rows, uint32_t width,
    uint32_t group_size, uint32_t scales_transposed, void* stream);

int loom_cuda_bridge_greedy_sample_logprobs(
    uint32_t dtype, const void* logits, uint64_t logits_elements,
    int32_t* token_ids, uint64_t token_id_elements, float* logprobs,
    uint64_t logprob_elements, int64_t* ranks, uint64_t rank_elements,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_bridge_selected_token_logprobs(
    uint32_t dtype, const void* logits, uint64_t logits_elements,
    const int64_t* token_ids, uint64_t token_id_elements, float* logprobs,
    uint64_t logprob_elements, int64_t* ranks, uint64_t rank_elements,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_bridge_min_p_filter(
    uint32_t dtype, void* logits, uint64_t logits_elements,
    const float* min_p, uint64_t min_p_elements, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, void* stream);

int loom_cuda_bridge_rope_paged_kv_write(
    uint32_t dtype, void* query, uint64_t query_elements, void* key,
    uint64_t key_elements, const void* value, uint64_t value_elements,
    const int64_t* positions, uint64_t position_elements,
    const void* cos_sin_cache, uint64_t cos_sin_cache_elements,
    void* key_cache, uint64_t key_cache_elements, void* value_cache,
    uint64_t value_cache_elements, const int64_t* slot_mapping,
    uint64_t slot_mapping_elements, uint32_t tokens, uint32_t cache_tokens,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t rotary_dim, uint32_t max_position,
    uint32_t num_blocks, uint32_t block_size, uint64_t query_token_stride,
    uint64_t query_head_stride, uint64_t key_token_stride,
    uint64_t source_key_head_stride, uint64_t value_token_stride,
    uint64_t source_value_head_stride, uint64_t key_block_stride,
    uint64_t key_page_stride, uint64_t key_head_stride,
    uint64_t value_block_stride, uint64_t value_page_stride,
    uint64_t value_cache_head_stride, uint32_t is_neox, void* stream);

int loom_cuda_bridge_paged_decode_workspace_elements(
    uint32_t dtype, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, uint64_t* workspace_elements);

int loom_cuda_bridge_paged_decode_attention(
    uint32_t dtype, const void* query, uint64_t query_elements,
    const void* key_cache, uint64_t key_cache_elements,
    const void* value_cache, uint64_t value_cache_elements,
    const int32_t* block_tables, uint64_t block_table_elements,
    const int32_t* sequence_lengths, uint64_t sequence_length_elements,
    void* output, uint64_t output_elements, float* workspace,
    uint64_t workspace_elements, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size, uint64_t key_block_stride,
    uint64_t value_block_stride, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CUDA_BRIDGE_H_
