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

enum LoomCudaDType {
  LOOM_CUDA_FP16 = 0,
  LOOM_CUDA_BF16 = 1,
};

const char* loom_cuda_status_string(int status);

// Tensor layout for every entry point is contiguous row-major:
//   query/output: [rows, query_heads, head_dim]
//   tail K/V:     [tail_tokens, kv_heads, head_dim], shared across rows
//   LSE:          [rows, query_heads] in FP32

// Compute a mergeable output-plus-LSE state over the local active tail.
int loom_cuda_tail_attention_state(
    const void* query, const void* tail_key, const void* tail_value,
    void* tail_output, float* tail_lse, uint32_t rows, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t tail_tokens, float scale,
    enum LoomCudaDType dtype, void* stream);

// Merge two disjoint-KV attention states exactly.
int loom_cuda_merge_two_states(
    const void* left_output, const float* left_lse,
    const void* right_output, const float* right_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads,
    uint32_t head_dim, enum LoomCudaDType dtype, void* stream);

// Compute local-tail attention and merge it with a remote state without
// materializing the local output or local LSE tensors.
int loom_cuda_fused_tail_attention_merge(
    const void* query, const void* tail_key, const void* tail_value,
    const void* remote_output, const float* remote_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t tail_tokens, float scale,
    enum LoomCudaDType dtype, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_CUDA_H_
