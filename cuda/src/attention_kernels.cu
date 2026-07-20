#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 128;
constexpr uint32_t kMaxHeadDim = 256;
constexpr uint32_t kMaxTailTokens = 64;

template <typename T>
__device__ __forceinline__ float load_scalar(const T* values, size_t index);

template <>
__device__ __forceinline__ float load_scalar<__half>(const __half* values,
                                                     size_t index) {
  return __half2float(values[index]);
}

template <>
__device__ __forceinline__ float load_scalar<__nv_bfloat16>(
    const __nv_bfloat16* values, size_t index) {
  return __bfloat162float(values[index]);
}

template <typename T>
__device__ __forceinline__ void store_scalar(T* values, size_t index,
                                             float value);

template <>
__device__ __forceinline__ void store_scalar<__half>(__half* values,
                                                     size_t index,
                                                     float value) {
  values[index] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void store_scalar<__nv_bfloat16>(
    __nv_bfloat16* values, size_t index, float value) {
  values[index] = __float2bfloat16_rn(value);
}

template <typename T>
__global__ __launch_bounds__(kThreads) void tail_attention_state_kernel(
    const T* query, const T* tail_key, const T* tail_value, T* tail_output,
    float* tail_lse, uint32_t rows, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t tail_tokens, float scale) {
  const uint32_t row_head = blockIdx.x;
  const uint32_t row = row_head / query_heads;
  const uint32_t query_head = row_head % query_heads;
  if (row >= rows) {
    return;
  }
  const uint32_t query_group_size = query_heads / kv_heads;
  const uint32_t kv_head = query_head / query_group_size;
  const size_t query_base =
      (static_cast<size_t>(row) * query_heads + query_head) * head_dim;

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float logits[kMaxTailTokens];
  __shared__ float shard_lse;

  for (uint32_t token = 0; token < tail_tokens; ++token) {
    const size_t key_base =
        (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
    float partial = 0.0F;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      partial += load_scalar(query, query_base + dimension) *
                 load_scalar(tail_key, key_base + dimension);
    }
    const float dot = BlockReduce(reduce_storage).Sum(partial);
    if (threadIdx.x == 0) {
      logits[token] = dot * scale;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      maximum = fmaxf(maximum, logits[token]);
    }
    float sum = 0.0F;
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      sum += expf(logits[token] - maximum);
    }
    shard_lse = maximum + logf(sum);
    tail_lse[row_head] = shard_lse;
  }
  __syncthreads();

  for (uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float weighted_value = 0.0F;
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      const size_t value_index =
          (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim +
          dimension;
      weighted_value += expf(logits[token] - shard_lse) *
                        load_scalar(tail_value, value_index);
    }
    store_scalar(tail_output, query_base + dimension, weighted_value);
  }
}

template <typename T>
__global__ __launch_bounds__(kThreads) void merge_two_states_kernel(
    const T* left_output, const float* left_lse, const T* right_output,
    const float* right_lse, T* merged_output, float* merged_lse,
    uint32_t rows, uint32_t query_heads, uint32_t head_dim) {
  const uint32_t row_head = blockIdx.x;
  const uint32_t row = row_head / query_heads;
  const uint32_t query_head = row_head % query_heads;
  if (row >= rows) {
    return;
  }
  __shared__ float left_weight;
  __shared__ float right_weight;
  if (threadIdx.x == 0) {
    const float maximum = fmaxf(left_lse[row_head], right_lse[row_head]);
    const float left_exponential = expf(left_lse[row_head] - maximum);
    const float right_exponential = expf(right_lse[row_head] - maximum);
    const float lse = maximum + logf(left_exponential + right_exponential);
    merged_lse[row_head] = lse;
    left_weight = expf(left_lse[row_head] - lse);
    right_weight = expf(right_lse[row_head] - lse);
  }
  __syncthreads();

  const size_t output_base =
      (static_cast<size_t>(row) * query_heads + query_head) * head_dim;
  for (uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    const size_t index = output_base + dimension;
    const float value = left_weight * load_scalar(left_output, index) +
                        right_weight * load_scalar(right_output, index);
    store_scalar(merged_output, index, value);
  }
}

template <typename T>
__global__ __launch_bounds__(kThreads) void fused_tail_attention_merge_kernel(
    const T* query, const T* tail_key, const T* tail_value,
    const T* remote_output, const float* remote_lse, T* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t tail_tokens, float scale) {
  const uint32_t row_head = blockIdx.x;
  const uint32_t row = row_head / query_heads;
  const uint32_t query_head = row_head % query_heads;
  if (row >= rows) {
    return;
  }
  const uint32_t query_group_size = query_heads / kv_heads;
  const uint32_t kv_head = query_head / query_group_size;
  const size_t query_base =
      (static_cast<size_t>(row) * query_heads + query_head) * head_dim;

  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float logits[kMaxTailTokens];
  __shared__ float combined_lse;
  __shared__ float remote_weight;

  for (uint32_t token = 0; token < tail_tokens; ++token) {
    const size_t key_base =
        (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
    float partial = 0.0F;
    for (uint32_t dimension = threadIdx.x; dimension < head_dim;
         dimension += blockDim.x) {
      partial += load_scalar(query, query_base + dimension) *
                 load_scalar(tail_key, key_base + dimension);
    }
    const float dot = BlockReduce(reduce_storage).Sum(partial);
    if (threadIdx.x == 0) {
      logits[token] = dot * scale;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float maximum = -INFINITY;
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      maximum = fmaxf(maximum, logits[token]);
    }
    float sum = 0.0F;
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      sum += expf(logits[token] - maximum);
    }
    const float local_lse = maximum + logf(sum);
    const float merge_maximum = fmaxf(remote_lse[row_head], local_lse);
    combined_lse = merge_maximum +
                   logf(expf(remote_lse[row_head] - merge_maximum) +
                        expf(local_lse - merge_maximum));
    merged_lse[row_head] = combined_lse;
    remote_weight = expf(remote_lse[row_head] - combined_lse);
  }
  __syncthreads();

  for (uint32_t dimension = threadIdx.x; dimension < head_dim;
       dimension += blockDim.x) {
    float value = remote_weight *
                  load_scalar(remote_output, query_base + dimension);
    for (uint32_t token = 0; token < tail_tokens; ++token) {
      const size_t value_index =
          (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim +
          dimension;
      value += expf(logits[token] - combined_lse) *
               load_scalar(tail_value, value_index);
    }
    store_scalar(merged_output, query_base + dimension, value);
  }
}

bool valid_common_shape(uint32_t rows, uint32_t query_heads,
                        uint32_t head_dim) {
  const uint64_t blocks = static_cast<uint64_t>(rows) * query_heads;
  return rows > 0 && query_heads > 0 && head_dim > 0 &&
         head_dim <= kMaxHeadDim && blocks <= 0x7fffffffULL;
}

bool valid_tail_shape(uint32_t rows, uint32_t query_heads, uint32_t kv_heads,
                      uint32_t head_dim, uint32_t tail_tokens, float scale) {
  return valid_common_shape(rows, query_heads, head_dim) && kv_heads > 0 &&
         query_heads % kv_heads == 0 && tail_tokens > 0 &&
         tail_tokens <= kMaxTailTokens && std::isfinite(scale) && scale > 0.0F;
}

int launch_status() {
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename T>
int launch_tail_attention_state(
    const void* query, const void* tail_key, const void* tail_value,
    void* tail_output, float* tail_lse, uint32_t rows, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t tail_tokens, float scale,
    cudaStream_t stream) {
  const uint32_t blocks = rows * query_heads;
  tail_attention_state_kernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(query), static_cast<const T*>(tail_key),
      static_cast<const T*>(tail_value), static_cast<T*>(tail_output), tail_lse,
      rows, query_heads, kv_heads, head_dim, tail_tokens, scale);
  return launch_status();
}

template <typename T>
int launch_merge_two_states(
    const void* left_output, const float* left_lse,
    const void* right_output, const float* right_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads,
    uint32_t head_dim, cudaStream_t stream) {
  const uint32_t blocks = rows * query_heads;
  merge_two_states_kernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(left_output), left_lse,
      static_cast<const T*>(right_output), right_lse,
      static_cast<T*>(merged_output), merged_lse, rows, query_heads, head_dim);
  return launch_status();
}

template <typename T>
int launch_fused_tail_attention_merge(
    const void* query, const void* tail_key, const void* tail_value,
    const void* remote_output, const float* remote_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t tail_tokens, float scale,
    cudaStream_t stream) {
  const uint32_t blocks = rows * query_heads;
  fused_tail_attention_merge_kernel<T><<<blocks, kThreads, 0, stream>>>(
      static_cast<const T*>(query), static_cast<const T*>(tail_key),
      static_cast<const T*>(tail_value),
      static_cast<const T*>(remote_output), remote_lse,
      static_cast<T*>(merged_output), merged_lse, rows, query_heads, kv_heads,
      head_dim, tail_tokens, scale);
  return launch_status();
}

}  // namespace

extern "C" const char* loom_cuda_status_string(int status) {
  switch (status) {
    case LOOM_CUDA_SUCCESS:
      return "success";
    case LOOM_CUDA_INVALID_ARGUMENT:
      return "invalid argument";
    case LOOM_CUDA_UNSUPPORTED:
      return "unsupported operation";
    case LOOM_CUDA_LAUNCH_ERROR:
      return "CUDA kernel launch failed";
    case LOOM_CUDA_UNAVAILABLE:
      return "CUDA backend unavailable";
    default:
      return "unknown Loom CUDA status";
  }
}

extern "C" int loom_cuda_tail_attention_state(
    const void* query, const void* tail_key, const void* tail_value,
    void* tail_output, float* tail_lse, uint32_t rows, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t tail_tokens, float scale,
    enum LoomCudaDType dtype, void* stream) {
  if (query == nullptr || tail_key == nullptr || tail_value == nullptr ||
      tail_output == nullptr || tail_lse == nullptr ||
      !valid_tail_shape(rows, query_heads, kv_heads, head_dim, tail_tokens,
                        scale)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case LOOM_CUDA_FP16:
      return launch_tail_attention_state<__half>(
          query, tail_key, tail_value, tail_output, tail_lse, rows,
          query_heads, kv_heads, head_dim, tail_tokens, scale, cuda_stream);
    case LOOM_CUDA_BF16:
      return launch_tail_attention_state<__nv_bfloat16>(
          query, tail_key, tail_value, tail_output, tail_lse, rows,
          query_heads, kv_heads, head_dim, tail_tokens, scale, cuda_stream);
    default:
      return LOOM_CUDA_UNSUPPORTED;
  }
}

extern "C" int loom_cuda_merge_two_states(
    const void* left_output, const float* left_lse,
    const void* right_output, const float* right_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads,
    uint32_t head_dim, enum LoomCudaDType dtype, void* stream) {
  if (left_output == nullptr || left_lse == nullptr ||
      right_output == nullptr || right_lse == nullptr ||
      merged_output == nullptr || merged_lse == nullptr ||
      !valid_common_shape(rows, query_heads, head_dim)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case LOOM_CUDA_FP16:
      return launch_merge_two_states<__half>(
          left_output, left_lse, right_output, right_lse, merged_output,
          merged_lse, rows, query_heads, head_dim, cuda_stream);
    case LOOM_CUDA_BF16:
      return launch_merge_two_states<__nv_bfloat16>(
          left_output, left_lse, right_output, right_lse, merged_output,
          merged_lse, rows, query_heads, head_dim, cuda_stream);
    default:
      return LOOM_CUDA_UNSUPPORTED;
  }
}

extern "C" int loom_cuda_fused_tail_attention_merge(
    const void* query, const void* tail_key, const void* tail_value,
    const void* remote_output, const float* remote_lse, void* merged_output,
    float* merged_lse, uint32_t rows, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_dim, uint32_t tail_tokens, float scale,
    enum LoomCudaDType dtype, void* stream) {
  if (query == nullptr || tail_key == nullptr || tail_value == nullptr ||
      remote_output == nullptr || remote_lse == nullptr ||
      merged_output == nullptr || merged_lse == nullptr ||
      !valid_tail_shape(rows, query_heads, kv_heads, head_dim, tail_tokens,
                        scale)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (dtype) {
    case LOOM_CUDA_FP16:
      return launch_fused_tail_attention_merge<__half>(
          query, tail_key, tail_value, remote_output, remote_lse,
          merged_output, merged_lse, rows, query_heads, kv_heads, head_dim,
          tail_tokens, scale, cuda_stream);
    case LOOM_CUDA_BF16:
      return launch_fused_tail_attention_merge<__nv_bfloat16>(
          query, tail_key, tail_value, remote_output, remote_lse,
          merged_output, merged_lse, rows, query_heads, kv_heads, head_dim,
          tail_tokens, scale, cuda_stream);
    default:
      return LOOM_CUDA_UNSUPPORTED;
  }
}
