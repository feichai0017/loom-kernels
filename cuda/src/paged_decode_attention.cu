#include "loom_cuda.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr uint32_t kThreads = 256;
constexpr uint32_t kWarps = kThreads / 32;
constexpr uint32_t kMaxContext = 1024;

struct FloatOps {
  using Scalar = float;
  __device__ static float to_float(float value) { return value; }
  __device__ static float from_float(float value) { return value; }
};

struct HalfOps {
  using Scalar = __half;
  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static __half from_float(float value) {
    return __float2half_rn(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;
  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16(value);
  }
};

struct Maximum {
  __device__ float operator()(float left, float right) const {
    return fmaxf(left, right);
  }
};

__device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

template <typename Ops>
__global__ void paged_decode_attention_kernel(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t block_size,
    uint32_t max_blocks_per_sequence, float scale) {
  const uint32_t sequence_head = blockIdx.x;
  const uint32_t sequence = sequence_head / query_heads;
  const uint32_t query_head = sequence_head % query_heads;
  const uint32_t queries_per_kv = query_heads / kv_heads;
  const uint32_t kv_head = query_head / queries_per_kv;
  const uint32_t sequence_length =
      static_cast<uint32_t>(sequence_lengths[sequence]);

  extern __shared__ float scores[];
  using BlockReduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  __shared__ float maximum;
  __shared__ float inverse_denominator;

  const uint32_t lane = threadIdx.x & 31U;
  const uint32_t warp = threadIdx.x >> 5U;
  const size_t query_offset =
      (static_cast<size_t>(sequence) * query_heads + query_head) * head_size;
  const size_t table_offset =
      static_cast<size_t>(sequence) * max_blocks_per_sequence;

  for (uint32_t position = warp; position < sequence_length;
       position += kWarps) {
    const uint32_t logical_block = position / block_size;
    const uint32_t block_offset = position % block_size;
    const uint32_t physical_block =
        static_cast<uint32_t>(block_tables[table_offset + logical_block]);
    const size_t key_offset =
        ((static_cast<size_t>(physical_block) * block_size + block_offset) *
             kv_heads +
         kv_head) *
        head_size;
    float partial = 0.0F;
    for (uint32_t dimension = lane; dimension < head_size;
         dimension += 32U) {
      partial += Ops::to_float(query[query_offset + dimension]) *
                 Ops::to_float(key_cache[key_offset + dimension]);
    }
    partial = warp_sum(partial);
    if (lane == 0U) {
      scores[position] = partial * scale;
    }
  }
  __syncthreads();

  float local_maximum = -CUDART_INF_F;
  for (uint32_t position = threadIdx.x; position < sequence_length;
       position += kThreads) {
    local_maximum = fmaxf(local_maximum, scores[position]);
  }
  const float reduced_maximum =
      BlockReduce(reduction_storage).Reduce(local_maximum, Maximum{});
  if (threadIdx.x == 0U) {
    maximum = reduced_maximum;
  }
  __syncthreads();

  float local_denominator = 0.0F;
  for (uint32_t position = threadIdx.x; position < sequence_length;
       position += kThreads) {
    const float weight = expf(scores[position] - maximum);
    scores[position] = weight;
    local_denominator += weight;
  }
  __syncthreads();
  const float denominator =
      BlockReduce(reduction_storage).Sum(local_denominator);
  if (threadIdx.x == 0U) {
    inverse_denominator = 1.0F / denominator;
  }
  __syncthreads();

  const size_t output_offset =
      (static_cast<size_t>(sequence) * query_heads + query_head) *
      value_head_size;
  for (uint32_t dimension = threadIdx.x; dimension < value_head_size;
       dimension += kThreads) {
    float accumulator = 0.0F;
    for (uint32_t position = 0; position < sequence_length; ++position) {
      const uint32_t logical_block = position / block_size;
      const uint32_t block_offset = position % block_size;
      const uint32_t physical_block =
          static_cast<uint32_t>(block_tables[table_offset + logical_block]);
      const size_t value_offset =
          ((static_cast<size_t>(physical_block) * block_size + block_offset) *
               kv_heads +
           kv_head) *
          value_head_size;
      accumulator += scores[position] *
                     Ops::to_float(value_cache[value_offset + dimension]);
    }
    output[output_offset + dimension] =
        Ops::from_float(accumulator * inverse_denominator);
  }
}

bool checked_product(const uint32_t* dimensions, size_t count) {
  size_t product = 1;
  for (size_t index = 0; index < count; ++index) {
    if (dimensions[index] == 0U ||
        product > std::numeric_limits<size_t>::max() / dimensions[index]) {
      return false;
    }
    product *= dimensions[index];
  }
  return true;
}

template <typename Ops>
int launch_paged_decode_attention(
    const typename Ops::Scalar* query, const typename Ops::Scalar* key_cache,
    const typename Ops::Scalar* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, typename Ops::Scalar* output,
    uint32_t sequences, uint32_t query_heads, uint32_t kv_heads,
    uint32_t head_size, uint32_t value_head_size, uint32_t num_blocks,
    uint32_t block_size, uint32_t max_blocks_per_sequence,
    uint32_t max_sequence_length, float scale, void* stream) {
  const uint32_t query_dimensions[] = {sequences, query_heads, head_size};
  const uint32_t output_dimensions[] = {sequences, query_heads,
                                        value_head_size};
  const uint32_t key_dimensions[] = {num_blocks, block_size, kv_heads,
                                     head_size};
  const uint32_t value_dimensions[] = {num_blocks, block_size, kv_heads,
                                       value_head_size};
  const uint32_t table_dimensions[] = {sequences,
                                       max_blocks_per_sequence};
  const uint64_t context_capacity =
      static_cast<uint64_t>(block_size) * max_blocks_per_sequence;
  const uint64_t grid_size =
      static_cast<uint64_t>(sequences) * query_heads;
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      block_tables == nullptr || sequence_lengths == nullptr ||
      output == nullptr || sequences == 0U || query_heads == 0U ||
      kv_heads == 0U || head_size == 0U || value_head_size == 0U ||
      num_blocks == 0U || block_size == 0U ||
      max_blocks_per_sequence == 0U || query_heads % kv_heads != 0U ||
      max_sequence_length == 0U || max_sequence_length > kMaxContext ||
      max_sequence_length > context_capacity || !isfinite(scale) ||
      scale <= 0.0F ||
      grid_size > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
      !checked_product(query_dimensions, 3) ||
      !checked_product(output_dimensions, 3) ||
      !checked_product(key_dimensions, 4) ||
      !checked_product(value_dimensions, 4) ||
      !checked_product(table_dimensions, 2)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  const size_t shared_bytes =
      static_cast<size_t>(max_sequence_length) * sizeof(float);
  paged_decode_attention_kernel<Ops>
      <<<static_cast<uint32_t>(grid_size), kThreads, shared_bytes,
         static_cast<cudaStream_t>(stream)>>>(
          query, key_cache, value_cache, block_tables, sequence_lengths,
          output, query_heads, kv_heads, head_size, value_head_size, block_size,
          max_blocks_per_sequence, scale);
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_paged_decode_attention_f32(
    const float* query, const float* key_cache, const float* value_cache,
    const int32_t* block_tables, const int32_t* sequence_lengths,
    float* output, uint32_t sequences, uint32_t query_heads,
    uint32_t kv_heads, uint32_t head_size, uint32_t value_head_size,
    uint32_t num_blocks, uint32_t block_size,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<FloatOps>(
      query, key_cache, value_cache, block_tables, sequence_lengths, output,
      sequences, query_heads, kv_heads, head_size, value_head_size, num_blocks,
      block_size, max_blocks_per_sequence, max_sequence_length, scale, stream);
}

extern "C" int loom_cuda_paged_decode_attention_f16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<HalfOps>(
      reinterpret_cast<const __half*>(query),
      reinterpret_cast<const __half*>(key_cache),
      reinterpret_cast<const __half*>(value_cache), block_tables,
      sequence_lengths, reinterpret_cast<__half*>(output), sequences,
      query_heads, kv_heads, head_size, value_head_size, num_blocks, block_size,
      max_blocks_per_sequence, max_sequence_length, scale, stream);
}

extern "C" int loom_cuda_paged_decode_attention_bf16(
    const uint16_t* query, const uint16_t* key_cache,
    const uint16_t* value_cache, const int32_t* block_tables,
    const int32_t* sequence_lengths, uint16_t* output, uint32_t sequences,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_size,
    uint32_t value_head_size, uint32_t num_blocks, uint32_t block_size,
    uint32_t max_blocks_per_sequence, uint32_t max_sequence_length,
    float scale, void* stream) {
  return launch_paged_decode_attention<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(query),
      reinterpret_cast<const __nv_bfloat16*>(key_cache),
      reinterpret_cast<const __nv_bfloat16*>(value_cache), block_tables,
      sequence_lengths, reinterpret_cast<__nv_bfloat16*>(output), sequences,
      query_heads, kv_heads, head_size, value_head_size, num_blocks, block_size,
      max_blocks_per_sequence, max_sequence_length, scale, stream);
}
