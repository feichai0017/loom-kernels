#include "loom_cuda.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

struct FloatOps {
  using Scalar = float;
  __device__ static float to_float(Scalar value) { return value; }
};

struct HalfOps {
  using Scalar = __half;
  __device__ static float to_float(Scalar value) {
    return __half2float(value);
  }
};

struct Bfloat16Ops {
  using Scalar = __nv_bfloat16;
  __device__ static float to_float(Scalar value) {
    return __bfloat162float(value);
  }
};

struct LogSumExpState {
  float maximum;
  float exponential_sum;
  uint32_t maximum_index;
  uint32_t maximum_count;
};

struct CombineLogSumExp {
  __device__ LogSumExpState operator()(const LogSumExpState& left,
                                       const LogSumExpState& right) const {
    if (left.maximum > right.maximum) {
      return {left.maximum,
              left.exponential_sum +
                  right.exponential_sum *
                      expf(right.maximum - left.maximum),
              left.maximum_index, left.maximum_count};
    }
    if (right.maximum > left.maximum) {
      return {right.maximum,
              right.exponential_sum +
                  left.exponential_sum * expf(left.maximum - right.maximum),
              right.maximum_index, right.maximum_count};
    }
    return {left.maximum, left.exponential_sum + right.exponential_sum,
            min(left.maximum_index, right.maximum_index),
            left.maximum_count + right.maximum_count};
  }
};

struct SelectedLogprobState {
  float maximum;
  float exponential_sum;
  uint32_t rank;
};

struct CombineSelectedLogprob {
  __device__ SelectedLogprobState operator()(
      const SelectedLogprobState& left,
      const SelectedLogprobState& right) const {
    if (left.maximum > right.maximum) {
      return {left.maximum,
              left.exponential_sum +
                  right.exponential_sum *
                      expf(right.maximum - left.maximum),
              left.rank + right.rank};
    }
    if (right.maximum > left.maximum) {
      return {right.maximum,
              right.exponential_sum +
                  left.exponential_sum * expf(left.maximum - right.maximum),
              left.rank + right.rank};
    }
    return {left.maximum, left.exponential_sum + right.exponential_sum,
            left.rank + right.rank};
  }
};

__device__ void update_state(LogSumExpState* state, float value,
                             uint32_t index) {
  if (value > state->maximum) {
    state->exponential_sum =
        state->exponential_sum * expf(state->maximum - value) + 1.0F;
    state->maximum = value;
    state->maximum_index = index;
    state->maximum_count = 1U;
  } else if (value == state->maximum) {
    state->exponential_sum += 1.0F;
    state->maximum_index = min(state->maximum_index, index);
    state->maximum_count += 1U;
  } else {
    state->exponential_sum += expf(value - state->maximum);
  }
}

template <typename Ops, int Threads>
__global__ __launch_bounds__(Threads) void greedy_sample_logprobs_kernel(
    const typename Ops::Scalar* logits, int32_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t vocab_size,
    uint64_t row_stride) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * row_stride;
  LogSumExpState local = {-FLT_MAX, 0.0F, 0xffffffffU, 0U};
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    update_state(&local, Ops::to_float(logits[row_offset + column]), column);
  }

  using BlockReduce = cub::BlockReduce<LogSumExpState, Threads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  const LogSumExpState row =
      BlockReduce(reduction_storage).Reduce(local, CombineLogSumExp{});
  if (threadIdx.x == 0) {
    token_ids[blockIdx.x] = static_cast<int32_t>(row.maximum_index);
    logprobs[blockIdx.x] = -logf(row.exponential_sum);
    // vLLM defines the sampled-token rank as the number of logprobs greater
    // than or equal to the sampled value. For greedy sampling this is the
    // number of tokens tied at the maximum, rather than always one.
    ranks[blockIdx.x] = static_cast<int64_t>(row.maximum_count);
  }
}

template <typename Ops>
int launch_greedy_sample_logprobs(const typename Ops::Scalar* logits,
                                  int32_t* token_ids, float* logprobs,
                                  int64_t* ranks, uint32_t rows,
                                  uint32_t vocab_size, uint64_t row_stride,
                                  void* stream) {
  if (logits == nullptr || token_ids == nullptr || logprobs == nullptr ||
      ranks == nullptr || rows == 0 || vocab_size == 0 ||
      row_stride < vocab_size ||
      vocab_size > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (vocab_size >= 65536U) {
    greedy_sample_logprobs_kernel<Ops, 1024>
        <<<rows, 1024, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                        vocab_size, row_stride);
  } else {
    greedy_sample_logprobs_kernel<Ops, 256>
        <<<rows, 256, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                       vocab_size, row_stride);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

template <typename Ops, int Threads>
__global__ __launch_bounds__(Threads) void selected_token_logprobs_kernel(
    const typename Ops::Scalar* logits, const int64_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t vocab_size,
    uint64_t row_stride) {
  const int64_t selected_index = token_ids[blockIdx.x];
  if (selected_index < 0 ||
      selected_index >= static_cast<int64_t>(vocab_size)) {
    if (threadIdx.x == 0) {
      logprobs[blockIdx.x] = __int_as_float(0x7fffffff);
      ranks[blockIdx.x] = 0;
    }
    return;
  }

  const size_t row_offset = static_cast<size_t>(blockIdx.x) * row_stride;
  const float selected = Ops::to_float(
      logits[row_offset + static_cast<size_t>(selected_index)]);
  SelectedLogprobState local = {-FLT_MAX, 0.0F, 0U};
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    const float value = Ops::to_float(logits[row_offset + column]);
    local.rank += static_cast<uint32_t>(value >= selected);
    if (value > local.maximum) {
      local.exponential_sum =
          local.exponential_sum * expf(local.maximum - value) + 1.0F;
      local.maximum = value;
    } else {
      local.exponential_sum += expf(value - local.maximum);
    }
  }

  using BlockReduce = cub::BlockReduce<SelectedLogprobState, Threads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  const SelectedLogprobState row = BlockReduce(reduction_storage)
                                       .Reduce(local, CombineSelectedLogprob{});
  if (threadIdx.x == 0) {
    logprobs[blockIdx.x] =
        selected - row.maximum - logf(row.exponential_sum);
    ranks[blockIdx.x] = static_cast<int64_t>(row.rank);
  }
}

template <typename Ops>
int launch_selected_token_logprobs(
    const typename Ops::Scalar* logits, const int64_t* token_ids,
    float* logprobs, int64_t* ranks, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, void* stream) {
  if (logits == nullptr || token_ids == nullptr || logprobs == nullptr ||
      ranks == nullptr || rows == 0 || vocab_size == 0 ||
      row_stride < vocab_size ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (vocab_size >= 65536U) {
    selected_token_logprobs_kernel<Ops, 1024>
        <<<rows, 1024, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                        vocab_size, row_stride);
  } else {
    selected_token_logprobs_kernel<Ops, 256>
        <<<rows, 256, 0, cuda_stream>>>(logits, token_ids, logprobs, ranks,
                                       vocab_size, row_stride);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_greedy_sample_logprobs_f32(
    const float* logits, int32_t* token_ids, float* logprobs, int64_t* ranks,
    uint32_t rows, uint32_t vocab_size, uint64_t row_stride, void* stream) {
  return launch_greedy_sample_logprobs<FloatOps>(
      logits, token_ids, logprobs, ranks, rows, vocab_size, row_stride,
      stream);
}

extern "C" int loom_cuda_greedy_sample_logprobs_f16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_greedy_sample_logprobs<HalfOps>(
      reinterpret_cast<const __half*>(logits), token_ids, logprobs, ranks,
      rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_greedy_sample_logprobs_bf16(
    const uint16_t* logits, int32_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_greedy_sample_logprobs<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(logits), token_ids, logprobs,
      ranks, rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_selected_token_logprobs_f32(
    const float* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<FloatOps>(
      logits, token_ids, logprobs, ranks, rows, vocab_size, row_stride,
      stream);
}

extern "C" int loom_cuda_selected_token_logprobs_f16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<HalfOps>(
      reinterpret_cast<const __half*>(logits), token_ids, logprobs, ranks,
      rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_selected_token_logprobs_bf16(
    const uint16_t* logits, const int64_t* token_ids, float* logprobs,
    int64_t* ranks, uint32_t rows, uint32_t vocab_size, uint64_t row_stride,
    void* stream) {
  return launch_selected_token_logprobs<Bfloat16Ops>(
      reinterpret_cast<const __nv_bfloat16*>(logits), token_ids, logprobs,
      ranks, rows, vocab_size, row_stride, stream);
}
