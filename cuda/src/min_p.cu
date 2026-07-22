#include "loom_cuda.h"

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

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

template <typename Ops, int Threads>
__global__ void min_p_filter_kernel(typename Ops::Scalar* logits,
                                    const float* min_p,
                                    uint32_t vocab_size,
                                    uint64_t row_stride) {
  const size_t row_offset = static_cast<size_t>(blockIdx.x) * row_stride;
  float local_maximum = -CUDART_INF_F;
  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    local_maximum =
        fmaxf(local_maximum, Ops::to_float(logits[row_offset + column]));
  }

  using BlockReduce = cub::BlockReduce<float, Threads>;
  __shared__ typename BlockReduce::TempStorage reduction_storage;
  __shared__ float threshold;
  const float row_maximum =
      BlockReduce(reduction_storage).Reduce(local_maximum, Maximum{});
  if (threadIdx.x == 0) {
    const float probability = min_p[blockIdx.x];
    threshold = probability == 0.0F
                    ? -CUDART_INF_F
                    : row_maximum + logf(probability);
  }
  __syncthreads();

  for (uint32_t column = threadIdx.x; column < vocab_size;
       column += blockDim.x) {
    const size_t index = row_offset + column;
    if (Ops::to_float(logits[index]) < threshold) {
      logits[index] = Ops::from_float(-CUDART_INF_F);
    }
  }
}

template <typename Ops>
int launch_min_p_filter(typename Ops::Scalar* logits, const float* min_p,
                        uint32_t rows, uint32_t vocab_size,
                        uint64_t row_stride, void* stream) {
  if (logits == nullptr || min_p == nullptr || rows == 0 ||
      vocab_size == 0 || row_stride < vocab_size ||
      rows > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      row_stride > std::numeric_limits<size_t>::max() ||
      static_cast<size_t>(rows - 1U) >
          (std::numeric_limits<size_t>::max() - vocab_size) /
              static_cast<size_t>(row_stride)) {
    return LOOM_CUDA_INVALID_ARGUMENT;
  }

  cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
  if (vocab_size >= 65536U) {
    min_p_filter_kernel<Ops, 1024>
        <<<rows, 1024, 0, cuda_stream>>>(logits, min_p, vocab_size,
                                        row_stride);
  } else {
    min_p_filter_kernel<Ops, 256>
        <<<rows, 256, 0, cuda_stream>>>(logits, min_p, vocab_size,
                                       row_stride);
  }
  return cudaGetLastError() == cudaSuccess ? LOOM_CUDA_SUCCESS
                                           : LOOM_CUDA_LAUNCH_ERROR;
}

}  // namespace

extern "C" int loom_cuda_min_p_filter_f32(
    float* logits, const float* min_p, uint32_t rows, uint32_t vocab_size,
    uint64_t row_stride, void* stream) {
  return launch_min_p_filter<FloatOps>(logits, min_p, rows, vocab_size,
                                      row_stride, stream);
}

extern "C" int loom_cuda_min_p_filter_f16(
    uint16_t* logits, const float* min_p, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, void* stream) {
  return launch_min_p_filter<HalfOps>(reinterpret_cast<__half*>(logits), min_p,
                                     rows, vocab_size, row_stride, stream);
}

extern "C" int loom_cuda_min_p_filter_bf16(
    uint16_t* logits, const float* min_p, uint32_t rows,
    uint32_t vocab_size, uint64_t row_stride, void* stream) {
  return launch_min_p_filter<Bfloat16Ops>(
      reinterpret_cast<__nv_bfloat16*>(logits), min_p, rows, vocab_size,
      row_stride, stream);
}
