#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "loom_cuda.h"

namespace {

void require_cuda_tensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::tuple<at::Tensor, at::Tensor> fused_tail_attention_merge(
    const at::Tensor& query, const at::Tensor& tail_key,
    const at::Tensor& tail_value, const at::Tensor& remote_output,
    const at::Tensor& remote_lse, double scale) {
  require_cuda_tensor(query, "query");
  require_cuda_tensor(tail_key, "tail_key");
  require_cuda_tensor(tail_value, "tail_value");
  require_cuda_tensor(remote_output, "remote_output");
  require_cuda_tensor(remote_lse, "remote_lse");
  TORCH_CHECK(query.dim() == 3,
              "query must have shape [rows, query_heads, head_dim]");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0,
              "rows and query heads must be positive");
  TORCH_CHECK(tail_key.dim() == 3,
              "tail_key must have shape [tokens, kv_heads, head_dim]");
  TORCH_CHECK(tail_value.sizes() == tail_key.sizes(),
              "tail_key and tail_value shapes must match");
  TORCH_CHECK(remote_output.sizes() == query.sizes(),
              "remote_output shape must match query");
  TORCH_CHECK(remote_lse.dim() == 2 &&
                  remote_lse.size(0) == query.size(0) &&
                  remote_lse.size(1) == query.size(1),
              "remote_lse must have shape [rows, query_heads]");
  TORCH_CHECK(tail_key.size(2) == query.size(2),
              "tail and query head dimensions must match");
  TORCH_CHECK(tail_key.size(0) > 0 && tail_key.size(0) <= 64,
              "tail token count must be in 1..=64");
  TORCH_CHECK(query.size(2) > 0 && query.size(2) <= 256,
              "head dimension must be in 1..=256");
  TORCH_CHECK(tail_key.size(1) > 0 &&
                  query.size(1) % tail_key.size(1) == 0,
              "KV heads must divide query heads");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half ||
                  query.scalar_type() == at::ScalarType::BFloat16,
              "query dtype must be FP16 or BF16");
  TORCH_CHECK(tail_key.scalar_type() == query.scalar_type() &&
                  tail_value.scalar_type() == query.scalar_type() &&
                  remote_output.scalar_type() == query.scalar_type(),
              "query, tail K/V, and remote output dtypes must match");
  TORCH_CHECK(remote_lse.scalar_type() == at::ScalarType::Float,
              "remote_lse must be FP32");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "attention scale must be finite and positive");
  for (const auto dimension : {query.size(0), query.size(1), tail_key.size(1),
                               query.size(2), tail_key.size(0)}) {
    TORCH_CHECK(
        dimension <= std::numeric_limits<uint32_t>::max(),
        "fused-attention dimensions must fit in an unsigned 32-bit integer");
  }
  const auto device = query.device();
  for (const auto& tensor :
       {tail_key, tail_value, remote_output, remote_lse}) {
    TORCH_CHECK(tensor.device() == device,
                "all fused-attention tensors must share one CUDA device");
  }

  c10::cuda::CUDAGuard guard(device);
  auto merged_output = at::empty_like(query);
  auto merged_lse = at::empty_like(remote_lse);
  const auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  const auto dtype = query.scalar_type() == at::ScalarType::Half
                         ? LOOM_CUDA_FP16
                         : LOOM_CUDA_BF16;
  const int status = loom_cuda_fused_tail_attention_merge(
      query.data_ptr(), tail_key.data_ptr(), tail_value.data_ptr(),
      remote_output.data_ptr(), static_cast<const float*>(remote_lse.data_ptr()),
      merged_output.data_ptr(), static_cast<float*>(merged_lse.data_ptr()),
      static_cast<uint32_t>(query.size(0)),
      static_cast<uint32_t>(query.size(1)),
      static_cast<uint32_t>(tail_key.size(1)),
      static_cast<uint32_t>(query.size(2)),
      static_cast<uint32_t>(tail_key.size(0)), static_cast<float>(scale), dtype,
      reinterpret_cast<void*>(stream));
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS, "Loom CUDA kernel failed: ",
              loom_cuda_status_string(status));
  return {std::move(merged_output), std::move(merged_lse)};
}

}  // namespace

TORCH_LIBRARY(loom, library) {
  library.def(
      "fused_tail_attention_merge(Tensor query, Tensor tail_key, Tensor "
      "tail_value, Tensor remote_output, Tensor remote_lse, float scale) -> "
      "(Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(loom, CUDA, library) {
  library.impl("fused_tail_attention_merge",
               TORCH_FN(fused_tail_attention_merge));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
