#include "loom_cuda_bridge.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace {

uint32_t bridge_dtype(const at::Tensor& tensor) {
  if (tensor.scalar_type() == at::kFloat) {
    return LOOM_CUDA_BRIDGE_F32;
  }
  if (tensor.scalar_type() == at::kHalf) {
    return LOOM_CUDA_BRIDGE_F16;
  }
  if (tensor.scalar_type() == at::kBFloat16) {
    return LOOM_CUDA_BRIDGE_BF16;
  }
  TORCH_CHECK(false, "unsupported Loom bridge dtype");
}

uint64_t storage_span_elements(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.numel() > 0,
              "Loom bridge tensors must contain at least one element");
  uint64_t span = 1;
  for (int64_t dimension = 0; dimension < tensor.dim(); ++dimension) {
    TORCH_CHECK(tensor.size(dimension) > 0 && tensor.stride(dimension) > 0,
                "Loom bridge requires positive tensor sizes and strides");
    const auto extent =
        static_cast<uint64_t>(tensor.size(dimension) - 1);
    const auto stride = static_cast<uint64_t>(tensor.stride(dimension));
    TORCH_CHECK(
        extent == 0 ||
            stride <=
                (std::numeric_limits<uint64_t>::max() - span) / extent,
        "Loom tensor storage span exceeds the bridge ABI");
    span += extent * stride;
  }
  return span;
}

void check_bridge_status(int status, const char* operation) {
  TORCH_CHECK(status == LOOM_CUDA_BRIDGE_SUCCESS, "Loom Rust ", operation,
              " bridge failed: ", loom_cuda_bridge_last_error_message(),
              " (status ", status, ")");
}

bool byte_ranges_overlap(const at::Tensor& left, const at::Tensor& right) {
  const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  const auto left_end = left_begin + left.nbytes();
  const auto right_end = right_begin + right.nbytes();
  return left_begin < right_end && right_begin < left_end;
}

bool has_dense_nhd_inner_strides(const at::Tensor& tensor) {
  if (tensor.dim() != 4) {
    return false;
  }
  const int64_t block_elements =
      tensor.size(1) * tensor.size(2) * tensor.size(3);
  return tensor.stride(3) == 1 &&
         tensor.stride(2) == tensor.size(3) &&
         tensor.stride(1) == tensor.size(2) * tensor.size(3) &&
         tensor.stride(0) >= block_elements;
}

void check_rms_norm_contract(const at::Tensor& input,
                             const at::Tensor& weight,
                             const at::Tensor& output, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Loom RMSNorm input must be CUDA");
  TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device(),
              "Loom RMSNorm tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
                  output.scalar_type() == input.scalar_type(),
              "Loom RMSNorm tensors must have matching dtypes");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom RMSNorm supports F32, FP16, and BF16");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "Loom RMSNorm tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm input must be non-empty");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm weight must match the hidden dimension");
  TORCH_CHECK(output.sizes() == input.sizes(),
              "Loom RMSNorm output shape must match input");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm epsilon must be finite and positive");
  TORCH_CHECK(!input.requires_grad() && !weight.requires_grad(),
              "Loom RMSNorm is an inference-only operator");
  TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight),
              "Loom RMSNorm output storage must not overlap inputs");
}

void rms_norm(const at::Tensor& input, const at::Tensor& weight,
              at::Tensor output, double epsilon) {
  check_rms_norm_contract(input, weight, output, epsilon);
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm shape exceeds the bridge ABI");
  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const int status = loom_cuda_bridge_rms_norm(
      bridge_dtype(input), input.data_ptr(),
      static_cast<uint64_t>(input.numel()), weight.data_ptr(),
      static_cast<uint64_t>(weight.numel()), output.data_ptr(),
      static_cast<uint64_t>(output.numel()),
      static_cast<uint32_t>(rows_i64),
      static_cast<uint32_t>(hidden_size_i64), static_cast<float>(epsilon),
      stream.stream());
  check_bridge_status(status, "RMSNorm");
}

void check_contract(const at::Tensor& input, const at::Tensor& residual,
                    const at::Tensor& weight, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Loom Add+RMSNorm input must be CUDA");
  TORCH_CHECK(residual.device() == input.device() &&
                  weight.device() == input.device(),
              "Loom Add+RMSNorm tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == residual.scalar_type() &&
                  input.scalar_type() == weight.scalar_type(),
              "Loom Add+RMSNorm tensors must have matching dtypes");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom Add+RMSNorm supports F32, FP16, and BF16");
  TORCH_CHECK(input.is_contiguous() && residual.is_contiguous() &&
                  weight.is_contiguous(),
              "Loom Add+RMSNorm tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom Add+RMSNorm input must be non-empty");
  TORCH_CHECK(input.sizes() == residual.sizes(),
              "Loom Add+RMSNorm input/residual shapes must match");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom Add+RMSNorm weight must match the hidden dimension");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom Add+RMSNorm epsilon must be finite and positive");
  TORCH_CHECK(!byte_ranges_overlap(input, residual) &&
                  !byte_ranges_overlap(input, weight) &&
                  !byte_ranges_overlap(residual, weight),
              "Loom Add+RMSNorm tensor storage ranges must not overlap");
}

void launch_add_rms_norm(at::Tensor input, at::Tensor residual,
                         const at::Tensor& weight, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom Add+RMSNorm shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto residual_elements = static_cast<uint64_t>(residual.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  const int status = loom_cuda_bridge_add_rms_norm(
      bridge_dtype(input), input.data_ptr(), input_elements,
      residual.data_ptr(), residual_elements, weight.data_ptr(),
      weight_elements, rows, hidden_size, epsilon_f32, stream.stream());
  check_bridge_status(status, "Add+RMSNorm");
}

void add_rms_norm_mut(at::Tensor input, at::Tensor residual,
                      const at::Tensor& weight, double epsilon) {
  check_contract(input, residual, weight, epsilon);
  launch_add_rms_norm(input, residual, weight, epsilon);
}

void check_dynamic_fp8_contract(const at::Tensor& input,
                                const at::Tensor& weight,
                                const at::Tensor& output,
                                const at::Tensor& scales, double epsilon) {
  TORCH_CHECK(input.is_cuda(), "Loom RMSNorm+FP8 input must be CUDA");
  TORCH_CHECK(weight.device() == input.device() &&
                  output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom RMSNorm+FP8 tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
              "Loom RMSNorm+FP8 input and weight dtypes must match");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom RMSNorm+FP8 supports F32, FP16, and BF16 inputs");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "Loom RMSNorm+FP8 output must use torch.float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == at::kFloat,
              "Loom RMSNorm+FP8 scales must use F32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous() && scales.is_contiguous(),
              "Loom RMSNorm+FP8 tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom RMSNorm+FP8 input must be non-empty");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(-1),
              "Loom RMSNorm+FP8 weight must match the hidden dimension");
  TORCH_CHECK(output.sizes() == input.sizes(),
              "Loom RMSNorm+FP8 output shape must match input");
  const int64_t rows = input.numel() / input.size(-1);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == 1,
              "Loom RMSNorm+FP8 scales must have shape [rows, 1]");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
              "Loom RMSNorm+FP8 epsilon must be finite and positive");
  TORCH_CHECK(!byte_ranges_overlap(output, input) &&
                  !byte_ranges_overlap(output, weight) &&
                  !byte_ranges_overlap(output, scales) &&
                  !byte_ranges_overlap(scales, input) &&
                  !byte_ranges_overlap(scales, weight),
              "Loom RMSNorm+FP8 mutable tensor storage must not overlap");
}

void launch_rms_norm_dynamic_fp8(const at::Tensor& input,
                                 const at::Tensor& weight, at::Tensor output,
                                 at::Tensor scales, double epsilon) {
  const int64_t hidden_size_i64 = input.size(-1);
  const int64_t rows_i64 = input.numel() / hidden_size_i64;
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  hidden_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom RMSNorm+FP8 shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto hidden_size = static_cast<uint32_t>(hidden_size_i64);
  const auto input_elements = static_cast<uint64_t>(input.numel());
  const auto weight_elements = static_cast<uint64_t>(weight.numel());
  const auto output_elements = static_cast<uint64_t>(output.numel());
  const auto scale_elements = static_cast<uint64_t>(scales.numel());
  const auto epsilon_f32 = static_cast<float>(epsilon);
  auto* output_bytes = reinterpret_cast<uint8_t*>(output.data_ptr());
  auto* scale_values = scales.data_ptr<float>();
  const int status = loom_cuda_bridge_rms_norm_dynamic_fp8(
      bridge_dtype(input), input.data_ptr(), input_elements,
      weight.data_ptr(), weight_elements, output_bytes, output_elements,
      scale_values, scale_elements, rows, hidden_size, epsilon_f32,
      stream.stream());
  check_bridge_status(status, "RMSNorm+FP8");
}

void rms_norm_dynamic_fp8(const at::Tensor& input, const at::Tensor& weight,
                          at::Tensor output, at::Tensor scales,
                          double epsilon) {
  check_dynamic_fp8_contract(input, weight, output, scales, epsilon);
  launch_rms_norm_dynamic_fp8(input, weight, output, scales, epsilon);
}

void check_silu_and_mul_contract(const at::Tensor& input,
                                 const at::Tensor& output) {
  TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul input must be CUDA");
  TORCH_CHECK(output.device() == input.device(),
              "Loom SiLU-and-Mul tensors must be on the same CUDA device");
  TORCH_CHECK(output.scalar_type() == input.scalar_type(),
              "Loom SiLU-and-Mul input/output dtypes must match");
  TORCH_CHECK(input.scalar_type() == at::kFloat ||
                  input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom SiLU-and-Mul supports F32, FP16, and BF16");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul tensors must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul input must be non-empty");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul input last dimension must be even");
  TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul output prefix shape must match input");
  }
  TORCH_CHECK(output.size(-1) == input.size(-1) / 2,
              "Loom SiLU-and-Mul output last dimension must be half input");
  TORCH_CHECK(!byte_ranges_overlap(input, output),
              "Loom SiLU-and-Mul input/output storage must not overlap");
}

void launch_silu_and_mul(const at::Tensor& input, at::Tensor output) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const int status = loom_cuda_bridge_silu_and_mul(
      bridge_dtype(input), input.data_ptr(),
      static_cast<uint64_t>(input.numel()), output.data_ptr(),
      static_cast<uint64_t>(output.numel()), rows, width, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul");
}

void silu_and_mul(const at::Tensor& input, at::Tensor output) {
  check_silu_and_mul_contract(input, output);
  launch_silu_and_mul(input, output);
}

void check_silu_and_mul_dynamic_fp8_contract(const at::Tensor& input,
                                              const at::Tensor& output,
                                              const at::Tensor& scales,
                                              int64_t group_size,
                                              bool scales_transposed = false) {
  TORCH_CHECK(input.is_cuda(), "Loom SiLU-and-Mul+FP8 input must be CUDA");
  TORCH_CHECK(output.device() == input.device() &&
                  scales.device() == input.device(),
              "Loom SiLU-and-Mul+FP8 tensors must be on the same CUDA device");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
                  input.scalar_type() == at::kBFloat16,
              "Loom SiLU-and-Mul+FP8 supports FP16 and BF16 input");
  TORCH_CHECK(output.scalar_type() == at::kFloat8_e4m3fn,
              "Loom SiLU-and-Mul+FP8 output must use torch.float8_e4m3fn");
  TORCH_CHECK(scales.scalar_type() == at::kFloat,
              "Loom SiLU-and-Mul+FP8 scales must use F32");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "Loom SiLU-and-Mul+FP8 input/output must be contiguous");
  TORCH_CHECK(input.dim() >= 1 && input.numel() > 0,
              "Loom SiLU-and-Mul+FP8 input must be non-empty");
  TORCH_CHECK(input.size(-1) % 2 == 0,
              "Loom SiLU-and-Mul+FP8 input last dimension must be even");
  TORCH_CHECK(group_size == 64 || group_size == 128,
              "Loom SiLU-and-Mul+FP8 group size must be 64 or 128");
  const int64_t width = input.size(-1) / 2;
  TORCH_CHECK(width % group_size == 0,
              "Loom SiLU-and-Mul+FP8 width must be divisible by group size");
  TORCH_CHECK(output.dim() == input.dim(),
              "Loom SiLU-and-Mul+FP8 output rank must match input");
  for (int64_t dimension = 0; dimension + 1 < input.dim(); ++dimension) {
    TORCH_CHECK(output.size(dimension) == input.size(dimension),
                "Loom SiLU-and-Mul+FP8 output prefix shape must match input");
  }
  TORCH_CHECK(output.size(-1) == width,
              "Loom SiLU-and-Mul+FP8 output last dimension must be half input");
  const int64_t rows = input.numel() / input.size(-1);
  TORCH_CHECK(scales.dim() == 2 && scales.size(0) == rows &&
                  scales.size(1) == width / group_size,
              "Loom SiLU-and-Mul+FP8 scales must have shape "
              "[rows, width / group_size]");
  if (scales_transposed) {
    TORCH_CHECK(scales.stride(0) == 1 && scales.stride(1) == rows,
                "Loom transposed FP8 scales must use group-major storage");
  } else {
    TORCH_CHECK(scales.is_contiguous(),
                "Loom row-major FP8 scales must be contiguous");
  }
  TORCH_CHECK(!byte_ranges_overlap(input, output) &&
                  !byte_ranges_overlap(input, scales) &&
                  !byte_ranges_overlap(output, scales),
              "Loom SiLU-and-Mul+FP8 mutable tensor storage must not overlap");
}

void launch_silu_and_mul_dynamic_fp8_layout(
    const at::Tensor& input, at::Tensor output, at::Tensor scales,
    int64_t group_size_i64, const std::optional<at::Tensor>& scale_ub,
    bool scales_transposed) {
  const int64_t width_i64 = input.size(-1) / 2;
  const int64_t rows_i64 = input.numel() / input.size(-1);
  TORCH_CHECK(rows_i64 <= std::numeric_limits<uint32_t>::max() &&
                  width_i64 <= std::numeric_limits<uint32_t>::max() &&
                  group_size_i64 <= std::numeric_limits<uint32_t>::max(),
              "Loom SiLU-and-Mul+FP8 shape exceeds the CUDA ABI");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto rows = static_cast<uint32_t>(rows_i64);
  const auto width = static_cast<uint32_t>(width_i64);
  const auto group_size = static_cast<uint32_t>(group_size_i64);
  auto* output_bytes = reinterpret_cast<uint8_t*>(output.data_ptr());
  auto* scale_values = scales.data_ptr<float>();
  const float* scale_ub_value =
      scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr;
  const int status = loom_cuda_bridge_silu_and_mul_dynamic_fp8(
      bridge_dtype(input), input.data_ptr(),
      static_cast<uint64_t>(input.numel()), output_bytes,
      static_cast<uint64_t>(output.numel()), scale_values,
      static_cast<uint64_t>(scales.numel()), scale_ub_value,
      scale_ub.has_value() ? static_cast<uint64_t>(scale_ub->numel()) : 0U,
      rows, width, group_size, scales_transposed ? 1U : 0U, stream.stream());
  check_bridge_status(status, "SiLU-and-Mul+FP8");
}

void launch_silu_and_mul_dynamic_fp8(const at::Tensor& input,
                                      at::Tensor output, at::Tensor scales,
                                      int64_t group_size) {
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         std::nullopt, false);
}

void silu_and_mul_dynamic_fp8(const at::Tensor& input, at::Tensor output,
                              at::Tensor scales, int64_t group_size) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size);
  launch_silu_and_mul_dynamic_fp8(input, output, scales, group_size);
}

void vllm_silu_and_mul_per_block_fp8(
    at::Tensor output, const at::Tensor& input, at::Tensor scales,
    int64_t group_size, const std::optional<at::Tensor>& scale_ub,
    bool scales_transposed) {
  check_silu_and_mul_dynamic_fp8_contract(input, output, scales, group_size,
                                          scales_transposed);
  if (scale_ub.has_value()) {
    TORCH_CHECK(scale_ub->device() == input.device() &&
                    scale_ub->scalar_type() == at::kFloat &&
                    scale_ub->numel() == 1 && scale_ub->is_contiguous(),
                "Loom FP8 scale upper bound must be one same-device F32 value");
  }
  launch_silu_and_mul_dynamic_fp8_layout(input, output, scales, group_size,
                                         scale_ub, scales_transposed);
}

void check_greedy_sample_logprobs_shape(const at::Tensor& logits) {
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom greedy sampling logits must be non-empty rank-2");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<int32_t>::max(),
              "Loom greedy sampling shape exceeds the CUDA ABI");
}

void check_greedy_sample_logprobs_contract(const at::Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  TORCH_CHECK(logits.is_cuda(), "Loom greedy sampling logits must be CUDA");
  TORCH_CHECK(logits.scalar_type() == at::kFloat ||
                  logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kBFloat16,
              "Loom greedy sampling supports F32, FP16, and BF16 logits");
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom greedy sampling logits require unit vocabulary stride "
              "and non-overlapping positive row stride");
  TORCH_CHECK(!logits.requires_grad(),
              "Loom greedy sampling is an inference-only operator");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
launch_greedy_sample_logprobs(const at::Tensor& logits) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const auto logits_elements = storage_span_elements(logits);
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  at::Tensor token_ids =
      at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
  at::Tensor logprobs =
      at::empty({logits.size(0)}, logits.options().dtype(at::kFloat));
  at::Tensor ranks =
      at::empty({logits.size(0)}, logits.options().dtype(at::kLong));

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream(logits.device().index());
  const int status = loom_cuda_bridge_greedy_sample_logprobs(
      bridge_dtype(logits), logits.data_ptr(), logits_elements,
      token_ids.data_ptr<int32_t>(), output_elements,
      logprobs.data_ptr<float>(), output_elements, ranks.data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "greedy-sampling");
  return {token_ids, logprobs, ranks};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> greedy_sample_logprobs(
    const at::Tensor& logits) {
  check_greedy_sample_logprobs_contract(logits);
  return launch_greedy_sample_logprobs(logits);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> greedy_sample_logprobs_meta(
    const at::Tensor& logits) {
  check_greedy_sample_logprobs_shape(logits);
  return {
      at::empty({logits.size(0)}, logits.options().dtype(at::kInt)),
      at::empty({logits.size(0)}, logits.options().dtype(at::kFloat)),
      at::empty({logits.size(0)}, logits.options().dtype(at::kLong)),
  };
}

void check_selected_token_logprobs_shape(const at::Tensor& logits,
                                         const at::Tensor& token_ids) {
  check_greedy_sample_logprobs_shape(logits);
  TORCH_CHECK(token_ids.dim() == 1 && token_ids.size(0) == logits.size(0),
              "Loom selected token IDs must contain one value per logits row");
}

void check_selected_token_logprobs_contract(const at::Tensor& logits,
                                            const at::Tensor& token_ids) {
  check_greedy_sample_logprobs_contract(logits);
  check_selected_token_logprobs_shape(logits, token_ids);
  TORCH_CHECK(token_ids.device() == logits.device(),
              "Loom selected token IDs and logits must share a CUDA device");
  TORCH_CHECK(token_ids.scalar_type() == at::kLong,
              "Loom selected token IDs must be int64");
  TORCH_CHECK(token_ids.is_contiguous(),
              "Loom selected token IDs must be contiguous");
  TORCH_CHECK(!token_ids.requires_grad(),
              "Loom selected token IDs must not require gradients");
}

std::tuple<at::Tensor, at::Tensor> launch_selected_token_logprobs(
    const at::Tensor& logits, const at::Tensor& token_ids) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  at::Tensor logprobs =
      at::empty({logits.size(0)}, logits.options().dtype(at::kFloat));
  at::Tensor ranks =
      at::empty({logits.size(0)}, logits.options().dtype(at::kLong));

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream(logits.device().index());
  const auto output_elements = static_cast<uint64_t>(logits.size(0));
  const int status = loom_cuda_bridge_selected_token_logprobs(
      bridge_dtype(logits), logits.data_ptr(), storage_span_elements(logits),
      token_ids.data_ptr<int64_t>(), static_cast<uint64_t>(token_ids.numel()),
      logprobs.data_ptr<float>(), output_elements, ranks.data_ptr<int64_t>(),
      output_elements, rows, vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "selected-token logprob");
  return {logprobs, ranks};
}

std::tuple<at::Tensor, at::Tensor> selected_token_logprobs(
    const at::Tensor& logits, const at::Tensor& token_ids) {
  check_selected_token_logprobs_contract(logits, token_ids);
  return launch_selected_token_logprobs(logits, token_ids);
}

std::tuple<at::Tensor, at::Tensor> selected_token_logprobs_meta(
    const at::Tensor& logits, const at::Tensor& token_ids) {
  check_selected_token_logprobs_shape(logits, token_ids);
  return {
      at::empty({logits.size(0)}, logits.options().dtype(at::kFloat)),
      at::empty({logits.size(0)}, logits.options().dtype(at::kLong)),
  };
}

void check_min_p_filter_shape(const at::Tensor& logits,
                              const at::Tensor& min_p) {
  TORCH_CHECK(logits.dim() == 2 && logits.size(0) > 0 && logits.size(1) > 0,
              "Loom min-p logits must be non-empty rank-2");
  TORCH_CHECK(logits.size(0) <= std::numeric_limits<uint32_t>::max() &&
                  logits.size(1) <= std::numeric_limits<uint32_t>::max(),
              "Loom min-p shape exceeds the CUDA ABI");
  TORCH_CHECK((min_p.dim() == 1 && min_p.size(0) == logits.size(0)) ||
                  (min_p.dim() == 2 && min_p.size(0) == logits.size(0) &&
                   min_p.size(1) == 1),
              "Loom min-p probabilities must have shape [rows] or [rows, 1]");
}

void check_min_p_filter_contract(const at::Tensor& logits,
                                 const at::Tensor& min_p) {
  check_min_p_filter_shape(logits, min_p);
  TORCH_CHECK(logits.is_cuda(), "Loom min-p logits must be CUDA");
  TORCH_CHECK(min_p.device() == logits.device(),
              "Loom min-p probabilities and logits must share a CUDA device");
  TORCH_CHECK(logits.scalar_type() == at::kFloat ||
                  logits.scalar_type() == at::kHalf ||
                  logits.scalar_type() == at::kBFloat16,
              "Loom min-p supports F32, FP16, and BF16 logits");
  TORCH_CHECK(min_p.scalar_type() == at::kFloat,
              "Loom min-p probabilities must use F32");
  TORCH_CHECK(logits.stride(1) == 1 && logits.stride(0) >= logits.size(1),
              "Loom min-p logits require unit vocabulary stride and "
              "non-overlapping positive row stride");
  TORCH_CHECK(min_p.is_contiguous(),
              "Loom min-p probabilities must be contiguous");
  TORCH_CHECK(!logits.requires_grad() && !min_p.requires_grad(),
              "Loom min-p filtering is an inference-only operator");
  TORCH_CHECK(!byte_ranges_overlap(logits, min_p),
              "Loom min-p logits and probabilities must not overlap");
}

void launch_min_p_filter(at::Tensor logits, const at::Tensor& min_p) {
  const auto rows = static_cast<uint32_t>(logits.size(0));
  const auto vocab_size = static_cast<uint32_t>(logits.size(1));
  const auto row_stride = static_cast<uint64_t>(logits.stride(0));
  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream(logits.device().index());
  const int status = loom_cuda_bridge_min_p_filter(
      bridge_dtype(logits), logits.data_ptr(), storage_span_elements(logits),
      min_p.data_ptr<float>(), static_cast<uint64_t>(min_p.numel()), rows,
      vocab_size, row_stride, stream.stream());
  check_bridge_status(status, "min-p");
}

void min_p_filter(at::Tensor logits, const at::Tensor& min_p) {
  check_min_p_filter_contract(logits, min_p);
  launch_min_p_filter(logits, min_p);
}

void check_rope_paged_kv_write_contract(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& slot_mapping) {
  TORCH_CHECK(query.is_cuda(), "Loom RoPE+paged-KV query must be CUDA");
  TORCH_CHECK(key.device() == query.device() &&
                  value.device() == query.device() &&
                  positions.device() == query.device() &&
                  cos_sin_cache.device() == query.device() &&
                  key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  slot_mapping.device() == query.device(),
              "Loom RoPE+paged-KV tensors must be on one CUDA device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                  query.scalar_type() == value.scalar_type() &&
                  query.scalar_type() == cos_sin_cache.scalar_type() &&
                  query.scalar_type() == key_cache.scalar_type() &&
                  query.scalar_type() == value_cache.scalar_type(),
              "Loom RoPE+paged-KV data and native caches must share a dtype");
  TORCH_CHECK(query.scalar_type() == at::kFloat ||
                  query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "Loom RoPE+paged-KV supports F32, FP16, and BF16 native caches");
  TORCH_CHECK(positions.scalar_type() == at::kLong &&
                  slot_mapping.scalar_type() == at::kLong,
              "Loom RoPE+paged-KV positions and slot mapping must be int64");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
              "Loom RoPE+paged-KV Q/K/V must have rank 3");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0,
              "Loom RoPE+paged-KV query must be non-empty");
  TORCH_CHECK(key.size(0) == query.size(0) &&
                  value.size(0) == query.size(0),
              "Loom RoPE+paged-KV Q/K/V token counts must match");
  TORCH_CHECK(key.size(1) > 0 && key.size(1) == value.size(1),
              "Loom RoPE+paged-KV K/V head counts must match");
  TORCH_CHECK(key.size(2) == query.size(2),
              "Loom RoPE+paged-KV Q/K head sizes must match");
  TORCH_CHECK(value.size(2) > 0,
              "Loom RoPE+paged-KV value head size must be positive");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1 &&
                  value.stride(2) == 1 && query.stride(0) > 0 &&
                  query.stride(1) > 0 && key.stride(0) > 0 &&
                  key.stride(1) > 0 && value.stride(0) > 0 &&
                  value.stride(1) > 0 && positions.is_contiguous() &&
                  cos_sin_cache.is_contiguous() &&
                  slot_mapping.is_contiguous(),
              "Loom RoPE+paged-KV sources require unit dim stride and positive "
              "token/head strides; metadata must be contiguous");
  TORCH_CHECK(positions.dim() == 1 &&
                  positions.numel() == query.size(0) &&
                  slot_mapping.dim() == 1 &&
                  slot_mapping.numel() <= query.size(0),
              "Loom RoPE positions must cover every token and slot_mapping "
              "must not exceed the padded token count");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(0) > 0 &&
                  cos_sin_cache.size(1) > 0 &&
                  cos_sin_cache.size(1) % 2 == 0 &&
                  cos_sin_cache.size(1) <= query.size(2),
              "Loom RoPE+paged-KV cos/sin cache must be "
              "[max_position, even rotary_dim <= head_size]");
  TORCH_CHECK(key_cache.dim() == 4 && value_cache.dim() == 4,
              "Loom paged K/V cache views must have rank 4");
  TORCH_CHECK(key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) == key.size(1) &&
                  key_cache.size(3) == key.size(2),
              "Loom key cache must have logical shape "
              "[blocks, block_size, kv_heads, head_size]");
  TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == value.size(1) &&
                  value_cache.size(3) == value.size(2),
              "Loom value cache must have logical shape "
              "[blocks, block_size, kv_heads, value_head_size]");
  TORCH_CHECK(key_cache.stride(3) == 1 && value_cache.stride(3) == 1 &&
                  key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
                  key_cache.stride(2) > 0 && value_cache.stride(0) > 0 &&
                  value_cache.stride(1) > 0 && value_cache.stride(2) > 0,
              "Loom paged K/V caches require unit dim stride and positive "
              "block/page/head strides");
  TORCH_CHECK(!query.requires_grad() && !key.requires_grad() &&
                  !value.requires_grad() && !cos_sin_cache.requires_grad(),
              "Loom RoPE+paged-KV is an inference-only operator");
}

void launch_rope_paged_kv_write(
    at::Tensor query, at::Tensor key, const at::Tensor& value,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    at::Tensor key_cache, at::Tensor value_cache,
    const at::Tensor& slot_mapping, bool is_neox) {
  const int64_t limits[] = {
      query.size(0),       query.size(1),       key.size(1),
      query.size(2),       value.size(2),       cos_sin_cache.size(1),
      cos_sin_cache.size(0), key_cache.size(0), key_cache.size(1),
  };
  for (const int64_t value_to_check : limits) {
    TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom RoPE+paged-KV shape exceeds the CUDA ABI");
  }

  const c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  const auto tokens = static_cast<uint32_t>(query.size(0));
  const auto cache_tokens = static_cast<uint32_t>(slot_mapping.numel());
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key.size(1));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value.size(2));
  const auto rotary_dim = static_cast<uint32_t>(cos_sin_cache.size(1));
  const auto max_position = static_cast<uint32_t>(cos_sin_cache.size(0));
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto query_token_stride = static_cast<uint64_t>(query.stride(0));
  const auto query_head_stride = static_cast<uint64_t>(query.stride(1));
  const auto key_token_stride = static_cast<uint64_t>(key.stride(0));
  const auto key_head_stride = static_cast<uint64_t>(key.stride(1));
  const auto value_token_stride = static_cast<uint64_t>(value.stride(0));
  const auto value_head_stride = static_cast<uint64_t>(value.stride(1));
  const auto key_block_stride = static_cast<uint64_t>(key_cache.stride(0));
  const auto key_page_stride = static_cast<uint64_t>(key_cache.stride(1));
  const auto key_cache_head_stride =
      static_cast<uint64_t>(key_cache.stride(2));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto value_page_stride =
      static_cast<uint64_t>(value_cache.stride(1));
  const auto value_cache_head_stride =
      static_cast<uint64_t>(value_cache.stride(2));

  const int status = loom_cuda_bridge_rope_paged_kv_write(
      bridge_dtype(query), query.data_ptr(), storage_span_elements(query),
      key.data_ptr(), storage_span_elements(key), value.data_ptr(),
      storage_span_elements(value), positions.data_ptr<int64_t>(),
      static_cast<uint64_t>(positions.numel()), cos_sin_cache.data_ptr(),
      static_cast<uint64_t>(cos_sin_cache.numel()), key_cache.data_ptr(),
      storage_span_elements(key_cache), value_cache.data_ptr(),
      storage_span_elements(value_cache), slot_mapping.data_ptr<int64_t>(),
      static_cast<uint64_t>(slot_mapping.numel()), tokens, cache_tokens,
      query_heads, kv_heads, head_size, value_head_size, rotary_dim,
      max_position, num_blocks, block_size, query_token_stride,
      query_head_stride, key_token_stride, key_head_stride,
      value_token_stride, value_head_stride, key_block_stride,
      key_page_stride, key_cache_head_stride, value_block_stride,
      value_page_stride, value_cache_head_stride, is_neox ? 1U : 0U,
      stream.stream());
  check_bridge_status(status, "RoPE+paged-KV");
}

void rope_paged_kv_write(
    at::Tensor query, at::Tensor key, const at::Tensor& value,
    const at::Tensor& positions, const at::Tensor& cos_sin_cache,
    at::Tensor key_cache, at::Tensor value_cache,
    const at::Tensor& slot_mapping, bool is_neox) {
  check_rope_paged_kv_write_contract(query, key, value, positions,
                                     cos_sin_cache, key_cache, value_cache,
                                     slot_mapping);
  launch_rope_paged_kv_write(query, key, value, positions, cos_sin_cache,
                             key_cache, value_cache, slot_mapping, is_neox);
}

void check_paged_decode_attention_contract(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths, const at::Tensor& output,
    int64_t max_sequence_length, double scale) {
  TORCH_CHECK(query.is_cuda(), "Loom paged decode query must be CUDA");
  TORCH_CHECK(key_cache.device() == query.device() &&
                  value_cache.device() == query.device() &&
                  block_tables.device() == query.device() &&
                  sequence_lengths.device() == query.device() &&
                  output.device() == query.device(),
              "Loom paged decode tensors must be on one CUDA device");
  TORCH_CHECK(query.scalar_type() == at::kFloat ||
                  query.scalar_type() == at::kHalf ||
                  query.scalar_type() == at::kBFloat16,
              "Loom paged decode supports F32, FP16, and BF16 native caches");
  TORCH_CHECK(key_cache.scalar_type() == query.scalar_type() &&
                  value_cache.scalar_type() == query.scalar_type() &&
                  output.scalar_type() == query.scalar_type(),
              "Loom paged decode data tensors must share a dtype");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt &&
                  sequence_lengths.scalar_type() == at::kInt,
              "Loom paged decode metadata must use int32");
  TORCH_CHECK(query.dim() == 3 && key_cache.dim() == 4 &&
                  value_cache.dim() == 4 && block_tables.dim() == 2 &&
                  sequence_lengths.dim() == 1 && output.dim() == 3,
              "Loom paged decode requires rank-3 query/output, rank-4 K/V "
              "caches, rank-2 block tables, and rank-1 sequence lengths");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                  key_cache.size(0) > 0 && key_cache.size(1) > 0 &&
                  key_cache.size(2) > 0 && value_cache.size(3) > 0,
              "Loom paged decode dimensions must be positive");
  TORCH_CHECK(key_cache.size(3) == query.size(2),
              "Loom paged decode Q/K head sizes must match");
  TORCH_CHECK(value_cache.size(0) == key_cache.size(0) &&
                  value_cache.size(1) == key_cache.size(1) &&
                  value_cache.size(2) == key_cache.size(2),
              "Loom paged decode K/V cache prefixes must match");
  TORCH_CHECK(query.size(1) % key_cache.size(2) == 0,
              "Loom paged decode query heads must be divisible by KV heads");
  TORCH_CHECK(block_tables.size(0) == query.size(0) &&
                  block_tables.size(1) > 0 &&
                  sequence_lengths.size(0) == query.size(0),
              "Loom paged decode metadata batch dimensions must match query");
  TORCH_CHECK(output.size(0) == query.size(0) &&
                  output.size(1) == query.size(1) &&
                  output.size(2) == value_cache.size(3),
              "Loom paged decode output must have shape [B, Hq, Dv]");
  TORCH_CHECK(query.is_contiguous() &&
                  has_dense_nhd_inner_strides(key_cache) &&
                  has_dense_nhd_inner_strides(value_cache) &&
                  block_tables.is_contiguous() &&
                  sequence_lengths.is_contiguous() && output.is_contiguous(),
              "Loom paged decode requires contiguous query/output/metadata "
              "and dense-inner NHD caches with an optional block stride");
  TORCH_CHECK(max_sequence_length > 0 && max_sequence_length <= 1024 &&
                  max_sequence_length <=
                      block_tables.size(1) * key_cache.size(1),
              "Loom paged decode max_sequence_length must be within table "
              "capacity and the first-kernel limit 1024");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "Loom paged decode scale must be finite and positive");
  TORCH_CHECK(!query.requires_grad() && !key_cache.requires_grad() &&
                  !value_cache.requires_grad(),
              "Loom paged decode is an inference-only operator");
  TORCH_CHECK(!byte_ranges_overlap(output, query) &&
                  !byte_ranges_overlap(output, key_cache) &&
                  !byte_ranges_overlap(output, value_cache) &&
                  !byte_ranges_overlap(output, block_tables) &&
                  !byte_ranges_overlap(output, sequence_lengths),
              "Loom paged decode output storage must not overlap inputs");

  const int64_t limits[] = {
      query.size(0),       query.size(1),      key_cache.size(2),
      query.size(2),       value_cache.size(3), key_cache.size(0),
      key_cache.size(1),   block_tables.size(1), max_sequence_length,
  };
  for (const int64_t value_to_check : limits) {
    TORCH_CHECK(value_to_check > 0 &&
                    value_to_check <= std::numeric_limits<uint32_t>::max(),
                "Loom paged decode shape exceeds the CUDA ABI");
  }
  TORCH_CHECK(query.size(0) <=
                  std::numeric_limits<int32_t>::max() / query.size(1),
              "Loom paged decode grid exceeds the CUDA ABI");
}

void launch_paged_decode_attention(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths, at::Tensor output,
    int64_t max_sequence_length, double scale) {
  const auto sequences = static_cast<uint32_t>(query.size(0));
  const auto query_heads = static_cast<uint32_t>(query.size(1));
  const auto kv_heads = static_cast<uint32_t>(key_cache.size(2));
  const auto head_size = static_cast<uint32_t>(query.size(2));
  const auto value_head_size = static_cast<uint32_t>(value_cache.size(3));
  const auto num_blocks = static_cast<uint32_t>(key_cache.size(0));
  const auto block_size = static_cast<uint32_t>(key_cache.size(1));
  const auto key_block_stride =
      static_cast<uint64_t>(key_cache.stride(0));
  const auto value_block_stride =
      static_cast<uint64_t>(value_cache.stride(0));
  const auto max_blocks_per_sequence =
      static_cast<uint32_t>(block_tables.size(1));
  const auto max_context = static_cast<uint32_t>(max_sequence_length);
  const auto scale_f32 = static_cast<float>(scale);
  const c10::cuda::CUDAGuard device_guard(query.device());
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  uint64_t split_k_workspace_elements = 0;
  int status = loom_cuda_bridge_paged_decode_workspace_elements(
      bridge_dtype(query), sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, max_blocks_per_sequence,
      max_context, scale_f32, &split_k_workspace_elements);
  check_bridge_status(status, "paged decode workspace query");
  TORCH_CHECK(split_k_workspace_elements <=
                  static_cast<uint64_t>(
                      std::numeric_limits<int64_t>::max()),
              "Loom paged decode split-K workspace exceeds PyTorch limits");
  at::Tensor split_k_workspace;
  if (split_k_workspace_elements != 0U) {
    split_k_workspace = at::empty(
        {static_cast<int64_t>(split_k_workspace_elements)},
        query.options().dtype(at::kFloat));
  }
  float* split_k_workspace_pointer =
      split_k_workspace.defined() ? split_k_workspace.data_ptr<float>()
                                  : nullptr;

  status = loom_cuda_bridge_paged_decode_attention(
      bridge_dtype(query), query.data_ptr(),
      static_cast<uint64_t>(query.numel()), key_cache.data_ptr(),
      storage_span_elements(key_cache), value_cache.data_ptr(),
      storage_span_elements(value_cache), block_tables.data_ptr<int32_t>(),
      static_cast<uint64_t>(block_tables.numel()),
      sequence_lengths.data_ptr<int32_t>(),
      static_cast<uint64_t>(sequence_lengths.numel()), output.data_ptr(),
      static_cast<uint64_t>(output.numel()), split_k_workspace_pointer,
      split_k_workspace_elements, sequences, query_heads, kv_heads, head_size,
      value_head_size, num_blocks, block_size, key_block_stride,
      value_block_stride, max_blocks_per_sequence, max_context, scale_f32,
      stream.stream());
  check_bridge_status(status, "paged decode attention");
}

void paged_decode_attention(
    const at::Tensor& query, const at::Tensor& key_cache,
    const at::Tensor& value_cache, const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths, at::Tensor output,
    int64_t max_sequence_length, double scale) {
  check_paged_decode_attention_contract(
      query, key_cache, value_cache, block_tables, sequence_lengths, output,
      max_sequence_length, scale);
  launch_paged_decode_attention(query, key_cache, value_cache, block_tables,
                                sequence_lengths, output,
                                max_sequence_length, scale);
}

int64_t bridge_abi_version() {
  return static_cast<int64_t>(loom_cuda_bridge_abi_version());
}

int64_t bridge_launch_count(int64_t operation) {
  TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_PAGED_DECODE_ATTENTION,
              "Loom bridge operator id is out of range");
  uint64_t count = 0;
  const int status = loom_cuda_bridge_launch_count(
      static_cast<uint32_t>(operation), &count);
  check_bridge_status(status, "telemetry query");
  TORCH_CHECK(
      count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "Loom bridge launch count exceeds int64");
  return static_cast<int64_t>(count);
}

void reset_bridge_launch_count(int64_t operation) {
  TORCH_CHECK(operation >= 0 &&
                  operation <= LOOM_CUDA_BRIDGE_PAGED_DECODE_ATTENTION,
              "Loom bridge operator id is out of range");
  const int status =
      loom_cuda_bridge_reset_launch_count(static_cast<uint32_t>(operation));
  check_bridge_status(status, "telemetry reset");
}

}  // namespace

TORCH_LIBRARY(loom_kernels, library) {
  library.def(
      "rms_norm(Tensor input_tensor, Tensor weight, Tensor(a!) output, "
      "float epsilon) -> ()");
  library.def(
      "add_rms_norm_mut(Tensor(a!) input_tensor, Tensor(b!) residual, "
      "Tensor weight, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "silu_and_mul(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8(Tensor input_tensor, Tensor(a!) output, "
      "Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_per_block_fp8(Tensor(a!) out, Tensor input, "
      "Tensor(b!) scales, int group_size, Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  library.def(
      "greedy_sample_logprobs(Tensor logits) -> (Tensor token_ids, Tensor "
      "logprobs, Tensor ranks)");
  library.def(
      "selected_token_logprobs(Tensor logits, Tensor token_ids) -> (Tensor "
      "logprobs, Tensor ranks)");
  library.def("min_p_filter_(Tensor(a!) logits, Tensor min_p) -> ()");
  library.def(
      "paged_decode_attention(Tensor query, Tensor key_cache, Tensor "
      "value_cache, Tensor block_tables, Tensor sequence_lengths, "
      "Tensor(a!) output, int max_sequence_length, float scale) -> ()");
  library.def(
      "rope_paged_kv_write_(Tensor(a!) query, Tensor(b!) key, Tensor value, "
      "Tensor positions, Tensor cos_sin_cache, Tensor(c!) key_cache, "
      "Tensor(d!) value_cache, Tensor slot_mapping, bool is_neox) -> ()");
  library.def("bridge_abi_version() -> int", &bridge_abi_version);
  library.def("bridge_launch_count(int operation) -> int",
              &bridge_launch_count);
  library.def("reset_bridge_launch_count(int operation) -> ()",
              &reset_bridge_launch_count);
}

TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl("rms_norm", &rms_norm);
  library.impl("add_rms_norm_mut", &add_rms_norm_mut);
  library.impl("rms_norm_dynamic_fp8", &rms_norm_dynamic_fp8);
  library.impl("silu_and_mul", &silu_and_mul);
  library.impl("silu_and_mul_dynamic_fp8", &silu_and_mul_dynamic_fp8);
  library.impl("silu_and_mul_per_block_fp8",
               &vllm_silu_and_mul_per_block_fp8);
  library.impl("greedy_sample_logprobs", &greedy_sample_logprobs);
  library.impl("selected_token_logprobs", &selected_token_logprobs);
  library.impl("min_p_filter_", &min_p_filter);
  library.impl("paged_decode_attention", &paged_decode_attention);
  library.impl("rope_paged_kv_write_", &rope_paged_kv_write);
}

TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl("greedy_sample_logprobs", &greedy_sample_logprobs_meta);
  library.impl("selected_token_logprobs", &selected_token_logprobs_meta);
}
