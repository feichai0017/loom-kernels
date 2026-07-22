#include "loom_cuda.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace {

std::atomic<int64_t> vllm_silu_and_mul_per_block_fp8_launches{0};
std::atomic<int64_t> rope_paged_kv_write_launches{0};
std::atomic<int64_t> greedy_sample_logprobs_launches{0};
std::atomic<int64_t> selected_token_logprobs_launches{0};
std::atomic<int64_t> min_p_filter_launches{0};

bool byte_ranges_overlap(const at::Tensor& left, const at::Tensor& right) {
  const auto left_begin = reinterpret_cast<uintptr_t>(left.data_ptr());
  const auto right_begin = reinterpret_cast<uintptr_t>(right.data_ptr());
  const auto left_end = left_begin + left.nbytes();
  const auto right_end = right_begin + right.nbytes();
  return left_begin < right_end && right_begin < left_end;
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
  const auto epsilon_f32 = static_cast<float>(epsilon);
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_add_rms_norm_f32(
        input.data_ptr<float>(), residual.data_ptr<float>(),
        weight.data_ptr<float>(), rows, hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_add_rms_norm_f16(
        reinterpret_cast<uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::Half>()), rows,
        hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_add_rms_norm_bf16(
        reinterpret_cast<uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()), rows,
        hidden_size, epsilon_f32, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA Add+RMSNorm launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
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
  const auto epsilon_f32 = static_cast<float>(epsilon);
  auto* output_bytes = reinterpret_cast<uint8_t*>(output.data_ptr());
  auto* scale_values = scales.data_ptr<float>();
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_rms_norm_dynamic_fp8_f32(
        input.data_ptr<float>(), weight.data_ptr<float>(), output_bytes,
        scale_values, rows, hidden_size, epsilon_f32, stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_rms_norm_dynamic_fp8_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::Half>()),
        output_bytes, scale_values, rows, hidden_size, epsilon_f32,
        stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_rms_norm_dynamic_fp8_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
        output_bytes, scale_values, rows, hidden_size, epsilon_f32,
        stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA RMSNorm+FP8 launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
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
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kFloat) {
    status = loom_cuda_silu_and_mul_f32(
        input.data_ptr<float>(), output.data_ptr<float>(), rows, width,
        stream.stream());
  } else if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_silu_and_mul_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(output.data_ptr<at::Half>()), rows, width,
        stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_silu_and_mul_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(output.data_ptr<at::BFloat16>()), rows,
        width, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA SiLU-and-Mul launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
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
  int status = LOOM_CUDA_UNSUPPORTED;
  if (input.scalar_type() == at::kHalf) {
    status = loom_cuda_silu_and_mul_dynamic_fp8_f16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::Half>()),
        output_bytes, scale_values, rows, width, group_size, scale_ub_value,
        scales_transposed ? 1U : 0U, stream.stream());
  } else if (input.scalar_type() == at::kBFloat16) {
    status = loom_cuda_silu_and_mul_dynamic_fp8_bf16(
        reinterpret_cast<const uint16_t*>(input.data_ptr<at::BFloat16>()),
        output_bytes, scale_values, rows, width, group_size, scale_ub_value,
        scales_transposed ? 1U : 0U, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA SiLU-and-Mul+FP8 launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
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
  vllm_silu_and_mul_per_block_fp8_launches.fetch_add(
      1, std::memory_order_relaxed);
}

int64_t vllm_silu_and_mul_per_block_fp8_launch_count() {
  return vllm_silu_and_mul_per_block_fp8_launches.load(
      std::memory_order_relaxed);
}

void reset_vllm_silu_and_mul_per_block_fp8_launch_count() {
  vllm_silu_and_mul_per_block_fp8_launches.store(0,
                                                 std::memory_order_relaxed);
}

int64_t rope_paged_kv_write_launch_count() {
  return rope_paged_kv_write_launches.load(std::memory_order_relaxed);
}

void reset_rope_paged_kv_write_launch_count() {
  rope_paged_kv_write_launches.store(0, std::memory_order_relaxed);
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
  at::Tensor token_ids =
      at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
  at::Tensor logprobs =
      at::empty({logits.size(0)}, logits.options().dtype(at::kFloat));
  at::Tensor ranks =
      at::empty({logits.size(0)}, logits.options().dtype(at::kLong));

  const c10::cuda::CUDAGuard device_guard(logits.device());
  const auto stream = at::cuda::getCurrentCUDAStream(logits.device().index());
  int status = LOOM_CUDA_UNSUPPORTED;
  if (logits.scalar_type() == at::kFloat) {
    status = loom_cuda_greedy_sample_logprobs_f32(
        logits.data_ptr<float>(), token_ids.data_ptr<int32_t>(),
        logprobs.data_ptr<float>(), ranks.data_ptr<int64_t>(), rows, vocab_size,
        row_stride, stream.stream());
  } else if (logits.scalar_type() == at::kHalf) {
    status = loom_cuda_greedy_sample_logprobs_f16(
        reinterpret_cast<const uint16_t*>(logits.data_ptr<at::Half>()),
        token_ids.data_ptr<int32_t>(), logprobs.data_ptr<float>(),
        ranks.data_ptr<int64_t>(), rows, vocab_size, row_stride,
        stream.stream());
  } else if (logits.scalar_type() == at::kBFloat16) {
    status = loom_cuda_greedy_sample_logprobs_bf16(
        reinterpret_cast<const uint16_t*>(logits.data_ptr<at::BFloat16>()),
        token_ids.data_ptr<int32_t>(), logprobs.data_ptr<float>(),
        ranks.data_ptr<int64_t>(), rows, vocab_size, row_stride,
        stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA greedy sampling launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
  greedy_sample_logprobs_launches.fetch_add(1, std::memory_order_relaxed);
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

int64_t greedy_sample_logprobs_launch_count() {
  return greedy_sample_logprobs_launches.load(std::memory_order_relaxed);
}

void reset_greedy_sample_logprobs_launch_count() {
  greedy_sample_logprobs_launches.store(0, std::memory_order_relaxed);
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
  int status = LOOM_CUDA_UNSUPPORTED;
  if (logits.scalar_type() == at::kFloat) {
    status = loom_cuda_selected_token_logprobs_f32(
        logits.data_ptr<float>(), token_ids.data_ptr<int64_t>(),
        logprobs.data_ptr<float>(), ranks.data_ptr<int64_t>(), rows, vocab_size,
        row_stride, stream.stream());
  } else if (logits.scalar_type() == at::kHalf) {
    status = loom_cuda_selected_token_logprobs_f16(
        reinterpret_cast<const uint16_t*>(logits.data_ptr<at::Half>()),
        token_ids.data_ptr<int64_t>(), logprobs.data_ptr<float>(),
        ranks.data_ptr<int64_t>(), rows, vocab_size, row_stride,
        stream.stream());
  } else if (logits.scalar_type() == at::kBFloat16) {
    status = loom_cuda_selected_token_logprobs_bf16(
        reinterpret_cast<const uint16_t*>(logits.data_ptr<at::BFloat16>()),
        token_ids.data_ptr<int64_t>(), logprobs.data_ptr<float>(),
        ranks.data_ptr<int64_t>(), rows, vocab_size, row_stride,
        stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA selected-token logprob launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
  selected_token_logprobs_launches.fetch_add(1, std::memory_order_relaxed);
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

int64_t selected_token_logprobs_launch_count() {
  return selected_token_logprobs_launches.load(std::memory_order_relaxed);
}

void reset_selected_token_logprobs_launch_count() {
  selected_token_logprobs_launches.store(0, std::memory_order_relaxed);
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
  int status = LOOM_CUDA_UNSUPPORTED;
  if (logits.scalar_type() == at::kFloat) {
    status = loom_cuda_min_p_filter_f32(
        logits.data_ptr<float>(), min_p.data_ptr<float>(), rows, vocab_size,
        row_stride, stream.stream());
  } else if (logits.scalar_type() == at::kHalf) {
    status = loom_cuda_min_p_filter_f16(
        reinterpret_cast<uint16_t*>(logits.data_ptr<at::Half>()),
        min_p.data_ptr<float>(), rows, vocab_size, row_stride, stream.stream());
  } else if (logits.scalar_type() == at::kBFloat16) {
    status = loom_cuda_min_p_filter_bf16(
        reinterpret_cast<uint16_t*>(logits.data_ptr<at::BFloat16>()),
        min_p.data_ptr<float>(), rows, vocab_size, row_stride, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA min-p launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
  min_p_filter_launches.fetch_add(1, std::memory_order_relaxed);
}

void min_p_filter(at::Tensor logits, const at::Tensor& min_p) {
  check_min_p_filter_contract(logits, min_p);
  launch_min_p_filter(logits, min_p);
}

int64_t min_p_filter_launch_count() {
  return min_p_filter_launches.load(std::memory_order_relaxed);
}

void reset_min_p_filter_launch_count() {
  min_p_filter_launches.store(0, std::memory_order_relaxed);
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

  int status = LOOM_CUDA_UNSUPPORTED;
  if (query.scalar_type() == at::kFloat) {
    status = loom_cuda_rope_paged_kv_write_f32(
        query.data_ptr<float>(), key.data_ptr<float>(), value.data_ptr<float>(),
        positions.data_ptr<int64_t>(), cos_sin_cache.data_ptr<float>(),
        key_cache.data_ptr<float>(), value_cache.data_ptr<float>(),
        slot_mapping.data_ptr<int64_t>(), tokens, cache_tokens, query_heads,
        kv_heads, head_size, value_head_size, rotary_dim, max_position,
        num_blocks, block_size, query_token_stride, query_head_stride,
        key_token_stride, key_head_stride, value_token_stride,
        value_head_stride, key_block_stride, key_page_stride,
        key_cache_head_stride, value_block_stride, value_page_stride,
        value_cache_head_stride, is_neox ? 1U : 0U, stream.stream());
  } else if (query.scalar_type() == at::kHalf) {
    status = loom_cuda_rope_paged_kv_write_f16(
        reinterpret_cast<uint16_t*>(query.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(value.data_ptr<at::Half>()),
        positions.data_ptr<int64_t>(),
        reinterpret_cast<const uint16_t*>(
            cos_sin_cache.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr<at::Half>()),
        slot_mapping.data_ptr<int64_t>(), tokens, cache_tokens, query_heads,
        kv_heads, head_size, value_head_size, rotary_dim, max_position,
        num_blocks, block_size, query_token_stride, query_head_stride,
        key_token_stride, key_head_stride, value_token_stride,
        value_head_stride, key_block_stride, key_page_stride,
        key_cache_head_stride, value_block_stride, value_page_stride,
        value_cache_head_stride, is_neox ? 1U : 0U, stream.stream());
  } else if (query.scalar_type() == at::kBFloat16) {
    status = loom_cuda_rope_paged_kv_write_bf16(
        reinterpret_cast<uint16_t*>(query.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(key.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(value.data_ptr<at::BFloat16>()),
        positions.data_ptr<int64_t>(),
        reinterpret_cast<const uint16_t*>(
            cos_sin_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(key_cache.data_ptr<at::BFloat16>()),
        reinterpret_cast<uint16_t*>(value_cache.data_ptr<at::BFloat16>()),
        slot_mapping.data_ptr<int64_t>(), tokens, cache_tokens, query_heads,
        kv_heads, head_size, value_head_size, rotary_dim, max_position,
        num_blocks, block_size, query_token_stride, query_head_stride,
        key_token_stride, key_head_stride, value_token_stride,
        value_head_stride, key_block_stride, key_page_stride,
        key_cache_head_stride, value_block_stride, value_page_stride,
        value_cache_head_stride, is_neox ? 1U : 0U, stream.stream());
  }
  TORCH_CHECK(status == LOOM_CUDA_SUCCESS,
              "Loom CUDA RoPE+paged-KV launch failed: ",
              loom_cuda_status_string(status), " (status ", status, ")");
  rope_paged_kv_write_launches.fetch_add(1, std::memory_order_relaxed);
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

}  // namespace

TORCH_LIBRARY(loom_kernels, library) {
  library.def(
      "add_rms_norm_mut(Tensor(a!) input_tensor, Tensor(b!) residual, "
      "Tensor weight, float epsilon) -> ()");
  library.def(
      "add_rms_norm_mut_unchecked(Tensor(a!) input_tensor, Tensor(b!) "
      "residual, Tensor weight, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "rms_norm_dynamic_fp8_unchecked(Tensor input_tensor, Tensor weight, "
      "Tensor(a!) output, Tensor(b!) scales, float epsilon) -> ()");
  library.def(
      "silu_and_mul(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_unchecked(Tensor input_tensor, Tensor(a!) output) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8(Tensor input_tensor, Tensor(a!) output, "
      "Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_dynamic_fp8_unchecked(Tensor input_tensor, "
      "Tensor(a!) output, Tensor(b!) scales, int group_size) -> ()");
  library.def(
      "silu_and_mul_per_block_fp8(Tensor(a!) out, Tensor input, "
      "Tensor(b!) scales, int group_size, Tensor? scale_ub=None, "
      "bool is_scale_transposed=False) -> ()");
  library.def("vllm_silu_and_mul_per_block_fp8_launch_count() -> int",
              &vllm_silu_and_mul_per_block_fp8_launch_count);
  library.def("reset_vllm_silu_and_mul_per_block_fp8_launch_count() -> ()",
              &reset_vllm_silu_and_mul_per_block_fp8_launch_count);
  library.def("rope_paged_kv_write_launch_count() -> int",
              &rope_paged_kv_write_launch_count);
  library.def("reset_rope_paged_kv_write_launch_count() -> ()",
              &reset_rope_paged_kv_write_launch_count);
  library.def(
      "greedy_sample_logprobs(Tensor logits) -> (Tensor token_ids, Tensor "
      "logprobs, Tensor ranks)");
  library.def("greedy_sample_logprobs_launch_count() -> int",
              &greedy_sample_logprobs_launch_count);
  library.def("reset_greedy_sample_logprobs_launch_count() -> ()",
              &reset_greedy_sample_logprobs_launch_count);
  library.def(
      "selected_token_logprobs(Tensor logits, Tensor token_ids) -> (Tensor "
      "logprobs, Tensor ranks)");
  library.def("selected_token_logprobs_launch_count() -> int",
              &selected_token_logprobs_launch_count);
  library.def("reset_selected_token_logprobs_launch_count() -> ()",
              &reset_selected_token_logprobs_launch_count);
  library.def("min_p_filter_(Tensor(a!) logits, Tensor min_p) -> ()");
  library.def(
      "min_p_filter_unchecked_(Tensor(a!) logits, Tensor min_p) -> ()");
  library.def("min_p_filter_launch_count() -> int",
              &min_p_filter_launch_count);
  library.def("reset_min_p_filter_launch_count() -> ()",
              &reset_min_p_filter_launch_count);
  library.def(
      "rope_paged_kv_write_(Tensor(a!) query, Tensor(b!) key, Tensor value, "
      "Tensor positions, Tensor cos_sin_cache, Tensor(c!) key_cache, "
      "Tensor(d!) value_cache, Tensor slot_mapping, bool is_neox) -> ()");
  library.def(
      "rope_paged_kv_write_unchecked_(Tensor(a!) query, Tensor(b!) key, "
      "Tensor value, Tensor positions, Tensor cos_sin_cache, "
      "Tensor(c!) key_cache, Tensor(d!) value_cache, Tensor slot_mapping, "
      "bool is_neox) -> ()");
}

TORCH_LIBRARY_IMPL(loom_kernels, CUDA, library) {
  library.impl("add_rms_norm_mut", &add_rms_norm_mut);
  library.impl("add_rms_norm_mut_unchecked", &launch_add_rms_norm);
  library.impl("rms_norm_dynamic_fp8", &rms_norm_dynamic_fp8);
  library.impl("rms_norm_dynamic_fp8_unchecked",
               &launch_rms_norm_dynamic_fp8);
  library.impl("silu_and_mul", &silu_and_mul);
  library.impl("silu_and_mul_unchecked", &launch_silu_and_mul);
  library.impl("silu_and_mul_dynamic_fp8", &silu_and_mul_dynamic_fp8);
  library.impl("silu_and_mul_dynamic_fp8_unchecked",
               &launch_silu_and_mul_dynamic_fp8);
  library.impl("silu_and_mul_per_block_fp8",
               &vllm_silu_and_mul_per_block_fp8);
  library.impl("greedy_sample_logprobs", &greedy_sample_logprobs);
  library.impl("selected_token_logprobs", &selected_token_logprobs);
  library.impl("min_p_filter_", &min_p_filter);
  library.impl("min_p_filter_unchecked_", &launch_min_p_filter);
  library.impl("rope_paged_kv_write_", &rope_paged_kv_write);
  library.impl("rope_paged_kv_write_unchecked_",
               &launch_rope_paged_kv_write);
}

TORCH_LIBRARY_IMPL(loom_kernels, Meta, library) {
  library.impl("greedy_sample_logprobs", &greedy_sample_logprobs_meta);
  library.impl("selected_token_logprobs", &selected_token_logprobs_meta);
}
