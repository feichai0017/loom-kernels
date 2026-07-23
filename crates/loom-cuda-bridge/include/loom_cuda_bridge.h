#ifndef LOOM_CUDA_BRIDGE_H_
#define LOOM_CUDA_BRIDGE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum loom_cuda_bridge_status {
  LOOM_CUDA_BRIDGE_SUCCESS = 0,
  LOOM_CUDA_BRIDGE_INVALID_ARGUMENT = 1,
  LOOM_CUDA_BRIDGE_UNSUPPORTED = 2,
  LOOM_CUDA_BRIDGE_LAUNCH_ERROR = 3,
  LOOM_CUDA_BRIDGE_UNAVAILABLE = 4,
};

// Return the detailed error recorded by the most recent failed bridge call on
// this host thread. The pointer remains valid until the next bridge failure on
// the same thread.
const char* loom_cuda_bridge_last_error_message(void);

// Checked Rust-runtime Add+RMSNorm entrypoints. Element counts describe the
// full pointed-to regions and are validated against rows * hidden_size.
// Launches are asynchronous on the supplied framework-owned CUDA stream.
int loom_cuda_bridge_add_rms_norm_f32(
    float* input, uint64_t input_elements, float* residual,
    uint64_t residual_elements, const float* weight, uint64_t weight_elements,
    uint32_t rows, uint32_t hidden_size, float epsilon, void* stream);

int loom_cuda_bridge_add_rms_norm_f16(
    uint16_t* input, uint64_t input_elements, uint16_t* residual,
    uint64_t residual_elements, const uint16_t* weight,
    uint64_t weight_elements, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

int loom_cuda_bridge_add_rms_norm_bf16(
    uint16_t* input, uint64_t input_elements, uint16_t* residual,
    uint64_t residual_elements, const uint16_t* weight,
    uint64_t weight_elements, uint32_t rows, uint32_t hidden_size,
    float epsilon, void* stream);

uint64_t loom_cuda_bridge_add_rms_norm_launch_count(void);
void loom_cuda_bridge_reset_add_rms_norm_launch_count(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CUDA_BRIDGE_H_
