"""vLLM IR provider registration for Loom Kernels."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from ._torch_extension import load_torch_extension, torch_extension_available


DEFAULT_PROVIDER = "loom_cuda"
SUPPORTED_VLLM_SERIES = ((0, 24), (0, 25))
SILU_OVERRIDE_KEY = "SiluAndMul"
SILU_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL"
ACT_QUANT_OVERRIDE_KEY = "silu_and_mul_dynamic_fp8"
ACT_QUANT_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8"
ROPE_PAGED_KV_OVERRIDE_KEY = "rope_paged_kv"
GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY = "greedy_sample_logprobs"
SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY = "selected_token_logprobs"
MIN_P_OVERRIDE_KEY = "min_p_filter"
MIN_P_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_MIN_P"
MIN_P_FAST_PATH_MIN_ROWS = 32
MIN_P_FAST_PATH_MIN_VOCAB_SIZE = 65536
PAGED_DECODE_OVERRIDE_KEY = "paged_decode_attention"
PAGED_DECODE_OVERRIDE_ENV = "LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION"
PAGED_DECODE_FAST_PATH_MAX_BATCH = 128
PAGED_DECODE_FAST_PATH_MAX_CONTEXT = 32
_SILU_OVERRIDE_CLASS: type | None = None
_ACT_QUANT_OVERRIDE_REGISTERED = False
_ROPE_PAGED_KV_REGISTERED = False
_ROPE_PAGED_KV_FIRST_CONTRACT: dict[str, Any] | None = None
_GREEDY_SAMPLE_LOGPROBS_REGISTERED = False
_GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH: Any | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_REGISTERED = False
_SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD: Any | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT: dict[str, Any] | None = None
_SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION: dict[str, Any] | None = None
_MIN_P_REGISTERED = False
_MIN_P_ORIGINAL_APPLY: Any | None = None
_PAGED_DECODE_REGISTERED = False
_PAGED_DECODE_ORIGINAL_FORWARD: Any | None = None
_PAGED_DECODE_CAN_USE_FAST_PATH: Any | None = None
_PAGED_DECODE_FIRST_CONTRACT: dict[str, Any] | None = None
_PAGED_DECODE_FIRST_REJECTION: dict[str, Any] | None = None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def installed_vllm_version() -> str | None:
    """Return the installed vLLM release, if the package is present."""
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def supports_installed_vllm() -> bool:
    """Return whether the installed vLLM series is qualified by Loom."""
    release = installed_vllm_version()
    if release is None:
        return False
    components = release.split(".", 2)
    if len(components) < 2:
        return False
    try:
        series = (int(components[0]), int(components[1]))
    except ValueError:
        return False
    return series in SUPPORTED_VLLM_SERIES


def _silu_override_requested() -> bool:
    return _env_enabled(SILU_OVERRIDE_ENV)


def _act_quant_override_requested() -> bool:
    return _env_enabled(ACT_QUANT_OVERRIDE_ENV)


def _min_p_override_requested() -> bool:
    return _env_enabled(MIN_P_OVERRIDE_ENV)


def _paged_decode_override_requested() -> bool:
    return _env_enabled(PAGED_DECODE_OVERRIDE_ENV)


def supports_vllm_paged_decode_shape(
    *,
    dtype: torch.dtype,
    batch: int,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    max_sequence_length: int,
) -> bool:
    """Return whether a shape is inside the H20-qualified FA3 win region."""
    return bool(
        dtype in (torch.float16, torch.bfloat16)
        and 0 < batch <= PAGED_DECODE_FAST_PATH_MAX_BATCH
        and query_heads == 32
        and kv_heads == 8
        and head_size == 128
        and block_size in (16, 32)
        and 0 < max_sequence_length <= PAGED_DECODE_FAST_PATH_MAX_CONTEXT
    )


def register_vllm_silu_and_mul() -> str | None:
    """Override vLLM's standard SwiGLU layer with the Loom CUDA operator."""
    global _SILU_OVERRIDE_CLASS
    if _SILU_OVERRIDE_CLASS is not None:
        return SILU_OVERRIDE_KEY
    if not torch_extension_available():
        return None
    if not supports_installed_vllm():
        return None

    from vllm.model_executor.custom_op import CustomOp
    from vllm.model_executor.layers.activation import SiluAndMul

    load_torch_extension()
    implementation = torch.ops.loom_kernels.silu_and_mul.default

    @CustomOp.register_oot(name=SILU_OVERRIDE_KEY)
    class LoomSiluAndMul(SiluAndMul):
        def __init__(self, *, compile_native: bool = True):
            # vLLM may globally disable CustomOp kernels while compiling its
            # native fallback.  An out-of-tree replacement must opt back in,
            # otherwise the registered class exists but never reaches Loom.
            del compile_native
            CustomOp.__init__(self, enforce_enable=True, compile_native=False)

        def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
            width = x.shape[-1] // 2
            output = torch.empty(
                (*x.shape[:-1], width), dtype=x.dtype, device=x.device
            )
            implementation(x, output)
            return output

    _SILU_OVERRIDE_CLASS = LoomSiluAndMul
    return SILU_OVERRIDE_KEY


def register_vllm_silu_and_mul_dynamic_fp8() -> str | None:
    """Route vLLM's 64/128-element activation-quant fusions to Loom."""
    global _ACT_QUANT_OVERRIDE_REGISTERED
    if _ACT_QUANT_OVERRIDE_REGISTERED:
        return ACT_QUANT_OVERRIDE_KEY
    if not torch_extension_available():
        return None
    if not supports_installed_vllm():
        return None

    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic64Sym,
        kFp8Dynamic128Sym,
    )

    load_torch_extension()
    implementation = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    FUSED_OPS[kFp8Dynamic64Sym] = implementation
    FUSED_OPS[kFp8Dynamic128Sym] = implementation
    _ACT_QUANT_OVERRIDE_REGISTERED = True
    return ACT_QUANT_OVERRIDE_KEY


def register_vllm_rope_paged_kv() -> str | None:
    """Teach vLLM 0.24/0.25 CUDA attention backends to call Loom's fused op.

    Registration only installs the backend capability and implementation. Use
    :func:`configure_vllm_rope_paged_kv` before constructing ``vllm.LLM`` to
    opt the compilation graph into vLLM's existing RoPE+KV fusion pass.
    """
    global _ROPE_PAGED_KV_REGISTERED
    if _ROPE_PAGED_KV_REGISTERED:
        return ROPE_PAGED_KV_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    if not supports_installed_vllm():
        return None

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    load_torch_extension()
    implementation = torch.ops.loom_kernels.rope_paged_kv_write_.default
    native_cache_dtypes = {
        "auto",
        "float16",
        "half",
        "bfloat16",
        "float32",
        "float",
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }

    def supported(attention: Any) -> bool:
        return bool(
            getattr(attention, "kv_sharing_target_layer_name", None) is None
            and getattr(attention, "kv_cache_dtype", None) in native_cache_dtypes
            and getattr(attention, "attn_type", AttentionType.DECODER)
            == AttentionType.DECODER
        )

    def do_rope_and_kv_cache_update(
        attention: Any,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ) -> None:
        global _ROPE_PAGED_KV_FIRST_CONTRACT
        del attention, layer
        key_cache, value_cache = kv_cache.unbind(1)
        if _ROPE_PAGED_KV_FIRST_CONTRACT is None:
            _ROPE_PAGED_KV_FIRST_CONTRACT = {
                "query": {
                    "shape": list(query.shape),
                    "stride": list(query.stride()),
                    "dtype": str(query.dtype),
                },
                "key": {
                    "shape": list(key.shape),
                    "stride": list(key.stride()),
                    "dtype": str(key.dtype),
                },
                "value": {
                    "shape": list(value.shape),
                    "stride": list(value.stride()),
                    "dtype": str(value.dtype),
                },
                "positions": list(positions.shape),
                "cos_sin_cache": {
                    "shape": list(cos_sin_cache.shape),
                    "dtype": str(cos_sin_cache.dtype),
                },
                "kv_cache": {
                    "shape": list(kv_cache.shape),
                    "stride": list(kv_cache.stride()),
                    "dtype": str(kv_cache.dtype),
                },
                "slot_mapping": list(layer_slot_mapping.shape),
                "is_neox": is_neox,
            }
        implementation(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            layer_slot_mapping,
            is_neox,
        )

    for implementation_class in (FlashAttentionImpl, FlashInferImpl):
        implementation_class.fused_rope_kvcache_supported = supported
        implementation_class.do_rope_and_kv_cache_update = (
            do_rope_and_kv_cache_update
        )

    _ROPE_PAGED_KV_REGISTERED = True
    return ROPE_PAGED_KV_OVERRIDE_KEY


def configure_vllm_rope_paged_kv(
    compilation_config: Any | None = None,
    *,
    max_token_num: int = 256,
) -> Any:
    """Return a vLLM compilation config with Loom RoPE+KV fusion enabled.

    vLLM labels this pass ROCm-only during initial ``PassConfig``
    validation. Setting the flag after constructing ``CompilationConfig`` is
    intentional: Loom supplies the missing CUDA backend implementation.
    """
    if max_token_num <= 0:
        raise ValueError("max_token_num must be positive")
    if register_vllm_rope_paged_kv() is None:
        raise RuntimeError(
            "Loom RoPE+paged-KV requires vLLM 0.24/0.25 and the C++ "
            "dispatcher bridge"
        )

    from vllm.config import CompilationConfig

    if compilation_config is None:
        configured = CompilationConfig()
    elif isinstance(compilation_config, dict):
        configured = CompilationConfig(**compilation_config)
    elif isinstance(compilation_config, CompilationConfig):
        configured = compilation_config
    else:
        raise TypeError("compilation_config must be a dict or CompilationConfig")

    if "+rotary_embedding" not in configured.custom_ops:
        configured.custom_ops.append("+rotary_embedding")
    if configured.splitting_ops is None:
        configured.splitting_ops = []
    configured.pass_config.fuse_rope_kvcache = True
    configured.pass_config.rope_kvcache_fusion_max_token_num = max_token_num
    return configured


def register_vllm_paged_decode_attention() -> str | None:
    """Install a measured-shape vLLM 0.24/0.25 FlashAttention decode path."""
    global _PAGED_DECODE_CAN_USE_FAST_PATH
    global _PAGED_DECODE_ORIGINAL_FORWARD
    global _PAGED_DECODE_REGISTERED
    if _PAGED_DECODE_REGISTERED:
        return PAGED_DECODE_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    from .torch_ops import supports_paged_decode_attention

    if not supports_installed_vllm():
        return None

    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    implementation = torch.ops.loom_kernels.paged_decode_attention.default
    original_forward = FlashAttentionImpl.forward
    native_cache_dtypes = {
        "auto",
        "float16",
        "half",
        "bfloat16",
        torch.float16,
        torch.bfloat16,
    }

    def can_use_fast_path(
        attention: Any,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None,
        output_block_scale: torch.Tensor | None,
    ) -> bool:
        if attn_metadata is None or kv_cache.dim() != 5:
            return False
        sequences = int(attn_metadata.seq_lens.shape[0])
        block_size = int(kv_cache.shape[2])
        if not supports_vllm_paged_decode_shape(
            dtype=query.dtype,
            batch=sequences,
            query_heads=attention.num_heads,
            kv_heads=attention.num_kv_heads,
            head_size=attention.head_size,
            block_size=block_size,
            max_sequence_length=attn_metadata.max_seq_len,
        ):
            return False
        # FA3's AOT scheduler tensor and the non-DCP zero context length are
        # execution hints, not attention semantics. Loom owns its scheduling;
        # the DCP world-size/cache-length and cascade gates below remain strict.
        return bool(
            output_scale is None
            and output_block_scale is None
            and attention.attn_type == AttentionType.DECODER
            and attention.num_heads == 32
            and attention.num_kv_heads == 8
            and attention.head_size == 128
            and attention.alibi_slopes is None
            and tuple(attention.sliding_window) == (-1, -1)
            and attention.logits_soft_cap == 0
            and attention.sinks is None
            and attention.kv_sharing_target_layer_name is None
            and attention.kv_cache_dtype in native_cache_dtypes
            and getattr(attention, "dcp_world_size", 1) == 1
            and attn_metadata.max_query_len == 1
            and attn_metadata.num_actual_tokens == sequences
            and attn_metadata.query_start_loc.shape[0] == sequences + 1
            and attn_metadata.block_table.shape[0] == sequences
            and not attn_metadata.use_cascade
            and attn_metadata.common_prefix_len == 0
            and attn_metadata.dcp_context_kv_lens is None
            and attn_metadata.prefix_scheduler_metadata is None
            and attn_metadata.causal is True
            and attn_metadata.mm_prefix_range_tensor is None
            and query.device.type == "cuda"
            and query.dtype == kv_cache.dtype
            and query.dim() == 3
            and query.shape[0] >= sequences
            and tuple(query.shape[1:]) == (32, 128)
            and query.is_contiguous()
            and not query.requires_grad
            and output.device == query.device
            and output.dtype == query.dtype
            and output.dim() == 3
            and output.shape[0] >= sequences
            and tuple(output.shape[1:]) == (32, 128)
            and output.is_contiguous()
            and kv_cache.device == query.device
            and tuple(kv_cache.shape[1:])
            == (2, block_size, 8, 128)
        )

    def forward(
        attention: Any,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global _PAGED_DECODE_FIRST_CONTRACT
        global _PAGED_DECODE_FIRST_REJECTION
        if not can_use_fast_path(
            attention,
            query,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        ):
            if _PAGED_DECODE_FIRST_REJECTION is None and attn_metadata is not None:
                _PAGED_DECODE_FIRST_REJECTION = {
                    "query_shape": list(query.shape),
                    "query_dtype": str(query.dtype),
                    "kv_cache_shape": list(kv_cache.shape),
                    "num_actual_tokens": attn_metadata.num_actual_tokens,
                    "max_query_len": attn_metadata.max_query_len,
                    "max_seq_len": attn_metadata.max_seq_len,
                    "use_cascade": attn_metadata.use_cascade,
                }
            return original_forward(
                attention,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        sequences = int(attn_metadata.seq_lens.shape[0])
        query_view = query[:sequences]
        output_view = output[:sequences]
        key_cache, value_cache = kv_cache.unbind(1)
        if not supports_paged_decode_attention(
            query_view,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            max_sequence_length=attn_metadata.max_seq_len,
        ):
            return original_forward(
                attention,
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if _PAGED_DECODE_FIRST_CONTRACT is None:
            _PAGED_DECODE_FIRST_CONTRACT = {
                "query_shape": list(query_view.shape),
                "query_stride": list(query_view.stride()),
                "dtype": str(query_view.dtype),
                "kv_cache_shape": list(kv_cache.shape),
                "kv_cache_stride": list(kv_cache.stride()),
                "block_table_shape": list(attn_metadata.block_table.shape),
                "max_seq_len": attn_metadata.max_seq_len,
            }
        implementation(
            query_view,
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            output_view,
            attn_metadata.max_seq_len,
            attention.scale,
        )
        return output

    forward.__module__ = __name__
    _PAGED_DECODE_ORIGINAL_FORWARD = original_forward
    _PAGED_DECODE_CAN_USE_FAST_PATH = can_use_fast_path
    FlashAttentionImpl.forward = forward
    _PAGED_DECODE_REGISTERED = True
    return PAGED_DECODE_OVERRIDE_KEY


def register_vllm_greedy_sample_logprobs() -> str | None:
    """Install the deterministic vLLM 0.24/0.25 greedy+logprob fast path.

    The override is deliberately narrow: all requests must be greedy, request
    only the sampled token's raw logprob (`max_num_logprobs == 0`), and have no
    logits mutation from masks, bad words, penalties, or processors. Every
    other sampler contract executes vLLM's original implementation.
    """
    global _GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD
    global _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH
    global _GREEDY_SAMPLE_LOGPROBS_REGISTERED
    if _GREEDY_SAMPLE_LOGPROBS_REGISTERED:
        return GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    if not supports_installed_vllm():
        return None

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
    from vllm.v1.sample.logits_processor.builtin import (
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
    )
    from vllm.v1.sample.sampler import Sampler

    load_torch_extension()
    implementation = torch.ops.loom_kernels.greedy_sample_logprobs.default
    original_forward = Sampler.forward

    def non_argmax_processors_are_inactive(processors: list[Any]) -> bool:
        for processor in processors:
            if isinstance(processor, MinTokensLogitsProcessor):
                if not processor.min_toks:
                    continue
            elif isinstance(processor, LogitBiasLogitsProcessor):
                if not processor.biases:
                    continue
            elif isinstance(processor, AdapterLogitsProcessor):
                if not processor.req_info:
                    continue
            return False
        return True

    def can_use_fast_path(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool,
        logprobs_mode_override: Any,
    ) -> bool:
        logprobs_mode = logprobs_mode_override or sampler.logprobs_mode
        holder = sampling_metadata.thinking_budget_state_holder
        thinking_active = holder is not None and holder.has_tracked_requests()
        return bool(
            logprobs_mode == "raw_logprobs"
            and sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs == 0
            and not sampling_metadata.logprob_token_ids
            and sampling_metadata.no_penalties
            and sampling_metadata.allowed_token_ids_mask is None
            and not sampling_metadata.bad_words_token_ids
            and non_argmax_processors_are_inactive(
                sampling_metadata.logitsprocs.non_argmax_invariant
            )
            and not thinking_active
            and not predict_bonus_token
            and logits.device.type == "cuda"
            and logits.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and logits.dim() == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.shape[1] <= 0x7FFF_FFFF
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and not logits.requires_grad
        )

    def forward(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        global _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT
        global _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION
        use_fast_path = can_use_fast_path(
            sampler,
            logits,
            sampling_metadata,
            predict_bonus_token,
            logprobs_mode_override,
        )
        if not use_fast_path:
            if (
                _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION is None
                and (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.all_greedy
                )
            ):
                holder = sampling_metadata.thinking_budget_state_holder
                _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION = {
                    "shape": list(logits.shape),
                    "stride": list(logits.stride()),
                    "dtype": str(logits.dtype),
                    "logprobs_mode": (
                        logprobs_mode_override or sampler.logprobs_mode
                    ),
                    "max_num_logprobs": sampling_metadata.max_num_logprobs,
                    "has_logprob_token_ids": bool(
                        sampling_metadata.logprob_token_ids
                    ),
                    "all_greedy": sampling_metadata.all_greedy,
                    "no_penalties": sampling_metadata.no_penalties,
                    "has_allowed_mask": (
                        sampling_metadata.allowed_token_ids_mask is not None
                    ),
                    "has_bad_words": bool(sampling_metadata.bad_words_token_ids),
                    "non_argmax_processors": len(
                        sampling_metadata.logitsprocs.non_argmax_invariant
                    ),
                    "thinking_active": (
                        holder is not None and holder.has_tracked_requests()
                    ),
                    "predict_bonus_token": predict_bonus_token,
                    "is_contiguous": logits.is_contiguous(),
                    "requires_grad": logits.requires_grad,
                }
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT is None:
            _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "stride": list(logits.stride()),
                "dtype": str(logits.dtype),
                "max_num_logprobs": sampling_metadata.max_num_logprobs,
                "all_greedy": sampling_metadata.all_greedy,
            }
        token_ids, logprobs, ranks = implementation(logits)
        token_ids = token_ids.unsqueeze(-1)
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=token_ids,
            logprobs=logprobs.unsqueeze(-1),
            selected_token_ranks=ranks,
        )
        return SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=logprobs_tensors,
        )

    _GREEDY_SAMPLE_LOGPROBS_ORIGINAL_FORWARD = original_forward
    _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH = can_use_fast_path
    Sampler.forward = forward
    _GREEDY_SAMPLE_LOGPROBS_REGISTERED = True
    return GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY


def register_vllm_selected_token_logprobs() -> str | None:
    """Avoid full-vocabulary raw log-softmax after vLLM 0.24/0.25 sampling.

    vLLM remains responsible for masks, processors, penalties, temperature,
    top-k/top-p, RNG, and token selection. For BF16/FP16 logits requesting
    only the sampled token's raw logprob (`max_num_logprobs == 0`), Loom scans
    the preserved raw logits once after sampling and returns just that token's
    normalized logprob and tie-aware rank. Other contracts execute vLLM's
    original implementation. All-greedy batches retain Loom's narrower fused
    argmax+logprob path.
    """
    global _SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD
    global _SELECTED_TOKEN_LOGPROBS_REGISTERED
    if _SELECTED_TOKEN_LOGPROBS_REGISTERED:
        return SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY
    if register_vllm_greedy_sample_logprobs() is None:
        return None

    if not supports_installed_vllm():
        return None

    from vllm.v1.outputs import LogprobsTensors, SamplerOutput
    from vllm.v1.sample.sampler import Sampler

    load_torch_extension()
    implementation = torch.ops.loom_kernels.selected_token_logprobs.default
    original_forward = Sampler.forward

    def can_use_fast_path(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        logprobs_mode_override: Any,
    ) -> bool:
        logprobs_mode = logprobs_mode_override or sampler.logprobs_mode
        topk_topp_mode = getattr(
            sampler.topk_topp_sampler, "logprobs_mode", sampler.logprobs_mode
        )
        return bool(
            sampler.logprobs_mode == "raw_logprobs"
            and topk_topp_mode == "raw_logprobs"
            and logprobs_mode == "raw_logprobs"
            and sampling_metadata.max_num_logprobs == 0
            and not sampling_metadata.logprob_token_ids
            and logits.device.type == "cuda"
            and logits.dtype in (torch.float16, torch.bfloat16)
            and logits.dim() == 2
            and logits.shape[0] > 0
            and logits.shape[1] > 0
            and logits.shape[1] <= 0x7FFF_FFFF
            and logits.stride(1) == 1
            and logits.stride(0) >= logits.shape[1]
            and not logits.requires_grad
        )

    def forward(
        sampler: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: Any = None,
    ) -> Any:
        global _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT
        global _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION
        if not can_use_fast_path(
            sampler, logits, sampling_metadata, logprobs_mode_override
        ):
            if (
                _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION is None
                and sampling_metadata.max_num_logprobs is not None
                and not sampling_metadata.all_greedy
            ):
                _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION = {
                    "shape": list(logits.shape),
                    "stride": list(logits.stride()),
                    "dtype": str(logits.dtype),
                    "sampler_logprobs_mode": sampler.logprobs_mode,
                    "logprobs_mode": (
                        logprobs_mode_override or sampler.logprobs_mode
                    ),
                    "max_num_logprobs": sampling_metadata.max_num_logprobs,
                    "has_logprob_token_ids": bool(
                        sampling_metadata.logprob_token_ids
                    ),
                    "all_greedy": sampling_metadata.all_greedy,
                    "requires_grad": logits.requires_grad,
                }
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if (
            sampling_metadata.all_greedy
            and _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH is not None
            and _GREEDY_SAMPLE_LOGPROBS_CAN_USE_FAST_PATH(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )
        ):
            return original_forward(
                sampler,
                logits,
                sampling_metadata,
                predict_bonus_token,
                logprobs_mode_override,
            )

        if _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT is None:
            _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT = {
                "shape": list(logits.shape),
                "stride": list(logits.stride()),
                "dtype": str(logits.dtype),
                "max_num_logprobs": sampling_metadata.max_num_logprobs,
                "all_random": sampling_metadata.all_random,
                "has_top_k": sampling_metadata.top_k is not None,
                "has_top_p": sampling_metadata.top_p is not None,
                "no_penalties": sampling_metadata.no_penalties,
                "predict_bonus_token": predict_bonus_token,
            }

        raw_logits = logits
        sampling_logits = logits.to(torch.float32)
        sampling_logits = sampler.apply_logits_processors(
            sampling_logits, sampling_metadata, predict_bonus_token
        )
        sampled, processed_logprobs = sampler.sample(
            sampling_logits, sampling_metadata
        )
        if processed_logprobs is not None:
            raise RuntimeError(
                "vLLM returned processed logprobs under Loom's raw-logprob "
                "selected-token contract"
            )
        sampled = sampled.long().contiguous()
        logprobs, ranks = implementation(raw_logits, sampled)
        sampled = sampled.to(torch.int32)
        sampled_column = sampled.unsqueeze(-1)
        return SamplerOutput(
            sampled_token_ids=sampled_column,
            logprobs_tensors=LogprobsTensors(
                logprob_token_ids=sampled_column,
                logprobs=logprobs.unsqueeze(-1),
                selected_token_ranks=ranks,
            ),
        )

    _SELECTED_TOKEN_LOGPROBS_ORIGINAL_FORWARD = original_forward
    Sampler.forward = forward
    _SELECTED_TOKEN_LOGPROBS_REGISTERED = True
    return SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY


def register_vllm_min_p() -> str | None:
    """Replace vLLM 0.24/0.25 allocating min-p with Loom's in-place kernel."""
    global _MIN_P_ORIGINAL_APPLY
    global _MIN_P_REGISTERED
    if _MIN_P_REGISTERED:
        return MIN_P_OVERRIDE_KEY
    if not torch_extension_available():
        return None

    from .torch_ops import supports_min_p_filter

    if not supports_installed_vllm():
        return None

    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    implementation = torch.ops.loom_kernels.min_p_filter_.default
    original_apply = MinPLogitsProcessor.apply

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits
        if (
            not supports_min_p_filter(logits, self.min_p)
            or logits.shape[0] < MIN_P_FAST_PATH_MIN_ROWS
            or logits.shape[1] < MIN_P_FAST_PATH_MIN_VOCAB_SIZE
        ):
            return original_apply(self, logits)
        implementation(logits, self.min_p)
        return logits

    apply.__module__ = __name__
    _MIN_P_ORIGINAL_APPLY = original_apply
    MinPLogitsProcessor.apply = apply
    _MIN_P_REGISTERED = True
    return MIN_P_OVERRIDE_KEY


def register_vllm_ir(provider: str = DEFAULT_PROVIDER) -> str | None:
    """Register Loom as an in-place fused_add_rms_norm IR provider."""
    if not supports_installed_vllm() or not torch_extension_available():
        return None

    from vllm import ir
    import vllm.ir.ops.layernorm  # noqa: F401 - registers the IR operation

    from .torch_ops import supports_add_rms_norm

    if _silu_override_requested():
        register_vllm_silu_and_mul()
    if _act_quant_override_requested():
        register_vllm_silu_and_mul_dynamic_fp8()
    if _min_p_override_requested():
        register_vllm_min_p()
    if _paged_decode_override_requested():
        register_vllm_paged_decode_attention()

    operation = ir.ops.fused_add_rms_norm
    implementations = getattr(operation, "impls", {})
    if provider in implementations:
        return provider

    def implementation(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if weight is None or variance_size is not None:
            raise ValueError("unsupported Loom Add+RMSNorm contract reached dispatch")
        torch.ops.loom_kernels.add_rms_norm_mut.default(
            x, x_residual, weight, epsilon
        )
        return x, x_residual

    def supports(
        x: torch.Tensor,
        x_residual: torch.Tensor,
        weight: torch.Tensor | None,
        epsilon: float,
        variance_size: int | None = None,
    ) -> bool:
        return supports_add_rms_norm(
            x, x_residual, weight, epsilon, variance_size
        )

    decorator = operation.register_impl(
        provider,
        supported=torch_extension_available(),
        supports_args=supports,
        inplace=True,
    )
    decorator(implementation)
    return provider


def provider_metadata() -> dict[str, Any]:
    return {
        "provider": DEFAULT_PROVIDER,
        "vllm_version": installed_vllm_version(),
        "vllm_supported": supports_installed_vllm(),
        "supported_vllm_series": [
            f"{major}.{minor}" for major, minor in SUPPORTED_VLLM_SERIES
        ],
        "extension_available": torch_extension_available(),
        "operator": "fused_add_rms_norm",
        "inplace": True,
        "silu_and_mul_override_requested": _silu_override_requested(),
        "silu_and_mul_override": _SILU_OVERRIDE_CLASS is not None,
        "silu_and_mul_fp8_override_requested": _act_quant_override_requested(),
        "silu_and_mul_fp8_override": _ACT_QUANT_OVERRIDE_REGISTERED,
        "min_p_override_requested": _min_p_override_requested(),
        "min_p_override": _MIN_P_REGISTERED,
        "min_p_fast_path_min_rows": MIN_P_FAST_PATH_MIN_ROWS,
        "min_p_fast_path_min_vocab_size": MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
        "paged_decode_override_requested": _paged_decode_override_requested(),
        "paged_decode_override": _PAGED_DECODE_REGISTERED,
        "paged_decode_fast_path_max_batch": PAGED_DECODE_FAST_PATH_MAX_BATCH,
        "paged_decode_fast_path_max_context": (
            PAGED_DECODE_FAST_PATH_MAX_CONTEXT
        ),
        "paged_decode_first_contract": _PAGED_DECODE_FIRST_CONTRACT,
        "paged_decode_first_rejection": _PAGED_DECODE_FIRST_REJECTION,
        "rope_paged_kv_override": _ROPE_PAGED_KV_REGISTERED,
        "rope_paged_kv_first_contract": _ROPE_PAGED_KV_FIRST_CONTRACT,
        "greedy_sample_logprobs_override": _GREEDY_SAMPLE_LOGPROBS_REGISTERED,
        "greedy_sample_logprobs_first_contract": (
            _GREEDY_SAMPLE_LOGPROBS_FIRST_CONTRACT
        ),
        "greedy_sample_logprobs_first_rejection": (
            _GREEDY_SAMPLE_LOGPROBS_FIRST_REJECTION
        ),
        "selected_token_logprobs_override": _SELECTED_TOKEN_LOGPROBS_REGISTERED,
        "selected_token_logprobs_first_contract": (
            _SELECTED_TOKEN_LOGPROBS_FIRST_CONTRACT
        ),
        "selected_token_logprobs_first_rejection": (
            _SELECTED_TOKEN_LOGPROBS_FIRST_REJECTION
        ),
    }


__all__ = [
    "ACT_QUANT_OVERRIDE_ENV",
    "ACT_QUANT_OVERRIDE_KEY",
    "DEFAULT_PROVIDER",
    "GREEDY_SAMPLE_LOGPROBS_OVERRIDE_KEY",
    "MIN_P_FAST_PATH_MIN_ROWS",
    "MIN_P_FAST_PATH_MIN_VOCAB_SIZE",
    "MIN_P_OVERRIDE_ENV",
    "MIN_P_OVERRIDE_KEY",
    "PAGED_DECODE_FAST_PATH_MAX_BATCH",
    "PAGED_DECODE_FAST_PATH_MAX_CONTEXT",
    "PAGED_DECODE_OVERRIDE_ENV",
    "PAGED_DECODE_OVERRIDE_KEY",
    "ROPE_PAGED_KV_OVERRIDE_KEY",
    "SELECTED_TOKEN_LOGPROBS_OVERRIDE_KEY",
    "SILU_OVERRIDE_ENV",
    "SILU_OVERRIDE_KEY",
    "SUPPORTED_VLLM_SERIES",
    "configure_vllm_rope_paged_kv",
    "installed_vllm_version",
    "provider_metadata",
    "register_vllm_ir",
    "register_vllm_min_p",
    "register_vllm_paged_decode_attention",
    "register_vllm_greedy_sample_logprobs",
    "register_vllm_rope_paged_kv",
    "register_vllm_selected_token_logprobs",
    "register_vllm_silu_and_mul",
    "register_vllm_silu_and_mul_dynamic_fp8",
    "supports_installed_vllm",
    "supports_vllm_paged_decode_shape",
]
