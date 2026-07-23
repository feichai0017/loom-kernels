"""vLLM fused RoPE and paged-KV registration."""

from __future__ import annotations

from typing import Any

import torch

from .._torch_extension import load_torch_extension, torch_extension_available
from ._runtime import supports_installed_vllm

ROPE_PAGED_KV_OVERRIDE_KEY = "rope_paged_kv"

_ROPE_PAGED_KV_REGISTERED = False
_ROPE_PAGED_KV_FIRST_CONTRACT: dict[str, Any] | None = None


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
    fp8_cache_dtypes = {"fp8", "fp8_e4m3"}
    supported_cache_dtypes = native_cache_dtypes | fp8_cache_dtypes

    def supported(attention: Any) -> bool:
        # FlashInfer rejects non-decoder construction but does not retain an
        # attn_type field; FlashAttention retains the explicit enum.
        attention_type = getattr(attention, "attn_type", AttentionType.DECODER)
        return bool(
            getattr(attention, "kv_sharing_target_layer_name", None) is None
            and getattr(attention, "kv_cache_dtype", None)
            in supported_cache_dtypes
            and attention_type == AttentionType.DECODER
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
        del attention
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
                "key_scales": {
                    "shape": list(layer._k_scale.shape),
                    "dtype": str(layer._k_scale.dtype),
                },
                "value_scales": {
                    "shape": list(layer._v_scale.shape),
                    "dtype": str(layer._v_scale.dtype),
                },
                "is_neox": is_neox,
            }
        implementation(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            kv_cache,
            layer._k_scale,
            layer._v_scale,
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

    # vLLM's FP8 FlashAttention path quantizes Q after RoPE. Keep both
    # operations opaque so its official RopeStaticQQuantKVCachePattern can
    # match RoPE + static-Q quant + the paged-KV update as one unit.
    for custom_op in ("+rotary_embedding", "+quant_fp8"):
        if custom_op not in configured.custom_ops:
            configured.custom_ops.append(custom_op)
    if configured.splitting_ops is None:
        configured.splitting_ops = []
    configured.pass_config.fuse_rope_kvcache = True
    configured.pass_config.rope_kvcache_fusion_max_token_num = max_token_num
    return configured


def _metadata() -> dict[str, object]:
    return {
        "rope_paged_kv_override": _ROPE_PAGED_KV_REGISTERED,
        "rope_paged_kv_first_contract": _ROPE_PAGED_KV_FIRST_CONTRACT,
    }
