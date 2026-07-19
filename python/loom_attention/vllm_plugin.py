"""Out-of-tree vLLM attention backend that delegates to FlashAttention.

The plugin registers `AttentionBackendEnum.CUSTOM`. M1 keeps all tensor math in
vLLM's native FlashAttention implementation while Loom validates the
local contract and records forward telemetry. Remote execution is introduced by
later backends, not by changing this delegate's behavior.
"""

from __future__ import annotations

import os
from typing import Any

from .local_delegate import LocalForwardObserver
from .step_metadata import StepMetadataObserver, StepMetadataSnapshot

_REGISTERED = False
_STEP_SNAPSHOT_ATTRIBUTE = "loom_step_snapshot"


def register() -> None:
    """Register the Loom backend with vLLM's documented OOT registry."""

    global _REGISTERED
    global LoomFlashAttentionBackend
    global LoomFlashAttentionImpl
    global LoomFlashAttentionMetadataBuilder

    if _REGISTERED:
        return
    delegate = os.environ.get("LOOM_VLLM_DELEGATE", "flash_attn")
    if delegate != "flash_attn":
        raise RuntimeError(
            "M1 supports only LOOM_VLLM_DELEGATE=flash_attn; "
            f"got {delegate!r}"
        )

    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionBackend,
            FlashAttentionImpl,
            FlashAttentionMetadataBuilder,
        )
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
    except ImportError as error:
        raise RuntimeError(
            "Loom's vLLM plugin requires vLLM 0.25.x with the V1 "
            "attention backend registry"
        ) from error

    class _LoomFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._loom_step_observer = _step_observer_from_builder(
                self, args, kwargs
            )

        @property
        def loom_step_observer(self) -> StepMetadataObserver:
            return self._loom_step_observer

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: Any,
            fast_build: bool = False,
        ) -> Any:
            metadata = super().build(
                common_prefix_len,
                common_attn_metadata,
                fast_build=fast_build,
            )
            snapshot = self._loom_step_observer.capture(
                common_prefix_tokens=int(common_prefix_len),
                common_metadata=common_attn_metadata,
                fast_build=bool(fast_build),
            )
            setattr(metadata, _STEP_SNAPSHOT_ATTRIBUTE, snapshot)
            return metadata

        def update_block_table(
            self,
            metadata: Any,
            blk_table: Any,
            slot_mapping: Any,
        ) -> Any:
            updated = super().update_block_table(metadata, blk_table, slot_mapping)
            previous = getattr(metadata, _STEP_SNAPSHOT_ATTRIBUTE, None)
            if not isinstance(previous, StepMetadataSnapshot):
                raise RuntimeError(
                    "Loom metadata is missing its node-local step snapshot"
                )
            snapshot = self._loom_step_observer.update_block_table(
                previous,
                block_table=blk_table,
                slot_mapping=slot_mapping,
            )
            setattr(updated, _STEP_SNAPSHOT_ATTRIBUTE, snapshot)
            return updated

    class _LoomFlashAttentionImpl(FlashAttentionImpl):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._loom_observer = _observer_from_init(args, kwargs)

        @property
        def loom_observer(self) -> LocalForwardObserver:
            return self._loom_observer

        def forward(
            self,
            layer: Any,
            query: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            attn_metadata: Any,
            output: Any,
            output_scale: Any = None,
            output_block_scale: Any = None,
        ) -> Any:
            token = self._loom_observer.before_forward(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                output=output,
            )
            try:
                result = super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale=output_scale,
                    output_block_scale=output_block_scale,
                )
            except BaseException:
                self._loom_observer.after_forward(token, failed=True)
                raise
            self._loom_observer.after_forward(token)
            return result

    class _LoomFlashAttentionBackend(FlashAttentionBackend):
        @staticmethod
        def get_name() -> str:
            return "LOOM_FLASH_ATTN"

        @staticmethod
        def get_impl_cls() -> type[_LoomFlashAttentionImpl]:
            return _LoomFlashAttentionImpl

        @staticmethod
        def get_builder_cls() -> type[_LoomFlashAttentionMetadataBuilder]:
            return _LoomFlashAttentionMetadataBuilder

    # vLLM resolves registered classes by module-qualified name. Publish the
    # dynamic subclasses under stable names before updating the registry.
    _LoomFlashAttentionImpl.__name__ = "LoomFlashAttentionImpl"
    _LoomFlashAttentionImpl.__qualname__ = "LoomFlashAttentionImpl"
    _LoomFlashAttentionImpl.__module__ = __name__
    LoomFlashAttentionImpl = _LoomFlashAttentionImpl

    _LoomFlashAttentionMetadataBuilder.__name__ = (
        "LoomFlashAttentionMetadataBuilder"
    )
    _LoomFlashAttentionMetadataBuilder.__qualname__ = (
        "LoomFlashAttentionMetadataBuilder"
    )
    _LoomFlashAttentionMetadataBuilder.__module__ = __name__
    LoomFlashAttentionMetadataBuilder = (
        _LoomFlashAttentionMetadataBuilder
    )

    _LoomFlashAttentionBackend.__name__ = "LoomFlashAttentionBackend"
    _LoomFlashAttentionBackend.__qualname__ = "LoomFlashAttentionBackend"
    _LoomFlashAttentionBackend.__module__ = __name__
    LoomFlashAttentionBackend = _LoomFlashAttentionBackend

    register_backend(
        AttentionBackendEnum.CUSTOM,
        class_path=f"{__name__}.LoomFlashAttentionBackend",
    )
    _REGISTERED = True


def _step_observer_from_builder(
    builder: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> StepMetadataObserver:
    if "layer_names" in kwargs:
        layer_names = kwargs["layer_names"]
    elif len(args) > 1:
        layer_names = args[1]
    else:
        raise RuntimeError("vLLM did not provide layer_names to the metadata builder")
    return StepMetadataObserver(
        layer_names=tuple(str(name) for name in layer_names),
        block_size=int(builder.block_size),
        num_attention_heads=int(builder.num_heads_q),
        num_kv_heads=int(builder.num_heads_kv),
        head_size=int(builder.headdim),
        kv_cache_dtype=str(builder.kv_cache_dtype),
    )


def _observer_from_init(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> LocalForwardObserver:
    def argument(name: str, index: int, default: Any = None) -> Any:
        if name in kwargs:
            return kwargs[name]
        if index < len(args):
            return args[index]
        return default

    num_heads = argument("num_heads", 0)
    head_size = argument("head_size", 1)
    if num_heads is None or head_size is None:
        raise RuntimeError(
            "vLLM did not provide num_heads/head_size to the attention implementation"
        )
    return LocalForwardObserver(
        num_heads=int(num_heads),
        head_size=int(head_size),
        num_kv_heads=argument("num_kv_heads", 3),
        kv_cache_dtype=str(argument("kv_cache_dtype", 6, "auto")),
        attention_type=str(argument("attn_type", 8, "decoder")),
        validate_every_call=_boolean_environment(
            "LOOM_VALIDATE_EVERY_FORWARD", default=False
        ),
    )


def _boolean_environment(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value, got {value!r}")
