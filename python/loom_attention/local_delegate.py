"""Process-local validation and telemetry for delegated attention calls.

This module has no torch or vLLM dependency. It inspects tensor-like objects by
their public shape/device/dtype attributes, so the contract can be tested in CI
without installing a GPU framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from threading import Lock
from time import perf_counter_ns
from typing import Any


class TensorContractError(RuntimeError):
    """Raised before delegation when an engine tensor violates the layout."""


@dataclass(frozen=True)
class ForwardToken:
    call_id: int
    started_ns: int


@dataclass(frozen=True)
class ForwardSnapshot:
    calls: int
    failures: int
    total_ns: int
    layout_digest: str
    last_device: str | None


class LocalForwardObserver:
    """Validate a local attention call without entering a remote control path."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None,
        kv_cache_dtype: str,
        attention_type: str,
        validate_every_call: bool = True,
    ) -> None:
        if num_heads <= 0 or head_size <= 0:
            raise ValueError("num_heads and head_size must be positive")
        resolved_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if resolved_kv_heads <= 0 or num_heads % resolved_kv_heads != 0:
            raise ValueError("num_kv_heads must be positive and divide num_heads")

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = resolved_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attention_type = attention_type
        self.validate_every_call = validate_every_call
        layout = {
            "attention_type": attention_type,
            "head_size": head_size,
            "kv_cache_dtype": kv_cache_dtype,
            "num_heads": num_heads,
            "num_kv_heads": resolved_kv_heads,
        }
        encoded = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        self.layout_digest = sha256(encoded).hexdigest()

        self._lock = Lock()
        self._next_call_id = 1
        self._calls = 0
        self._failures = 0
        self._total_ns = 0
        self._last_device: str | None = None

    def before_forward(
        self,
        *,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        output: Any,
    ) -> ForwardToken:
        if self.validate_every_call or self._last_device is None:
            device = self._validate_tensors(
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                output=output,
            )
        else:
            device = self._last_device

        with self._lock:
            token = ForwardToken(self._next_call_id, perf_counter_ns())
            self._next_call_id += 1
            self._last_device = device
        return token

    def _validate_tensors(
        self,
        *,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        output: Any,
    ) -> str:
        tensors = {
            "query": query,
            "key": key,
            "value": value,
            "kv_cache": kv_cache,
            "output": output,
        }
        for name, tensor in tensors.items():
            self._require_tensor(name, tensor)

        self._require_width("query", query, self.num_heads)
        self._require_width("key", key, self.num_kv_heads)
        self._require_width("value", value, self.num_kv_heads)
        self._require_width("output", output, self.num_heads)

        devices = {
            self._device(tensor)
            for tensor in tensors.values()
            if self._numel(tensor) > 0
        }
        if len(devices) != 1:
            raise TensorContractError(
                f"local attention tensors must share one device, got {sorted(devices)}"
            )
        return next(iter(devices))

    def after_forward(self, token: ForwardToken, *, failed: bool = False) -> None:
        elapsed = perf_counter_ns() - token.started_ns
        with self._lock:
            self._calls += 1
            self._failures += int(failed)
            self._total_ns += elapsed

    def snapshot(self) -> ForwardSnapshot:
        with self._lock:
            return ForwardSnapshot(
                calls=self._calls,
                failures=self._failures,
                total_ns=self._total_ns,
                layout_digest=self.layout_digest,
                last_device=self._last_device,
            )

    def _require_width(self, name: str, tensor: Any, heads: int) -> None:
        shape = tuple(int(dimension) for dimension in tensor.shape)
        accepted = {self.head_size, heads * self.head_size}
        if not shape or shape[-1] not in accepted:
            raise TensorContractError(
                f"{name} last dimension {shape[-1] if shape else None} "
                f"does not match head_size/head_count layout {sorted(accepted)}"
            )
        if len(shape) >= 3 and shape[-1] == self.head_size and shape[-2] != heads:
            raise TensorContractError(
                f"{name} head dimension {shape[-2]} does not match expected {heads}"
            )

    @staticmethod
    def _require_tensor(name: str, tensor: Any) -> None:
        missing = [
            attribute
            for attribute in ("shape", "device", "dtype")
            if not hasattr(tensor, attribute)
        ]
        if missing:
            raise TensorContractError(
                f"{name} is missing tensor attributes: {', '.join(missing)}"
            )

    @staticmethod
    def _device(tensor: Any) -> str:
        return str(tensor.device)

    @staticmethod
    def _numel(tensor: Any) -> int:
        numel = getattr(tensor, "numel", None)
        if callable(numel):
            return int(numel())
        product = 1
        for dimension in tensor.shape:
            product *= int(dimension)
        return product
