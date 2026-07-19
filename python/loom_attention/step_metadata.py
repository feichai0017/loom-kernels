"""Zero-copy description of one vLLM paged-attention step.

The metadata builder may inspect the CPU query offsets already maintained by
vLLM. Device tensors stay opaque: only public shape, dtype, device, and data
pointer attributes are captured, so observation cannot introduce a GPU-to-CPU
synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from threading import Lock
from typing import Any


class StepMetadataContractError(RuntimeError):
    """Raised when engine metadata cannot form a safe node-local step view."""


@dataclass(frozen=True)
class TensorDescriptor:
    shape: tuple[int, ...]
    device: str
    dtype: str
    data_ptr: int
    numel: int
    element_size: int
    bytes: int


@dataclass(frozen=True)
class StepMetadataSnapshot:
    generation: int
    layout_digest: str
    layer_names: tuple[str, ...]
    block_size: int
    request_count: int
    query_tokens: int
    num_actual_tokens: int
    max_query_tokens: int
    max_sequence_tokens: int
    common_prefix_tokens: int
    fast_build: bool
    query_start_offsets: tuple[int, ...]
    query_start_locations: TensorDescriptor
    sequence_lengths: TensorDescriptor
    block_table: TensorDescriptor
    slot_mapping: TensorDescriptor


class StepMetadataObserver:
    """Build generation-checked snapshots outside the per-layer forward path."""

    def __init__(
        self,
        *,
        layer_names: tuple[str, ...],
        block_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_size: int,
        kv_cache_dtype: str,
    ) -> None:
        if not layer_names or any(not name for name in layer_names):
            raise ValueError("layer_names must contain at least one non-empty name")
        if block_size <= 0 or head_size <= 0:
            raise ValueError("block_size and head_size must be positive")
        if (
            num_attention_heads <= 0
            or num_kv_heads <= 0
            or num_attention_heads % num_kv_heads != 0
        ):
            raise ValueError(
                "num_kv_heads must be positive and divide num_attention_heads"
            )

        self.layer_names = layer_names
        self.block_size = block_size
        layout = {
            "block_size": block_size,
            "head_size": head_size,
            "kv_cache_dtype": kv_cache_dtype,
            "layer_names": layer_names,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
        }
        encoded = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()
        self.layout_digest = sha256(encoded).hexdigest()
        self._lock = Lock()
        self._next_generation = 1

    def capture(
        self,
        *,
        common_prefix_tokens: int,
        common_metadata: Any,
        fast_build: bool,
    ) -> StepMetadataSnapshot:
        request_count = self._integer_attribute(common_metadata, "num_reqs")
        num_actual_tokens = self._integer_attribute(
            common_metadata, "num_actual_tokens"
        )
        max_query_tokens = self._integer_attribute(common_metadata, "max_query_len")
        max_sequence_tokens = self._integer_attribute(
            common_metadata, "max_seq_len"
        )
        query_start_offsets = self._cpu_int_values(
            "query_start_loc_cpu", common_metadata.query_start_loc_cpu
        )
        query_start_locations = self._tensor_descriptor(
            "query_start_loc", common_metadata.query_start_loc
        )
        sequence_lengths = self._tensor_descriptor(
            "seq_lens", common_metadata.seq_lens
        )
        block_table = self._tensor_descriptor(
            "block_table_tensor", common_metadata.block_table_tensor
        )
        slot_mapping = self._tensor_descriptor(
            "slot_mapping", common_metadata.slot_mapping
        )
        for name, descriptor in (
            ("query_start_loc", query_start_locations),
            ("seq_lens", sequence_lengths),
            ("block_table_tensor", block_table),
            ("slot_mapping", slot_mapping),
        ):
            self._require_integer_tensor(name, descriptor)

        query_tokens = self._validate_common_metadata(
            request_count=request_count,
            num_actual_tokens=num_actual_tokens,
            max_query_tokens=max_query_tokens,
            max_sequence_tokens=max_sequence_tokens,
            common_prefix_tokens=common_prefix_tokens,
            query_start_offsets=query_start_offsets,
            query_start_locations=query_start_locations,
            sequence_lengths=sequence_lengths,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )
        return StepMetadataSnapshot(
            generation=self._allocate_generation(),
            layout_digest=self.layout_digest,
            layer_names=self.layer_names,
            block_size=self.block_size,
            request_count=request_count,
            query_tokens=query_tokens,
            num_actual_tokens=num_actual_tokens,
            max_query_tokens=max_query_tokens,
            max_sequence_tokens=max_sequence_tokens,
            common_prefix_tokens=common_prefix_tokens,
            fast_build=fast_build,
            query_start_offsets=query_start_offsets,
            query_start_locations=query_start_locations,
            sequence_lengths=sequence_lengths,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )

    def update_block_table(
        self,
        previous: StepMetadataSnapshot,
        *,
        block_table: Any,
        slot_mapping: Any,
    ) -> StepMetadataSnapshot:
        if previous.layout_digest != self.layout_digest:
            raise StepMetadataContractError(
                "cannot update block table across different KV layouts"
            )
        block_descriptor = self._tensor_descriptor("block_table", block_table)
        slot_descriptor = self._tensor_descriptor("slot_mapping", slot_mapping)
        self._require_integer_tensor("block_table", block_descriptor)
        self._require_integer_tensor("slot_mapping", slot_descriptor)
        self._validate_table_shapes(
            request_count=previous.request_count,
            query_tokens=previous.query_tokens,
            block_table=block_descriptor,
            slot_mapping=slot_descriptor,
        )
        if block_descriptor.device != previous.sequence_lengths.device:
            raise StepMetadataContractError(
                "updated block table must stay on the attention metadata device"
            )
        return replace(
            previous,
            generation=self._allocate_generation(),
            block_table=block_descriptor,
            slot_mapping=slot_descriptor,
        )

    def _allocate_generation(self) -> int:
        with self._lock:
            generation = self._next_generation
            self._next_generation += 1
        return generation

    def _validate_common_metadata(
        self,
        *,
        request_count: int,
        num_actual_tokens: int,
        max_query_tokens: int,
        max_sequence_tokens: int,
        common_prefix_tokens: int,
        query_start_offsets: tuple[int, ...],
        query_start_locations: TensorDescriptor,
        sequence_lengths: TensorDescriptor,
        block_table: TensorDescriptor,
        slot_mapping: TensorDescriptor,
    ) -> int:
        scalar_values = {
            "request_count": request_count,
            "num_actual_tokens": num_actual_tokens,
            "max_query_tokens": max_query_tokens,
            "max_sequence_tokens": max_sequence_tokens,
            "common_prefix_tokens": common_prefix_tokens,
        }
        negative = [name for name, value in scalar_values.items() if value < 0]
        if negative:
            raise StepMetadataContractError(
                f"step metadata values must be non-negative: {', '.join(negative)}"
            )
        if len(query_start_offsets) != request_count + 1:
            raise StepMetadataContractError(
                "query_start_loc_cpu length must equal request_count + 1"
            )
        if not query_start_offsets or query_start_offsets[0] != 0:
            raise StepMetadataContractError("query_start_loc_cpu must start at zero")
        if any(
            left > right
            for left, right in zip(query_start_offsets, query_start_offsets[1:])
        ):
            raise StepMetadataContractError(
                "query_start_loc_cpu offsets must be monotonically increasing"
            )

        query_tokens = query_start_offsets[-1]
        if query_tokens > num_actual_tokens:
            raise StepMetadataContractError(
                "query offsets exceed the number of actual or padded tokens"
            )
        query_lengths = [
            right - left
            for left, right in zip(query_start_offsets, query_start_offsets[1:])
        ]
        if query_lengths and max(query_lengths) > max_query_tokens:
            raise StepMetadataContractError(
                "max_query_len is smaller than a request query length"
            )
        if common_prefix_tokens > max_sequence_tokens:
            raise StepMetadataContractError(
                "common prefix length exceeds the maximum sequence length"
            )

        self._require_vector(
            "query_start_loc", query_start_locations, request_count + 1
        )
        self._require_vector("seq_lens", sequence_lengths, request_count)
        self._validate_table_shapes(
            request_count=request_count,
            query_tokens=query_tokens,
            block_table=block_table,
            slot_mapping=slot_mapping,
        )
        devices = {
            query_start_locations.device,
            sequence_lengths.device,
            block_table.device,
            slot_mapping.device,
        }
        if len(devices) != 1:
            raise StepMetadataContractError(
                f"paged-attention metadata tensors must share one device, got {sorted(devices)}"
            )
        return query_tokens

    def _validate_table_shapes(
        self,
        *,
        request_count: int,
        query_tokens: int,
        block_table: TensorDescriptor,
        slot_mapping: TensorDescriptor,
    ) -> None:
        if len(block_table.shape) != 2 or block_table.shape[0] < request_count:
            raise StepMetadataContractError(
                "block table must be rank two with at least one row per request"
            )
        if request_count > 0 and block_table.shape[1] == 0:
            raise StepMetadataContractError(
                "block table must expose at least one block column"
            )
        self._require_vector("slot_mapping", slot_mapping, query_tokens)
        if block_table.device != slot_mapping.device:
            raise StepMetadataContractError(
                "block table and slot mapping must share one device"
            )

    @staticmethod
    def _require_vector(
        name: str, descriptor: TensorDescriptor, minimum_items: int
    ) -> None:
        if len(descriptor.shape) != 1 or descriptor.numel < minimum_items:
            raise StepMetadataContractError(
                f"{name} must be a vector with at least {minimum_items} items"
            )

    @staticmethod
    def _require_integer_tensor(name: str, descriptor: TensorDescriptor) -> None:
        if "int32" not in descriptor.dtype and "int64" not in descriptor.dtype:
            raise StepMetadataContractError(
                f"{name} must use an int32 or int64 dtype, got {descriptor.dtype}"
            )

    @staticmethod
    def _integer_attribute(value: Any, name: str) -> int:
        if not hasattr(value, name):
            raise StepMetadataContractError(f"common metadata is missing {name}")
        try:
            return int(getattr(value, name))
        except (TypeError, ValueError) as error:
            raise StepMetadataContractError(
                f"common metadata {name} must be an integer"
            ) from error

    @classmethod
    def _cpu_int_values(cls, name: str, tensor: Any) -> tuple[int, ...]:
        cls._require_tensor_attributes(name, tensor)
        if not str(tensor.device).startswith("cpu"):
            raise StepMetadataContractError(
                f"{name} must already be on CPU; refusing an implicit device sync"
            )
        values = getattr(tensor, "tolist", None)
        if not callable(values):
            raise StepMetadataContractError(f"{name} does not provide tolist()")
        raw_values = values()
        if not isinstance(raw_values, (list, tuple)) or any(
            isinstance(item, (list, tuple)) for item in raw_values
        ):
            raise StepMetadataContractError(f"{name} must be a one-dimensional vector")
        return tuple(int(item) for item in raw_values)

    @classmethod
    def _tensor_descriptor(cls, name: str, tensor: Any) -> TensorDescriptor:
        cls._require_tensor_attributes(name, tensor)
        shape = tuple(int(dimension) for dimension in tensor.shape)
        numel = cls._numel(tensor, shape)
        data_ptr_method = getattr(tensor, "data_ptr", None)
        if not callable(data_ptr_method):
            raise StepMetadataContractError(f"{name} does not provide data_ptr()")
        data_ptr = int(data_ptr_method())
        if numel > 0 and data_ptr <= 0:
            raise StepMetadataContractError(
                f"{name} has no live storage pointer for a non-empty tensor"
            )
        contiguous_method = getattr(tensor, "is_contiguous", None)
        if not callable(contiguous_method) or not bool(contiguous_method()):
            raise StepMetadataContractError(
                f"{name} must be contiguous before it can become a tensor handle"
            )
        element_size_method = getattr(tensor, "element_size", None)
        if not callable(element_size_method):
            raise StepMetadataContractError(f"{name} does not provide element_size()")
        element_size = int(element_size_method())
        if element_size <= 0:
            raise StepMetadataContractError(f"{name} has an invalid element size")
        return TensorDescriptor(
            shape=shape,
            device=str(tensor.device),
            dtype=str(tensor.dtype),
            data_ptr=data_ptr,
            numel=numel,
            element_size=element_size,
            bytes=numel * element_size,
        )

    @staticmethod
    def _require_tensor_attributes(name: str, tensor: Any) -> None:
        missing = [
            attribute
            for attribute in ("shape", "device", "dtype")
            if not hasattr(tensor, attribute)
        ]
        if missing:
            raise StepMetadataContractError(
                f"{name} is missing tensor attributes: {', '.join(missing)}"
            )

    @staticmethod
    def _numel(tensor: Any, shape: tuple[int, ...]) -> int:
        method = getattr(tensor, "numel", None)
        if callable(method):
            return int(method())
        product = 1
        for dimension in shape:
            product *= dimension
        return product
