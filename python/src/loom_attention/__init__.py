"""Engine integration and distributed-attention utilities for Loom."""

from .block_binding import (
    BindingStep,
    BlockBindingContractError,
    BlockBindingRegistry,
    BlockBindingSnapshot,
    CacheTensorDescriptor,
    ExternalBlockBinding,
    PoolObjectRef,
    ReadLeaseProof,
    RequestBlockState,
    RequestBlockUpdate,
    binding_telemetry_snapshot,
    registry_for_engine,
)

from .local_delegate import LocalForwardObserver, TensorContractError
from .cuda_ops import fused_tail_attention_merge, load_cuda_extension
from .paged_executor import (
    FlashInferPagedExecutor,
    PagedKvContractError,
    PagedKvView,
)
from .step_metadata import (
    StepMetadataContractError,
    StepMetadataObserver,
    StepMetadataSnapshot,
    TensorDescriptor,
)

__all__ = [
    "BindingStep",
    "BlockBindingContractError",
    "BlockBindingRegistry",
    "BlockBindingSnapshot",
    "CacheTensorDescriptor",
    "ExternalBlockBinding",
    "fused_tail_attention_merge",
    "LocalForwardObserver",
    "load_cuda_extension",
    "FlashInferPagedExecutor",
    "PagedKvContractError",
    "PagedKvView",
    "PoolObjectRef",
    "ReadLeaseProof",
    "RequestBlockState",
    "RequestBlockUpdate",
    "StepMetadataContractError",
    "StepMetadataObserver",
    "StepMetadataSnapshot",
    "TensorContractError",
    "TensorDescriptor",
    "binding_telemetry_snapshot",
    "registry_for_engine",
]
