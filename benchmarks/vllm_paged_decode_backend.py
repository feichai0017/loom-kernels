#!/usr/bin/env python3
"""Benchmark Loom's shape-gated vLLM FlashAttention backend override."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from loom_kernels.torch_ops import (
    Operator,
    bridge_abi_version,
    launch_count,
    reset_launch_count,
)
from loom_kernels.vllm import (
    PAGED_DECODE_OVERRIDE_KEY,
    register_vllm_paged_decode_attention,
    supports_vllm_paged_decode_shape,
)
from vllm_paged_decode_attention import DTYPES, measure_pair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--contexts", default="16,32,64")
    parser.add_argument("--dtypes", default="bf16,f16")
    parser.add_argument("--block-sizes", default="16,32")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2719)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def positive_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or min(values) <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return values


def benchmark_case(
    *,
    baseline_forward: Callable[..., torch.Tensor],
    loom_forward: Callable[..., torch.Tensor],
    attention: Any,
    metadata_class: type,
    dtype_name: str,
    batch: int,
    context: int,
    block_size: int,
    warmup: int,
    iterations: int,
    samples: int,
    seed: int,
) -> dict[str, object]:
    dtype = DTYPES[dtype_name]
    torch.manual_seed(seed + batch * 1009 + context * 17 + block_size)
    max_blocks = math.ceil(context / block_size)
    num_blocks = batch * max_blocks
    query = torch.randn((batch, 32, 128), device="cuda", dtype=dtype)
    key = torch.empty((batch, 8, 128), device="cuda", dtype=dtype)
    value = torch.empty_like(key)
    kv_cache = torch.randn(
        (num_blocks, 2, block_size, 8, 128), device="cuda", dtype=dtype
    )
    block_table = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
    block_table = block_table.reshape(batch, max_blocks).to(torch.int32)
    sequence_lengths = torch.tensor(
        [max(1, context - (index * 7) % max(1, 2 * block_size)) for index in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    sequence_lengths[0] = context
    metadata = metadata_class(
        num_actual_tokens=batch,
        max_query_len=1,
        query_start_loc=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        max_seq_len=context,
        seq_lens=sequence_lengths,
        block_table=block_table,
        slot_mapping=torch.arange(batch, device="cuda", dtype=torch.int64),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    layer = SimpleNamespace(_q_scale=scale, _k_scale=scale, _v_scale=scale)
    loom_output = torch.empty_like(query)
    baseline_output = torch.empty_like(query)

    def loom() -> torch.Tensor:
        return loom_forward(
            attention,
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            loom_output,
        )

    def baseline() -> torch.Tensor:
        return baseline_forward(
            attention,
            layer,
            query,
            key,
            value,
            kv_cache,
            metadata,
            baseline_output,
        )

    baseline()
    reset_launch_count(Operator.PAGED_DECODE_ATTENTION)
    loom()
    torch.cuda.synchronize()
    used_loom = launch_count(Operator.PAGED_DECODE_ATTENTION) == 1
    torch.testing.assert_close(
        loom_output, baseline_output, rtol=2.0e-2, atol=2.0e-2
    )

    loom_measurement, baseline_measurement = measure_pair(
        loom, baseline, warmup, iterations, samples
    )
    loom_eager = loom_measurement["eager_dispatch_latency"]["median_us"]
    baseline_eager = baseline_measurement["eager_dispatch_latency"]["median_us"]
    loom_graph = loom_measurement["cuda_graph_replay_latency"]["median_us"]
    baseline_graph = baseline_measurement["cuda_graph_replay_latency"]["median_us"]
    expected_fast_path = supports_vllm_paged_decode_shape(
        dtype=dtype,
        batch=batch,
        query_heads=32,
        kv_heads=8,
        head_size=128,
        block_size=block_size,
        max_sequence_length=context,
    )
    if used_loom != expected_fast_path:
        raise RuntimeError(
            f"route mismatch: expected_fast_path={expected_fast_path}, "
            f"used_loom={used_loom}"
        )
    return {
        "dtype": dtype_name,
        "batch": batch,
        "context": context,
        "block_size": block_size,
        "expected_fast_path": expected_fast_path,
        "used_loom": used_loom,
        "loom_route": loom_measurement,
        "vllm_flash_attention": baseline_measurement,
        "loom_eager_speedup": baseline_eager / loom_eager,
        "loom_cuda_graph_speedup": baseline_graph / loom_graph,
        "max_abs_error": float(
            (loom_output.float() - baseline_output.float()).abs().max().item()
        ),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.warmup, args.iterations, args.samples) <= 0:
        raise ValueError("warmup, iterations, and samples must be positive")
    batches = positive_ints(args.batches, "batches")
    contexts = positive_ints(args.contexts, "contexts")
    block_sizes = positive_ints(args.block_sizes, "block-sizes")
    dtype_names = [item for item in args.dtypes.split(",") if item]
    if not dtype_names or any(item not in DTYPES for item in dtype_names):
        raise ValueError(f"dtypes must use {sorted(DTYPES)}")

    import vllm
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )

    baseline_forward = FlashAttentionImpl.forward
    if register_vllm_paged_decode_attention() != PAGED_DECODE_OVERRIDE_KEY:
        raise RuntimeError("Loom vLLM paged-decode registration is unavailable")
    loom_forward = FlashAttentionImpl.forward
    attention = FlashAttentionImpl(
        num_heads=32,
        head_size=128,
        scale=128**-0.5,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    results = [
        benchmark_case(
            baseline_forward=baseline_forward,
            loom_forward=loom_forward,
            attention=attention,
            metadata_class=FlashAttentionMetadata,
            dtype_name=dtype_name,
            batch=batch,
            context=context,
            block_size=block_size,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            seed=args.seed,
        )
        for dtype_name in dtype_names
        for block_size in block_sizes
        for batch in batches
        for context in contexts
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "vLLM FlashAttention shape-gated paged decode backend",
        "candidate": "Loom register_vllm_paged_decode_attention",
        "baseline": f"vLLM {vllm.__version__} FlashAttentionImpl.forward",
        "scope": {
            "query_heads": 32,
            "kv_heads": 8,
            "head_size": 128,
            "dtypes": dtype_names,
            "block_sizes": block_sizes,
            "batches": batches,
            "contexts": contexts,
            "cache_storage": "vllm-interleaved",
            "tested_cases": len(results),
        },
        "timing": {
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "samples": args.samples,
            "provider_order": "alternated for every eager and graph sample",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi_version": bridge_abi_version(),
        },
        "acceptance": {
            "passed": True,
            "rtol": 2.0e-2,
            "atol": 2.0e-2,
            "route_matches_gate": True,
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
