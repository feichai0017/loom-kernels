#!/usr/bin/env python3
"""Compare Loom fused RoPE+paged-KV with vLLM's two native CUDA ops."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import bridge_abi_version, rope_paged_kv_write_


DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--layouts", default="NHD,HND")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--query-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-position", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_cos_sin_cache(
    max_position: int, rotary_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    inverse_frequency = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda")
            / rotary_dim
        )
    )
    positions = torch.arange(max_position, dtype=torch.float32, device="cuda")
    frequencies = torch.outer(positions, inverse_frequency)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).to(dtype)


def make_cache(
    num_blocks: int,
    block_size: int,
    kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (num_blocks, 2, block_size, kv_heads, head_size)
    if layout == "NHD":
        combined = torch.empty(shape, device="cuda", dtype=dtype)
    elif layout == "HND":
        combined = torch.empty_strided(
            shape,
            (
                2 * block_size * kv_heads * head_size,
                block_size * kv_heads * head_size,
                head_size,
                block_size * head_size,
                1,
            ),
            device="cuda",
            dtype=dtype,
        )
    else:
        raise ValueError(f"unsupported layout {layout!r}")
    combined.zero_()
    key_cache, value_cache = combined.unbind(1)
    return combined, key_cache, value_cache


def elapsed_microseconds(
    operation: Callable[[], None], warmup: int, iterations: int
) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def benchmark_case(
    *,
    tokens: int,
    layout: str,
    dtype: torch.dtype,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    rotary_dim: int,
    block_size: int,
    max_position: int,
    is_neox: bool,
    warmup: int,
    iterations: int,
    repeats: int,
    cos_sin_cache: torch.Tensor,
) -> dict[str, object]:
    num_blocks = max(1, math.ceil(tokens / block_size))
    positions = (
        torch.arange(tokens, device="cuda", dtype=torch.int64) + 37
    ) % max_position
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    query_source = torch.randn(
        (tokens, query_heads, head_size), device="cuda", dtype=dtype
    )
    key_source = torch.randn(
        (tokens, kv_heads, head_size), device="cuda", dtype=dtype
    )
    value = torch.randn(
        (tokens, kv_heads, head_size), device="cuda", dtype=dtype
    )

    baseline_query = query_source.clone()
    baseline_key = key_source.clone()
    baseline_combined, baseline_key_cache, baseline_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )
    loom_query = query_source.clone()
    loom_key = key_source.clone()
    loom_combined, loom_key_cache, loom_value_cache = make_cache(
        num_blocks, block_size, kv_heads, head_size, dtype, layout
    )

    def baseline() -> None:
        torch.ops._C.rotary_embedding(
            positions,
            baseline_query,
            baseline_key,
            head_size,
            cos_sin_cache,
            is_neox,
        )
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            baseline_key,
            value,
            baseline_key_cache,
            baseline_value_cache,
            slots,
            "auto",
            scale,
            scale,
        )

    def loom() -> None:
        rope_paged_kv_write_(
            loom_query,
            loom_key,
            value,
            positions,
            cos_sin_cache,
            loom_key_cache,
            loom_value_cache,
            slots,
            is_neox,
        )

    # One fresh-state correctness gate before repeated in-place benchmarking.
    baseline()
    loom()
    torch.cuda.synchronize()
    tolerance = {
        torch.float32: (1.0e-5, 1.0e-6),
        torch.float16: (1.0e-3, 1.0e-3),
        torch.bfloat16: (1.0e-2, 1.0e-2),
    }[dtype]
    torch.testing.assert_close(
        loom_query, baseline_query, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        loom_key, baseline_key, rtol=tolerance[0], atol=tolerance[1]
    )
    torch.testing.assert_close(
        loom_combined, baseline_combined, rtol=tolerance[0], atol=tolerance[1]
    )

    baseline_samples: list[float] = []
    loom_samples: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_samples.append(
                elapsed_microseconds(baseline, warmup, iterations)
            )
            loom_samples.append(elapsed_microseconds(loom, warmup, iterations))
        else:
            loom_samples.append(elapsed_microseconds(loom, warmup, iterations))
            baseline_samples.append(
                elapsed_microseconds(baseline, warmup, iterations)
            )

    baseline_median = statistics.median(baseline_samples)
    loom_median = statistics.median(loom_samples)
    return {
        "tokens": tokens,
        "layout": layout,
        "baseline_us": baseline_median,
        "loom_us": loom_median,
        "speedup": baseline_median / loom_median,
        "latency_reduction_percent":
            (baseline_median - loom_median) / baseline_median * 100.0,
        "baseline_samples_us": baseline_samples,
        "loom_samples_us": loom_samples,
        "baseline_p90_us": percentile(baseline_samples, 0.9),
        "loom_p90_us": percentile(loom_samples, 0.9),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("vLLM baseline requires query_heads divisible by kv_heads")
    if args.rotary_dim <= 0 or args.rotary_dim % 2 or args.rotary_dim > args.head_size:
        raise ValueError("rotary_dim must be positive, even, and <= head_size")

    import vllm
    import vllm._custom_ops  # noqa: F401 - registers baseline ops

    dtype = DTYPES[args.dtype]
    token_counts = [int(value) for value in args.tokens.split(",") if value]
    layouts = [value.strip() for value in args.layouts.split(",") if value.strip()]
    cos_sin_cache = make_cos_sin_cache(
        args.max_position, args.rotary_dim, dtype
    )
    results = [
        benchmark_case(
            tokens=tokens,
            layout=layout,
            dtype=dtype,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_size=args.head_size,
            rotary_dim=args.rotary_dim,
            block_size=args.block_size,
            max_position=args.max_position,
            is_neox=not args.interleaved,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            cos_sin_cache=cos_sin_cache,
        )
        for layout in layouts
        for tokens in token_counts
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "bridge_abi_version": bridge_abi_version(),
        "baseline": "vLLM _C.rotary_embedding + _C_cache_ops.reshape_and_cache_flash",
        "candidate": "Loom rope_paged_kv_write_",
        "dtype": args.dtype,
        "query_heads": args.query_heads,
        "kv_heads": args.kv_heads,
        "head_size": args.head_size,
        "rotary_dim": args.rotary_dim,
        "block_size": args.block_size,
        "is_neox": not args.interleaved,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
