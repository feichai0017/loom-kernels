#!/usr/bin/env python3
"""Compare Loom's in-place min-p filter with vLLM's allocating path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import adapter_backend, min_p_filter_
from loom_kernels.vllm import (
    MIN_P_FAST_PATH_MIN_ROWS,
    MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,8,32,128")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=0,
        help="zero uses vocab-size; larger values model padded vocabulary rows",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.vocab_size <= 0 or min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("vocab-size, warmup, iterations, and repeats must be positive")
    if args.row_stride == 0:
        args.row_stride = args.vocab_size
    if args.row_stride < args.vocab_size:
        parser.error("row-stride must be at least vocab-size")
    return args


def vllm_min_p(values: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """The exact MinPLogitsProcessor.apply implementation in vLLM 0.24."""
    probability_values = torch.nn.functional.softmax(values, dim=-1)
    max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
    adjusted_min_p = max_probabilities.mul_(min_p)
    invalid_token_mask = probability_values < adjusted_min_p
    values.masked_fill_(invalid_token_mask, -float("inf"))
    return values


def elapsed_microseconds(
    operation: Callable[[torch.Tensor], object],
    source: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    # Every timed invocation receives fresh logits. Preparing those tensors is
    # outside the events, so neither provider is diluted by a reset copy.
    warmup_values = source.clone()
    for _ in range(warmup):
        warmup_values.copy_(source)
        operation(warmup_values)
    workspaces = [source.clone() for _ in range(iterations)]
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for values in workspaces:
        operation(values)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end) * 1000.0 / iterations)

    del workspaces, warmup_values
    torch.cuda.empty_cache()
    return elapsed


def peak_temporary_bytes(
    operation: Callable[[torch.Tensor], object], source: torch.Tensor
) -> int:
    values = source.clone()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation(values)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del values
    torch.cuda.empty_cache()
    return max(0, peak - before)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def benchmark_case(
    rows: int,
    vocab_size: int,
    row_stride: int,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed + rows)
    storage = torch.randn((rows, row_stride), device="cuda", dtype=torch.float32)
    source = storage[:, :vocab_size]
    min_p = torch.linspace(0.05, 0.8, rows, device="cuda").unsqueeze(1)

    def baseline(values: torch.Tensor):
        return vllm_min_p(values, min_p)

    def loom(values: torch.Tensor):
        return min_p_filter_(values, min_p)

    expected = baseline(source.clone())
    actual = loom(source.clone())
    torch.cuda.synchronize()
    expected_mask = torch.isneginf(expected)
    actual_mask = torch.isneginf(actual)
    if not torch.equal(actual_mask, expected_mask):
        raise AssertionError("Loom min-p mask differs from vLLM")
    if not torch.equal(actual[~actual_mask], expected[~expected_mask]):
        raise AssertionError("Loom changed a retained logit")

    baseline_temporary_bytes = peak_temporary_bytes(baseline, source)
    loom_temporary_bytes = peak_temporary_bytes(loom, source)
    baseline_samples: list[float] = []
    loom_samples: list[float] = []
    for repeat in range(repeats):
        if repeat % 2 == 0:
            baseline_samples.append(
                elapsed_microseconds(baseline, source, warmup, iterations)
            )
            loom_samples.append(
                elapsed_microseconds(loom, source, warmup, iterations)
            )
        else:
            loom_samples.append(
                elapsed_microseconds(loom, source, warmup, iterations)
            )
            baseline_samples.append(
                elapsed_microseconds(baseline, source, warmup, iterations)
            )

    baseline_median = statistics.median(baseline_samples)
    loom_median = statistics.median(loom_samples)
    return {
        "rows": rows,
        "baseline_us": baseline_median,
        "loom_us": loom_median,
        "speedup": baseline_median / loom_median,
        "latency_reduction_percent": (
            (baseline_median - loom_median) / baseline_median * 100.0
        ),
        "baseline_samples_us": baseline_samples,
        "loom_samples_us": loom_samples,
        "baseline_p90_us": percentile(baseline_samples, 0.9),
        "loom_p90_us": percentile(loom_samples, 0.9),
        "baseline_peak_temporary_bytes": baseline_temporary_bytes,
        "loom_peak_temporary_bytes": loom_temporary_bytes,
        "filtered_fraction": float(actual_mask.float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import vllm

    rows = [int(value) for value in args.rows.split(",") if value]
    if not rows or min(rows) <= 0:
        raise ValueError("rows must contain positive integers")
    results = [
        benchmark_case(
            row_count,
            args.vocab_size,
            args.row_stride,
            args.warmup,
            args.iterations,
            args.repeats,
            args.seed,
        )
        for row_count in rows
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "adapter_backend": adapter_backend(),
        "baseline": "vLLM 0.24 MinPLogitsProcessor.apply",
        "candidate": "Loom min_p_filter_",
        "dtype": "f32 (vLLM sampling processor contract)",
        "vocab_size": args.vocab_size,
        "row_stride": args.row_stride,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "timing_method": (
            "CUDA events over independent preallocated logits; input creation "
            "and reset copies excluded; provider order alternated by repeat"
        ),
        "vllm_fast_path_gate": {
            "minimum_rows": MIN_P_FAST_PATH_MIN_ROWS,
            "minimum_vocab_size": MIN_P_FAST_PATH_MIN_VOCAB_SIZE,
            "smaller_shapes": "fall back to vLLM 0.24",
        },
        "acceptance": {
            "passed": True,
            "mask": "exact",
            "retained_logits": "exact",
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
