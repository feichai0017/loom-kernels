#!/usr/bin/env python3
"""Compare selected-token normalization with vLLM's full log-softmax path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import bridge_abi_version, selected_token_logprobs


DTYPES = {
    "f32": torch.float32,
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--rows", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--row-stride", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.vocab_size <= 0 or min(args.warmup, args.iterations, args.repeats) <= 0:
        parser.error("vocab-size, warmup, iterations, and repeats must be positive")
    if args.row_stride == 0:
        args.row_stride = args.vocab_size
    if args.row_stride < args.vocab_size:
        parser.error("row-stride must be at least vocab-size")
    return args


def elapsed_microseconds(
    operation: Callable[[], object], warmup: int, iterations: int
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
    rows: int,
    vocab_size: int,
    row_stride: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    from vllm.v1.sample.sampler import Sampler

    torch.manual_seed(seed + rows)
    storage = torch.randn((rows, row_stride), device="cuda", dtype=dtype)
    logits = storage[:, :vocab_size]
    token_ids = torch.randint(
        0, vocab_size, (rows,), device="cuda", dtype=torch.int64
    )

    def baseline():
        raw_logprobs = Sampler.compute_logprobs(logits)
        gathered = Sampler.gather_logprobs(raw_logprobs, 0, token_ids)
        return gathered.logprobs[:, 0], gathered.selected_token_ranks

    def loom():
        return selected_token_logprobs(logits, token_ids)

    expected = baseline()
    actual = loom()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=2.0e-5, atol=2.0e-5)
    if not torch.equal(actual[1], expected[1]):
        raise AssertionError("Loom selected-token ranks differ from vLLM")

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
        "maximum_logprob_error": float(
            (actual[0] - expected[0]).abs().max().item()
        ),
        "minimum_selected_rank": int(actual[1].min().item()),
        "maximum_selected_rank": int(actual[1].max().item()),
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
            DTYPES[args.dtype],
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
        "bridge_abi_version": bridge_abi_version(),
        "baseline": (
            "vLLM Sampler.compute_logprobs + "
            "gather_logprobs(num_logprobs=0, caller-selected IDs)"
        ),
        "candidate": "Loom selected_token_logprobs",
        "dtype": args.dtype,
        "vocab_size": args.vocab_size,
        "row_stride": args.row_stride,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "selection": "uniform caller-selected token IDs; sampling cost excluded",
        "acceptance": {
            "passed": True,
            "sampled_token_ranks": "exact, including ties",
            "logprob_rtol": 2.0e-5,
            "logprob_atol": 2.0e-5,
            "maximum_logprob_error": max(
                result["maximum_logprob_error"] for result in results
            ),
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
