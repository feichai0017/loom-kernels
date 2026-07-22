#!/usr/bin/env python3
"""Compare Loom short-context paged decode with vLLM FlashAttention."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Callable

import torch

from loom_kernels.torch_ops import PAGED_DECODE_MAX_CONTEXT, adapter_backend


DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--contexts", default="16,128,512")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=709)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def latency_summary(samples: list[float]) -> dict[str, object]:
    return {
        "minimum_us": min(samples),
        "median_us": statistics.median(samples),
        "maximum_us": max(samples),
        "samples_us": samples,
    }


def timed_sample(operation: Callable[[], object], iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iterations)


def measure_pair(
    loom: Callable[[], object],
    baseline: Callable[[], object],
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[dict[str, object], dict[str, object]]:
    operations = {"loom": loom, "baseline": baseline}
    for repeat in range(warmup):
        order = ("baseline", "loom") if repeat % 2 == 0 else ("loom", "baseline")
        for provider in order:
            operations[provider]()
    torch.cuda.synchronize()

    eager = {"loom": [], "baseline": []}
    for repeat in range(samples):
        order = ("baseline", "loom") if repeat % 2 == 0 else ("loom", "baseline")
        for provider in order:
            eager[provider].append(timed_sample(operations[provider], iterations))

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for provider in ("baseline", "loom"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operations[provider]()
        graphs[provider] = graph
    for repeat in range(warmup):
        order = ("loom", "baseline") if repeat % 2 == 0 else ("baseline", "loom")
        for provider in order:
            graphs[provider].replay()
    torch.cuda.synchronize()

    replay = {"loom": [], "baseline": []}
    for repeat in range(samples):
        order = ("loom", "baseline") if repeat % 2 == 0 else ("baseline", "loom")
        for provider in order:
            replay[provider].append(
                timed_sample(graphs[provider].replay, iterations)
            )

    def result(provider: str) -> dict[str, object]:
        return {
            "eager_dispatch_latency": latency_summary(eager[provider]),
            "cuda_graph_replay_latency": latency_summary(replay[provider]),
        }

    return result("loom"), result("baseline")


def benchmark_case(
    *,
    batch: int,
    context: int,
    dtype: torch.dtype,
    query_heads: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    warmup: int,
    iterations: int,
    samples: int,
    seed: int,
    flash_attn_varlen_func: Callable[..., torch.Tensor],
    fa_version: int,
) -> dict[str, object]:
    torch.manual_seed(seed + batch * 1009 + context)
    max_blocks = math.ceil(context / block_size)
    num_blocks = batch * max_blocks
    query = torch.randn(
        (batch, query_heads, head_size), device="cuda", dtype=dtype
    )
    key_cache = torch.randn(
        (num_blocks, block_size, kv_heads, head_size),
        device="cuda",
        dtype=dtype,
    )
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.randperm(num_blocks, device="cuda", dtype=torch.int64)
    block_tables = block_tables.reshape(batch, max_blocks).to(torch.int32)
    # Keep the batch ragged while retaining one sequence at the named maximum.
    sequence_lengths = torch.tensor(
        [max(1, context - (index * 7) % max(1, 2 * block_size)) for index in range(batch)],
        device="cuda",
        dtype=torch.int32,
    )
    sequence_lengths[0] = context
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    scale = head_size**-0.5
    loom_output = torch.empty_like(query)
    baseline_output = torch.empty_like(query)

    def loom() -> None:
        torch.ops.loom_kernels.paged_decode_attention_unchecked(
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            loom_output,
            context,
            scale,
        )

    def baseline() -> None:
        flash_attn_varlen_func(
            q=query,
            k=key_cache,
            v=value_cache,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=context,
            seqused_k=sequence_lengths,
            softmax_scale=scale,
            causal=True,
            block_table=block_tables,
            out=baseline_output,
            fa_version=fa_version,
        )

    loom()
    baseline()
    torch.cuda.synchronize()
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
    return {
        "batch": batch,
        "context": context,
        "minimum_sequence_length": int(sequence_lengths.min().item()),
        "loom": loom_measurement,
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
    if adapter_backend() != "cpp-dispatch":
        raise RuntimeError("benchmark requires the Loom C++ dispatcher bridge")
    if args.query_heads % args.kv_heads:
        raise ValueError("query-heads must be divisible by kv-heads")
    if min(args.warmup, args.iterations, args.samples) <= 0:
        raise ValueError("warmup, iterations, and samples must be positive")

    import vllm
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    batches = [int(value) for value in args.batches.split(",") if value]
    contexts = [int(value) for value in args.contexts.split(",") if value]
    if not batches or min(batches) <= 0:
        raise ValueError("batches must contain positive integers")
    if (
        not contexts
        or min(contexts) <= 0
        or max(contexts) > PAGED_DECODE_MAX_CONTEXT
    ):
        raise ValueError(
            f"contexts must be within [1, {PAGED_DECODE_MAX_CONTEXT}]"
        )
    fa_version = get_flash_attn_version()
    dtype = DTYPES[args.dtype]
    results = [
        benchmark_case(
            batch=batch,
            context=context,
            dtype=dtype,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_size=args.head_size,
            block_size=args.block_size,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.samples,
            seed=args.seed,
            flash_attn_varlen_func=flash_attn_varlen_func,
            fa_version=fa_version,
        )
        for batch in batches
        for context in contexts
    ]
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "single-token paged MQA/GQA decode attention",
        "candidate": "Loom paged_decode_attention_unchecked",
        "baseline": f"vLLM {vllm.__version__} FlashAttention varlen FA{fa_version}",
        "scope": {
            "dtype": args.dtype,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_size": args.head_size,
            "value_head_size": args.head_size,
            "block_size": args.block_size,
            "cache_layout": "NHD",
            "maximum_context": PAGED_DECODE_MAX_CONTEXT,
        },
        "timing": {
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "samples": args.samples,
            "provider_order": "alternated for every eager and graph sample",
            "eager": "CUDA events over Python-dispatched provider calls",
            "cuda_graph": "CUDA events over graph replay",
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "flash_attention_version": fa_version,
            "adapter_backend": adapter_backend(),
        },
        "acceptance": {
            "passed": True,
            "rtol": 2.0e-2,
            "atol": 2.0e-2,
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
