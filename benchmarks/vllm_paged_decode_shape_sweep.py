#!/usr/bin/env python3
"""Sweep Loom paged decode across engine-relevant attention geometries."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from loom_kernels.torch_ops import (
    PAGED_DECODE_MAX_CONTEXT,
    bridge_abi_version,
)
from vllm_paged_decode_attention import DTYPES, benchmark_case


DEFAULT_SHAPES = (
    # GQA-ratio sweep at the Qwen-style head size.
    "bf16:32:32:128:16",
    "bf16:32:16:128:16",
    "bf16:32:8:128:16",
    "bf16:32:4:128:16",
    "bf16:32:1:128:16",
    # Head-size, dtype, and page-size sensitivity around Hq/Hkv=32/8.
    "bf16:32:8:64:16",
    "bf16:32:8:256:16",
    "f16:32:8:128:16",
    "bf16:32:8:128:32",
    # Query-head count and even/odd GQA packing ratios.
    "bf16:16:4:128:16",
    "bf16:40:8:128:16",
    "bf16:48:8:128:16",
    "bf16:64:8:128:16",
)


@dataclass(frozen=True)
class AttentionShape:
    dtype: str
    query_heads: int
    kv_heads: int
    head_size: int
    block_size: int

    @classmethod
    def parse(cls, value: str) -> "AttentionShape":
        parts = value.split(":")
        if len(parts) != 5:
            raise argparse.ArgumentTypeError(
                "shape must be DTYPE:Hq:Hkv:HEAD_SIZE:BLOCK_SIZE"
            )
        dtype = parts[0]
        if dtype not in DTYPES:
            raise argparse.ArgumentTypeError(
                f"shape dtype must be one of {sorted(DTYPES)}"
            )
        try:
            query_heads, kv_heads, head_size, block_size = map(int, parts[1:])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "shape dimensions must be integers"
            ) from error
        if min(query_heads, kv_heads, head_size, block_size) <= 0:
            raise argparse.ArgumentTypeError(
                "shape dimensions must be positive"
            )
        if query_heads % kv_heads:
            raise argparse.ArgumentTypeError(
                "shape query heads must be divisible by KV heads"
            )
        return cls(dtype, query_heads, kv_heads, head_size, block_size)


def parse_positive_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or min(values) <= 0:
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=AttentionShape.parse,
        help=(
            "repeatable DTYPE:Hq:Hkv:HEAD_SIZE:BLOCK_SIZE geometry; "
            "uses the curated matrix when omitted"
        ),
    )
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--contexts", default="16,32,64,128")
    parser.add_argument(
        "--cache-storage",
        choices=("separate", "vllm-interleaved"),
        default="vllm-interleaved",
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.warmup, args.iterations, args.samples) <= 0:
        raise ValueError("warmup, iterations, and samples must be positive")

    batches = parse_positive_list(args.batches, "batches")
    contexts = parse_positive_list(args.contexts, "contexts")
    if max(contexts) > PAGED_DECODE_MAX_CONTEXT:
        raise ValueError(
            f"contexts must be at most {PAGED_DECODE_MAX_CONTEXT}"
        )
    shapes = args.shape or [AttentionShape.parse(value) for value in DEFAULT_SHAPES]
    if len(set(shapes)) != len(shapes):
        raise ValueError("shapes must not contain duplicates")

    import vllm
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    fa_version = get_flash_attn_version()
    results: list[dict[str, object]] = []
    for shape_index, shape in enumerate(shapes):
        for batch in batches:
            for context in contexts:
                result = benchmark_case(
                    batch=batch,
                    context=context,
                    dtype=DTYPES[shape.dtype],
                    query_heads=shape.query_heads,
                    kv_heads=shape.kv_heads,
                    head_size=shape.head_size,
                    block_size=shape.block_size,
                    cache_storage=args.cache_storage,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    samples=args.samples,
                    seed=args.seed + shape_index * 1_000_003,
                    flash_attn_varlen_func=flash_attn_varlen_func,
                    fa_version=fa_version,
                )
                results.append({"shape": asdict(shape), **result})

    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "single-token paged MQA/GQA decode attention shape sweep",
        "candidate": "Loom paged_decode_attention",
        "baseline": (
            f"vLLM {vllm.__version__} FlashAttention varlen FA{fa_version}"
        ),
        "scope": {
            "shapes": [asdict(shape) for shape in shapes],
            "batches": batches,
            "contexts": contexts,
            "cache_layout": "NHD",
            "cache_storage": args.cache_storage,
            "value_head_size": "equal to head_size",
            "maximum_context": PAGED_DECODE_MAX_CONTEXT,
            "tested_cases": len(results),
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
            "bridge_abi_version": bridge_abi_version(),
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
