"""Command-line interface for the two-GPU Route-Q acceptance benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

from .two_gpu_benchmark import (
    BENCHMARK_ATTENTION_BACKENDS,
    DTYPE_BYTES,
    ROUTE_STRATEGIES,
    BenchmarkConfig,
    projected_transfer_bytes,
    run_benchmark,
)


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--tail-tokens", type=int, default=16)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--precondition-dimension", type=int, default=4096)
    parser.add_argument("--precondition-iterations", type=int, default=100)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float16")
    parser.add_argument(
        "--attention-backend",
        choices=BENCHMARK_ATTENTION_BACKENDS,
        default="reference",
    )
    parser.add_argument(
        "--route-strategy",
        choices=ROUTE_STRATEGIES,
        default="sequential",
        help="sequential, stream-overlapped, or handwritten fused tail/merge",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--timeout-seconds", type=int, default=120)


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        prefix_tokens=args.prefix_tokens,
        tail_tokens=args.tail_tokens,
        rows=args.rows,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        precondition_dimension=args.precondition_dimension,
        precondition_iterations=args.precondition_iterations,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        route_strategy=args.route_strategy,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        timeout_seconds=args.timeout_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Loom Route-Q against Stage-KV on two CUDA GPUs"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print payload sizes without CUDA")
    _add_workload_arguments(plan)
    run = commands.add_parser("run", help="run the two-GPU NCCL acceptance gate")
    _add_workload_arguments(run)
    run.add_argument(
        "--report",
        type=Path,
        default=Path("build/two-gpu-smoke/report.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _config_from_args(args)
        config.validate()
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        "workload": asdict(config),
                        "payload_bytes": projected_transfer_bytes(config),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        report = run_benchmark(config, args.report)
        print(
            "Loom two-GPU smoke "
            f"{'PASSED' if report['passed'] else 'FAILED'}; "
            f"Route-Q p50={report['route_query']['p50_ms']:.3f} ms, "
            f"Stage-KV p50={report['stage_kv']['p50_ms']:.3f} ms"
        )
        return 0 if report["passed"] else 1
    except (RuntimeError, ValueError) as error:
        print(f"integration.two_gpu_smoke: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
