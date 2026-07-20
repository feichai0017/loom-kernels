"""Single-GPU correctness gate for Loom's fused tail-attention CUDA op."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from typing import Any

from loom_attention.attention_state import compute_attention_state
from loom_attention.cuda_ops import fused_tail_attention_merge

from .two_gpu_benchmark import BenchmarkConfig, _route_engine


class _ImmediateRequest:
    def wait(self) -> None:
        return None


class _ImmediateDist:
    """Single-process stand-in with prepopulated receive buffers."""

    @staticmethod
    def send(_tensor: Any, dst: int) -> None:
        del dst

    @staticmethod
    def irecv(_tensor: Any, source: int) -> None:
        del source

    @staticmethod
    def P2POp(operation: Any, tensor: Any, peer: int) -> tuple[Any, ...]:
        return operation, tensor, peer

    @staticmethod
    def batch_isend_irecv(operations: list[Any]) -> list[_ImmediateRequest]:
        return [_ImmediateRequest() for _ in operations]


def run_case(
    torch: Any,
    *,
    dtype: Any,
    rows: int,
    prefix_tokens: int,
    tail_tokens: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scale = head_dim**-0.5

    def random(shape: tuple[int, ...]) -> Any:
        return (
            torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
            * 0.1
        ).contiguous()

    query = random((rows, query_heads, head_dim))
    prefix_key = random((prefix_tokens, kv_heads, head_dim))
    prefix_value = random((prefix_tokens, kv_heads, head_dim))
    tail_key = random((tail_tokens, kv_heads, head_dim))
    tail_value = random((tail_tokens, kv_heads, head_dim))
    remote = compute_attention_state(
        torch,
        query,
        prefix_key,
        prefix_value,
        kv_heads=kv_heads,
        scale=scale,
        backend="reference",
    )
    fused = fused_tail_attention_merge(
        query,
        tail_key,
        tail_value,
        remote[0],
        remote[1],
        scale=scale,
    )
    full = compute_attention_state(
        torch,
        query,
        torch.cat((prefix_key, tail_key)),
        torch.cat((prefix_value, tail_value)),
        kv_heads=kv_heads,
        scale=scale,
        backend="reference",
    )
    torch.cuda.synchronize()
    output_delta = (fused[0].float() - full[0].float()).abs()
    lse_delta = (fused[1] - full[1]).abs()
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    fused_passed = bool(
        torch.allclose(fused[0], full[0], atol=tolerance, rtol=tolerance)
        and torch.allclose(fused[1], full[1], atol=2e-4, rtol=2e-4)
    )
    route_strategies = []
    for strategy in ("sequential", "overlap", "fused"):
        config = BenchmarkConfig(
            prefix_tokens=prefix_tokens,
            tail_tokens=tail_tokens,
            rows=rows,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype="float16" if dtype == torch.float16 else "bfloat16",
            route_strategy=strategy,
        )
        local_stream = (
            torch.cuda.Stream() if strategy == "overlap" else None
        )
        output = _route_engine(
            torch,
            _ImmediateDist(),
            query,
            (remote[0].clone(), remote[1].clone()),
            (tail_key, tail_value),
            config,
            local_tail_stream=local_stream,
        )
        strategy_delta = (output.float() - full[0].float()).abs()
        route_strategies.append(
            {
                "strategy": strategy,
                "passed": bool(
                    torch.allclose(
                        output, full[0], atol=tolerance, rtol=tolerance
                    )
                ),
                "output_max_absolute_error": float(
                    strategy_delta.max().item()
                ),
            }
        )
    passed = fused_passed and all(
        strategy["passed"] for strategy in route_strategies
    )
    return {
        "dtype": str(dtype),
        "passed": passed,
        "output_max_absolute_error": float(output_delta.max().item()),
        "lse_max_absolute_error": float(lse_delta.max().item()),
        "output_tolerance": tolerance,
        "route_strategy_single_gpu_emulation": route_strategies,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--tail-tokens", type=int, default=16)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (
        args.rows <= 0
        or args.prefix_tokens <= 0
        or not 0 < args.tail_tokens <= 64
        or args.query_heads <= 0
        or args.kv_heads <= 0
        or args.query_heads % args.kv_heads
        or not 0 < args.head_dim <= 256
    ):
        raise ValueError("invalid fused-tail smoke shape")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the fused-tail smoke requires one CUDA GPU")
    cases = [
        run_case(
            torch,
            dtype=dtype,
            rows=args.rows,
            prefix_tokens=args.prefix_tokens,
            tail_tokens=args.tail_tokens,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            seed=args.seed,
        )
        for dtype in (torch.float16, torch.bfloat16)
    ]
    report = {
        "schema_version": 1,
        "passed": all(case["passed"] for case in cases),
        "environment": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "workload": {
            "rows": args.rows,
            "prefix_tokens": args.prefix_tokens,
            "tail_tokens": args.tail_tokens,
            "query_heads": args.query_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "scale": math.sqrt(args.head_dim) ** -1,
            "seed": args.seed,
        },
        "cases": cases,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
