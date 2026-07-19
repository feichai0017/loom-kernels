"""Two-GPU NCCL acceptance harness for the Route-Q data path.

The remote rank owns a sealed KV prefix. The engine rank sends only Q, receives
online-softmax partials, merges them with a local active tail, and compares the
result with full attention. A Stage-KV baseline transfers the remote prefix in
the opposite direction under the same workload.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Sequence


DTYPE_BYTES = {"float16": 2, "bfloat16": 2}


@dataclass(frozen=True)
class BenchmarkConfig:
    prefix_tokens: int = 4096
    tail_tokens: int = 16
    rows: int = 1
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"
    warmup: int = 5
    iterations: int = 20
    seed: int = 7
    atol: float = 1e-4
    rtol: float = 1e-4
    timeout_seconds: int = 120

    def validate(self) -> None:
        positive = {
            "prefix_tokens": self.prefix_tokens,
            "rows": self.rows,
            "query_heads": self.query_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "iterations": self.iterations,
            "timeout_seconds": self.timeout_seconds,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"values must be positive: {', '.join(invalid)}")
        if self.tail_tokens < 0 or self.warmup < 0:
            raise ValueError("tail_tokens and warmup must be non-negative")
        if self.query_heads % self.kv_heads != 0:
            raise ValueError("kv_heads must divide query_heads")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.atol < 0.0 or self.rtol < 0.0:
            raise ValueError("atol and rtol must be non-negative")


def projected_transfer_bytes(config: BenchmarkConfig) -> dict[str, int]:
    config.validate()
    element_bytes = DTYPE_BYTES[config.dtype]
    query = config.rows * config.query_heads * config.head_dim * element_bytes
    partial = (
        config.rows
        * config.query_heads
        * (2 + config.head_dim)
        * 4
    )
    staged_kv = (
        2
        * config.prefix_tokens
        * config.kv_heads
        * config.head_dim
        * element_bytes
    )
    return {
        "query": query,
        "partial": partial,
        "route_query_total": query + partial,
        "stage_kv_total": staged_kv,
    }


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _latency_summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "p50_ms": float(median(samples)),
        "p99_ms": percentile(samples, 0.99),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _make_inputs(torch: Any, config: BenchmarkConfig, device: Any) -> tuple[Any, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    dtype = _torch_dtype(torch, config.dtype)

    def sample(shape: tuple[int, ...]) -> Any:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .to(dtype=dtype)
            .to(device=device)
            .contiguous()
        )

    query = sample((config.rows, config.query_heads, config.head_dim))
    prefix_key = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    prefix_value = sample((config.prefix_tokens, config.kv_heads, config.head_dim))
    tail_key = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    tail_value = sample((config.tail_tokens, config.kv_heads, config.head_dim))
    return query, prefix_key, prefix_value, tail_key, tail_value


def _partial_attention(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    *,
    kv_heads: int,
    scale: float,
) -> tuple[Any, Any, Any]:
    rows, query_heads, head_dim = query.shape
    if key.shape[0] == 0:
        raise ValueError("partial attention requires at least one KV token")
    groups = query_heads // kv_heads
    grouped_query = query.float().reshape(rows, kv_heads, groups, head_dim)
    scores = torch.einsum("rhgd,thd->rhgt", grouped_query, key.float()) * scale
    max_logits = scores.amax(dim=-1)
    weights = torch.exp(scores - max_logits.unsqueeze(-1))
    exp_sums = weights.sum(dim=-1)
    weighted_values = torch.einsum("rhgt,thd->rhgd", weights, value.float())
    return (
        max_logits.reshape(rows, query_heads).contiguous(),
        exp_sums.reshape(rows, query_heads).contiguous(),
        weighted_values.reshape(rows, query_heads, head_dim).contiguous(),
    )


def _merge_partials(torch: Any, partials: Sequence[tuple[Any, Any, Any]]) -> Any:
    if not partials:
        raise ValueError("at least one partial is required")
    global_max = torch.stack([partial[0] for partial in partials]).amax(dim=0)
    denominator = torch.zeros_like(partials[0][1])
    numerator = torch.zeros_like(partials[0][2])
    for max_logits, exp_sums, weighted_values in partials:
        correction = torch.exp(max_logits - global_max)
        denominator.add_(correction * exp_sums)
        numerator.add_(correction.unsqueeze(-1) * weighted_values)
    return numerator / denominator.unsqueeze(-1)


def _batch_send(dist: Any, tensors: Sequence[Any], destination: int) -> None:
    operations = [dist.P2POp(dist.isend, tensor, destination) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _batch_receive(dist: Any, tensors: Sequence[Any], source: int) -> None:
    operations = [dist.P2POp(dist.irecv, tensor, source) for tensor in tensors]
    for request in dist.batch_isend_irecv(operations):
        request.wait()


def _route_engine(
    torch: Any,
    dist: Any,
    query: Any,
    remote_buffers: tuple[Any, Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    dist.send(query, dst=1)
    _batch_receive(dist, remote_buffers, source=1)
    partials = [remote_buffers]
    if config.tail_tokens:
        partials.append(
            _partial_attention(
                torch,
                query,
                local_tail[0],
                local_tail[1],
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
            )
        )
    return _merge_partials(torch, partials)


def _route_worker(
    torch: Any,
    dist: Any,
    query_buffer: Any,
    prefix: tuple[Any, Any],
    config: BenchmarkConfig,
) -> None:
    dist.recv(query_buffer, src=0)
    partial = _partial_attention(
        torch,
        query_buffer,
        prefix[0],
        prefix[1],
        kv_heads=config.kv_heads,
        scale=config.head_dim**-0.5,
    )
    _batch_send(dist, partial, destination=0)


def _stage_engine(
    torch: Any,
    dist: Any,
    query: Any,
    receive_buffers: tuple[Any, Any],
    full_buffers: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    _batch_receive(dist, receive_buffers, source=1)
    return _merge_partials(
        torch,
        [
            _partial_attention(
                torch,
                query,
                full_buffers[0],
                full_buffers[1],
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
            )
        ],
    )


def _stage_worker(dist: Any, prefix: tuple[Any, Any]) -> None:
    _batch_send(dist, prefix, destination=0)


def _full_attention(
    torch: Any,
    query: Any,
    prefix: tuple[Any, Any],
    local_tail: tuple[Any, Any],
    config: BenchmarkConfig,
) -> Any:
    if config.tail_tokens:
        key = torch.cat((prefix[0], local_tail[0]), dim=0)
        value = torch.cat((prefix[1], local_tail[1]), dim=0)
    else:
        key, value = prefix
    return _merge_partials(
        torch,
        [
            _partial_attention(
                torch,
                query,
                key,
                value,
                kv_heads=config.kv_heads,
                scale=config.head_dim**-0.5,
            )
        ],
    )


def _timed_cuda(torch: Any, operation: Any) -> float:
    torch.cuda.synchronize()
    started = perf_counter()
    operation()
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1_000.0


def _environment(torch: Any) -> dict[str, Any]:
    devices = []
    for index in range(2):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": properties.total_memory,
                "multiprocessor_count": properties.multi_processor_count,
            }
        )
    nccl_version = torch.cuda.nccl.version()
    if isinstance(nccl_version, tuple):
        nccl_version = ".".join(str(part) for part in nccl_version)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": nccl_version,
        "device_peer_access": torch.cuda.can_device_access_peer(0, 1),
        "devices": devices,
    }


def _write_report(
    torch: Any,
    config: BenchmarkConfig,
    route_output: Any,
    stage_output: Any,
    expected: Any,
    route_samples: Sequence[float],
    stage_samples: Sequence[float],
    report_path: str,
) -> None:
    def correctness(output: Any) -> dict[str, Any]:
        difference = (output - expected).abs()
        return {
            "passed": bool(
                torch.allclose(output, expected, atol=config.atol, rtol=config.rtol)
            ),
            "max_absolute_error": float(difference.max().item()),
            "max_relative_error": float(
                (difference / expected.abs().clamp_min(1e-6)).max().item()
            ),
        }

    route_correctness = correctness(route_output)
    stage_correctness = correctness(stage_output)
    passed = route_correctness["passed"] and stage_correctness["passed"]
    route = _latency_summary(route_samples)
    stage = _latency_summary(stage_samples)
    route["payload_bytes"] = projected_transfer_bytes(config)["route_query_total"]
    stage["payload_bytes"] = projected_transfer_bytes(config)["stage_kv_total"]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "implementation": {
            "transport": "torch.distributed NCCL point-to-point",
            "attention_kernel": "PyTorch CUDA einsum online-softmax reference",
            "production_kernel": False,
        },
        "environment": _environment(torch),
        "workload": asdict(config),
        "correctness": {
            "atol": config.atol,
            "rtol": config.rtol,
            "route_query": route_correctness,
            "stage_kv": stage_correctness,
        },
        "route_query": route,
        "stage_kv": stage,
        "stage_over_route_p50": stage["p50_ms"] / route["p50_ms"],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def _run_rank(
    rank: int,
    config: BenchmarkConfig,
    init_method: str,
    report_path: str,
) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=config.timeout_seconds),
    )
    try:
        query, prefix_key, prefix_value, tail_key, tail_value = _make_inputs(
            torch, config, device
        )
        prefix = (prefix_key, prefix_value)
        local_tail = (tail_key, tail_value)

        if rank == 0:
            remote_buffers = (
                torch.empty(
                    (config.rows, config.query_heads),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    (config.rows, config.query_heads),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    (config.rows, config.query_heads, config.head_dim),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            full_tokens = config.prefix_tokens + config.tail_tokens
            stage_key = torch.empty(
                (full_tokens, config.kv_heads, config.head_dim),
                dtype=prefix_key.dtype,
                device=device,
            )
            stage_value = torch.empty_like(stage_key)
            stage_receive_buffers = (
                stage_key[: config.prefix_tokens],
                stage_value[: config.prefix_tokens],
            )
            stage_full_buffers = (stage_key, stage_value)
            if config.tail_tokens:
                stage_key[config.prefix_tokens :].copy_(tail_key)
                stage_value[config.prefix_tokens :].copy_(tail_value)
        else:
            query_buffer = torch.empty_like(query)

        dist.barrier()
        if rank == 0:
            route_output = _route_engine(
                torch, dist, query, remote_buffers, local_tail, config
            )
            expected = _full_attention(torch, query, prefix, local_tail, config)
        else:
            _route_worker(torch, dist, query_buffer, prefix, config)

        dist.barrier()
        route_samples = []
        for iteration in range(config.warmup + config.iterations):
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _route_engine(
                        torch, dist, query, remote_buffers, local_tail, config
                    ),
                )
                if iteration >= config.warmup:
                    route_samples.append(elapsed)
            else:
                _route_worker(torch, dist, query_buffer, prefix, config)

        dist.barrier()
        if rank == 0:
            stage_output = _stage_engine(
                torch,
                dist,
                query,
                stage_receive_buffers,
                stage_full_buffers,
                config,
            )
        else:
            _stage_worker(dist, prefix)

        dist.barrier()
        stage_samples = []
        for iteration in range(config.warmup + config.iterations):
            if rank == 0:
                elapsed = _timed_cuda(
                    torch,
                    lambda: _stage_engine(
                        torch,
                        dist,
                        query,
                        stage_receive_buffers,
                        stage_full_buffers,
                        config,
                    ),
                )
                if iteration >= config.warmup:
                    stage_samples.append(elapsed)
            else:
                _stage_worker(dist, prefix)

        dist.barrier()
        if rank == 0:
            _write_report(
                torch,
                config,
                route_output,
                stage_output,
                expected,
                route_samples,
                stage_samples,
                report_path,
            )
    finally:
        dist.destroy_process_group()


def _require_cuda_environment() -> Any:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as error:
        raise RuntimeError("PyTorch is required; install './python[cuda]'") from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("two CUDA devices are required")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("a PyTorch build with torch.distributed NCCL is required")
    return torch


def _run(config: BenchmarkConfig, report_path: Path) -> int:
    config.validate()
    torch = _require_cuda_environment()
    import torch.multiprocessing as multiprocessing

    with TemporaryDirectory(prefix="attnarc-two-gpu-") as directory:
        rendezvous = Path(directory) / "nccl-rendezvous"
        init_method = f"file://{rendezvous}"
        multiprocessing.spawn(
            _run_rank,
            args=(config, init_method, str(report_path)),
            nprocs=2,
            join=True,
        )
    report = json.loads(report_path.read_text())
    print(
        "AttnArc two-GPU smoke "
        f"{'PASSED' if report['passed'] else 'FAILED'}; "
        f"Route-Q p50={report['route_query']['p50_ms']:.3f} ms, "
        f"Stage-KV p50={report['stage_kv']['p50_ms']:.3f} ms"
    )
    return 0 if report["passed"] else 1


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--tail-tokens", type=int, default=16)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--timeout-seconds", type=int, default=120)


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        prefix_tokens=args.prefix_tokens,
        tail_tokens=args.tail_tokens,
        rows=args.rows,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        timeout_seconds=args.timeout_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate AttnArc Route-Q against Stage-KV on two CUDA GPUs"
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
        return _run(config, args.report)
    except (RuntimeError, ValueError) as error:
        print(f"attnarc-two-gpu-smoke: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
