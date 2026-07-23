"""Compare Loom and vLLM SiLU-and-Mul at one PyTorch dispatcher boundary."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--width", type=int, default=11008)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--provider-order",
        choices=("loom-first", "vllm-first"),
        default="loom-first",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def latency_summary(samples_us: list[float]) -> dict[str, object]:
    return {
        "minimum_us": min(samples_us),
        "median_us": statistics.median(samples_us),
        "maximum_us": max(samples_us),
        "samples_us": samples_us,
    }


def warm_gpu(torch, seconds: float) -> None:
    if seconds == 0.0:
        return
    side = 4096
    left = torch.randn((side, side), device="cuda", dtype=torch.bfloat16)
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    deadline = time.perf_counter() + seconds
    while True:
        for _ in range(8):
            torch.mm(left, right, out=output)
        torch.cuda.synchronize()
        if time.perf_counter() >= deadline:
            break


def benchmark_provider(
    torch,
    operation: Callable[[], None],
    output,
    expected,
    args: argparse.Namespace,
) -> dict[str, object]:
    operation()
    torch.cuda.synchronize()
    difference = (output.float() - expected.float()).abs()
    max_abs_error = difference.max().item()
    max_rel_error = (
        difference / expected.float().abs().clamp_min(1.0e-8)
    ).max().item()
    bitwise_equal = torch.equal(output, expected)

    warm_gpu(torch, args.gpu_warmup_seconds)
    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()

    eager_samples_us: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            operation()
        end.record()
        end.synchronize()
        eager_samples_us.append(start.elapsed_time(end) * 1000.0 / args.iterations)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()

    graph_samples_us: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            graph.replay()
        end.record()
        end.synchronize()
        graph_samples_us.append(start.elapsed_time(end) * 1000.0 / args.iterations)

    return {
        "eager_dispatch_latency": latency_summary(eager_samples_us),
        "cuda_graph_replay_latency": latency_summary(graph_samples_us),
        "bitwise_equal_to_vllm": bitwise_equal,
        "max_abs_error_vs_vllm": max_abs_error,
        "max_rel_error_vs_vllm": max_rel_error,
    }


def main() -> None:
    args = parse_args()
    for name in ("rows", "width", "warmup", "iterations", "samples"):
        require_positive(name, getattr(args, name))
    if args.gpu_warmup_seconds < 0.0:
        raise ValueError(
            "gpu_warmup_seconds must be non-negative, "
            f"got {args.gpu_warmup_seconds}"
        )

    import torch
    import vllm

    from loom_kernels.torch_ops import bridge_abi_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(61)
    input_tensor = torch.randn(
        args.rows, args.width * 2, device="cuda", dtype=dtype
    )
    expected = torch.empty(args.rows, args.width, device="cuda", dtype=dtype)
    torch.ops._C.silu_and_mul(expected, input_tensor)
    torch.cuda.synchronize()

    provider_names = (
        ("loom_cuda", "vllm_c")
        if args.provider_order == "loom-first"
        else ("vllm_c", "loom_cuda")
    )
    providers: dict[str, object] = {}
    for provider in provider_names:
        output = torch.empty_like(expected)
        if provider == "loom_cuda":
            operation = lambda: torch.ops.loom_kernels.silu_and_mul(
                input_tensor, output
            )
        else:
            operation = lambda: torch.ops._C.silu_and_mul(output, input_tensor)
        providers[provider] = benchmark_provider(
            torch, operation, output, expected, args
        )

    loom_eager = providers["loom_cuda"]["eager_dispatch_latency"]["median_us"]
    vllm_eager = providers["vllm_c"]["eager_dispatch_latency"]["median_us"]
    loom_graph = providers["loom_cuda"]["cuda_graph_replay_latency"]["median_us"]
    vllm_graph = providers["vllm_c"]["cuda_graph_replay_latency"]["median_us"]
    report = {
        "benchmark": "silu_and_mul",
        "dispatch": "preallocated PyTorch C++ custom operators",
        "input_layout": "split-half [rows, 2 * width]",
        "dtype": args.dtype,
        "rows": args.rows,
        "width": args.width,
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "gpu_warmup_seconds_per_provider": args.gpu_warmup_seconds,
        "provider_order": args.provider_order,
        "providers": providers,
        "loom_eager_speedup_vs_vllm_c": vllm_eager / loom_eager,
        "loom_cuda_graph_speedup_vs_vllm_c": vllm_graph / loom_graph,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(0)
            ),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "bridge_abi_version": bridge_abi_version(),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
