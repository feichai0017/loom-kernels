"""Compare Loom fused SwiGLU+block-FP8 against vLLM on one dispatcher boundary."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--width", type=int, default=11008)
    parser.add_argument("--group-size", type=int, choices=(64, 128), default=128)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--gpu-warmup-seconds", type=float, default=1.0)
    parser.add_argument(
        "--provider-order", choices=("forward", "reverse"), default="forward"
    )
    return parser.parse_args()


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
    scales,
    expected_output,
    expected_scales,
    args: argparse.Namespace,
) -> dict[str, object]:
    operation()
    torch.cuda.synchronize()
    byte_mismatches = (
        output.view(torch.uint8) != expected_output.view(torch.uint8)
    ).sum().item()
    scale_difference = (scales - expected_scales).abs()
    max_scale_abs_error = scale_difference.max().item()
    max_scale_rel_error = (
        scale_difference / expected_scales.abs().clamp_min(1.0e-12)
    ).max().item()
    dequantized = output.float() * scales.repeat_interleave(
        args.group_size, dim=-1
    )
    expected_dequantized = expected_output.float() * expected_scales.repeat_interleave(
        args.group_size, dim=-1
    )
    max_dequantized_abs_error = (
        dequantized - expected_dequantized
    ).abs().max().item()

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
        "output_byte_mismatches_vs_vllm_fused": byte_mismatches,
        "max_scale_abs_error_vs_vllm_fused": max_scale_abs_error,
        "max_scale_rel_error_vs_vllm_fused": max_scale_rel_error,
        "max_dequantized_abs_error_vs_vllm_fused": max_dequantized_abs_error,
    }


def main() -> None:
    args = parse_args()
    for name in ("rows", "width", "group_size", "warmup", "iterations", "samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.width % args.group_size != 0:
        raise ValueError("width must be divisible by group_size")
    if args.gpu_warmup_seconds < 0.0:
        raise ValueError("gpu_warmup_seconds must be non-negative")

    import torch
    import vllm

    from loom_kernels.torch_ops import bridge_abi_version

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(79)
    input_tensor = torch.randn(
        args.rows, args.width * 2, device="cuda", dtype=dtype
    )
    expected_output = torch.empty(
        args.rows, args.width, device="cuda", dtype=torch.float8_e4m3fn
    )
    expected_scales = torch.empty(
        args.rows,
        args.width // args.group_size,
        device="cuda",
        dtype=torch.float32,
    )
    torch.ops._C.silu_and_mul_per_block_quant(
        expected_output,
        input_tensor,
        expected_scales,
        args.group_size,
        None,
        False,
    )
    torch.cuda.synchronize()

    forward_order = ("loom_cuda", "vllm_fused", "vllm_composed")
    provider_names = (
        forward_order if args.provider_order == "forward" else forward_order[::-1]
    )
    providers: dict[str, object] = {}
    for provider in provider_names:
        output = torch.empty_like(expected_output)
        scales = torch.empty_like(expected_scales)
        intermediate = torch.empty(
            args.rows, args.width, device="cuda", dtype=dtype
        )
        if provider == "loom_cuda":
            operation = lambda: torch.ops.loom_kernels.silu_and_mul_dynamic_fp8(
                input_tensor, output, scales, args.group_size
            )
        elif provider == "vllm_fused":
            operation = lambda: torch.ops._C.silu_and_mul_per_block_quant(
                output, input_tensor, scales, args.group_size, None, False
            )
        else:

            def operation() -> None:
                torch.ops._C.silu_and_mul(intermediate, input_tensor)
                torch.ops._C.per_token_group_fp8_quant(
                    intermediate,
                    output,
                    scales,
                    args.group_size,
                    1.0e-10,
                    -448.0,
                    448.0,
                    False,
                    False,
                    False,
                )

        providers[provider] = benchmark_provider(
            torch,
            operation,
            output,
            scales,
            expected_output,
            expected_scales,
            args,
        )

    def median(provider: str, mode: str) -> float:
        return providers[provider][mode]["median_us"]

    report = {
        "benchmark": "silu_and_mul_dynamic_per_block_fp8",
        "dispatch": "preallocated PyTorch C++ custom operators",
        "dtype": args.dtype,
        "rows": args.rows,
        "width": args.width,
        "group_size": args.group_size,
        "scale_layout": "row-major",
        "warmup": args.warmup,
        "iterations_per_sample": args.iterations,
        "samples": args.samples,
        "gpu_warmup_seconds_per_provider": args.gpu_warmup_seconds,
        "provider_order": args.provider_order,
        "providers": providers,
        "loom_eager_speedup_vs_vllm_fused": median(
            "vllm_fused", "eager_dispatch_latency"
        )
        / median("loom_cuda", "eager_dispatch_latency"),
        "loom_graph_speedup_vs_vllm_fused": median(
            "vllm_fused", "cuda_graph_replay_latency"
        )
        / median("loom_cuda", "cuda_graph_replay_latency"),
        "loom_eager_speedup_vs_vllm_composed": median(
            "vllm_composed", "eager_dispatch_latency"
        )
        / median("loom_cuda", "eager_dispatch_latency"),
        "loom_graph_speedup_vs_vllm_composed": median(
            "vllm_composed", "cuda_graph_replay_latency"
        )
        / median("loom_cuda", "cuda_graph_replay_latency"),
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
