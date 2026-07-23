#!/usr/bin/env python3
"""Run an isolated vLLM engine A/B for Loom's paged-decode fast path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


PROVIDERS = ("vllm", "loom")


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    input_len: int
    output_len: int

    @property
    def label(self) -> str:
        return f"b{self.batch_size}-in{self.input_len}-out{self.output_len}"

    @property
    def argument(self) -> str:
        return f"{self.batch_size}x{self.input_len}x{self.output_len}"


def parse_case(value: str) -> BenchmarkCase:
    try:
        dimensions = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT") from error
    if len(dimensions) != 3 or min(dimensions) <= 0:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    return BenchmarkCase(*dimensions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--block-size", type=int, choices=(16, 32), default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--provider-order",
        choices=("baseline-first", "loom-first"),
        default="baseline-first",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--internal-provider", choices=PROVIDERS, help=argparse.SUPPRESS)
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-cache-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 16, 16),
            BenchmarkCase(8, 16, 16),
            BenchmarkCase(32, 16, 16),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error("internal runs require result and cache paths")
    return args


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def make_prompts(case: BenchmarkCase) -> list[dict[str, list[int]]]:
    return [
        {
            "prompt_token_ids": [
                3 + ((batch * 17 + position * 13) % 1000)
                for position in range(case.input_len)
            ]
        }
        for batch in range(case.batch_size)
    ]


def run_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    import torch

    prompts = make_prompts(case)
    sampling = sampling_type(
        temperature=0.0, max_tokens=case.output_len, ignore_eos=True
    )
    for _ in range(warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    latencies_ms: list[float] = []
    throughputs: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)
        throughputs.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        token_ids = [list(request.outputs[0].token_ids) for request in outputs]
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")

    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "batch_latency_ms": summary(latencies_ms),
        "output_tokens_per_second": summary(throughputs),
        "token_ids": token_ids,
    }


def prepare_environment(cache_root: Path, provider: str) -> None:
    os.environ["LOOM_KERNELS_ENABLE_PAGED_DECODE_ATTENTION"] = (
        "1" if provider == "loom" else "0"
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    required = [str(Path(sys.executable).absolute().parent), str(cuda_home / "bin")]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required if entry not in current_entries] + current_entries
    )


def model_contract(model: str) -> dict[str, Any]:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return {"kind": "external", "config": None}
    config = json.loads(config_path.read_text())
    return {
        "kind": config.get("loom_fixture", "local-model"),
        "architecture": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "layers": config.get("num_hidden_layers"),
        "query_heads": config.get("num_attention_heads"),
        "kv_heads": config.get("num_key_value_heads"),
        "head_size": (
            config.get("hidden_size", 0) // config.get("num_attention_heads", 1)
        ),
        "vocab_size": config.get("vocab_size"),
    }


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve(), provider)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import provider_metadata

    reset_launch_count(Operator.PAGED_DECODE_ATTENTION)
    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    max_model_len = max(case.input_len + case.output_len for case in args.cases)
    engine = LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max(case.batch_size for case in args.cases),
        block_size=args.block_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        disable_log_stats=True,
    )
    launches_after_engine_init = launch_count(Operator.PAGED_DECODE_ATTENTION)
    cases = [
        run_case(engine, SamplingParams, case, args.warmup, args.repeats)
        for case in args.cases
    ]
    host_launch_count = launch_count(Operator.PAGED_DECODE_ATTENTION)
    report = {
        "provider": provider,
        "model": model,
        "model_contract": model_contract(model),
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "block_size": args.block_size,
        "cases": cases,
        "loom_path": {
            "launches_after_engine_init": launches_after_engine_init,
            "host_launch_count": host_launch_count,
            "provider_metadata": provider_metadata(),
            "counter_semantics": (
                "host submissions during graph construction or eager execution; "
                "CUDA Graph replays do not increment this counter"
            ),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "attention_backend": os.environ["VLLM_ATTENTION_BACKEND"],
        },
    }
    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"provider={provider} host_launch_count={host_launch_count}",
        file=sys.stderr,
    )
    return report


def child_command(
    args: argparse.Namespace, provider: str, result: Path, cache_root: Path
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--internal-provider",
        provider,
        "--internal-result",
        str(result),
        "--internal-cache-root",
        str(cache_root),
    ]
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ("vllm", "loom")
        if args.provider_order == "baseline-first"
        else ("loom", "vllm")
    )
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="loom-vllm-paged-decode-") as directory:
        root = Path(directory)
        for provider in order:
            result = root / f"{provider}.json"
            subprocess.run(
                child_command(args, provider, result, root / f"{provider}-cache"),
                check=True,
            )
            reports[provider] = json.loads(result.read_text())

    comparisons: list[dict[str, Any]] = []
    tokens_match = True
    for baseline, loom in zip(
        reports["vllm"]["cases"], reports["loom"]["cases"], strict=True
    ):
        case_tokens_match = baseline["token_ids"] == loom["token_ids"]
        tokens_match = tokens_match and case_tokens_match
        baseline_latency = baseline["batch_latency_ms"]["median"]
        loom_latency = loom["batch_latency_ms"]["median"]
        comparisons.append(
            {
                "case": baseline["case"],
                "token_ids_match": case_tokens_match,
                "baseline_over_loom_batch_latency": baseline_latency / loom_latency,
                "loom_over_baseline_output_throughput": (
                    loom["output_tokens_per_second"]["median"]
                    / baseline["output_tokens_per_second"]["median"]
                ),
            }
        )

    baseline_launches = reports["vllm"]["loom_path"]["host_launch_count"]
    loom_launches = reports["loom"]["loom_path"]["host_launch_count"]
    accepted = tokens_match and baseline_launches == 0 and loom_launches > 0
    report = {
        "benchmark": "vllm_engine_paged_decode_ab",
        "model": args.model,
        "model_contract": reports["vllm"]["model_contract"],
        "provider_order": list(order),
        "acceptance": {
            "passed": accepted,
            "token_ids_match": tokens_match,
            "baseline_host_launch_count": baseline_launches,
            "loom_host_launch_count": loom_launches,
        },
        "comparisons": comparisons,
        "providers": reports,
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n")
    print(rendered)
    if not accepted:
        raise SystemExit("vLLM engine paged-decode acceptance gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is None:
        run_controller(args)
    else:
        run_provider(args)


if __name__ == "__main__":
    main()
