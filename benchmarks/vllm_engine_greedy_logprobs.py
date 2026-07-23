"""Run isolated real-engine A/B for Loom sampled-token logprobs."""

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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--sampling-mode",
        choices=("greedy", "top-k-top-p"),
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
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
            BenchmarkCase(1, 32, 64),
            BenchmarkCase(8, 32, 64),
            BenchmarkCase(32, 32, 32),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.temperature <= 0.0 or args.top_k <= 0 or not 0.0 < args.top_p <= 1.0:
        parser.error("temperature/top-k must be positive and top-p must be in (0, 1]")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error("internal runs require result and cache paths")
    return args


def summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
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


def request_metrics(outputs: list[Any]) -> tuple[list[float], list[float]]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        if metrics.first_token_latency > 0.0:
            ttft_ms.append(metrics.first_token_latency * 1000.0)
        generated = metrics.num_generation_tokens
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        if generated > 1 and decode_seconds >= 0.0:
            tpot_ms.append(decode_seconds * 1000.0 / (generated - 1))
    return ttft_ms, tpot_ms


def selected_logprob_data(
    outputs: list[Any],
) -> tuple[list[list[float]], list[list[int]]]:
    values: list[list[float]] = []
    ranks: list[list[int]] = []
    for request in outputs:
        completion = request.outputs[0]
        if completion.logprobs is None:
            raise RuntimeError("vLLM did not return requested sampled-token logprobs")
        request_values: list[float] = []
        request_ranks: list[int] = []
        for token_id, step in zip(
            completion.token_ids, completion.logprobs, strict=True
        ):
            selected = step[token_id]
            if selected.rank is None:
                raise RuntimeError("vLLM did not return a sampled-token rank")
            request_values.append(float(selected.logprob))
            request_ranks.append(int(selected.rank))
        values.append(request_values)
        ranks.append(request_ranks)
    return values, ranks


def run_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    warmup: int,
    repeats: int,
    sampling_mode: str,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> dict[str, Any]:
    import torch

    prompts = make_prompts(case)
    sampling_arguments: dict[str, Any] = {
        "max_tokens": case.output_len,
        "ignore_eos": True,
        "logprobs": 0,
    }
    if sampling_mode == "greedy":
        sampling_arguments["temperature"] = 0.0
    else:
        sampling_arguments.update(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
    sampling = sampling_type(**sampling_arguments)
    for _ in range(warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    latencies_ms: list[float] = []
    throughputs: list[float] = []
    all_ttft_ms: list[float] = []
    all_tpot_ms: list[float] = []
    token_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    ranks: list[list[int]] = []
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
        ttft_ms, tpot_ms = request_metrics(outputs)
        all_ttft_ms.extend(ttft_ms)
        all_tpot_ms.extend(tpot_ms)
        token_ids = [list(request.outputs[0].token_ids) for request in outputs]
        logprobs, ranks = selected_logprob_data(outputs)
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")

    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "batch_latency_ms": summary(latencies_ms),
        "request_ttft_ms": summary(all_ttft_ms),
        "request_tpot_ms": summary(all_tpot_ms),
        "output_tokens_per_second": summary(throughputs),
        "token_ids": token_ids,
        "sampled_token_logprobs": logprobs,
        "sampled_token_ranks": ranks,
    }


def prepare_environment(cache_root: Path) -> None:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        raise RuntimeError(f"nvcc was not found under {cuda_home}")
    os.environ["CUDA_HOME"] = str(cuda_home)
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    required = [str(Path(sys.executable).absolute().parent), str(cuda_home / "bin")]
    os.environ["PATH"] = os.pathsep.join(
        [entry for entry in required if entry not in current_entries] + current_entries
    )


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve())

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import (
        provider_metadata,
        register_vllm_greedy_sample_logprobs,
        register_vllm_selected_token_logprobs,
    )

    explicit_registration = None
    if provider == "loom":
        registration = (
            register_vllm_greedy_sample_logprobs
            if args.sampling_mode == "greedy"
            else register_vllm_selected_token_logprobs
        )
        explicit_registration = registration()
        if explicit_registration is None:
            raise RuntimeError("Loom sampled-token logprob registration failed")
    if args.sampling_mode == "greedy":
        operator = Operator.GREEDY_SAMPLE_LOGPROBS
    else:
        operator = Operator.SELECTED_TOKEN_LOGPROBS
    reset_launch_count(operator)
    launch_count_fn = lambda: launch_count(operator)

    model_path = Path(args.model).expanduser()
    model = str(model_path.resolve()) if model_path.exists() else args.model
    max_model_len = max(case.input_len + case.output_len for case in args.cases)
    engine = LLM(
        model=model,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        max_num_seqs=max(case.batch_size for case in args.cases),
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        disable_log_stats=False,
    )
    launches_after_engine_init = launch_count_fn()
    cases = [
        run_case(
            engine,
            SamplingParams,
            case,
            args.warmup,
            args.repeats,
            args.sampling_mode,
            args.seed,
            args.temperature,
            args.top_k,
            args.top_p,
        )
        for case in args.cases
    ]
    host_launch_count = launch_count_fn()
    report = {
        "provider": provider,
        "model": model,
        "dtype": "bfloat16 logits observed by the sampler",
        "sampling": (
            "temperature=0, logprobs=0, ignore_eos=true"
            if args.sampling_mode == "greedy"
            else (
                f"temperature={args.temperature}, top_k={args.top_k}, "
                f"top_p={args.top_p}, seed={args.seed}, logprobs=0, "
                "ignore_eos=true"
            )
        ),
        "sampling_mode": args.sampling_mode,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "cases": cases,
        "loom_path": {
            "explicit_registration": explicit_registration,
            "launches_after_engine_init": launches_after_engine_init,
            "host_launch_count": host_launch_count,
            "provider_metadata": provider_metadata(),
            "counter_semantics": (
                "host submissions through the sampler fast path; this sampler "
                "boundary executes outside the model CUDA Graph"
            ),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
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
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--sampling-mode",
        args.sampling_mode,
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
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


def maximum_logprob_error(
    baseline: list[list[float]], loom: list[list[float]]
) -> float:
    return max(
        abs(expected - actual)
        for expected_request, actual_request in zip(baseline, loom, strict=True)
        for expected, actual in zip(expected_request, actual_request, strict=True)
    )


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ("vllm", "loom")
        if args.provider_order == "baseline-first"
        else ("loom", "vllm")
    )
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="loom-vllm-greedy-logprobs-") as directory:
        root = Path(directory)
        for provider in order:
            result = root / f"{provider}.json"
            subprocess.run(
                child_command(args, provider, result, root / f"{provider}-cache"),
                check=True,
            )
            reports[provider] = json.loads(result.read_text())

    comparisons: list[dict[str, Any]] = []
    outputs_match = True
    for baseline, loom in zip(
        reports["vllm"]["cases"], reports["loom"]["cases"], strict=True
    ):
        token_ids_match = baseline["token_ids"] == loom["token_ids"]
        ranks_match = (
            baseline["sampled_token_ranks"] == loom["sampled_token_ranks"]
        )
        logprob_error = maximum_logprob_error(
            baseline["sampled_token_logprobs"], loom["sampled_token_logprobs"]
        )
        case_matches = token_ids_match and ranks_match and logprob_error <= 2.0e-5
        outputs_match = outputs_match and case_matches
        baseline_latency = baseline["batch_latency_ms"]["median"]
        loom_latency = loom["batch_latency_ms"]["median"]
        baseline_tpot = baseline["request_tpot_ms"]["median"]
        loom_tpot = loom["request_tpot_ms"]["median"]
        comparisons.append(
            {
                "case": baseline["case"],
                "token_ids_match": token_ids_match,
                "sampled_token_ranks_match": ranks_match,
                "maximum_sampled_logprob_error": logprob_error,
                "baseline_over_loom_batch_latency": (
                    baseline_latency / loom_latency
                ),
                "baseline_over_loom_tpot": baseline_tpot / loom_tpot,
            }
        )

    baseline_launches = reports["vllm"]["loom_path"]["host_launch_count"]
    loom_launches = reports["loom"]["loom_path"]["host_launch_count"]
    accepted = outputs_match and baseline_launches == 0 and loom_launches > 0
    report = {
        "benchmark": f"vllm_engine_{args.sampling_mode}_sample_logprobs_ab",
        "model": args.model,
        "provider_order": list(order),
        "acceptance": {
            "passed": accepted,
            "outputs_match": outputs_match,
            "logprob_atol": 2.0e-5,
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
        raise SystemExit("vLLM greedy sampled-logprob acceptance gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is None:
        run_controller(args)
    else:
        run_provider(args)


if __name__ == "__main__":
    main()
