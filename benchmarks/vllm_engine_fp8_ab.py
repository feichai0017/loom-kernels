"""Run a provider-isolated real-model vLLM FP8 A/B benchmark.

The controller starts the native vLLM baseline and Loom in separate Python
processes so model teardown, CUDA Graph state, and fusion-table registration
cannot leak across providers. Each child uses the same pretrained checkpoint,
online FP8 block quantization, prompts, and generation settings.
"""

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


MODEL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION_DEFAULT = "7ae557604adf67be50417f59c2c2f167def9a775"
LINEAR_BACKEND_DEFAULT = "cutlass"
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
    parts = value.lower().split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("case must be BATCHxINPUTxOUTPUT")
    try:
        batch_size, input_len, output_len = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case dimensions must be positive integers"
        ) from error
    if min(batch_size, input_len, output_len) <= 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive")
    return BenchmarkCase(batch_size, input_len, output_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--model-revision",
        default=MODEL_REVISION_DEFAULT,
        help="Pinned Hugging Face revision; recorded for local snapshots too.",
    )
    parser.add_argument("--quantization", default="fp8_per_block")
    parser.add_argument(
        "--linear-backend",
        default=LINEAR_BACKEND_DEFAULT,
        help=(
            "vLLM FP8 GEMM backend. Cutlass keeps activation quantization as "
            "a graph node so the SiLU+quant provider can be compared."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help="Repeatable BATCHxINPUTxOUTPUT workload.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--provider-order",
        choices=("baseline-first", "loom-first"),
        default="baseline-first",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-provider", choices=PROVIDERS, help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-cache-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 128, 128),
            BenchmarkCase(8, 128, 128),
            BenchmarkCase(32, 128, 64),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.internal_provider is not None and (
        args.internal_result is None or args.internal_cache_root is None
    ):
        parser.error(
            "internal provider runs require --internal-result and "
            "--internal-cache-root"
        )
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
                3 + ((batch_index * 17 + position * 13) % 1000)
                for position in range(case.input_len)
            ]
        }
        for batch_index in range(case.batch_size)
    ]


def request_metrics(outputs: list[Any]) -> tuple[list[float], list[float], list[float]]:
    ttft_ms: list[float] = []
    tpot_ms: list[float] = []
    e2e_ms: list[float] = []
    for output in outputs:
        metrics = output.metrics
        if metrics is None or metrics.is_corrupted:
            continue
        if metrics.first_token_latency > 0.0:
            ttft = metrics.first_token_latency * 1000.0
            ttft_ms.append(ttft)
        else:
            ttft = 0.0
        generated = metrics.num_generation_tokens
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        if generated > 1 and decode_seconds >= 0.0:
            tpot_ms.append(decode_seconds * 1000.0 / (generated - 1))
        if ttft > 0.0 and decode_seconds >= 0.0:
            e2e_ms.append(ttft + decode_seconds * 1000.0)
    return ttft_ms, tpot_ms, e2e_ms


def run_case(
    engine: Any,
    sampling_type: Any,
    case: BenchmarkCase,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    prompts = make_prompts(case)
    sampling = sampling_type(
        temperature=0.0,
        max_tokens=case.output_len,
        ignore_eos=True,
    )
    for _ in range(args.warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    latency_ms: list[float] = []
    throughput: list[float] = []
    all_ttft_ms: list[float] = []
    all_tpot_ms: list[float] = []
    all_e2e_ms: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latency_ms.append(elapsed_ms)
        throughput.append(
            case.batch_size * case.output_len / (elapsed_ms / 1000.0)
        )
        ttft_ms, tpot_ms, e2e_ms = request_metrics(outputs)
        all_ttft_ms.extend(ttft_ms)
        all_tpot_ms.extend(tpot_ms)
        all_e2e_ms.extend(e2e_ms)
        token_ids = [list(request.outputs[0].token_ids) for request in outputs]
        if any(len(tokens) != case.output_len for tokens in token_ids):
            raise RuntimeError("vLLM returned an unexpected output length")

    return {
        "case": case.label,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "batch_latency_ms": summary(latency_ms),
        "request_ttft_ms": summary(all_ttft_ms),
        "request_tpot_ms": summary(all_tpot_ms),
        "request_e2e_ms": summary(all_e2e_ms),
        "output_tokens_per_second": summary(throughput),
        "token_ids": token_ids,
    }


def run_provider(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.internal_provider
    assert provider is not None
    os.environ["LOOM_KERNELS_ENABLE_SILU_AND_MUL"] = "0"
    os.environ["LOOM_KERNELS_ENABLE_SILU_AND_MUL_FP8"] = (
        "1" if provider == "loom" else "0"
    )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.pop("VLLM_DISABLED_KERNELS", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    assert args.internal_cache_root is not None
    cache_root = args.internal_cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CACHE_ROOT"] = str(cache_root / "vllm")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "torchinductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache_root / "triton")

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(
            f"nvcc was not found at {nvcc}; set CUDA_HOME to the active toolkit"
        )
    os.environ["CUDA_HOME"] = str(cuda_home)
    venv_bin = str(Path(sys.executable).absolute().parent)
    current_path = os.environ.get("PATH", "")
    path_entries = current_path.split(os.pathsep)
    required_entries = [venv_bin, str(cuda_home / "bin")]
    missing_entries = [
        entry for entry in required_entries if entry not in path_entries
    ]
    if missing_entries:
        os.environ["PATH"] = os.pathsep.join(missing_entries + path_entries)

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
        register_vllm_silu_and_mul_dynamic_fp8,
    )
    from vllm.compilation.passes.fusion.act_quant_fusion import FUSED_OPS
    from vllm.compilation.passes.vllm_inductor_pass import get_match_table
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
    )

    loom_operator = torch.ops.loom_kernels.silu_and_mul_per_block_fp8.default
    fusion_table_uses_loom_before_registration = (
        FUSED_OPS[kFp8Dynamic128Sym] == loom_operator
    )
    explicit_registration = None
    if provider == "loom":
        # vLLM captures the selected replacement while constructing its fusion
        # pass. Install this process-local override before constructing LLM so
        # the benchmark does not depend on plugin-discovery timing.
        explicit_registration = register_vllm_silu_and_mul_dynamic_fp8()
        if explicit_registration is None:
            raise RuntimeError("Loom FP8 activation fusion registration failed")
    fusion_table_uses_loom_before_engine = (
        FUSED_OPS[kFp8Dynamic128Sym] == loom_operator
    )
    reset_launch_count(Operator.SILU_AND_MUL_DYNAMIC_FP8)

    model_path = Path(args.model).expanduser()
    model_is_local = model_path.exists()
    model = str(model_path.resolve()) if model_is_local else args.model
    max_model_len = max(case.input_len + case.output_len for case in args.cases)
    max_num_seqs = max(case.batch_size for case in args.cases)
    engine_arguments: dict[str, Any] = {
        "model": model,
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "quantization": args.quantization,
        "linear_backend": args.linear_backend,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
        "compilation_config": {
            "custom_ops": ["+quant_fp8"],
            "pass_config": {"fuse_act_quant": True},
        },
    }
    if args.model_revision and not model_is_local:
        engine_arguments["revision"] = args.model_revision
    engine = LLM(**engine_arguments)
    launches_after_engine_init = launch_count(
        Operator.SILU_AND_MUL_DYNAMIC_FP8
    )
    fusion_matches_after_engine = get_match_table()

    fusion_table_uses_loom = FUSED_OPS[kFp8Dynamic128Sym] == loom_operator
    cases = [run_case(engine, SamplingParams, case, args) for case in args.cases]
    host_launch_count = launch_count(Operator.SILU_AND_MUL_DYNAMIC_FP8)
    fusion_matches_after_cases = get_match_table()
    report = {
        "provider": provider,
        "model": model,
        "model_source": args.model,
        "model_revision": args.model_revision,
        "model_kind": "local-checkpoint" if model_is_local else "huggingface",
        "dtype": "bfloat16",
        "quantization": args.quantization,
        "linear_backend": args.linear_backend,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "cases": cases,
        "loom_path": {
            "fusion_table_uses_loom_before_registration": (
                fusion_table_uses_loom_before_registration
            ),
            "explicit_registration": explicit_registration,
            "fusion_table_uses_loom_before_engine": (
                fusion_table_uses_loom_before_engine
            ),
            "fusion_table_uses_loom": fusion_table_uses_loom,
            "launches_after_engine_init": launches_after_engine_init,
            "host_launch_count": host_launch_count,
            "fusion_matches_after_engine": fusion_matches_after_engine,
            "fusion_matches_after_cases": fusion_matches_after_cases,
            "counter_semantics": (
                "host submissions during graph construction or eager execution; "
                "CUDA Graph replays do not increment this counter"
            ),
            "provider_metadata": provider_metadata(),
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm.__version__,
            "cuda_home": str(cuda_home),
            "v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
            "vllm_cache_root": os.environ["VLLM_CACHE_ROOT"],
        },
    }
    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"provider={provider} loom_table={fusion_table_uses_loom} "
        f"loom_host_launches={launch_count}",
        file=sys.stderr,
    )
    return report


def child_command(
    args: argparse.Namespace,
    provider: str,
    result: Path,
    cache_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--model-revision",
        args.model_revision,
        "--quantization",
        args.quantization,
        "--linear-backend",
        args.linear_backend,
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
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


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def median_metric(case: dict[str, Any], name: str) -> float | None:
    metric = case[name]
    return None if metric is None else float(metric["median"])


def compare_reports(
    reports: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    comparisons: list[dict[str, Any]] = []
    tokens_match = True
    for baseline, loom in zip(
        reports["vllm"]["cases"], reports["loom"]["cases"], strict=True
    ):
        if baseline["case"] != loom["case"]:
            raise RuntimeError("provider reports contain different cases")
        case_tokens_match = baseline["token_ids"] == loom["token_ids"]
        tokens_match = tokens_match and case_tokens_match
        baseline_batch = median_metric(baseline, "batch_latency_ms")
        loom_batch = median_metric(loom, "batch_latency_ms")
        baseline_ttft = median_metric(baseline, "request_ttft_ms")
        loom_ttft = median_metric(loom, "request_ttft_ms")
        baseline_tpot = median_metric(baseline, "request_tpot_ms")
        loom_tpot = median_metric(loom, "request_tpot_ms")
        baseline_throughput = median_metric(baseline, "output_tokens_per_second")
        loom_throughput = median_metric(loom, "output_tokens_per_second")
        comparisons.append(
            {
                "case": baseline["case"],
                "token_ids_match": case_tokens_match,
                "baseline_over_loom_batch_latency": ratio(
                    baseline_batch, loom_batch
                ),
                "baseline_over_loom_ttft": ratio(baseline_ttft, loom_ttft),
                "baseline_over_loom_tpot": ratio(baseline_tpot, loom_tpot),
                "loom_over_baseline_output_throughput": ratio(
                    loom_throughput, baseline_throughput
                ),
            }
        )
    return comparisons, tokens_match


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = (
        ["vllm", "loom"]
        if args.provider_order == "baseline-first"
        else ["loom", "vllm"]
    )
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="loom-vllm-fp8-ab-") as directory:
        root = Path(directory)
        for provider in order:
            result = root / f"{provider}.json"
            cache_root = root / f"{provider}-cache"
            subprocess.run(
                child_command(args, provider, result, cache_root), check=True
            )
            reports[provider] = json.loads(result.read_text(encoding="utf-8"))

    comparisons, tokens_match = compare_reports(reports)
    baseline_clean = not reports["vllm"]["loom_path"]["fusion_table_uses_loom"]
    loom_registered = reports["loom"]["loom_path"]["fusion_table_uses_loom"]
    loom_launched = reports["loom"]["loom_path"]["host_launch_count"] > 0
    baseline_matches = reports["vllm"]["loom_path"][
        "fusion_matches_after_engine"
    ].get("activation_quant_fusion_pass", 0)
    loom_matches = reports["loom"]["loom_path"][
        "fusion_matches_after_engine"
    ].get("activation_quant_fusion_pass", 0)
    providers_matched_same_graph = baseline_matches > 0 and (
        baseline_matches == loom_matches
    )
    accepted = (
        tokens_match
        and baseline_clean
        and loom_registered
        and loom_launched
        and providers_matched_same_graph
    )
    report = {
        "benchmark": "vllm_real_model_fp8_ab",
        "model": args.model,
        "model_revision": args.model_revision,
        "quantization": args.quantization,
        "linear_backend": args.linear_backend,
        "provider_order": order,
        "acceptance": {
            "passed": accepted,
            "token_ids_match": tokens_match,
            "baseline_uses_native_fusion": baseline_clean,
            "loom_fusion_registered": loom_registered,
            "loom_host_launch_observed": loom_launched,
            "baseline_activation_quant_matches": baseline_matches,
            "loom_activation_quant_matches": loom_matches,
            "providers_matched_same_graph": providers_matched_same_graph,
        },
        "comparisons": comparisons,
        "providers": reports,
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not accepted:
        raise SystemExit("real-model FP8 A/B acceptance gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_provider is not None:
        run_provider(args)
    else:
        run_controller(args)


if __name__ == "__main__":
    main()
