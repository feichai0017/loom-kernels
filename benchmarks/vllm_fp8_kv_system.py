"""Measure the system boundary of native and static-FP8 KV caches.

The benchmark keeps three concerns separate:

* ``native-vllm`` measures vLLM's BF16 KV-cache operator path;
* ``fp8-vllm`` measures vLLM's static FP8 KV-cache operator path;
* ``fp8-loom`` changes only the FP8 RoPE+cache-write fusion boundary.

Each variant runs in a fresh process. The report records cache capacity,
process-local CUDA memory, TTFT, TPOT, throughput, generated-token divergence,
and optional corpus perplexity. A fast cache-write kernel alone cannot pass the
system-value gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


VARIANTS = ("native-vllm", "fp8-vllm", "fp8-loom")
VARIANT_ORDERS = {
    "native-first": VARIANTS,
    "fp8-first": tuple(reversed(VARIANTS)),
}


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
    parser.add_argument(
        "--model-revision",
        help=(
            "Pinned model revision or checkpoint digest; required by the "
            "system-value gate and recorded even when --model is local."
        ),
    )
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-fused-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--variant-order",
        choices=tuple(VARIANT_ORDERS),
        default="native-first",
    )
    parser.add_argument(
        "--quality-jsonl",
        type=Path,
        help="Optional JSONL corpus with one non-empty 'text' field per row.",
    )
    parser.add_argument("--quality-max-sequences", type=int, default=64)
    parser.add_argument("--quality-max-tokens", type=int, default=2048)
    parser.add_argument("--min-capacity-ratio", type=float, default=1.8)
    parser.add_argument("--max-perplexity-ratio", type=float, default=1.02)
    parser.add_argument("--max-tpot-regression", type=float, default=1.05)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument(
        "--internal-variant", choices=VARIANTS, help=argparse.SUPPRESS
    )
    parser.add_argument("--internal-result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-cache-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cases is None:
        args.cases = [
            BenchmarkCase(1, 4096, 128),
            BenchmarkCase(8, 2048, 128),
            BenchmarkCase(32, 512, 128),
        ]
    if args.warmup <= 0 or args.repeats <= 0:
        parser.error("warmup and repeats must be positive")
    if args.max_fused_tokens <= 0:
        parser.error("max-fused-tokens must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        parser.error("gpu-memory-utilization must be between zero and one")
    if args.quality_max_sequences <= 0 or args.quality_max_tokens <= 1:
        parser.error("quality limits must be positive and include at least two tokens")
    if args.min_capacity_ratio <= 1.0:
        parser.error("min-capacity-ratio must be greater than one")
    if args.max_perplexity_ratio < 1.0 or args.max_tpot_regression < 1.0:
        parser.error("quality and TPOT limits must be at least one")

    required_model_len = max(
        max(case.input_len + case.output_len for case in args.cases),
        args.quality_max_tokens + 1 if args.quality_jsonl is not None else 0,
    )
    if args.max_model_len is None:
        args.max_model_len = required_model_len
    elif args.max_model_len < required_model_len:
        parser.error(
            f"max-model-len must be at least {required_model_len} for this workload"
        )
    if args.max_num_seqs is None:
        args.max_num_seqs = max(case.batch_size for case in args.cases)
    elif args.max_num_seqs < max(case.batch_size for case in args.cases):
        parser.error("max-num-seqs cannot be smaller than a benchmark batch")

    if args.quality_jsonl is not None:
        args.quality_jsonl = args.quality_jsonl.expanduser().resolve()
        if not args.quality_jsonl.is_file():
            parser.error(f"quality corpus does not exist: {args.quality_jsonl}")
    if args.internal_variant is not None and (
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


def cuda_memory_snapshot(torch: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
    }


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
        temperature=0.0,
        max_tokens=case.output_len,
        ignore_eos=True,
    )
    for _ in range(warmup):
        engine.generate(prompts, sampling, use_tqdm=False)

    torch.cuda.reset_peak_memory_stats()
    latency_ms: list[float] = []
    throughput: list[float] = []
    all_ttft_ms: list[float] = []
    all_tpot_ms: list[float] = []
    all_e2e_ms: list[float] = []
    token_ids: list[list[int]] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
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
        "cuda_memory": cuda_memory_snapshot(torch),
        "token_ids": token_ids,
    }


def cache_capacity(engine: Any, max_model_len: int, max_num_seqs: int) -> dict[str, Any]:
    cache = engine.llm_engine.vllm_config.cache_config
    required = {
        "num_gpu_blocks": cache.num_gpu_blocks,
        "block_size": cache.block_size,
        "kv_cache_size_tokens": cache.kv_cache_size_tokens,
        "kv_cache_max_concurrency": cache.kv_cache_max_concurrency,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "vLLM did not expose initialized cache capacity: " + ", ".join(missing)
        )
    capacity_tokens = int(cache.kv_cache_size_tokens)
    return {
        "cache_dtype": cache.cache_dtype,
        "num_gpu_blocks": int(cache.num_gpu_blocks),
        "block_size": int(cache.block_size),
        "kv_cache_size_tokens": capacity_tokens,
        "kv_cache_max_concurrency": float(cache.kv_cache_max_concurrency),
        "configured_max_model_len": max_model_len,
        "configured_max_num_seqs": max_num_seqs,
        "capacity_limited_concurrency": capacity_tokens / max_model_len,
        "effective_admitted_concurrency": min(
            capacity_tokens / max_model_len, float(max_num_seqs)
        ),
    }


def model_kv_geometry(engine: Any, cache_dtype: str) -> dict[str, Any]:
    config = engine.model_config.hf_config
    layers = int(config.num_hidden_layers)
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", attention_heads))
    head_size = int(
        getattr(config, "head_dim", config.hidden_size // attention_heads)
    )
    element_bytes = 1 if cache_dtype == "fp8" else 2
    bytes_per_token = 2 * layers * kv_heads * head_size * element_bytes
    return {
        "num_hidden_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_size": head_size,
        "cache_element_bytes": element_bytes,
        "theoretical_kv_bytes_per_token": bytes_per_token,
    }


def load_quality_texts(path: Path, limit: int) -> tuple[list[str], str]:
    payload = path.read_bytes()
    texts: list[str] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        text = value.get("text") if isinstance(value, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number} must contain non-empty text")
        texts.append(text)
        if len(texts) == limit:
            break
    if not texts:
        raise ValueError(f"{path} contains no quality texts")
    return texts, hashlib.sha256(payload).hexdigest()


def run_quality(
    engine: Any,
    sampling_type: Any,
    path: Path,
    max_sequences: int,
    max_tokens: int,
) -> dict[str, Any]:
    texts, dataset_sha256 = load_quality_texts(path, max_sequences)
    tokenizer = engine.get_tokenizer()
    tokenized: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    for text in texts:
        token_ids = list(tokenizer.encode(text, add_special_tokens=False))[:max_tokens]
        if len(token_ids) < 2:
            continue
        tokenized.append({"prompt_token_ids": token_ids})
        lengths.append(len(token_ids))
    if not tokenized:
        raise RuntimeError("quality corpus produced no sequences with two tokens")

    sampling = sampling_type(
        temperature=0.0,
        max_tokens=1,
        ignore_eos=True,
        prompt_logprobs=0,
    )
    outputs = engine.generate(tokenized, sampling, use_tqdm=False)
    total_nll = 0.0
    scored_tokens = 0
    sequence_mean_nll: list[float] = []
    for output in outputs:
        prompt_token_ids = output.prompt_token_ids
        prompt_logprobs = output.prompt_logprobs
        if prompt_token_ids is None or prompt_logprobs is None:
            raise RuntimeError("vLLM did not return requested prompt logprobs")
        if len(prompt_token_ids) != len(prompt_logprobs):
            raise RuntimeError("prompt token and logprob lengths differ")
        sequence_nll = 0.0
        sequence_tokens = 0
        for token_id, position in zip(
            prompt_token_ids[1:], prompt_logprobs[1:], strict=True
        ):
            if position is None or token_id not in position:
                raise RuntimeError("selected prompt token logprob is missing")
            sequence_nll -= float(position[token_id].logprob)
            sequence_tokens += 1
        total_nll += sequence_nll
        scored_tokens += sequence_tokens
        sequence_mean_nll.append(sequence_nll / sequence_tokens)
    mean_nll = total_nll / scored_tokens
    return {
        "dataset": str(path),
        "dataset_sha256": dataset_sha256,
        "sequences": len(outputs),
        "sequence_token_lengths": lengths,
        "scored_tokens": scored_tokens,
        "total_negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "sequence_mean_negative_log_likelihood": sequence_mean_nll,
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


def run_variant(args: argparse.Namespace) -> dict[str, Any]:
    variant = args.internal_variant
    assert variant is not None and args.internal_cache_root is not None
    prepare_environment(args.internal_cache_root.resolve())

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    from loom_kernels import native_build_info
    from loom_kernels.torch_ops import (
        Operator,
        launch_count,
        reset_launch_count,
    )
    from loom_kernels.vllm import configure_vllm_rope_paged_kv, provider_metadata

    cache_dtype = "auto" if variant == "native-vllm" else "fp8"
    use_loom = variant == "fp8-loom"
    # Keep graph partitioning and operator opacity identical across variants.
    # The only compiler difference is that fp8-loom enables vLLM's official
    # RoPE+KV fusion pass and replaces its fused implementation.
    compilation_config = CompilationConfig(
        custom_ops=["+rotary_embedding", "+quant_fp8"],
        splitting_ops=[],
    )
    if use_loom:
        compilation_config = configure_vllm_rope_paged_kv(
            compilation_config,
            max_token_num=args.max_fused_tokens,
        )

    reset_launch_count(Operator.ROPE_PAGED_KV_WRITE)
    model_path = Path(args.model).expanduser()
    model_is_local = model_path.exists()
    model = str(model_path.resolve()) if model_is_local else args.model
    engine_arguments: dict[str, Any] = {
        "model": model,
        "skip_tokenizer_init": args.quality_jsonl is None,
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": False,
        "enable_prefix_caching": False,
        "kv_cache_dtype": cache_dtype,
    }
    engine_arguments["compilation_config"] = compilation_config
    if args.model_revision is not None and not model_is_local:
        engine_arguments["revision"] = args.model_revision

    torch.cuda.reset_peak_memory_stats()
    init_started = time.perf_counter()
    engine = LLM(**engine_arguments)
    torch.cuda.synchronize()
    init_seconds = time.perf_counter() - init_started
    memory_after_init = cuda_memory_snapshot(torch)
    capacity = cache_capacity(engine, args.max_model_len, args.max_num_seqs)
    geometry = model_kv_geometry(engine, cache_dtype)
    launches_after_init = launch_count(Operator.ROPE_PAGED_KV_WRITE)

    cases = [
        run_case(
            engine,
            SamplingParams,
            case,
            args.warmup,
            args.repeats,
        )
        for case in args.cases
    ]
    quality = (
        run_quality(
            engine,
            SamplingParams,
            args.quality_jsonl,
            args.quality_max_sequences,
            args.quality_max_tokens,
        )
        if args.quality_jsonl is not None
        else None
    )
    host_launch_count = launch_count(Operator.ROPE_PAGED_KV_WRITE)
    report = {
        "variant": variant,
        "model": model,
        "model_source": args.model,
        "model_revision": args.model_revision,
        "model_kind": "local-checkpoint" if model_is_local else "huggingface",
        "dtype": "bfloat16",
        "kv_cache_dtype": cache_dtype,
        "engine_init_seconds": init_seconds,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cache_capacity": capacity,
        "model_kv_geometry": geometry,
        "memory_after_engine_init": memory_after_init,
        "cases": cases,
        "quality": quality,
        "loom_path": {
            "enabled": use_loom,
            "launches_after_engine_init": launches_after_init,
            "host_launch_count": host_launch_count,
            "provider_metadata": provider_metadata(),
            "native_build_info": native_build_info(),
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
            "cuda_home": os.environ["CUDA_HOME"],
        },
    }
    assert args.internal_result is not None
    args.internal_result.parent.mkdir(parents=True, exist_ok=True)
    args.internal_result.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"variant={variant} cache_tokens={capacity['kv_cache_size_tokens']} "
        f"loom_host_launches={host_launch_count}",
        file=sys.stderr,
    )
    return report


def child_command(
    args: argparse.Namespace,
    variant: str,
    result: Path,
    cache_root: Path,
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
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-fused-tokens",
        str(args.max_fused_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--quality-max-sequences",
        str(args.quality_max_sequences),
        "--quality-max-tokens",
        str(args.quality_max_tokens),
        "--min-capacity-ratio",
        str(args.min_capacity_ratio),
        "--max-perplexity-ratio",
        str(args.max_perplexity_ratio),
        "--max-tpot-regression",
        str(args.max_tpot_regression),
        "--internal-variant",
        variant,
        "--internal-result",
        str(result),
        "--internal-cache-root",
        str(cache_root),
    ]
    if args.model_revision is not None:
        command.extend(("--model-revision", args.model_revision))
    if args.quality_jsonl is not None:
        command.extend(("--quality-jsonl", str(args.quality_jsonl)))
    for case in args.cases:
        command.extend(("--case", case.argument))
    return command


def metric_median(case: dict[str, Any], name: str) -> float | None:
    value = case[name]
    return None if value is None else float(value["median"])


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return numerator / denominator


def sequence_agreement(
    reference: list[list[int]], candidate: list[list[int]]
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise RuntimeError("variant reports contain different request counts")
    exact_requests = 0
    matching_tokens = 0
    total_tokens = 0
    prefix_fractions: list[float] = []
    for expected, actual in zip(reference, candidate, strict=True):
        if len(expected) != len(actual):
            raise RuntimeError("variant reports contain different output lengths")
        exact_requests += int(expected == actual)
        matching_tokens += sum(left == right for left, right in zip(expected, actual))
        total_tokens += len(expected)
        common_prefix = 0
        for left, right in zip(expected, actual):
            if left != right:
                break
            common_prefix += 1
        prefix_fractions.append(common_prefix / len(expected))
    return {
        "exact_requests": exact_requests,
        "total_requests": len(reference),
        "exact_request_fraction": exact_requests / len(reference),
        "matching_token_fraction": matching_tokens / total_tokens,
        "mean_common_prefix_fraction": statistics.mean(prefix_fractions),
    }


def compare_variants(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    case_comparisons: list[dict[str, Any]] = []
    for expected, actual in zip(
        reference["cases"], candidate["cases"], strict=True
    ):
        if expected["case"] != actual["case"]:
            raise RuntimeError("variant reports contain different cases")
        case_comparisons.append(
            {
                "case": expected["case"],
                "generated_token_agreement": sequence_agreement(
                    expected["token_ids"], actual["token_ids"]
                ),
                "reference_over_candidate_batch_latency": ratio(
                    metric_median(expected, "batch_latency_ms"),
                    metric_median(actual, "batch_latency_ms"),
                ),
                "reference_over_candidate_ttft": ratio(
                    metric_median(expected, "request_ttft_ms"),
                    metric_median(actual, "request_ttft_ms"),
                ),
                "reference_over_candidate_tpot": ratio(
                    metric_median(expected, "request_tpot_ms"),
                    metric_median(actual, "request_tpot_ms"),
                ),
                "candidate_over_reference_output_throughput": ratio(
                    metric_median(actual, "output_tokens_per_second"),
                    metric_median(expected, "output_tokens_per_second"),
                ),
            }
        )

    reference_quality = reference["quality"]
    candidate_quality = candidate["quality"]
    quality = None
    if reference_quality is not None and candidate_quality is not None:
        if reference_quality["dataset_sha256"] != candidate_quality["dataset_sha256"]:
            raise RuntimeError("variant reports used different quality corpora")
        quality = {
            "dataset_sha256": reference_quality["dataset_sha256"],
            "reference_perplexity": reference_quality["perplexity"],
            "candidate_perplexity": candidate_quality["perplexity"],
            "candidate_over_reference_perplexity": ratio(
                candidate_quality["perplexity"],
                reference_quality["perplexity"],
            ),
            "mean_negative_log_likelihood_delta": (
                candidate_quality["mean_negative_log_likelihood"]
                - reference_quality["mean_negative_log_likelihood"]
            ),
        }

    reference_capacity = reference["cache_capacity"]
    candidate_capacity = candidate["cache_capacity"]
    return {
        "reference": reference["variant"],
        "candidate": candidate["variant"],
        "cache_capacity": {
            "candidate_over_reference_tokens": ratio(
                candidate_capacity["kv_cache_size_tokens"],
                reference_capacity["kv_cache_size_tokens"],
            ),
            "candidate_over_reference_max_concurrency": ratio(
                candidate_capacity["kv_cache_max_concurrency"],
                reference_capacity["kv_cache_max_concurrency"],
            ),
            "reference_tokens": reference_capacity["kv_cache_size_tokens"],
            "candidate_tokens": candidate_capacity["kv_cache_size_tokens"],
        },
        "quality": quality,
        "cases": case_comparisons,
    }


def system_gate(
    args: argparse.Namespace,
    operational_passed: bool,
    native_vs_fp8_loom: dict[str, Any],
) -> dict[str, Any]:
    quality = native_vs_fp8_loom["quality"]
    if quality is None:
        return {
            "status": "not_run",
            "passed": False,
            "reason": "quality-jsonl was not supplied",
        }
    if args.model_revision is None:
        return {
            "status": "not_run",
            "passed": False,
            "reason": "model-revision was not supplied",
        }
    capacity_ratio = native_vs_fp8_loom["cache_capacity"][
        "candidate_over_reference_tokens"
    ]
    perplexity_ratio = quality["candidate_over_reference_perplexity"]
    tpot_regressions: list[float] = []
    for case in native_vs_fp8_loom["cases"]:
        native_over_fp8 = case["reference_over_candidate_tpot"]
        if native_over_fp8 is not None:
            tpot_regressions.append(1.0 / native_over_fp8)
    worst_tpot_regression = max(tpot_regressions) if tpot_regressions else None
    passed = (
        operational_passed
        and capacity_ratio is not None
        and capacity_ratio >= args.min_capacity_ratio
        and perplexity_ratio is not None
        and perplexity_ratio <= args.max_perplexity_ratio
        and worst_tpot_regression is not None
        and worst_tpot_regression <= args.max_tpot_regression
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "thresholds": {
            "minimum_cache_capacity_ratio": args.min_capacity_ratio,
            "maximum_perplexity_ratio": args.max_perplexity_ratio,
            "maximum_tpot_regression": args.max_tpot_regression,
        },
        "observed": {
            "cache_capacity_ratio": capacity_ratio,
            "perplexity_ratio": perplexity_ratio,
            "worst_tpot_regression": worst_tpot_regression,
        },
    }


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    order = VARIANT_ORDERS[args.variant_order]
    reports: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="loom-fp8-kv-system-") as directory:
        root = Path(directory)
        for variant in order:
            result = root / f"{variant}.json"
            subprocess.run(
                child_command(args, variant, result, root / f"{variant}-cache"),
                check=True,
                stdout=sys.stderr,
            )
            reports[variant] = json.loads(result.read_text())

    native_vs_fp8_vllm = compare_variants(
        reports["native-vllm"], reports["fp8-vllm"]
    )
    native_vs_fp8_loom = compare_variants(
        reports["native-vllm"], reports["fp8-loom"]
    )
    fp8_vllm_vs_loom = compare_variants(
        reports["fp8-vllm"], reports["fp8-loom"]
    )
    launch_counts = {
        variant: reports[variant]["loom_path"]["host_launch_count"]
        for variant in VARIANTS
    }
    fp8_fusion_tokens_match = all(
        case["generated_token_agreement"]["exact_request_fraction"] == 1.0
        for case in fp8_vllm_vs_loom["cases"]
    )
    operational_passed = (
        launch_counts["native-vllm"] == 0
        and launch_counts["fp8-vllm"] == 0
        and launch_counts["fp8-loom"] > 0
        and fp8_fusion_tokens_match
    )
    report = {
        "benchmark": "vllm_fp8_kv_system",
        "model": args.model,
        "model_revision": args.model_revision,
        "variant_order": list(order),
        "claim_boundary": (
            "native-vs-FP8 cache capacity, quality, TTFT, TPOT, throughput, "
            "and process-local CUDA memory; FP8-vLLM vs FP8-Loom separately "
            "isolates the fused write boundary"
        ),
        "operational_acceptance": {
            "passed": operational_passed,
            "launch_counts": launch_counts,
            "fp8_vllm_and_loom_generated_tokens_match": fp8_fusion_tokens_match,
        },
        "system_value_gate": system_gate(
            args, operational_passed, native_vs_fp8_loom
        ),
        "comparisons": {
            "native_vllm_vs_fp8_vllm": native_vs_fp8_vllm,
            "native_vllm_vs_fp8_loom": native_vs_fp8_loom,
            "fp8_vllm_vs_fp8_loom": fp8_vllm_vs_loom,
        },
        "variants": reports,
    }
    rendered = json.dumps(report, indent=2)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(rendered + "\n")
    print(rendered)
    if not operational_passed:
        raise SystemExit("FP8 KV system benchmark operational gate failed")
    return report


def main() -> None:
    args = parse_args()
    if args.internal_variant is None:
        run_controller(args)
    else:
        run_variant(args)


if __name__ == "__main__":
    main()
