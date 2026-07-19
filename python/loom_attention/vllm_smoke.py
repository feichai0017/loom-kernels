"""CUDA acceptance test for the Loom vLLM attention backend.

The public ``compare`` command runs native FlashAttention and the Loom
delegate in separate processes. This avoids reusing CUDA/vLLM global state and
turns M1 correctness into a machine-readable token and logprob comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Sequence

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_REVISION = "83212e1e2b3cfd6958f3707877bb878945dea8ee"
DEFAULT_PROMPTS = (
    "Loom validates a shared long-context prefix before attention. "
    "The runtime tracks object identity, generation, layout, and leases. "
    "For this deterministic test, summarize the role of the page table:",
    "Loom validates a shared long-context prefix before attention. "
    "The runtime tracks object identity, generation, layout, and leases. "
    "For this deterministic test, summarize the role of the scheduler:",
)
REPORT_SCHEMA = 1


class SmokeFailure(RuntimeError):
    """Raised when the CUDA environment or vLLM output violates the gate."""


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SmokeFailure(f"report {path} is not a JSON object")
    return payload


def _chosen_logprobs(token_ids: Sequence[int], tables: Any) -> list[float]:
    if tables is None or len(tables) != len(token_ids):
        raise SmokeFailure("vLLM did not return one sampled logprob table per token")
    values: list[float] = []
    for token_id, table in zip(token_ids, tables):
        entry = table.get(token_id)
        if entry is None:
            entry = table.get(str(token_id))
        if entry is None or not hasattr(entry, "logprob"):
            raise SmokeFailure(f"sampled token {token_id} has no returned logprob")
        values.append(float(entry.logprob))
    return values


def _run_backend(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as error:
        raise SmokeFailure(
            "install the GPU package with `python -m pip install -e './python[vllm]'`"
        ) from error

    if not torch.cuda.is_available():
        raise SmokeFailure("this acceptance test requires a real CUDA device")
    version_parts = vllm.__version__.split(".")
    if version_parts[:2] != ["0", "25"]:
        raise SmokeFailure(
            f"expected vLLM 0.25.x, found {vllm.__version__}; plugin APIs are pinned"
        )
    if args.backend == "CUSTOM":
        from .vllm_plugin import register

        register()

    prompts = list(DEFAULT_PROMPTS)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
        ignore_eos=True,
        logprobs=1,
        seed=args.seed,
        detokenize=False,
    )
    started = perf_counter()
    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        seed=args.seed,
        enforce_eager=True,
        attention_config={"backend": args.backend},
        block_size=16,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    startup_seconds = perf_counter() - started

    for _ in range(args.warmup):
        llm.generate(prompts, sampling, use_tqdm=False)

    sequences: list[dict[str, Any]] = []
    generation_seconds: list[float] = []
    for repetition in range(args.repetitions):
        started = perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        generation_seconds.append(perf_counter() - started)
        if len(outputs) != len(prompts):
            raise SmokeFailure("vLLM returned a different number of requests")
        for prompt_index, (prompt, request) in enumerate(zip(prompts, outputs)):
            if len(request.outputs) != 1:
                raise SmokeFailure(
                    "smoke test expects one generated sequence per prompt"
                )
            if request.prompt_token_ids is None:
                raise SmokeFailure("vLLM did not return prompt token IDs")
            sample = request.outputs[0]
            token_ids = [int(token_id) for token_id in sample.token_ids]
            if len(token_ids) != args.max_tokens:
                raise SmokeFailure(
                    f"expected {args.max_tokens} output tokens, got {len(token_ids)}"
                )
            sequences.append(
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "prompt_token_ids": [
                        int(token_id) for token_id in request.prompt_token_ids
                    ],
                    "output_token_ids": token_ids,
                    "sampled_logprobs": _chosen_logprobs(token_ids, sample.logprobs),
                }
            )

    return {
        "schema": REPORT_SCHEMA,
        "backend": args.backend,
        "model": args.model,
        "revision": args.revision,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "startup_seconds": startup_seconds,
        "generation_seconds": generation_seconds,
        "median_generation_seconds": statistics.median(generation_seconds),
        "sequences": sequences,
    }


def _compare_run_payloads(
    native: dict[str, Any], custom: dict[str, Any], *, logprob_atol: float
) -> dict[str, Any]:
    differences: list[str] = []
    if native.get("backend") != "FLASH_ATTN":
        differences.append("native report did not run the FLASH_ATTN backend")
    if custom.get("backend") != "CUSTOM":
        differences.append("custom report did not run the CUSTOM backend")
    for field in (
        "schema",
        "model",
        "revision",
        "dtype",
        "seed",
        "max_tokens",
        "vllm_version",
        "cuda_device",
    ):
        if native.get(field) != custom.get(field):
            differences.append(
                f"configuration field {field!r} differs: "
                f"{native.get(field)!r} != {custom.get(field)!r}"
            )

    native_sequences = native.get("sequences", [])
    custom_sequences = custom.get("sequences", [])
    if len(native_sequences) != len(custom_sequences):
        differences.append(
            "sequence count differs: "
            f"{len(native_sequences)} != {len(custom_sequences)}"
        )

    max_logprob_delta = 0.0
    for index, (left, right) in enumerate(zip(native_sequences, custom_sequences)):
        for field in ("repetition", "prompt_index", "prompt", "prompt_token_ids"):
            if left.get(field) != right.get(field):
                differences.append(f"sequence {index} field {field!r} differs")
        if left.get("output_token_ids") != right.get("output_token_ids"):
            differences.append(f"sequence {index} generated token IDs differ")
        left_logprobs = left.get("sampled_logprobs", [])
        right_logprobs = right.get("sampled_logprobs", [])
        if len(left_logprobs) != len(right_logprobs):
            differences.append(f"sequence {index} sampled logprob count differs")
            continue
        for token_index, (a, b) in enumerate(zip(left_logprobs, right_logprobs)):
            left_value = float(a)
            right_value = float(b)
            if not math.isfinite(left_value) or not math.isfinite(right_value):
                differences.append(
                    f"sequence {index} token {token_index} logprob is not finite"
                )
                continue
            delta = abs(left_value - right_value)
            max_logprob_delta = max(max_logprob_delta, delta)
            if delta > logprob_atol:
                differences.append(
                    f"sequence {index} token {token_index} logprob delta "
                    f"{delta:.8g} exceeds {logprob_atol:.8g}"
                )

    native_median = float(native.get("median_generation_seconds", 0.0))
    custom_median = float(custom.get("median_generation_seconds", 0.0))
    return {
        "schema": REPORT_SCHEMA,
        "passed": not differences,
        "logprob_atol": logprob_atol,
        "max_logprob_delta": max_logprob_delta,
        "native_median_generation_seconds": native_median,
        "custom_median_generation_seconds": custom_median,
        "custom_over_native_time_ratio": (
            custom_median / native_median if native_median > 0.0 else None
        ),
        "differences": differences,
        "native": native,
        "custom": custom,
    }


def _child_command(args: argparse.Namespace, backend: str, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "loom_attention.vllm_smoke",
        "run",
        "--backend",
        backend,
        "--output",
        str(output),
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--warmup",
        str(args.warmup),
        "--repetitions",
        str(args.repetitions),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]


def _compare(args: argparse.Namespace) -> int:
    environment = os.environ.copy()
    environment["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    environment["VLLM_PLUGINS"] = "loom"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    with tempfile.TemporaryDirectory(prefix="loom-vllm-smoke-") as directory:
        directory_path = Path(directory)
        native_path = directory_path / "native.json"
        custom_path = directory_path / "custom.json"
        for backend, output in (("FLASH_ATTN", native_path), ("CUSTOM", custom_path)):
            completed = subprocess.run(
                _child_command(args, backend, output),
                env=environment,
                check=False,
            )
            if completed.returncode != 0:
                raise SmokeFailure(
                    f"{backend} child exited with status {completed.returncode}"
                )
        report = _compare_run_payloads(
            _read_json(native_path),
            _read_json(custom_path),
            logprob_atol=args.logprob_atol,
        )
    _write_json(args.report, report)
    print(
        f"Loom vLLM smoke {'PASSED' if report['passed'] else 'FAILED'}; "
        f"report: {args.report}"
    )
    for difference in report["differences"]:
        print(f"- {difference}")
    return 0 if report["passed"] else 1


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.model or not args.revision:
        raise SmokeFailure("model and revision must be non-empty")
    if args.max_tokens <= 0 or args.max_model_len <= 0:
        raise SmokeFailure("max-tokens and max-model-len must be positive")
    if args.warmup < 0 or args.repetitions <= 0:
        raise SmokeFailure("warmup must be non-negative and repetitions positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise SmokeFailure("gpu-memory-utilization must be in (0, 1]")
    if args.command == "compare" and args.logprob_atol < 0.0:
        raise SmokeFailure("logprob-atol must be non-negative")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare native and Loom-delegated vLLM attention"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="run the complete CUDA A/B gate")
    _add_workload_arguments(compare)
    compare.add_argument(
        "--report", type=Path, default=Path("build/vllm-smoke/report.json")
    )
    compare.add_argument("--logprob-atol", type=float, default=1e-5)

    run = commands.add_parser("run", help="run one backend (used by compare)")
    _add_workload_arguments(run)
    run.add_argument("--backend", required=True, choices=("FLASH_ATTN", "CUSTOM"))
    run.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_arguments(args)
        if args.command == "compare":
            return _compare(args)
        _write_json(args.output, _run_backend(args))
        return 0
    except SmokeFailure as error:
        print(f"vLLM smoke error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
