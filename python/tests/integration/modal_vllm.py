"""Run the real-model vLLM CUSTOM-backend acceptance gate on Modal."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import modal


REMOTE_ROOT = Path("/workspace")
WORKLOAD_PYTHON = "/usr/bin/python3"
ENTRYPOINT_PATH = Path(__file__).resolve()
LOCAL_PYTHON = (
    ENTRYPOINT_PATH.parents[3] / "python"
    if len(ENTRYPOINT_PATH.parents) > 3
    else REMOTE_ROOT / "python"
)
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_REVISION = "83212e1e2b3cfd6958f3707877bb878945dea8ee"

image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.25.0", add_python="3.12"
    )
    .entrypoint([])
    .env(
        {
            "PYTHONPATH": "/workspace/python/src:/workspace/python/tests",
            "PYTHONUNBUFFERED": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(LOCAL_PYTHON, remote_path=str(REMOTE_ROOT / "python"))
)

model_cache = modal.Volume.from_name(
    "loom-huggingface-cache", create_if_missing=True
)
app = modal.App("loom-vllm-gate")


def _probe(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32_768,
    timeout=45 * 60,
    volumes={"/root/.cache/huggingface": model_cache},
)
def run_gate(
    model: str,
    revision: str,
    dtype: str,
    seed: int,
    max_tokens: int,
    max_model_len: int,
    warmup: int,
    repetitions: int,
    gpu_memory_utilization: float,
    logprob_atol: float,
) -> dict[str, Any]:
    report_path = Path("/tmp/loom-vllm-smoke.json")
    command = [
        WORKLOAD_PYTHON,
        "-m",
        "integration.vllm_smoke",
        "compare",
        "--model",
        model,
        "--revision",
        revision,
        "--dtype",
        dtype,
        "--seed",
        str(seed),
        "--max-tokens",
        str(max_tokens),
        "--max-model-len",
        str(max_model_len),
        "--warmup",
        str(warmup),
        "--repetitions",
        str(repetitions),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--logprob-atol",
        str(logprob_atol),
        "--report",
        str(report_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    versions = _probe(
        [
            WORKLOAD_PYTHON,
            "-c",
            (
                "import torch, vllm; "
                "print('python=' + __import__('platform').python_version()); "
                "print('torch=' + torch.__version__); "
                "print('cuda=' + str(torch.version.cuda)); "
                "print('vllm=' + vllm.__version__)"
            ),
        ]
    )
    inventory = _probe(["nvidia-smi", "-L"])
    if completed.returncode != 0 or not report_path.exists():
        raise RuntimeError(
            "vLLM acceptance gate failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            f"versions:\n{json.dumps(versions, indent=2)}\n"
            f"inventory:\n{json.dumps(inventory, indent=2)}"
        )

    report = json.loads(report_path.read_text())
    report["cloud"] = {
        "provider": "Modal",
        "requested_gpu": "L4",
        "image": "vllm/vllm-openai:v0.25.0",
        "versions": versions,
        "inventory": inventory,
        "runner_stdout": completed.stdout,
        "runner_stderr": completed.stderr,
    }
    return report


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    dtype: str = "float16",
    seed: int = 7,
    max_tokens: int = 8,
    max_model_len: int = 1024,
    warmup: int = 1,
    repetitions: int = 3,
    gpu_memory_utilization: float = 0.45,
    logprob_atol: float = 1e-5,
    report: str = "build/modal/vllm-l4.json",
) -> None:
    result = run_gate.remote(
        model,
        revision,
        dtype,
        seed,
        max_tokens,
        max_model_len,
        warmup,
        repetitions,
        gpu_memory_utilization,
        logprob_atol,
    )
    output = Path(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    telemetry = result["custom"]["loom_telemetry"]
    print(f"report={output}")
    print(f"passed={result['passed']}")
    print(f"forward_calls={telemetry['forward_calls']}")
    print(f"max_step_generation={telemetry['max_step_generation']}")


if __name__ == "__main__":
    raise SystemExit("run with: modal run python/tests/integration/modal_vllm.py")
