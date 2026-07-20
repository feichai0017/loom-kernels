"""Run the two-GPU paged-attention acceptance gate on Modal."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import modal


REMOTE_ROOT = Path("/workspace")
ENTRYPOINT_PATH = Path(__file__).resolve()
LOCAL_PYTHON = (
    ENTRYPOINT_PATH.parents[3] / "python"
    if len(ENTRYPOINT_PATH.parents) > 3
    else REMOTE_ROOT / "python"
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.9,<2.10",
        "flashinfer-python>=0.6,<0.7",
    )
    .env(
        {
            "PYTHONPATH": "/workspace/python/src:/workspace/python/tests",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        LOCAL_PYTHON,
        remote_path=str(REMOTE_ROOT / "python"),
    )
)

app = modal.App("loom-two-gpu-gate")


def _run_gate_process(
    prefix_tokens: int,
    tail_tokens: int,
    rows: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    route_strategy: str,
    page_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    report_path = Path(f"/tmp/loom-two-gpu-{prefix_tokens}-report.json")
    command = [
        sys.executable,
        "-m",
        "integration.two_gpu_smoke",
        "run",
        "--prefix-tokens",
        str(prefix_tokens),
        "--tail-tokens",
        str(tail_tokens),
        "--rows",
        str(rows),
        "--query-heads",
        str(query_heads),
        "--kv-heads",
        str(kv_heads),
        "--head-dim",
        str(head_dim),
        "--dtype",
        dtype,
        "--attention-backend",
        "flashinfer-paged",
        "--route-strategy",
        route_strategy,
        "--page-size",
        str(page_size),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
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
    topology_result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=False,
        capture_output=True,
        text=True,
    )
    topology = topology_result.stdout
    if topology_result.returncode != 0:
        inventory = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        topology = (
            f"topo_exit_code={topology_result.returncode}\n"
            f"topo_stderr={topology_result.stderr.strip()}\n"
            f"inventory_exit_code={inventory.returncode}\n"
            f"{inventory.stdout}{inventory.stderr}"
        )
    versions_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import flashinfer, torch; "
                "print('torch=' + torch.__version__); "
                "print('cuda=' + str(torch.version.cuda)); "
                "print('flashinfer=' + flashinfer.__version__)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    versions = (
        f"exit_code={versions_result.returncode}\n"
        f"{versions_result.stdout}{versions_result.stderr}"
    )
    if completed.returncode != 0 or not report_path.exists():
        raise RuntimeError(
            "two-GPU gate failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}\n"
            f"topology:\n{topology}\n"
            f"versions:\n{versions}"
        )

    report = json.loads(report_path.read_text())
    report["cloud"] = {
        "provider": "Modal",
        "requested_gpu": "L4:2",
        "topology": topology,
        "versions": versions,
        "runner_stdout": completed.stdout,
        "runner_stderr": completed.stderr,
    }
    return report


@app.function(
    image=image,
    gpu="L4:2",
    cpu=4,
    memory=16_384,
    timeout=30 * 60,
)
def run_gate(
    prefix_tokens: int,
    tail_tokens: int,
    rows: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    route_strategy: str,
    page_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    return _run_gate_process(
        prefix_tokens,
        tail_tokens,
        rows,
        query_heads,
        kv_heads,
        head_dim,
        dtype,
        route_strategy,
        page_size,
        warmup,
        iterations,
    )


@app.function(
    image=image,
    gpu="L4:2",
    cpu=4,
    memory=16_384,
    timeout=30 * 60,
)
def run_sweep(
    prefix_tokens: list[int],
    tail_tokens: int,
    rows: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    route_strategy: str,
    page_size: int,
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    return [
        _run_gate_process(
            prefix_length,
            tail_tokens,
            rows,
            query_heads,
            kv_heads,
            head_dim,
            dtype,
            route_strategy,
            page_size,
            warmup,
            iterations,
        )
        for prefix_length in prefix_tokens
    ]


@app.local_entrypoint()
def main(
    prefix_tokens: int = 4096,
    prefix_sweep: str = "",
    tail_tokens: int = 16,
    rows: int = 1,
    query_heads: int = 32,
    kv_heads: int = 8,
    head_dim: int = 128,
    dtype: str = "float16",
    route_strategy: str = "sequential",
    page_size: int = 16,
    warmup: int = 10,
    iterations: int = 100,
    report: str = "build/modal/two-gpu-l4.json",
) -> None:
    if prefix_sweep:
        prefixes = [
            int(value.strip())
            for value in prefix_sweep.split(",")
            if value.strip()
        ]
        if not prefixes:
            raise ValueError("prefix_sweep must contain at least one integer")
        reports = run_sweep.remote(
            prefixes,
            tail_tokens,
            rows,
            query_heads,
            kv_heads,
            head_dim,
            dtype,
            route_strategy,
            page_size,
            warmup,
            iterations,
        )
        result: dict[str, Any] = {
            "schema_version": 1,
            "sweep_variable": "prefix_tokens",
            "prefix_tokens": prefixes,
            "passed": all(item["passed"] for item in reports),
            "reports": reports,
        }
    else:
        result = run_gate.remote(
            prefix_tokens,
            tail_tokens,
            rows,
            query_heads,
            kv_heads,
            head_dim,
            dtype,
            route_strategy,
            page_size,
            warmup,
            iterations,
        )
    output = Path(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"report={output}")
    print(f"passed={result['passed']}")
    if prefix_sweep:
        for item in result["reports"]:
            prefix_length = item["workload"]["prefix_tokens"]
            print(
                f"prefix_tokens={prefix_length} "
                f"route_query_p50_ms={item['route_query']['p50_ms']:.3f} "
                f"stage_kv_p50_ms={item['stage_kv']['p50_ms']:.3f}"
            )
    else:
        print(f"route_query_p50_ms={result['route_query']['p50_ms']:.3f}")
        print(f"stage_kv_p50_ms={result['stage_kv']['p50_ms']:.3f}")


if __name__ == "__main__":
    raise SystemExit("run with: modal run python/tests/integration/modal_two_gpu.py")
