#!/usr/bin/env python3
"""Build the dependency-light Loom CUDA shared library for Python adapters."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(
            os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
        ),
    )
    parser.add_argument(
        "--archs",
        default=os.environ.get("LOOM_CUDA_ARCHS", "80,89,90"),
        help="comma-separated CUDA SM numbers",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = args.output or repository / "build" / "libloom_kernels_cuda.so"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    nvcc = args.cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        discovered = shutil.which("nvcc")
        if discovered is None:
            raise FileNotFoundError(
                f"nvcc not found below {args.cuda_home}; set CUDA_HOME"
            )
        nvcc = Path(discovered)

    archs = [arch.strip() for arch in args.archs.split(",") if arch.strip()]
    if not archs or any(not arch.isdigit() for arch in archs):
        raise ValueError(f"invalid CUDA architecture list: {args.archs!r}")

    command = [
        str(nvcc),
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        "-I",
        str(repository / "cuda" / "include"),
    ]
    for arch in archs:
        command.extend(
            ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
        )
    command.extend(
        [
            str(repository / "cuda" / "src" / "rms_norm.cu"),
            str(repository / "cuda" / "src" / "rms_norm_quant.cu"),
            str(repository / "cuda" / "src" / "add_rms_norm.cu"),
            str(repository / "cuda" / "src" / "silu_and_mul.cu"),
            str(repository / "cuda" / "src" / "silu_and_mul_quant.cu"),
            str(repository / "cuda" / "src" / "greedy_sample.cu"),
            str(repository / "cuda" / "src" / "min_p.cu"),
            str(repository / "cuda" / "src" / "rope_paged_kv.cu"),
            "-o",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    print(output)


if __name__ == "__main__":
    main()
