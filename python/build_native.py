#!/usr/bin/env python3
"""Build the single Rust-owned CUDA backend shared library."""

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
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_dir = (args.output_dir or repository / "build").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    nvcc = args.cuda_home / "bin" / "nvcc"
    if not nvcc.is_file() and shutil.which("nvcc") is None:
        raise FileNotFoundError(
            f"nvcc not found below {args.cuda_home}; set CUDA_HOME"
        )

    archs = [arch.strip() for arch in args.archs.split(",") if arch.strip()]
    if not archs or any(not arch.isdigit() for arch in archs):
        raise ValueError(f"invalid CUDA architecture list: {args.archs!r}")

    cargo = shutil.which("cargo")
    if cargo is None:
        raise FileNotFoundError("cargo was not found in PATH")

    environment = os.environ.copy()
    environment["CUDA_HOME"] = str(args.cuda_home.resolve())
    environment["LOOM_CUDA_ARCHS"] = ",".join(archs)
    subprocess.run(
        [
            cargo,
            "build",
            "--release",
            "--locked",
            "-p",
            "loom-cuda-bridge",
            "--features",
            "cuda",
        ],
        cwd=repository,
        env=environment,
        check=True,
    )

    source = repository / "target" / "release" / "libloom_cuda_bridge.so"
    if not source.is_file():
        raise FileNotFoundError(f"Cargo did not produce {source}")
    output = output_dir / source.name
    shutil.copy2(source, output)
    print(output)


if __name__ == "__main__":
    main()
