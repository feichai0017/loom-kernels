#!/usr/bin/env python3
"""Build the CUDA library and checked Rust bridge for Python adapters."""

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
    cuda_sources = repository / "crates" / "loom-cuda-sys" / "cuda"
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
        str(cuda_sources / "include"),
    ]
    for arch in archs:
        command.extend(
            ["-gencode", f"arch=compute_{arch},code=sm_{arch}"]
        )
    command.extend(
        [
            str(cuda_sources / "src" / "rms_norm.cu"),
            str(cuda_sources / "src" / "rms_norm_quant.cu"),
            str(cuda_sources / "src" / "add_rms_norm.cu"),
            str(cuda_sources / "src" / "silu_and_mul.cu"),
            str(cuda_sources / "src" / "silu_and_mul_quant.cu"),
            str(cuda_sources / "src" / "greedy_sample.cu"),
            str(cuda_sources / "src" / "min_p.cu"),
            str(cuda_sources / "src" / "paged_decode_attention.cu"),
            str(cuda_sources / "src" / "rope_paged_kv.cu"),
            "-o",
            str(output),
        ]
    )
    subprocess.run(command, check=True)

    cargo = shutil.which("cargo")
    if cargo is None:
        cargo_candidate = Path.home() / ".cargo" / "bin" / "cargo"
        if cargo_candidate.is_file():
            cargo = str(cargo_candidate)
        else:
            raise FileNotFoundError(
                "cargo was not found; install Rust before building the checked bridge"
            )

    rust_environment = os.environ.copy()
    rust_environment["CUDA_HOME"] = str(args.cuda_home.resolve())
    rust_environment["LOOM_CUDA_ARCHS"] = ",".join(archs)
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
        env=rust_environment,
        check=True,
    )
    rust_bridge_source = repository / "target" / "release" / "libloom_cuda_bridge.so"
    if not rust_bridge_source.is_file():
        raise FileNotFoundError(
            f"Rust bridge build completed without producing {rust_bridge_source}"
        )
    rust_bridge_output = output.parent / rust_bridge_source.name
    shutil.copy2(rust_bridge_source, rust_bridge_output)

    print(output)
    print(rust_bridge_output)


if __name__ == "__main__":
    main()
