"""Discovery, loading, and ABI validation for the PyTorch extension."""

from __future__ import annotations

import os
from pathlib import Path
import threading

import torch


BRIDGE_ABI_VERSION = 1
_LOCK = threading.Lock()
_LOADED_PATH: Path | None = None


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (
            parent / "crates" / "loom-cuda-bridge"
        ).is_dir():
            return parent
    return None


def _candidates() -> tuple[Path, ...]:
    candidates: list[Path] = []
    configured = os.environ.get("LOOM_KERNELS_TORCH_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(
        Path(__file__).resolve().parent / "lib" / "libloom_kernels_torch.so"
    )
    repository = _repository_root()
    if repository is not None:
        candidates.append(repository / "build" / "libloom_kernels_torch.so")
    return tuple(candidates)


def torch_extension_path() -> Path | None:
    """Return the first installed Loom PyTorch extension path."""
    for candidate in _candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def torch_extension_available() -> bool:
    """Return whether the single supported framework extension is installed."""
    return torch_extension_path() is not None


def load_torch_extension() -> Path:
    """Load the extension once and reject an incompatible Rust bridge ABI."""
    global _LOADED_PATH
    if _LOADED_PATH is not None:
        return _LOADED_PATH

    path = torch_extension_path()
    if path is None:
        searched = "\n".join(f"  - {candidate}" for candidate in _candidates())
        raise RuntimeError(
            "Loom Kernels requires its compiled PyTorch extension; no Python "
            "or ctypes fallback exists. Run `python python/build_native.py` "
            "and `python python/build_torch_extension.py`, or set "
            f"LOOM_KERNELS_TORCH_LIBRARY. Searched:\n{searched}"
        )

    with _LOCK:
        if _LOADED_PATH is None:
            torch.ops.load_library(str(path))
            actual_abi = int(torch.ops.loom_kernels.bridge_abi_version())
            if actual_abi != BRIDGE_ABI_VERSION:
                raise RuntimeError(
                    "Loom Kernels bridge ABI mismatch: Python expects "
                    f"{BRIDGE_ABI_VERSION}, extension reports {actual_abi}. "
                    "Rebuild the Rust bridge and PyTorch extension together."
                )
            _LOADED_PATH = path
    return _LOADED_PATH


__all__ = [
    "BRIDGE_ABI_VERSION",
    "load_torch_extension",
    "torch_extension_available",
    "torch_extension_path",
]
