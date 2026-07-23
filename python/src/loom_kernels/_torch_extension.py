"""Discovery and one-time loading of the C++ PyTorch dispatcher shim."""

from __future__ import annotations

import os
from pathlib import Path
import threading

import torch


_LOCK = threading.Lock()
_LOADED_PATH: Path | None = None


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Cargo.toml").is_file() and (
            parent / "crates" / "loom-cuda-sys" / "cuda"
        ).is_dir():
            return parent
    return None


def _candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("LOOM_KERNELS_TORCH_LIBRARY")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parent / "lib" / "libloom_kernels_torch.so")
    repository = _repository_root()
    if repository is not None:
        candidates.append(repository / "build" / "libloom_kernels_torch.so")
    return candidates


def torch_extension_path() -> Path | None:
    for candidate in _candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_torch_extension() -> Path | None:
    global _LOADED_PATH
    if _LOADED_PATH is not None:
        return _LOADED_PATH
    path = torch_extension_path()
    if path is None:
        return None

    with _LOCK:
        if _LOADED_PATH is None:
            torch.ops.load_library(str(path))
            _LOADED_PATH = path
    return _LOADED_PATH


__all__ = ["load_torch_extension", "torch_extension_path"]
