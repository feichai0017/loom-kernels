"""Shared PyTorch tensor metadata."""

import torch


_DTYPE_NAMES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
