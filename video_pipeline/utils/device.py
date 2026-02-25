"""Device and performance utilities."""

from __future__ import annotations

import torch


def configure_torch_backends() -> None:
    """Enable TF32 for better throughput on NVIDIA GPUs."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def clear_device_cache() -> None:
    """Clear CUDA cache when available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
