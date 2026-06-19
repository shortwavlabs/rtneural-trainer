from __future__ import annotations

import platform
from typing import Any

from rttrainer import __version__


def inspect_device() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "trainer_version": __version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_status": "not_installed",
        "cuda_available": False,
        "mps_available": False,
        "mps_built": False,
        "selected_device": "cpu",
    }

    try:
        import torch
    except Exception:
        return payload

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(torch.backends.mps.is_available())
    mps_built = bool(torch.backends.mps.is_built())
    if cuda_available:
        selected = "cuda"
    elif mps_available and mps_built:
        selected = "mps"
    else:
        selected = "cpu"

    payload.update(
        {
            "torch_status": "available",
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "mps_built": mps_built,
            "selected_device": selected,
        }
    )
    return payload


def require_torch():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for training/export. Install the trainer with "
            "the 'training' extra or provide a Python environment with torch."
        ) from exc
    return torch


def choose_device(preferred: str | None = None):
    torch = require_torch()
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
