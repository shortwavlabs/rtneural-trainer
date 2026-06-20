from __future__ import annotations

import platform
from importlib import metadata
from typing import Any

from rttrainer import __version__


def inspect_device() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "trainer_version": __version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_available": True,
        "tensorflow_status": "not_installed",
        "torch_status": "not_installed",
        "cuda_available": False,
        "mps_available": False,
        "mps_built": False,
        "selected_device": "cpu",
        "package_versions": package_versions(),
    }

    try:
        import tensorflow as tf
    except Exception:
        pass
    else:
        tf_gpus = tf.config.list_physical_devices("GPU")
        payload.update(
            {
                "tensorflow_status": "available",
                "tensorflow_version": tf.__version__,
                "keras_version": str(
                    getattr(tf.keras, "__version__", getattr(tf, "__version__", "unknown"))
                ),
                "tensorflow_gpus": [device.name for device in tf_gpus],
                "selected_device": f"tensorflow-gpu:{tf_gpus[0].name}"
                if tf_gpus
                else "tensorflow-cpu",
            }
        )

    try:
        import torch
    except Exception:
        return payload

    cuda_available = bool(torch.cuda.is_available())
    cuda_devices = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else []
    )
    mps_available = bool(torch.backends.mps.is_available())
    mps_built = bool(torch.backends.mps.is_built())
    if cuda_available:
        selected = "cuda"
    elif mps_available and mps_built:
        selected = "mps"
    else:
        selected = "cpu"
    selected_device = (
        payload["selected_device"]
        if payload.get("tensorflow_status") == "available"
        else selected
    )

    payload.update(
        {
            "torch_status": "available",
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": len(cuda_devices),
            "cuda_devices": cuda_devices,
            "mps_available": mps_available,
            "mps_built": mps_built,
            "torch_selected_device": selected,
            "selected_device": selected_device,
        }
    )
    return payload


def package_versions() -> dict[str, str]:
    packages = {
        "rttrainer": __version__,
        "python": platform.python_version(),
    }
    for name in ("tensorflow", "keras", "torch", "numpy", "scipy", "soundfile"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return packages


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
