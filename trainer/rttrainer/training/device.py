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
        tensorflow_metal_available = package_versions().get("tensorflow-metal") not in {
            None,
            "not installed",
        }
        is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        mps_available = bool(is_apple_silicon and tensorflow_metal_available and tf_gpus)
        cuda_available = bool(tf_gpus and not is_apple_silicon)
        payload.update(
            {
                "tensorflow_status": "available",
                "tensorflow_version": tf.__version__,
                "keras_version": str(
                    getattr(tf.keras, "__version__", getattr(tf, "__version__", "unknown"))
                ),
                "tensorflow_gpus": [device.name for device in tf_gpus],
                "cuda_available": cuda_available,
                "cuda_device_count": len(tf_gpus) if cuda_available else 0,
                "cuda_devices": [device.name for device in tf_gpus] if cuda_available else [],
                "mps_available": mps_available,
                "mps_built": bool(is_apple_silicon and tensorflow_metal_available),
                "selected_device": f"tensorflow-gpu:{tf_gpus[0].name}"
                if tf_gpus
                else "tensorflow-cpu",
            }
        )
    return payload


def package_versions() -> dict[str, str]:
    packages = {
        "rttrainer": __version__,
        "python": platform.python_version(),
    }
    for name in (
        "tensorflow",
        "tensorflow-metal",
        "keras",
        "numpy",
        "scipy",
        "soundfile",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return packages


def normalize_device_preference(preferred: str | None) -> str:
    if preferred is None:
        return "auto"
    normalized = str(preferred).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"cpu", "tensorflow-cpu"} or normalized.startswith("tensorflow-cpu:"):
        return "cpu"
    if normalized in {"mps", "metal"} or normalized.startswith(
        ("tensorflow-mps:", "tensorflow-metal:")
    ):
        return "mps"
    if normalized.startswith("tensorflow-gpu:"):
        return "auto"
    if normalized in {"cuda", "gpu", "tensorflow-gpu"} or normalized.startswith("cuda:"):
        return "cuda"
    raise RuntimeError("Device must be 'auto', 'cpu', 'mps', or 'cuda'.")
