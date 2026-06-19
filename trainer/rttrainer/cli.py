from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rttrainer import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rttrainer")
    parser.add_argument("--version", action="version", version=f"rttrainer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-device")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    for command in ("prepare", "train", "evaluate", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)

    try:
        if args.command == "inspect-device":
            payload = inspect_device()
            if args.as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"{payload['selected_device']} ({payload['torch_status']})")
            return 0

        manifest = read_manifest(Path(args.manifest))
        if args.command == "prepare":
            return prepare(manifest)
        if args.command == "train":
            return train(manifest)
        if args.command == "evaluate":
            return evaluate(manifest)
        if args.command == "export":
            return export(manifest)
    except Exception as exc:  # pragma: no cover - CLI guard
        emit({"type": "error", "message": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1

    return 2


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
        import torch  # type: ignore
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


def prepare(manifest: dict[str, Any]) -> int:
    output_dir = ensure_dir(manifest, "output_dir", "prepared")
    input_path = str(manifest.get("input_path", "input.wav"))
    target_path = str(manifest.get("target_path", "target.wav"))
    warnings = []
    if not input_path.lower().endswith(".wav"):
        warnings.append("Input should be a WAV file for v1.")
    if not target_path.lower().endswith(".wav"):
        warnings.append("Target should be a WAV file for v1.")

    report = {
        "schema_version": 1,
        "input": audio_report(input_path, -1.2, -18.4),
        "target": audio_report(target_path, -0.8, -15.6),
        "latency": {
            "estimated_samples": 123,
            "confidence": 0.94 if not warnings else 0.4,
            "method": "simulated-impulse",
        },
        "warnings": warnings,
        "status": "ready" if not warnings else "warning",
    }
    write_json(output_dir / "preparation-report.json", report)
    emit({"type": "prepare_finished", "report_path": str(output_dir / "preparation-report.json")})
    return 0 if not warnings else 3


def train(manifest: dict[str, Any]) -> int:
    run_dir = ensure_dir(manifest, "run_dir", "run")
    checkpoint_dir = mkdir(run_dir / "checkpoints")
    preview_dir = mkdir(run_dir / "previews")
    preset = str(manifest.get("preset", "lstm_standard"))
    run_id = str(manifest.get("run_id", f"run_{int(time.time())}"))
    device = inspect_device()["selected_device"]
    metrics = metrics_for_preset(preset)

    emit({"type": "run_started", "run_id": run_id, "preset": preset, "device": device})
    for epoch, multiplier in ((20, 1.8), (40, 1.2), (60, 1.0)):
        event = {
            "type": "epoch",
            "run_id": run_id,
            "epoch": epoch,
            "total_epochs": 60,
            "train_loss": round(metrics["esr"] * multiplier * 1.15, 6),
            "val_esr": round(metrics["esr"] * multiplier, 6),
            "timestamp": now(),
        }
        emit(event)

    write_json(
        checkpoint_dir / "best-checkpoint.json",
        {
            "schema_version": 1,
            "preset": preset,
            "format": "simulated-state-dict",
            "device": device,
        },
    )
    write_json(run_dir / "metrics.json", metrics)
    for name in ("target", "prediction", "residual"):
        (preview_dir / f"{name}.wav").write_bytes(fake_wav_bytes())

    emit({"type": "checkpoint", "path": str(checkpoint_dir / "best-checkpoint.json"), "is_best": True})
    emit({"type": "run_finished", "run_id": run_id, "status": "completed"})
    return 0


def evaluate(manifest: dict[str, Any]) -> int:
    output_dir = ensure_dir(manifest, "output_dir", "evaluation")
    metrics = metrics_for_preset(str(manifest.get("preset", "lstm_standard")))
    write_json(output_dir / "metrics.json", metrics)
    emit({"type": "evaluation_finished", "metrics_path": str(output_dir / "metrics.json")})
    return 0


def export(manifest: dict[str, Any]) -> int:
    export_dir = ensure_dir(manifest, "export_dir", "export")
    preset = str(manifest.get("preset", "lstm_standard"))
    metrics = metrics_for_preset(preset)
    write_json(
        export_dir / "model.rtneural.json",
        {
            "in_shape": [None, None, 1],
            "layers": [
                {"type": "lstm", "activation": "", "shape": [None, None, 16], "weights": []},
                {"type": "dense", "activation": "", "shape": [None, None, 1], "weights": []},
            ],
            "metadata": {
                "schema_version": 1,
                "sample_rate": 48000,
                "latency_samples": int(manifest.get("latency_samples", 0)),
                "architecture": preset,
                "loss": metrics,
                "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d",
            },
        },
    )
    write_json(
        export_dir / "validation-report.json",
        {"schema_version": 1, "status": "pass", "max_abs_error": 0.000001, "rmse": 0.0000003},
    )
    write_json(
        export_dir / "benchmark-report.json",
        {
            "schema_version": 1,
            "status": "pass",
            "backend": "simulated-eigen",
            "sample_rate": 48000,
            "realtime_factor": metrics["realtime_factor"],
        },
    )
    emit({"type": "export_finished", "export_dir": str(export_dir)})
    return 0


def read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(manifest: dict[str, Any], key: str, fallback: str) -> Path:
    path = Path(str(manifest.get(key, fallback))).expanduser()
    return mkdir(path)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def emit(payload: dict[str, Any]) -> None:
    payload.setdefault("timestamp", now())
    print(json.dumps(payload), flush=True)


def audio_report(path: str, peak_dbfs: float, rms_dbfs: float) -> dict[str, Any]:
    return {
        "sample_rate": 48000,
        "channels": 1,
        "duration_seconds": 95.4,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "clipped_samples": 0,
        "dc_offset": 0.0002,
        "path": path,
    }


def metrics_for_preset(preset: str) -> dict[str, float]:
    esr = {
        "heavy_recurrent": 0.028,
        "lstm_standard": 0.044,
        "dense_memoryless": 0.061,
    }.get(preset, 0.072)
    return {
        "esr": esr,
        "mae": esr / 2.8,
        "rmse": esr / 1.8,
        "peak_residual": esr * 2.6,
        "rms_residual": esr / 2.1,
        "realtime_factor": 24.0 if preset == "heavy_recurrent" else 118.0,
    }


def fake_wav_bytes(sample_rate: int = 48_000, samples: int = 480) -> bytes:
    data = bytearray()
    for index in range(samples):
        value = int(math.sin(index / 12.0) * 12000)
        data.extend(value.to_bytes(2, "little", signed=True))

    byte_rate = sample_rate * 2
    block_align = 2
    data_size = len(data)
    return (
        b"RIFF"
        + (36 + data_size).to_bytes(4, "little")
        + b"WAVEfmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + data_size.to_bytes(4, "little")
        + bytes(data)
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
