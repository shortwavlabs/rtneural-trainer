from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.export_rtneural.keras_exporter import ArrayEncoder, save_keras_model_json
from rttrainer.metrics.aliasing import analyze_rtneural_json_aliasing
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.models.presets import get_preset
from rttrainer.training.device import require_torch
from rttrainer.training.runner import (
    estimate_realtime_factor,
    load_checkpoint,
    predict_loaded_sequence,
    resolve_checkpoint_path,
)
from rttrainer.utils import mkdir, now, write_json
from rttrainer.validation.parity import validate_export_parity

RTNEURAL_COMMIT = "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d"


def export_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    export_dir = mkdir(Path(str(manifest.get("export_dir", "export"))).expanduser())
    checkpoint_path = resolve_checkpoint_path(manifest)
    model, checkpoint = load_checkpoint(checkpoint_path)
    preset = get_preset(checkpoint["preset"])
    sample_rate = int(manifest.get("sample_rate", 48_000))
    latency_samples = int(manifest.get("latency_samples", 0))
    if checkpoint.get("backend") == "keras":
        model_json = build_keras_rtneural_json(
            model=model,
            preset_id=preset.preset_id,
            sample_rate=sample_rate,
            latency_samples=latency_samples,
            checkpoint_metrics=checkpoint.get("metrics", {}),
        )
    else:
        state = model.state_dict()
        model_json = build_rtneural_json(
            torch=require_torch(),
            state=state,
            preset_id=preset.preset_id,
            sample_rate=sample_rate,
            latency_samples=latency_samples,
            checkpoint_metrics=checkpoint.get("metrics", {}),
        )
    model_path = export_dir / "model.rtneural.json"
    write_json(model_path, model_json)

    input_path = resolve_parity_input(manifest, checkpoint_path)
    device_preference = export_device_preference(manifest, checkpoint)
    validation = validate_export_parity(
        checkpoint_path=checkpoint_path,
        model_json_path=model_path,
        input_path=input_path,
        tolerance=float(
            manifest.get("parity_tolerance", default_parity_tolerance(preset.preset_id))
        ),
        device_preference=device_preference,
    )
    validation_path = export_dir / "validation-report.json"
    write_json(validation_path, validation)

    parity_snapshot = write_parity_snapshot(
        export_dir=export_dir,
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        input_path=input_path,
        manifest=manifest,
        device_preference=device_preference,
    )

    benchmark = {
        "schema_version": 1,
        "status": "pass",
        "backend": "python-estimate",
        "sample_rate": sample_rate,
        "realtime_factor": estimate_realtime_factor(preset),
        "created_at": now(),
    }
    benchmark_path = export_dir / "benchmark-report.json"
    write_json(benchmark_path, benchmark)

    aliasing_path = export_dir / "aliasing-report.json"
    aliasing = analyze_rtneural_json_aliasing(
        model_json_path=model_path,
        sample_rate=sample_rate,
        report_path=aliasing_path,
    )

    created_at = now()
    package = {
        "schema_version": 2,
        "package_format": "rtneural-trainer-export",
        "name": str(manifest.get("name", "RTNeural model")),
        "status": "exported",
        "preset": preset.preset_id,
        "backend": checkpoint.get("backend", "pytorch"),
        "sample_rate": sample_rate,
        "latency_samples": latency_samples,
        "model": {
            "format": "rtneural-json",
            "path": model_path.name,
            "sample_rate": sample_rate,
            "latency_samples": latency_samples,
            "backend": checkpoint.get("backend", "pytorch"),
            "metadata": model_json.get("metadata", {}),
        },
        "artifacts": [
            artifact_metadata(export_dir, "model", model_path, "application/json"),
            artifact_metadata(export_dir, "validation_report", validation_path, "application/json"),
            artifact_metadata(export_dir, "benchmark_report", benchmark_path, "application/json"),
            artifact_metadata(export_dir, "aliasing_report", aliasing_path, "application/json"),
            artifact_metadata(
                export_dir,
                "parity_snapshot_manifest",
                export_dir / str(parity_snapshot["manifest_path"]),
                "application/json",
            ),
            artifact_metadata(
                export_dir,
                "parity_snapshot_input",
                export_dir / str(parity_snapshot["input_path"]),
                "audio/wav",
            ),
            artifact_metadata(
                export_dir,
                "parity_snapshot_expected",
                export_dir / str(parity_snapshot["expected_output_path"]),
                "audio/wav",
            ),
        ],
        "model_path": model_path.name,
        "validation_path": validation_path.name,
        "benchmark_path": benchmark_path.name,
        "aliasing_path": aliasing_path.name,
        "parity_snapshot": parity_snapshot,
        "package_path": "package.json",
        "quality": checkpoint.get("metrics", {}),
        "validation": validation,
        "benchmark": benchmark,
        "aliasing": aliasing,
        "training": {
            "preset": preset.preset_id,
            "backend": checkpoint.get("backend", "pytorch"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
        },
        "generated_by": {
            "app": "rttrainer",
            "pipeline": "rttrainer export",
        },
        "compatibility": {
            "rtneural_commit": RTNEURAL_COMMIT,
            "rtneural_json": True,
            "dynamic_json": True,
            "schema": "rttrainer-rtneural-json-v0",
            "aidax": {
                "status": "deferred",
                "reason": "Pending format and license review before emitting an AIDA-X envelope.",
            },
        },
        "created_at": created_at,
        "updated_at": created_at,
    }
    package_path = export_dir / "package.json"
    write_json(package_path, package)
    return {
        "export_dir": str(export_dir),
        "model_path": str(model_path),
        "validation_path": str(validation_path),
        "benchmark_path": str(benchmark_path),
        "aliasing_path": str(aliasing_path),
        "package_path": str(package_path),
        "validation": validation,
        "aliasing": aliasing,
        "parity_snapshot": parity_snapshot,
    }


def artifact_metadata(export_dir: Path, role: str, path: Path, media_type: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "role": role,
        "path": path.relative_to(export_dir).as_posix() if path.is_relative_to(export_dir) else str(path),
        "media_type": media_type,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def write_parity_snapshot(
    *,
    export_dir: Path,
    model,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    input_path: Path,
    manifest: dict[str, Any],
    device_preference: str | None = None,
) -> dict[str, Any]:
    input_audio = read_wav_mono(input_path)
    if not input_audio.samples:
        raise ValueError(f"Parity snapshot input is empty: {input_path}")
    requested_samples = int(manifest.get("parity_snapshot_samples", 8192))
    sample_count = min(max(1, requested_samples), len(input_audio.samples))
    samples = input_audio.samples[:sample_count]
    expected_output = predict_loaded_sequence(
        model,
        checkpoint,
        samples,
        device_preference or export_device_preference(manifest, checkpoint),
    )

    snapshot_input_path = export_dir / "parity-snapshot-input.wav"
    snapshot_expected_path = export_dir / "parity-snapshot-expected.wav"
    snapshot_manifest_path = export_dir / "parity-snapshot.json"
    write_wav_mono(snapshot_input_path, samples, input_audio.sample_rate)
    write_wav_mono(snapshot_expected_path, expected_output, input_audio.sample_rate)

    snapshot = {
        "schema_version": 1,
        "sample_rate": input_audio.sample_rate,
        "sample_count": sample_count,
        "input_path": snapshot_input_path.name,
        "expected_output_path": snapshot_expected_path.name,
        "manifest_path": snapshot_manifest_path.name,
        "source_input_path": str(input_path),
        "checkpoint_path": str(checkpoint_path),
        "backend": checkpoint.get("backend", "unknown"),
        "preset": checkpoint.get("preset", "unknown"),
        "input_samples": samples,
        "expected_output_samples": expected_output,
        "created_at": now(),
    }
    write_json(snapshot_manifest_path, snapshot)
    return snapshot


def build_rtneural_json(
    *,
    torch,
    state,
    preset_id: str,
    sample_rate: int,
    latency_samples: int,
    checkpoint_metrics: dict[str, Any],
) -> dict[str, Any]:
    weight_ih = state["lstm.weight_ih_l0"].detach().cpu()
    hidden_size = int(weight_ih.shape[0] // 4)
    input_size = int(weight_ih.shape[1])
    dense_weight = state["dense.weight"].detach().cpu()
    output_size = int(dense_weight.shape[0])

    return {
        "in_shape": [None, None, input_size],
        "layers": [
            {
                "type": "lstm",
                "activation": "",
                "shape": [None, None, hidden_size],
                "input_size": input_size,
                "hidden_size": hidden_size,
                "weights": [
                    tensor_to_list(state["lstm.weight_ih_l0"].transpose(0, 1)),
                    tensor_to_list(state["lstm.weight_hh_l0"].transpose(0, 1)),
                    tensor_to_list(state["lstm.bias_ih_l0"] + state["lstm.bias_hh_l0"]),
                ],
            },
            {
                "type": "dense",
                "activation": "",
                "shape": [None, None, output_size],
                "input_size": hidden_size,
                "output_size": output_size,
                "weights": [
                    tensor_to_list(state["dense.weight"].transpose(0, 1)),
                    tensor_to_list(state["dense.bias"]),
                ],
            },
        ],
        "metadata": {
            "schema_version": 1,
            "schema": "rttrainer-rtneural-json-v0",
            "sample_rate": sample_rate,
            "latency_samples": latency_samples,
            "architecture": preset_id,
            "loss": checkpoint_metrics,
            "rtneural_commit": RTNEURAL_COMMIT,
        },
    }


def build_keras_rtneural_json(
    *,
    model,
    preset_id: str,
    sample_rate: int,
    latency_samples: int,
    checkpoint_metrics: dict[str, Any],
) -> dict[str, Any]:
    preset = get_preset(preset_id)
    model_json = json.loads(json.dumps(save_keras_model_json(model), cls=ArrayEncoder))
    for layer in model_json.get("layers", []):
        if layer.get("type") == "lstm":
            kernel, recurrent_kernel, _bias = layer["weights"]
            layer["input_size"] = len(kernel)
            layer["hidden_size"] = len(recurrent_kernel)
        elif layer.get("type") == "gru":
            kernel, recurrent_kernel, _bias = layer["weights"]
            layer["input_size"] = len(kernel)
            layer["hidden_size"] = len(recurrent_kernel)
        elif layer.get("type") == "dense":
            kernel, _bias = layer["weights"]
            layer["input_size"] = len(kernel)
            layer["output_size"] = len(kernel[0]) if kernel else 0
        elif layer.get("type") == "conv1d":
            if preset.conv_activation == "tanh" and preset.conv_activation_alpha != 1.0:
                fold_scaled_tanh_conv1d_activation(layer, preset.conv_activation_alpha)
            kernel, _bias = layer["weights"]
            groups = int(layer.get("groups", 1) or 1)
            layer["input_size"] = (len(kernel[0]) if kernel else 0) * groups
            layer["output_size"] = len(kernel[0][0]) if kernel and kernel[0] else 0
        elif layer.get("type") in ("batchnorm", "prelu", "activation"):
            shape = layer.get("shape") or []
            if shape:
                layer["input_size"] = shape[-1]
                layer["output_size"] = shape[-1]

    metadata = {
        "schema_version": 1,
        "schema": "rttrainer-rtneural-json-v0",
        "sample_rate": sample_rate,
        "latency_samples": latency_samples,
        "architecture": preset_id,
        "backend": "keras",
        "loss": checkpoint_metrics,
        "rtneural_commit": RTNEURAL_COMMIT,
        "conv_activation": preset.conv_activation,
        "conv_activation_alpha": preset.conv_activation_alpha,
    }
    if preset.conv_kernel_sizes:
        metadata["conv_kernel_sizes"] = list(preset.conv_kernel_sizes)
    model_json["metadata"] = metadata
    if preset.conv_activation == "tanh" and preset.conv_activation_alpha != 1.0:
        model_json["metadata"]["activation_export"] = {
            "strategy": "fold_scaled_tanh_into_conv1d",
            "trained_activation": f"tanh(x / {preset.conv_activation_alpha:g})",
            "exported_activation": "tanh",
        }
    return model_json


def fold_scaled_tanh_conv1d_activation(layer: dict[str, Any], alpha: float) -> None:
    weights = layer.get("weights")
    if not isinstance(weights, list) or not weights:
        return
    scale = 1.0 / float(alpha)
    layer["weights"] = [
        scale_nested_numbers(weight, scale)
        for weight in weights
    ]
    layer["activation"] = "tanh"
    layer["activation_fold"] = {
        "source": f"tanh(x / {alpha:g})",
        "scale": scale,
    }


def scale_nested_numbers(value: Any, scale: float) -> Any:
    if isinstance(value, (int, float)):
        return float(value) * scale
    if isinstance(value, list):
        return [scale_nested_numbers(item, scale) for item in value]
    return value


def default_parity_tolerance(preset_id: str) -> float:
    preset = get_preset(preset_id)
    if preset.architecture in {"gru", "conv_gru"}:
        return 3.0e-4
    return 1.0e-5


def resolve_parity_input(manifest: dict[str, Any], checkpoint_path: Path) -> Path:
    if manifest.get("input_path"):
        return Path(str(manifest["input_path"])).expanduser()
    if manifest.get("parity_input_path"):
        return Path(str(manifest["parity_input_path"])).expanduser()
    return checkpoint_path.parent.parent / "test-input.wav"


def tensor_to_list(tensor) -> list[Any]:  # type: ignore[no-untyped-def]
    return tensor.detach().cpu().tolist()


def render_export_prediction(manifest: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(manifest)
    input_path = resolve_parity_input(manifest, checkpoint_path)
    target_path = Path(
        str(manifest.get("target_path", checkpoint_path.parent.parent / "test-target.wav"))
    ).expanduser()
    output_dir = mkdir(Path(str(manifest.get("output_dir", "export-preview"))).expanduser())
    model, _checkpoint = load_checkpoint(checkpoint_path)
    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    prediction = predict_loaded_sequence(
        model,
        _checkpoint,
        input_audio.samples,
        export_device_preference(manifest, _checkpoint),
    )
    metrics = compute_metrics(target_audio.samples, prediction)
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "metrics_path": str(output_dir / "metrics.json")}


def export_device_preference(
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
) -> str | None:
    manifest_device = manifest.get("device")
    if manifest_device is not None:
        return str(manifest_device)
    checkpoint_device = checkpoint.get("device")
    if checkpoint_device is not None:
        return str(checkpoint_device)
    return None
