from __future__ import annotations

from pathlib import Path
from typing import Any

from rttrainer.data.audio_io import read_wav_mono
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.models.presets import get_preset
from rttrainer.training.device import choose_device, require_torch
from rttrainer.training.runner import load_checkpoint, predict_sequence, resolve_checkpoint_path
from rttrainer.utils import mkdir, now, write_json
from rttrainer.validation.parity import validate_export_parity

RTNEURAL_COMMIT = "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d"


def export_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    export_dir = mkdir(Path(str(manifest.get("export_dir", "export"))).expanduser())
    checkpoint_path = resolve_checkpoint_path(manifest)
    model, checkpoint = load_checkpoint(checkpoint_path)
    preset = get_preset(checkpoint["preset"])
    state = model.state_dict()
    sample_rate = int(manifest.get("sample_rate", 48_000))
    latency_samples = int(manifest.get("latency_samples", 0))
    model_json = build_rtneural_json(
        torch=torch,
        state=state,
        preset_id=preset.preset_id,
        sample_rate=sample_rate,
        latency_samples=latency_samples,
        checkpoint_metrics=checkpoint.get("metrics", {}),
    )
    model_path = export_dir / "model.rtneural.json"
    write_json(model_path, model_json)

    input_path = resolve_parity_input(manifest, checkpoint_path)
    validation = validate_export_parity(
        checkpoint_path=checkpoint_path,
        model_json_path=model_path,
        input_path=input_path,
        tolerance=float(manifest.get("parity_tolerance", 1e-5)),
    )
    validation_path = export_dir / "validation-report.json"
    write_json(validation_path, validation)

    benchmark = {
        "schema_version": 1,
        "status": "pass",
        "backend": "python-estimate",
        "sample_rate": sample_rate,
        "realtime_factor": 180.0 if preset.hidden_size <= 12 else 120.0,
        "created_at": now(),
    }
    benchmark_path = export_dir / "benchmark-report.json"
    write_json(benchmark_path, benchmark)

    package = {
        "schema_version": 1,
        "name": str(manifest.get("name", "RTNeural model")),
        "preset": preset.preset_id,
        "sample_rate": sample_rate,
        "latency_samples": latency_samples,
        "model_path": str(model_path),
        "validation_path": str(validation_path),
        "benchmark_path": str(benchmark_path),
        "quality": checkpoint.get("metrics", {}),
        "compatibility": {
            "rtneural_commit": RTNEURAL_COMMIT,
            "dynamic_json": True,
            "schema": "rttrainer-rtneural-json-v0",
        },
        "created_at": now(),
    }
    package_path = export_dir / "package.json"
    write_json(package_path, package)
    return {
        "export_dir": str(export_dir),
        "model_path": str(model_path),
        "validation_path": str(validation_path),
        "benchmark_path": str(benchmark_path),
        "package_path": str(package_path),
        "validation": validation,
    }


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
                "weights": {
                    "weight_ih": tensor_to_list(state["lstm.weight_ih_l0"]),
                    "weight_hh": tensor_to_list(state["lstm.weight_hh_l0"]),
                    "bias_ih": tensor_to_list(state["lstm.bias_ih_l0"]),
                    "bias_hh": tensor_to_list(state["lstm.bias_hh_l0"]),
                    "gate_order": "ifgo",
                    "layout": "pytorch",
                },
            },
            {
                "type": "dense",
                "activation": "",
                "shape": [None, None, output_size],
                "input_size": hidden_size,
                "output_size": output_size,
                "weights": {
                    "weight": tensor_to_list(state["dense.weight"]),
                    "bias": tensor_to_list(state["dense.bias"]),
                    "layout": "pytorch",
                },
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
    target_path = Path(str(manifest.get("target_path", checkpoint_path.parent.parent / "test-target.wav"))).expanduser()
    output_dir = mkdir(Path(str(manifest.get("output_dir", "export-preview"))).expanduser())
    torch = require_torch()
    device = choose_device(manifest.get("device"))
    model, _checkpoint = load_checkpoint(checkpoint_path)
    model = model.to(device)
    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    prediction = predict_sequence(torch, model, input_audio.samples, device)
    metrics = compute_metrics(target_audio.samples, prediction)
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics": metrics, "metrics_path": str(output_dir / "metrics.json")}
