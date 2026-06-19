from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

from rttrainer.data.audio_io import write_wav_mono
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.models.presets import PresetConfig, build_model, get_preset
from rttrainer.training.dataset import build_windowed_dataset
from rttrainer.training.device import choose_device, require_torch
from rttrainer.utils import emit, mkdir, now, write_json


def run_training(manifest: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    run_dir = mkdir(Path(str(manifest.get("run_dir", "run"))).expanduser())
    checkpoint_dir = mkdir(run_dir / "checkpoints")
    preview_dir = mkdir(run_dir / "previews")
    preset = get_preset(str(manifest.get("preset", "lstm_light")))
    run_id = str(manifest.get("run_id", f"run_{int(time.time())}"))
    seed = int(manifest.get("seed", 1337))
    epochs = int(manifest.get("epochs", 20))
    batch_size = int(manifest.get("batch_size", 16))
    learning_rate = float(manifest.get("learning_rate", 1e-3))
    sequence_length = int(manifest.get("sequence_length", 1024))
    max_windows = int(manifest.get("max_windows", 512))
    input_path, target_path = resolve_audio_paths(manifest)
    if not input_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Training requires prepared input_path and target_path WAV files.")

    set_seed(torch, seed)
    device = choose_device(manifest.get("device"))
    dataset = build_windowed_dataset(input_path, target_path, sequence_length, max_windows, seed)
    model = build_model(preset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(dataset.train_x, dataset.train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    emit(
        {
            "type": "run_started",
            "run_id": run_id,
            "preset": preset.preset_id,
            "device": str(device),
            "epochs": epochs,
        }
    )
    best_esr = float("inf")
    best_checkpoint_path = checkpoint_dir / "best-checkpoint.pt"
    last_metrics: dict[str, float] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        val_prediction = predict_tensor(torch, model, dataset.val_x, device)
        val_target = dataset.val_y.squeeze(-1).detach().cpu().tolist()
        val_pred = val_prediction.squeeze(-1).detach().cpu().tolist()
        flat_target = flatten(val_target)
        flat_pred = flatten(val_pred)
        last_metrics = compute_metrics(flat_target, flat_pred)
        train_loss = sum(losses) / max(1, len(losses))
        is_best = last_metrics["esr"] < best_esr
        if is_best:
            best_esr = last_metrics["esr"]
            save_checkpoint(
                torch,
                best_checkpoint_path,
                preset,
                model,
                optimizer,
                epoch,
                last_metrics,
                seed,
                sequence_length,
            )

        emit(
            {
                "type": "epoch",
                "run_id": run_id,
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": train_loss,
                "val_esr": last_metrics["esr"],
                "is_best": is_best,
            }
        )

    if last_metrics is None:
        raise RuntimeError("Training did not produce metrics.")

    model, checkpoint = load_checkpoint(best_checkpoint_path)
    model = model.to(device)
    prediction = predict_sequence(torch, model, dataset.test_input, device)
    metrics = compute_metrics(dataset.test_target, prediction)
    metrics["realtime_factor"] = estimate_realtime_factor(preset)

    write_wav_mono(preview_dir / "target.wav", dataset.test_target, dataset.sample_rate)
    write_wav_mono(preview_dir / "prediction.wav", prediction, dataset.sample_rate)
    residual = [
        dataset.test_target[index] - prediction[index]
        for index in range(min(len(dataset.test_target), len(prediction)))
    ]
    write_wav_mono(preview_dir / "residual.wav", residual, dataset.sample_rate)
    write_wav_mono(run_dir / "test-input.wav", dataset.test_input, dataset.sample_rate)
    write_wav_mono(run_dir / "test-target.wav", dataset.test_target, dataset.sample_rate)
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "training-report.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "preset": preset.preset_id,
            "device": str(device),
            "epochs": epochs,
            "best_checkpoint_path": str(best_checkpoint_path),
            "metrics": metrics,
            "checkpoint_epoch": checkpoint["epoch"],
            "created_at": now(),
        },
    )
    emit({"type": "checkpoint", "path": str(best_checkpoint_path), "is_best": True})
    emit({"type": "run_finished", "run_id": run_id, "status": "completed"})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_checkpoint_path": str(best_checkpoint_path),
        "metrics": metrics,
    }


def evaluate_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    output_dir = mkdir(Path(str(manifest.get("output_dir", "evaluation"))).expanduser())
    checkpoint_path = resolve_checkpoint_path(manifest)
    input_path, target_path = resolve_audio_paths(manifest, allow_missing=True)
    if not input_path.is_file() or not target_path.is_file():
        run_dir = checkpoint_path.parent.parent
        input_path = run_dir / "test-input.wav"
        target_path = run_dir / "test-target.wav"
    from rttrainer.data.audio_io import read_wav_mono

    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    device = choose_device(manifest.get("device"))
    model, _checkpoint = load_checkpoint(checkpoint_path)
    model = model.to(device)
    prediction = predict_sequence(torch, model, input_audio.samples, device)
    metrics = compute_metrics(target_audio.samples, prediction)
    metrics["realtime_factor"] = estimate_realtime_factor(get_preset(_checkpoint["preset"]))

    write_wav_mono(output_dir / "target.wav", target_audio.samples, target_audio.sample_rate)
    write_wav_mono(output_dir / "prediction.wav", prediction, target_audio.sample_rate)
    residual = [
        target_audio.samples[index] - prediction[index]
        for index in range(min(len(target_audio.samples), len(prediction)))
    ]
    write_wav_mono(output_dir / "residual.wav", residual, target_audio.sample_rate)
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics_path": str(output_dir / "metrics.json"), "metrics": metrics}


def save_checkpoint(
    torch,
    path: Path,
    preset: PresetConfig,
    model,
    optimizer,
    epoch: int,
    metrics: dict[str, float],
    seed: int,
    sequence_length: int,
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "preset": preset.preset_id,
            "model_config": preset.__dict__,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "seed": seed,
            "sequence_length": sequence_length,
            "created_at": now(),
        },
        path,
    )


def load_checkpoint(path: Path):
    torch = require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    preset = get_preset(checkpoint["preset"])
    model = build_model(preset)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def resolve_checkpoint_path(manifest: dict[str, Any]) -> Path:
    if manifest.get("checkpoint_path"):
        return Path(str(manifest["checkpoint_path"])).expanduser()
    run_dir = Path(str(manifest.get("run_dir", ""))).expanduser()
    if run_dir:
        return run_dir / "checkpoints" / "best-checkpoint.pt"
    raise ValueError("checkpoint_path or run_dir is required")


def resolve_audio_paths(
    manifest: dict[str, Any],
    *,
    allow_missing: bool = False,
) -> tuple[Path, Path]:
    prepared_dir = manifest.get("prepared_dir")
    input_value = manifest.get("input_path", manifest.get("prepared_input_path"))
    target_value = manifest.get("target_path", manifest.get("prepared_target_path"))
    if prepared_dir and not input_value:
        input_value = str(Path(str(prepared_dir)).expanduser() / "input.wav")
    if prepared_dir and not target_value:
        target_value = str(Path(str(prepared_dir)).expanduser() / "target.wav")
    if allow_missing and (not input_value or not target_value):
        return Path(), Path()
    if not input_value or not target_value:
        raise ValueError("input_path/target_path or prepared_dir is required")
    return Path(str(input_value)).expanduser(), Path(str(target_value)).expanduser()


def predict_tensor(torch, model, tensor, device):  # type: ignore[no-untyped-def]
    model.eval()
    with torch.no_grad():
        return model(tensor.to(device)).cpu()


def predict_sequence(torch, model, samples: list[float], device) -> list[float]:  # type: ignore[no-untyped-def]
    model.eval()
    tensor = torch.tensor(samples, dtype=torch.float32).view(1, -1, 1).to(device)
    with torch.no_grad():
        prediction = model(tensor).squeeze(0).squeeze(-1).cpu().tolist()
    return [float(value) for value in prediction]


def flatten(values: list[list[float]]) -> list[float]:
    return [item for row in values for item in row]


def estimate_realtime_factor(preset: PresetConfig) -> float:
    # Placeholder until native RTNeural benchmarking is wired to the trainer.
    return 180.0 if preset.hidden_size <= 12 else 120.0


def set_seed(torch, seed: int) -> None:  # type: ignore[no-untyped-def]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
