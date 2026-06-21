from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, cast

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.models.presets import PresetConfig, build_keras_model, build_model, get_preset
from rttrainer.training.dataset import build_windowed_dataset
from rttrainer.training.device import choose_device, require_torch
from rttrainer.utils import emit, mkdir, now, read_json, write_json


def run_training(manifest: dict[str, Any]) -> dict[str, Any]:
    backend = str(manifest.get("backend", "keras")).lower()
    if backend in {"keras", "tensorflow", "tf"}:
        return run_keras_training(manifest)
    if backend in {"pytorch", "torch"}:
        return run_pytorch_training(manifest)
    raise ValueError("Training backend must be 'keras' or 'pytorch'.")


def run_keras_training(manifest: dict[str, Any]) -> dict[str, Any]:
    tf, numpy = require_tensorflow()
    run_dir = mkdir(Path(str(manifest.get("run_dir", "run"))).expanduser())
    checkpoint_dir = mkdir(run_dir / "checkpoints")
    preview_dir = mkdir(run_dir / "previews")
    preset = get_preset(str(manifest.get("preset", "lstm_light")))
    run_id = str(manifest.get("run_id", f"run_{int(time.time())}"))
    seed = int(manifest.get("seed", 1337))
    requested_epochs = int(manifest.get("epochs", 20))
    batch_size = int(manifest.get("batch_size", 16))
    learning_rate = float(manifest.get("learning_rate", 1e-3))
    sequence_length = int(manifest.get("sequence_length", 1024))
    max_windows = int(manifest.get("max_windows", 512))
    preview_seconds = float(manifest.get("preview_seconds", 3.0))
    early_stopping_patience = max(0, int(manifest.get("early_stopping_patience", 5)))
    early_stopping_min_delta = max(0.0, float(manifest.get("early_stopping_min_delta", 1e-4)))
    input_path, target_path = resolve_audio_paths(manifest)
    if not input_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Training requires prepared input_path and target_path WAV files.")

    set_keras_seed(tf, numpy, seed)
    dataset = build_windowed_dataset(
        input_path,
        target_path,
        sequence_length,
        max_windows,
        seed,
        backend="numpy",
        preview_seconds=preview_seconds,
    )
    best_model_path = checkpoint_dir / "best-model.keras"
    checkpoint_metadata_path = checkpoint_dir / "best-checkpoint.json"
    resume_checkpoint_path = resolve_resume_checkpoint_path(manifest, best_model_path)
    resumed_checkpoint: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        model, resumed_checkpoint = load_keras_checkpoint(resume_checkpoint_path)
    else:
        model = build_keras_model(preset, tf.keras)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    device_label = tensorflow_device_label(tf)
    resumed_epoch = int((resumed_checkpoint or {}).get("epoch", 0))
    target_epochs = target_epoch_count(manifest, resumed_epoch, requested_epochs)
    start_epoch = resumed_epoch + 1
    start_epoch = max(1, min(start_epoch, target_epochs + 1))
    resumed_metrics = numeric_metrics((resumed_checkpoint or {}).get("metrics", {}))
    if resume_checkpoint_path is not None and not best_model_path.exists():
        model.save(best_model_path)
        save_keras_checkpoint_metadata(
            checkpoint_metadata_path,
            model_path=best_model_path,
            preset=preset,
            epoch=resumed_epoch,
            metrics=resumed_metrics,
            seed=seed,
            sequence_length=sequence_length,
            tensorflow_version=tf.__version__,
            keras_version=keras_version(tf),
            device=device_label,
        )

    emit(
        {
            "type": "run_started",
            "run_id": run_id,
            "preset": preset.preset_id,
            "backend": "keras",
            "device": device_label,
            "epochs": target_epochs,
            "requested_epochs": requested_epochs,
            "start_epoch": start_epoch,
            "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        }
    )

    best_esr = float(resumed_metrics.get("esr", float("inf")))
    last_metrics: dict[str, float] | None = dict(resumed_metrics) if resumed_metrics else None
    best_epoch = resumed_epoch
    history: list[dict[str, float | int | bool]] = []
    stopped_early: dict[str, Any] | None = None
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, target_epochs + 1):
        last_epoch = epoch
        fit_history = model.fit(
            dataset.train_x,
            dataset.train_y,
            batch_size=batch_size,
            epochs=1,
            shuffle=True,
            verbose=0,
        )
        val_prediction = model.predict(dataset.val_x, verbose=0)
        last_metrics = compute_metrics(flatten_array(dataset.val_y), flatten_array(val_prediction))
        train_loss = float(fit_history.history.get("loss", [0.0])[-1])
        is_best = last_metrics["esr"] < best_esr - early_stopping_min_delta
        if is_best:
            best_esr = last_metrics["esr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save(best_model_path)
            save_keras_checkpoint_metadata(
                checkpoint_metadata_path,
                model_path=best_model_path,
                preset=preset,
                epoch=epoch,
                metrics=last_metrics,
                seed=seed,
                sequence_length=sequence_length,
                tensorflow_version=tf.__version__,
                keras_version=keras_version(tf),
                device=device_label,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_esr": last_metrics["esr"],
                "val_mae": last_metrics["mae"],
                "val_rmse": last_metrics["rmse"],
                "is_best": is_best,
            }
        )

        emit(
            {
                "type": "epoch",
                "run_id": run_id,
                "epoch": epoch,
                "total_epochs": target_epochs,
                "train_loss": train_loss,
                "val_esr": last_metrics["esr"],
                "val_mae": last_metrics["mae"],
                "val_rmse": last_metrics["rmse"],
                "is_best": is_best,
            }
        )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_early = {
                "stopped": True,
                "reason": "validation_esr_plateau",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
            }
            emit({"type": "early_stopping", "run_id": run_id, **stopped_early})
            break

    if last_metrics is None:
        raise RuntimeError("Training did not produce metrics.")

    model, checkpoint = load_checkpoint(best_model_path)
    prediction = predict_keras_sequence(model, dataset.test_input)
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
    write_json(run_dir / "history.json", {"schema_version": 1, "history": history})
    write_json(
        run_dir / "training-report.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "preset": preset.preset_id,
            "backend": "keras",
            "device": device_label,
            "epochs": last_epoch,
            "requested_epochs": requested_epochs,
            "target_epochs": target_epochs,
            "best_checkpoint_path": str(best_model_path),
            "checkpoint_metadata_path": str(checkpoint_metadata_path),
            "metrics": metrics,
            "quality_assessment": quality_assessment(metrics),
            "checkpoint_epoch": checkpoint.get("epoch", best_epoch),
            "history": history,
            "dataset": dataset.summary,
            "early_stopping": stopped_early
            or {
                "stopped": False,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "best_epoch": best_epoch,
            },
            "tensorflow_version": tf.__version__,
            "keras_version": keras_version(tf),
            "created_at": now(),
        },
    )
    emit({"type": "checkpoint", "path": str(best_model_path), "is_best": True})
    emit({"type": "run_finished", "run_id": run_id, "status": "completed"})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": "keras",
        "best_checkpoint_path": str(best_model_path),
        "metrics": metrics,
    }


def run_pytorch_training(manifest: dict[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    run_dir = mkdir(Path(str(manifest.get("run_dir", "run"))).expanduser())
    checkpoint_dir = mkdir(run_dir / "checkpoints")
    preview_dir = mkdir(run_dir / "previews")
    preset = get_preset(str(manifest.get("preset", "lstm_light")))
    run_id = str(manifest.get("run_id", f"run_{int(time.time())}"))
    seed = int(manifest.get("seed", 1337))
    requested_epochs = int(manifest.get("epochs", 20))
    batch_size = int(manifest.get("batch_size", 16))
    learning_rate = float(manifest.get("learning_rate", 1e-3))
    sequence_length = int(manifest.get("sequence_length", 1024))
    max_windows = int(manifest.get("max_windows", 512))
    preview_seconds = float(manifest.get("preview_seconds", 3.0))
    early_stopping_patience = max(0, int(manifest.get("early_stopping_patience", 5)))
    early_stopping_min_delta = max(0.0, float(manifest.get("early_stopping_min_delta", 1e-4)))
    input_path, target_path = resolve_audio_paths(manifest)
    if not input_path.is_file() or not target_path.is_file():
        raise FileNotFoundError("Training requires prepared input_path and target_path WAV files.")

    set_torch_seed(torch, seed)
    device = choose_device(manifest.get("device"))
    dataset = build_windowed_dataset(
        input_path,
        target_path,
        sequence_length,
        max_windows,
        seed,
        backend="torch",
        preview_seconds=preview_seconds,
    )
    model = build_model(preset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    best_checkpoint_path = checkpoint_dir / "best-checkpoint.pt"
    resume_checkpoint_path = resolve_resume_checkpoint_path(manifest, best_checkpoint_path)
    resumed_checkpoint: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        loaded_checkpoint = cast(
            dict[str, Any],
            torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False),
        )
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        resumed_checkpoint = loaded_checkpoint
    train_x = cast(Any, dataset.train_x)
    train_y = cast(Any, dataset.train_y)
    val_y = cast(Any, dataset.val_y)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    resumed_epoch = int((resumed_checkpoint or {}).get("epoch", 0))
    target_epochs = target_epoch_count(manifest, resumed_epoch, requested_epochs)
    start_epoch = resumed_epoch + 1
    start_epoch = max(1, min(start_epoch, target_epochs + 1))
    resumed_metrics = numeric_metrics((resumed_checkpoint or {}).get("metrics", {}))
    if resume_checkpoint_path is not None and not best_checkpoint_path.exists():
        save_torch_checkpoint(
            torch,
            best_checkpoint_path,
            preset,
            model,
            optimizer,
            resumed_epoch,
            resumed_metrics,
            seed,
            sequence_length,
        )

    emit(
        {
            "type": "run_started",
            "run_id": run_id,
            "preset": preset.preset_id,
            "backend": "pytorch",
            "device": str(device),
            "epochs": target_epochs,
            "requested_epochs": requested_epochs,
            "start_epoch": start_epoch,
            "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        }
    )
    best_esr = float(resumed_metrics.get("esr", float("inf")))
    last_metrics: dict[str, float] | None = dict(resumed_metrics) if resumed_metrics else None
    best_epoch = resumed_epoch
    history: list[dict[str, float | int | bool]] = []
    stopped_early: dict[str, Any] | None = None
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, target_epochs + 1):
        last_epoch = epoch
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

        val_prediction = predict_torch_tensor(torch, model, dataset.val_x, device)
        flat_target = flatten_array(val_y.squeeze(-1).detach().cpu().tolist())
        flat_pred = flatten_array(val_prediction.squeeze(-1).detach().cpu().tolist())
        last_metrics = compute_metrics(flat_target, flat_pred)
        train_loss = sum(losses) / max(1, len(losses))
        is_best = last_metrics["esr"] < best_esr - early_stopping_min_delta
        if is_best:
            best_esr = last_metrics["esr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_checkpoint(
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
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_esr": last_metrics["esr"],
                "val_mae": last_metrics["mae"],
                "val_rmse": last_metrics["rmse"],
                "is_best": is_best,
            }
        )

        emit(
            {
                "type": "epoch",
                "run_id": run_id,
                "epoch": epoch,
                "total_epochs": target_epochs,
                "train_loss": train_loss,
                "val_esr": last_metrics["esr"],
                "val_mae": last_metrics["mae"],
                "val_rmse": last_metrics["rmse"],
                "is_best": is_best,
            }
        )
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_early = {
                "stopped": True,
                "reason": "validation_esr_plateau",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
            }
            emit({"type": "early_stopping", "run_id": run_id, **stopped_early})
            break

    if last_metrics is None:
        raise RuntimeError("Training did not produce metrics.")

    model, checkpoint = load_checkpoint(best_checkpoint_path)
    model = model.to(device)
    prediction = predict_torch_sequence(torch, model, dataset.test_input, device)
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
    write_json(run_dir / "history.json", {"schema_version": 1, "history": history})
    write_json(
        run_dir / "training-report.json",
        {
            "schema_version": 1,
            "run_id": run_id,
            "preset": preset.preset_id,
            "backend": "pytorch",
            "device": str(device),
            "epochs": last_epoch,
            "requested_epochs": requested_epochs,
            "target_epochs": target_epochs,
            "best_checkpoint_path": str(best_checkpoint_path),
            "metrics": metrics,
            "quality_assessment": quality_assessment(metrics),
            "checkpoint_epoch": checkpoint["epoch"],
            "history": history,
            "dataset": dataset.summary,
            "early_stopping": stopped_early
            or {
                "stopped": False,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "best_epoch": best_epoch,
            },
            "created_at": now(),
        },
    )
    emit({"type": "checkpoint", "path": str(best_checkpoint_path), "is_best": True})
    emit({"type": "run_finished", "run_id": run_id, "status": "completed"})
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": "pytorch",
        "best_checkpoint_path": str(best_checkpoint_path),
        "metrics": metrics,
    }


def evaluate_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    output_dir = mkdir(Path(str(manifest.get("output_dir", "evaluation"))).expanduser())
    checkpoint_path = resolve_checkpoint_path(manifest)
    input_path, target_path = resolve_audio_paths(manifest, allow_missing=True)
    if not input_path.is_file() or not target_path.is_file():
        run_dir = checkpoint_path.parent.parent
        input_path = run_dir / "test-input.wav"
        target_path = run_dir / "test-target.wav"

    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    model, checkpoint = load_checkpoint(checkpoint_path)
    prediction = predict_loaded_sequence(model, checkpoint, input_audio.samples, manifest.get("device"))
    metrics = compute_metrics(target_audio.samples, prediction)
    metrics["realtime_factor"] = estimate_realtime_factor(get_preset(checkpoint["preset"]))

    write_wav_mono(output_dir / "target.wav", target_audio.samples, target_audio.sample_rate)
    write_wav_mono(output_dir / "prediction.wav", prediction, target_audio.sample_rate)
    residual = [
        target_audio.samples[index] - prediction[index]
        for index in range(min(len(target_audio.samples), len(prediction)))
    ]
    write_wav_mono(output_dir / "residual.wav", residual, target_audio.sample_rate)
    write_json(output_dir / "metrics.json", metrics)
    return {"metrics_path": str(output_dir / "metrics.json"), "metrics": metrics}


def save_keras_checkpoint_metadata(
    path: Path,
    *,
    model_path: Path,
    preset: PresetConfig,
    epoch: int,
    metrics: dict[str, float],
    seed: int,
    sequence_length: int,
    tensorflow_version: str,
    keras_version: str,
    device: str,
) -> None:
    write_json(
        path,
        {
            "schema_version": 1,
            "backend": "keras",
            "preset": preset.preset_id,
            "model_config": preset.__dict__,
            "model_path": str(model_path),
            "epoch": epoch,
            "metrics": metrics,
            "seed": seed,
            "sequence_length": sequence_length,
            "tensorflow_version": tensorflow_version,
            "keras_version": keras_version,
            "device": device,
            "created_at": now(),
        },
    )


def save_torch_checkpoint(
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
            "backend": "pytorch",
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
    if path.suffix == ".keras":
        return load_keras_checkpoint(path)
    if path.suffix == ".json":
        metadata = read_json(path)
        if metadata.get("backend") == "keras":
            return load_keras_checkpoint(Path(str(metadata["model_path"])).expanduser())
    return load_torch_checkpoint(path)


def load_keras_checkpoint(path: Path):
    tf, _numpy = require_tensorflow()
    model = tf.keras.models.load_model(path)
    metadata_path = path.parent / "best-checkpoint.json"
    if metadata_path.exists():
        checkpoint = read_json(metadata_path)
    else:
        checkpoint = {
            "schema_version": 1,
            "backend": "keras",
            "preset": path.parent.parent.name,
            "model_path": str(path),
            "epoch": 0,
            "metrics": {},
        }
    checkpoint["backend"] = "keras"
    checkpoint["model_path"] = str(path)
    return model, checkpoint


def load_torch_checkpoint(path: Path):
    torch = require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.setdefault("backend", "pytorch")
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
        keras_path = run_dir / "checkpoints" / "best-model.keras"
        if keras_path.exists():
            return keras_path
        torch_path = run_dir / "checkpoints" / "best-checkpoint.pt"
        if torch_path.exists():
            return torch_path
        return keras_path
    raise ValueError("checkpoint_path or run_dir is required")


def resolve_resume_checkpoint_path(manifest: dict[str, Any], default_path: Path) -> Path | None:
    if not bool(manifest.get("resume_from_checkpoint", False)):
        return None
    checkpoint_path = Path(str(manifest.get("checkpoint_path", default_path))).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def target_epoch_count(
    manifest: dict[str, Any],
    resumed_epoch: int,
    requested_epochs: int,
) -> int:
    if bool(manifest.get("resume_epochs_are_additional", False)) and resumed_epoch > 0:
        return resumed_epoch + max(1, requested_epochs)
    return requested_epochs


def numeric_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for key, metric in value.items():
        if isinstance(metric, (int, float)):
            metrics[str(key)] = float(metric)
    return metrics


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


def predict_loaded_sequence(
    model,
    checkpoint: dict[str, Any],
    samples: list[float],
    device_preference: str | None = None,
) -> list[float]:
    if checkpoint.get("backend") == "keras":
        return predict_keras_sequence(model, samples)
    torch = require_torch()
    device = choose_device(device_preference)
    return predict_torch_sequence(torch, model.to(device), samples, device)


def predict_tensor(torch, model, tensor, device):  # type: ignore[no-untyped-def]
    return predict_torch_tensor(torch, model, tensor, device)


def predict_torch_tensor(torch, model, tensor, device):  # type: ignore[no-untyped-def]
    model.eval()
    with torch.no_grad():
        return model(tensor.to(device)).cpu()


def predict_sequence(torch, model, samples: list[float], device) -> list[float]:  # type: ignore[no-untyped-def]
    return predict_torch_sequence(torch, model, samples, device)


def predict_torch_sequence(torch, model, samples: list[float], device) -> list[float]:  # type: ignore[no-untyped-def]
    model.eval()
    tensor = torch.tensor(samples, dtype=torch.float32).view(1, -1, 1).to(device)
    with torch.no_grad():
        prediction = model(tensor).squeeze(0).squeeze(-1).cpu().tolist()
    return [float(value) for value in prediction]


def predict_keras_sequence(model, samples: list[float]) -> list[float]:
    _tf, numpy = require_tensorflow()
    tensor = numpy.asarray(samples, dtype="float32").reshape(1, -1, 1)
    prediction = model.predict(tensor, verbose=0)
    return flatten_array(prediction)


def flatten(values: list[list[float]]) -> list[float]:
    return flatten_array(values)


def flatten_array(values) -> list[float]:  # type: ignore[no-untyped-def]
    if hasattr(values, "reshape"):
        return [float(value) for value in values.reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        flattened: list[float] = []
        for value in values:
            if isinstance(value, (list, tuple)):
                flattened.extend(flatten_array(value))
            else:
                flattened.append(float(value))
        return flattened
    return [float(values)]


def estimate_realtime_factor(preset: PresetConfig) -> float:
    # Placeholder until native RTNeural benchmarking is wired to the trainer.
    return 180.0 if preset.hidden_size <= 12 else 120.0


def quality_assessment(metrics: dict[str, float]) -> dict[str, str | float]:
    esr = float(metrics.get("esr", 1.0))
    rmse = float(metrics.get("rmse", 1.0))
    peak_residual = float(metrics.get("peak_residual", 1.0))
    realtime_factor = float(metrics.get("realtime_factor", 0.0))

    if esr <= 0.03 and rmse <= 0.03 and realtime_factor >= 40:
        verdict = "good"
        summary = "Good candidate for export."
        action = "Listen to the residual and export if the preview matches the target."
    elif esr <= 0.10 and rmse <= 0.08 and realtime_factor >= 20:
        verdict = "usable"
        summary = "Usable, but inspect before shipping."
        action = "Compare target and prediction. Try a richer preset if the residual is audible."
    else:
        verdict = "needs_work"
        summary = "Needs more work before export."
        action = (
            "Check alignment and gain staging, then train longer or choose a stronger preset."
        )

    if peak_residual > 0.5:
        verdict = "needs_work"
        summary = "Residual peaks are high."
        action = "Look for alignment slips, clipping, or missing capture dynamics."

    return {
        "verdict": verdict,
        "summary": summary,
        "action": action,
        "esr": esr,
        "rmse": rmse,
        "peak_residual": peak_residual,
        "realtime_factor": realtime_factor,
    }


def require_tensorflow():
    try:
        import numpy
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the canonical Keras training/export path. "
            "Install it with: uv sync --extra tensorflow"
        ) from exc
    return tf, numpy


def tensorflow_device_label(tf) -> str:  # type: ignore[no-untyped-def]
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return f"tensorflow-gpu:{gpus[0].name}"
    return "tensorflow-cpu"


def keras_version(tf) -> str:  # type: ignore[no-untyped-def]
    return str(getattr(tf.keras, "__version__", getattr(tf, "__version__", "unknown")))


def set_keras_seed(tf, numpy, seed: int) -> None:  # type: ignore[no-untyped-def]
    random.seed(seed)
    numpy.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_torch_seed(torch, seed: int) -> None:  # type: ignore[no-untyped-def]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
