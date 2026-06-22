from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, cast

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.models.presets import PresetConfig, build_keras_model, build_model, get_preset
from rttrainer.training.dataset import build_windowed_dataset
from rttrainer.training.device import choose_device, normalize_device_preference, require_torch
from rttrainer.utils import emit, mkdir, now, read_json, write_json

WINDOW_VALIDATION_SCORE_WEIGHT = 0.25
MIN_STREAM_PREDICTION_RMS_RATIO = 0.25
UNDERPOWERED_PREDICTION_PENALTY_WEIGHT = 0.75
STATE_DIAGNOSTIC_ESR_DELTA_THRESHOLD = 0.1
STATE_DIAGNOSTIC_CORRELATION_DELTA_THRESHOLD = 0.2
STATE_DIAGNOSTIC_MIN_CHUNK_CORRELATION = 0.25
DEFAULT_RECURRENT_CONTEXT_MULTIPLIER = 4
MAX_RECURRENT_CONTEXT_MULTIPLIER = 16
PREEMPHASIS_COEFFICIENT = 0.95
PREEMPHASIS_LOSS_WEIGHT = 0.35
MRSTFT_LOSS_WEIGHT = 0.02
MRSTFT_LOG_MAG_WEIGHT = 0.05
MRSTFT_FRAME_SIZES = (256, 1024, 2048)


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
    requested_learning_rate = float(manifest.get("learning_rate", 1e-3))
    loss_name = resolve_training_loss_name(manifest, preset)
    sequence_length = int(manifest.get("sequence_length", 1024))
    max_windows = int(manifest.get("max_windows", 512))
    preview_seconds = float(manifest.get("preview_seconds", 3.0))
    early_stopping_patience = max(0, int(manifest.get("early_stopping_patience", 5)))
    early_stopping_min_delta = max(0.0, float(manifest.get("early_stopping_min_delta", 1e-4)))
    device_preference = str(manifest.get("device", "auto"))
    context_training_enabled = recurrent_context_training_enabled(manifest, preset)
    context_multiplier = recurrent_context_training_multiplier(manifest)
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
        context_multiplier=context_multiplier,
    )
    best_model_path = checkpoint_dir / "best-model.keras"
    checkpoint_metadata_path = checkpoint_dir / "best-checkpoint.json"
    resume_checkpoint_path = resolve_resume_checkpoint_path(manifest, best_model_path)
    resumed_checkpoint: dict[str, Any] | None = None
    device_scope = tensorflow_device_scope(tf, device_preference)
    with tf.device(device_scope):
        if resume_checkpoint_path is not None:
            model, resumed_checkpoint = load_keras_checkpoint(resume_checkpoint_path)
        else:
            model = build_keras_model(preset, tf.keras)
        learning_rate = resolve_resume_learning_rate(
            manifest,
            requested_learning_rate,
            resumed_checkpoint,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=build_keras_loss(tf, loss_name),
        )
        lr_schedule = resolve_learning_rate_schedule(
            manifest,
            learning_rate,
            early_stopping_patience,
            early_stopping_min_delta,
        )
        lr_scheduler = build_keras_lr_scheduler(tf, model, lr_schedule)
    device_label = tensorflow_device_label(tf, device_preference, device_scope)
    resumed_epoch = int((resumed_checkpoint or {}).get("epoch", 0))
    target_epochs = target_epoch_count(manifest, resumed_epoch, requested_epochs)
    start_epoch = resumed_epoch + 1
    start_epoch = max(1, min(start_epoch, target_epochs + 1))
    resumed_metrics = numeric_metrics((resumed_checkpoint or {}).get("metrics", {}))
    if (resumed_checkpoint or {}).get("metric_basis") != "composite_validation_score":
        resumed_metrics = {}
    if resume_checkpoint_path is not None and not best_model_path.exists():
        save_keras_model_checkpoint(model, best_model_path)
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
            learning_rate=learning_rate,
            loss_name=loss_name,
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
            "learning_rate": learning_rate,
            "requested_learning_rate": requested_learning_rate,
            "loss": loss_name,
            "learning_rate_schedule": lr_schedule,
            "recurrent_context_training": recurrent_context_training_report(
                context_training_enabled,
                context_multiplier,
                len(dataset.context_train_input),
                dataset.sample_rate,
            ),
            "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        }
    )

    best_score = float(resumed_metrics.get("validation_score", float("inf")))
    last_metrics: dict[str, float] | None = dict(resumed_metrics) if resumed_metrics else None
    best_epoch = resumed_epoch
    history: list[dict[str, float | int | bool]] = []
    stopped_early: dict[str, Any] | None = None
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    lr_reductions: list[dict[str, float | int]] = []
    current_learning_rate = current_keras_learning_rate(tf, model)

    for epoch in range(start_epoch, target_epochs + 1):
        last_epoch = epoch
        epoch_learning_rate = current_keras_learning_rate(tf, model)
        with tf.device(device_scope):
            fit_history = model.fit(
                dataset.train_x,
                dataset.train_y,
                batch_size=batch_size,
                epochs=1,
                shuffle=True,
                verbose=0,
            )
            context_train_loss = fit_keras_context_sequence(
                numpy,
                model,
                dataset.context_train_input,
                dataset.context_train_target,
                context_training_enabled,
            )
            val_prediction = model.predict(dataset.val_x, verbose=0)
            stream_prediction = predict_keras_sequence(model, dataset.stream_val_input)
        window_metrics = compute_metrics(flatten_array(dataset.val_y), flatten_array(val_prediction))
        stream_metrics = compute_metrics(dataset.stream_val_target, stream_prediction)
        selection_metrics = validation_selection_metrics(
            stream_metrics,
            window_metrics,
            stream_prediction,
            dataset.stream_val_target,
        )
        last_metrics = stream_metrics
        checkpoint_metrics = validation_checkpoint_metrics(
            stream_metrics,
            window_metrics,
            selection_metrics,
        )
        window_train_loss = float(fit_history.history.get("loss", [0.0])[-1])
        train_loss = average_loss(window_train_loss, context_train_loss)
        validation_score = selection_metrics["validation_score"]
        is_best = validation_score < best_score - early_stopping_min_delta
        if is_best:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_keras_model_checkpoint(model, best_model_path)
            save_keras_checkpoint_metadata(
                checkpoint_metadata_path,
                model_path=best_model_path,
                preset=preset,
                epoch=epoch,
                metrics=checkpoint_metrics,
                seed=seed,
                sequence_length=sequence_length,
                tensorflow_version=tf.__version__,
                keras_version=keras_version(tf),
                device=device_label,
                learning_rate=epoch_learning_rate,
                loss_name=loss_name,
            )
        else:
            epochs_without_improvement += 1

        previous_learning_rate = epoch_learning_rate
        current_learning_rate = step_keras_lr_scheduler(
            tf,
            lr_scheduler,
            model,
            epoch,
            validation_score,
        )
        lr_reduced = current_learning_rate < previous_learning_rate - 1e-15
        if lr_reduced:
            reduction = {
                "epoch": epoch,
                "from": previous_learning_rate,
                "to": current_learning_rate,
                "factor": float(lr_schedule["factor"]),
                "patience": int(lr_schedule["patience"]),
            }
            lr_reductions.append(reduction)
            emit(
                {
                    "type": "learning_rate_reduced",
                    "run_id": run_id,
                    **reduction,
                }
            )

        history_point: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "window_train_loss": window_train_loss,
            "val_esr": last_metrics["esr"],
            "val_mae": last_metrics["mae"],
            "val_rmse": last_metrics["rmse"],
            "stream_val_esr": stream_metrics["esr"],
            "stream_val_mae": stream_metrics["mae"],
            "stream_val_rmse": stream_metrics["rmse"],
            "window_val_esr": window_metrics["esr"],
            "window_val_mae": window_metrics["mae"],
            "window_val_rmse": window_metrics["rmse"],
            "validation_score": validation_score,
            "prediction_rms_ratio": selection_metrics["prediction_rms_ratio"],
            "stream_prediction_rms_db": selection_metrics["prediction_rms_db"],
            "learning_rate": previous_learning_rate,
            "next_learning_rate": current_learning_rate,
            "learning_rate_reduced": lr_reduced,
            "is_best": is_best,
        }
        if context_train_loss is not None:
            history_point["context_train_loss"] = context_train_loss
        history.append(history_point)

        epoch_event: dict[str, Any] = {
            "type": "epoch",
            "run_id": run_id,
            "epoch": epoch,
            "total_epochs": target_epochs,
            "train_loss": train_loss,
            "window_train_loss": window_train_loss,
            "val_esr": last_metrics["esr"],
            "val_mae": last_metrics["mae"],
            "val_rmse": last_metrics["rmse"],
            "stream_val_esr": stream_metrics["esr"],
            "stream_val_mae": stream_metrics["mae"],
            "stream_val_rmse": stream_metrics["rmse"],
            "window_val_esr": window_metrics["esr"],
            "window_val_mae": window_metrics["mae"],
            "window_val_rmse": window_metrics["rmse"],
            "validation_score": validation_score,
            "prediction_rms_ratio": selection_metrics["prediction_rms_ratio"],
            "stream_prediction_rms_db": selection_metrics["prediction_rms_db"],
            "learning_rate": previous_learning_rate,
            "next_learning_rate": current_learning_rate,
            "learning_rate_reduced": lr_reduced,
            "is_best": is_best,
        }
        if context_train_loss is not None:
            epoch_event["context_train_loss"] = context_train_loss
        emit(epoch_event)
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_early = {
                "stopped": True,
                "reason": "validation_score_plateau",
                "metric": "validation_score",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
            }
            emit({"type": "early_stopping", "run_id": run_id, **stopped_early})
            break

    if last_metrics is None:
        raise RuntimeError("Training did not produce metrics.")

    with tf.device(device_scope):
        model, checkpoint = load_checkpoint(best_model_path)
        prediction = predict_keras_sequence(model, dataset.test_input)
        diagnostic_chunk_size = state_diagnostic_chunk_size(manifest, sequence_length)
        chunk_reset_prediction = (
            predict_keras_sequence_chunk_reset(
                model,
                dataset.test_input,
                diagnostic_chunk_size,
            )
            if recurrent_state_preset(preset)
            else []
        )
    metrics = compute_metrics(dataset.test_target, prediction)
    metrics["realtime_factor"] = estimate_realtime_factor(preset)
    state_diagnostic = state_reset_diagnostic(
        preset,
        target=dataset.test_target,
        continuous_prediction=prediction,
        chunk_reset_prediction=chunk_reset_prediction,
        chunk_size=diagnostic_chunk_size,
        sample_rate=dataset.sample_rate,
    )
    metrics.update(state_diagnostic_metrics(state_diagnostic))

    write_wav_mono(preview_dir / "target.wav", dataset.test_target, dataset.sample_rate)
    write_wav_mono(preview_dir / "prediction.wav", prediction, dataset.sample_rate)
    residual = [
        dataset.test_target[index] - prediction[index]
        for index in range(min(len(dataset.test_target), len(prediction)))
    ]
    write_wav_mono(preview_dir / "residual.wav", residual, dataset.sample_rate)
    if state_diagnostic["applies"]:
        write_wav_mono(
            preview_dir / "chunk-reset-prediction.wav",
            chunk_reset_prediction,
            dataset.sample_rate,
        )
        chunk_reset_residual = [
            dataset.test_target[index] - chunk_reset_prediction[index]
            for index in range(min(len(dataset.test_target), len(chunk_reset_prediction)))
        ]
        write_wav_mono(
            preview_dir / "chunk-reset-residual.wav",
            chunk_reset_residual,
            dataset.sample_rate,
        )
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
            "loss": loss_name,
            "epochs": last_epoch,
            "requested_epochs": requested_epochs,
            "target_epochs": target_epochs,
            "best_checkpoint_path": str(best_model_path),
            "checkpoint_metadata_path": str(checkpoint_metadata_path),
            "metrics": metrics,
            "quality_assessment": quality_assessment(metrics, state_diagnostic),
            "checkpoint_epoch": checkpoint.get("epoch", best_epoch),
            "validation_basis": "composite_validation_score",
            "state_diagnostic": state_diagnostic,
            "history": history,
            "dataset": dataset.summary,
            "early_stopping": stopped_early
            or {
                "stopped": False,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "best_epoch": best_epoch,
            },
            "learning_rate_schedule": {
                **lr_schedule,
                "initial_learning_rate": learning_rate,
                "requested_learning_rate": requested_learning_rate,
                "final_learning_rate": current_learning_rate,
                "reductions": lr_reductions,
            },
            "recurrent_context_training": recurrent_context_training_report(
                context_training_enabled,
                context_multiplier,
                len(dataset.context_train_input),
                dataset.sample_rate,
            ),
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
    requested_learning_rate = float(manifest.get("learning_rate", 1e-3))
    loss_name = resolve_training_loss_name(manifest, preset)
    sequence_length = int(manifest.get("sequence_length", 1024))
    max_windows = int(manifest.get("max_windows", 512))
    preview_seconds = float(manifest.get("preview_seconds", 3.0))
    early_stopping_patience = max(0, int(manifest.get("early_stopping_patience", 5)))
    early_stopping_min_delta = max(0.0, float(manifest.get("early_stopping_min_delta", 1e-4)))
    context_training_enabled = recurrent_context_training_enabled(manifest, preset)
    context_multiplier = recurrent_context_training_multiplier(manifest)
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
        context_multiplier=context_multiplier,
    )
    model = build_model(preset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=requested_learning_rate)
    criterion = build_torch_loss(torch, loss_name)
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
    learning_rate = resolve_resume_learning_rate(
        manifest,
        requested_learning_rate,
        resumed_checkpoint,
    )
    set_torch_learning_rate(optimizer, learning_rate)
    lr_schedule = resolve_learning_rate_schedule(
        manifest,
        learning_rate,
        early_stopping_patience,
        early_stopping_min_delta,
    )
    lr_scheduler = build_torch_lr_scheduler(torch, optimizer, lr_schedule)
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
    if (resumed_checkpoint or {}).get("metric_basis") != "composite_validation_score":
        resumed_metrics = {}
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
            loss_name,
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
            "learning_rate": learning_rate,
            "requested_learning_rate": requested_learning_rate,
            "loss": loss_name,
            "learning_rate_schedule": lr_schedule,
            "recurrent_context_training": recurrent_context_training_report(
                context_training_enabled,
                context_multiplier,
                len(dataset.context_train_input),
                dataset.sample_rate,
            ),
            "resumed_from_checkpoint": str(resume_checkpoint_path) if resume_checkpoint_path else None,
        }
    )
    best_score = float(resumed_metrics.get("validation_score", float("inf")))
    last_metrics: dict[str, float] | None = dict(resumed_metrics) if resumed_metrics else None
    best_epoch = resumed_epoch
    history: list[dict[str, float | int | bool]] = []
    stopped_early: dict[str, Any] | None = None
    epochs_without_improvement = 0
    last_epoch = start_epoch - 1
    lr_reductions: list[dict[str, float | int]] = []
    current_learning_rate = current_torch_learning_rate(optimizer)

    for epoch in range(start_epoch, target_epochs + 1):
        last_epoch = epoch
        epoch_learning_rate = current_torch_learning_rate(optimizer)
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

        context_train_loss = fit_torch_context_sequence(
            torch,
            model,
            optimizer,
            criterion,
            dataset.context_train_input,
            dataset.context_train_target,
            device,
            context_training_enabled,
        )
        val_prediction = predict_torch_tensor(torch, model, dataset.val_x, device)
        flat_target = flatten_array(val_y.squeeze(-1).detach().cpu().tolist())
        flat_pred = flatten_array(val_prediction.squeeze(-1).detach().cpu().tolist())
        window_metrics = compute_metrics(flat_target, flat_pred)
        stream_prediction = predict_torch_sequence(torch, model, dataset.stream_val_input, device)
        stream_metrics = compute_metrics(dataset.stream_val_target, stream_prediction)
        selection_metrics = validation_selection_metrics(
            stream_metrics,
            window_metrics,
            stream_prediction,
            dataset.stream_val_target,
        )
        last_metrics = stream_metrics
        checkpoint_metrics = validation_checkpoint_metrics(
            stream_metrics,
            window_metrics,
            selection_metrics,
        )
        window_train_loss = sum(losses) / max(1, len(losses))
        train_loss = average_loss(window_train_loss, context_train_loss)
        validation_score = selection_metrics["validation_score"]
        is_best = validation_score < best_score - early_stopping_min_delta
        if is_best:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_torch_checkpoint(
                torch,
                best_checkpoint_path,
                preset,
                model,
                optimizer,
                epoch,
                checkpoint_metrics,
                seed,
                sequence_length,
                loss_name,
            )
        else:
            epochs_without_improvement += 1

        previous_learning_rate = epoch_learning_rate
        current_learning_rate = step_torch_lr_scheduler(
            lr_scheduler,
            optimizer,
            validation_score,
        )
        lr_reduced = current_learning_rate < previous_learning_rate - 1e-15
        if lr_reduced:
            reduction = {
                "epoch": epoch,
                "from": previous_learning_rate,
                "to": current_learning_rate,
                "factor": float(lr_schedule["factor"]),
                "patience": int(lr_schedule["patience"]),
            }
            lr_reductions.append(reduction)
            emit(
                {
                    "type": "learning_rate_reduced",
                    "run_id": run_id,
                    **reduction,
                }
            )

        history_point: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "window_train_loss": window_train_loss,
            "val_esr": last_metrics["esr"],
            "val_mae": last_metrics["mae"],
            "val_rmse": last_metrics["rmse"],
            "stream_val_esr": stream_metrics["esr"],
            "stream_val_mae": stream_metrics["mae"],
            "stream_val_rmse": stream_metrics["rmse"],
            "window_val_esr": window_metrics["esr"],
            "window_val_mae": window_metrics["mae"],
            "window_val_rmse": window_metrics["rmse"],
            "validation_score": validation_score,
            "prediction_rms_ratio": selection_metrics["prediction_rms_ratio"],
            "stream_prediction_rms_db": selection_metrics["prediction_rms_db"],
            "learning_rate": previous_learning_rate,
            "next_learning_rate": current_learning_rate,
            "learning_rate_reduced": lr_reduced,
            "is_best": is_best,
        }
        if context_train_loss is not None:
            history_point["context_train_loss"] = context_train_loss
        history.append(history_point)

        epoch_event: dict[str, Any] = {
            "type": "epoch",
            "run_id": run_id,
            "epoch": epoch,
            "total_epochs": target_epochs,
            "train_loss": train_loss,
            "window_train_loss": window_train_loss,
            "val_esr": last_metrics["esr"],
            "val_mae": last_metrics["mae"],
            "val_rmse": last_metrics["rmse"],
            "stream_val_esr": stream_metrics["esr"],
            "stream_val_mae": stream_metrics["mae"],
            "stream_val_rmse": stream_metrics["rmse"],
            "window_val_esr": window_metrics["esr"],
            "window_val_mae": window_metrics["mae"],
            "window_val_rmse": window_metrics["rmse"],
            "validation_score": validation_score,
            "prediction_rms_ratio": selection_metrics["prediction_rms_ratio"],
            "stream_prediction_rms_db": selection_metrics["prediction_rms_db"],
            "learning_rate": previous_learning_rate,
            "next_learning_rate": current_learning_rate,
            "learning_rate_reduced": lr_reduced,
            "is_best": is_best,
        }
        if context_train_loss is not None:
            epoch_event["context_train_loss"] = context_train_loss
        emit(epoch_event)
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            stopped_early = {
                "stopped": True,
                "reason": "validation_score_plateau",
                "metric": "validation_score",
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
    diagnostic_chunk_size = state_diagnostic_chunk_size(manifest, sequence_length)
    chunk_reset_prediction = (
        predict_torch_sequence_chunk_reset(
            torch,
            model,
            dataset.test_input,
            device,
            diagnostic_chunk_size,
        )
        if recurrent_state_preset(preset)
        else []
    )
    metrics = compute_metrics(dataset.test_target, prediction)
    metrics["realtime_factor"] = estimate_realtime_factor(preset)
    state_diagnostic = state_reset_diagnostic(
        preset,
        target=dataset.test_target,
        continuous_prediction=prediction,
        chunk_reset_prediction=chunk_reset_prediction,
        chunk_size=diagnostic_chunk_size,
        sample_rate=dataset.sample_rate,
    )
    metrics.update(state_diagnostic_metrics(state_diagnostic))

    write_wav_mono(preview_dir / "target.wav", dataset.test_target, dataset.sample_rate)
    write_wav_mono(preview_dir / "prediction.wav", prediction, dataset.sample_rate)
    residual = [
        dataset.test_target[index] - prediction[index]
        for index in range(min(len(dataset.test_target), len(prediction)))
    ]
    write_wav_mono(preview_dir / "residual.wav", residual, dataset.sample_rate)
    if state_diagnostic["applies"]:
        write_wav_mono(
            preview_dir / "chunk-reset-prediction.wav",
            chunk_reset_prediction,
            dataset.sample_rate,
        )
        chunk_reset_residual = [
            dataset.test_target[index] - chunk_reset_prediction[index]
            for index in range(min(len(dataset.test_target), len(chunk_reset_prediction)))
        ]
        write_wav_mono(
            preview_dir / "chunk-reset-residual.wav",
            chunk_reset_residual,
            dataset.sample_rate,
        )
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
            "loss": loss_name,
            "epochs": last_epoch,
            "requested_epochs": requested_epochs,
            "target_epochs": target_epochs,
            "best_checkpoint_path": str(best_checkpoint_path),
            "metrics": metrics,
            "quality_assessment": quality_assessment(metrics, state_diagnostic),
            "checkpoint_epoch": checkpoint["epoch"],
            "validation_basis": "composite_validation_score",
            "state_diagnostic": state_diagnostic,
            "history": history,
            "dataset": dataset.summary,
            "early_stopping": stopped_early
            or {
                "stopped": False,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "best_epoch": best_epoch,
            },
            "learning_rate_schedule": {
                **lr_schedule,
                "initial_learning_rate": learning_rate,
                "requested_learning_rate": requested_learning_rate,
                "final_learning_rate": current_learning_rate,
                "reductions": lr_reductions,
            },
            "recurrent_context_training": recurrent_context_training_report(
                context_training_enabled,
                context_multiplier,
                len(dataset.context_train_input),
                dataset.sample_rate,
            ),
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
    learning_rate: float,
    loss_name: str,
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
            "metric_basis": "composite_validation_score",
            "seed": seed,
            "sequence_length": sequence_length,
            "learning_rate": learning_rate,
            "loss": loss_name,
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
    loss_name: str,
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
            "metric_basis": "composite_validation_score",
            "seed": seed,
            "sequence_length": sequence_length,
            "learning_rate": current_torch_learning_rate(optimizer),
            "loss": loss_name,
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
    model = tf.keras.models.load_model(path, compile=False)
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


def validation_checkpoint_metrics(
    stream_metrics: dict[str, float],
    window_metrics: dict[str, float],
    selection_metrics: dict[str, float],
) -> dict[str, float]:
    metrics = dict(stream_metrics)
    for key, value in window_metrics.items():
        metrics[f"window_{key}"] = value
    metrics["stream_val_esr"] = stream_metrics["esr"]
    metrics["stream_val_mae"] = stream_metrics["mae"]
    metrics["stream_val_rmse"] = stream_metrics["rmse"]
    metrics["window_val_esr"] = window_metrics["esr"]
    metrics["window_val_mae"] = window_metrics["mae"]
    metrics["window_val_rmse"] = window_metrics["rmse"]
    metrics.update(selection_metrics)
    return metrics


def validation_selection_metrics(
    stream_metrics: dict[str, float],
    window_metrics: dict[str, float],
    stream_prediction: list[float],
    stream_target: list[float],
) -> dict[str, float]:
    prediction_rms = rms_level(stream_prediction)
    target_rms = rms_level(stream_target)
    prediction_rms_ratio = prediction_rms / max(target_rms, 1.0e-12)
    underpowered_penalty = (
        max(0.0, MIN_STREAM_PREDICTION_RMS_RATIO - prediction_rms_ratio)
        * UNDERPOWERED_PREDICTION_PENALTY_WEIGHT
    )
    validation_score = (
        stream_metrics["esr"]
        + WINDOW_VALIDATION_SCORE_WEIGHT * window_metrics["esr"]
        + underpowered_penalty
    )
    return {
        "validation_score": validation_score,
        "stream_validation_score": stream_metrics["esr"],
        "window_validation_score": window_metrics["esr"],
        "window_validation_weight": WINDOW_VALIDATION_SCORE_WEIGHT,
        "underpowered_prediction_penalty": underpowered_penalty,
        "prediction_rms_ratio": prediction_rms_ratio,
        "prediction_rms": prediction_rms,
        "target_rms": target_rms,
        "prediction_rms_db": dbfs(prediction_rms),
        "target_rms_db": dbfs(target_rms),
    }


def state_diagnostic_chunk_size(manifest: dict[str, Any], sequence_length: int) -> int:
    requested = int(manifest.get("state_diagnostic_chunk_size", sequence_length))
    return max(1, requested)


def recurrent_state_preset(preset: PresetConfig) -> bool:
    return preset.architecture in {"gru", "lstm", "conv_gru"}


def recurrent_context_training_enabled(
    manifest: dict[str, Any],
    preset: PresetConfig,
) -> bool:
    return bool(
        manifest.get(
            "recurrent_context_training_enabled",
            recurrent_state_preset(preset),
        )
    )


def recurrent_context_training_multiplier(manifest: dict[str, Any]) -> int:
    requested = int(
        manifest.get(
            "recurrent_context_multiplier",
            DEFAULT_RECURRENT_CONTEXT_MULTIPLIER,
        )
    )
    return max(1, min(MAX_RECURRENT_CONTEXT_MULTIPLIER, requested))


def recurrent_context_training_report(
    enabled: bool,
    multiplier: int,
    samples: int,
    sample_rate: int,
) -> dict[str, bool | int | float]:
    return {
        "enabled": enabled,
        "multiplier": multiplier,
        "samples": samples,
        "seconds": samples / max(1, sample_rate),
    }


def average_loss(window_loss: float, context_loss: float | None) -> float:
    if context_loss is None:
        return window_loss
    return (window_loss + context_loss) / 2.0


def state_reset_diagnostic(
    preset: PresetConfig,
    *,
    target: list[float],
    continuous_prediction: list[float],
    chunk_reset_prediction: list[float],
    chunk_size: int,
    sample_rate: int,
) -> dict[str, Any]:
    continuous_metrics = compute_metrics(target, continuous_prediction)
    continuous_correlation = correlation_coefficient(target, continuous_prediction)
    applies = recurrent_state_preset(preset)
    if not applies:
        return {
            "schema_version": 1,
            "applies": False,
            "preset": preset.preset_id,
            "architecture": preset.architecture,
            "verdict": "finite_memory",
            "summary": "Preset has no recurrent state.",
            "action": "Use the normal continuous preview for export decisions.",
            "chunk_size": chunk_size,
            "chunk_seconds": chunk_size / max(1, sample_rate),
            "continuous_esr": continuous_metrics["esr"],
            "continuous_correlation": continuous_correlation,
        }

    chunk_metrics = compute_metrics(target, chunk_reset_prediction)
    chunk_correlation = correlation_coefficient(target, chunk_reset_prediction)
    esr_delta = continuous_metrics["esr"] - chunk_metrics["esr"]
    correlation_delta = chunk_correlation - continuous_correlation
    suspected = (
        esr_delta >= STATE_DIAGNOSTIC_ESR_DELTA_THRESHOLD
        and correlation_delta >= STATE_DIAGNOSTIC_CORRELATION_DELTA_THRESHOLD
        and chunk_correlation >= STATE_DIAGNOSTIC_MIN_CHUNK_CORRELATION
    )
    if suspected:
        verdict = "state_drift_suspected"
        summary = "Recurrent state drift suspected."
        action = (
            "Try the Conv1D finite-memory preset, or retrain the recurrent model "
            "with longer sequences before export."
        )
    else:
        verdict = "stable"
        summary = "Continuous and reset-chunk previews agree."
        action = "Use the normal continuous preview for export decisions."

    return {
        "schema_version": 1,
        "applies": True,
        "preset": preset.preset_id,
        "architecture": preset.architecture,
        "verdict": verdict,
        "summary": summary,
        "action": action,
        "chunk_size": chunk_size,
        "chunk_seconds": chunk_size / max(1, sample_rate),
        "continuous_esr": continuous_metrics["esr"],
        "chunk_reset_esr": chunk_metrics["esr"],
        "esr_delta": esr_delta,
        "continuous_correlation": continuous_correlation,
        "chunk_reset_correlation": chunk_correlation,
        "correlation_delta": correlation_delta,
        "continuous_rms_db": dbfs(rms_level(continuous_prediction)),
        "chunk_reset_rms_db": dbfs(rms_level(chunk_reset_prediction)),
    }


def state_diagnostic_metrics(diagnostic: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in [
        "continuous_esr",
        "chunk_reset_esr",
        "esr_delta",
        "continuous_correlation",
        "chunk_reset_correlation",
        "correlation_delta",
        "continuous_rms_db",
        "chunk_reset_rms_db",
    ]:
        value = diagnostic.get(key)
        if isinstance(value, (int, float)):
            metrics[f"state_{key}"] = float(value)
    if diagnostic.get("verdict") == "state_drift_suspected":
        metrics["state_drift_suspected"] = 1.0
    return metrics


def correlation_coefficient(left: list[float], right: list[float]) -> float:
    length = min(len(left), len(right))
    if length < 2:
        return 0.0
    left_slice = left[:length]
    right_slice = right[:length]
    left_mean = sum(left_slice) / length
    right_mean = sum(right_slice) / length
    numerator = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for left_value, right_value in zip(left_slice, right_slice, strict=False):
        left_centered = left_value - left_mean
        right_centered = right_value - right_mean
        numerator += left_centered * right_centered
        left_energy += left_centered * left_centered
        right_energy += right_centered * right_centered
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 1.0e-18:
        return 0.0
    return numerator / denominator


def rms_level(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1.0e-12))


def resolve_resume_learning_rate(
    manifest: dict[str, Any],
    requested_learning_rate: float,
    resumed_checkpoint: dict[str, Any] | None,
) -> float:
    if resumed_checkpoint is None or bool(manifest.get("allow_resume_learning_rate_increase", False)):
        return requested_learning_rate
    checkpoint_learning_rate = finite_float(resumed_checkpoint.get("learning_rate"), float("nan"))
    if math.isfinite(checkpoint_learning_rate) and checkpoint_learning_rate > 0:
        return min(requested_learning_rate, checkpoint_learning_rate)
    return min(requested_learning_rate, 1e-4)


def resolve_training_loss_name(
    manifest: dict[str, Any],
    preset: PresetConfig | None = None,
) -> str:
    raw_loss = manifest.get("loss")
    default_loss = preset.default_loss if preset is not None else "mse"
    loss_name = str(raw_loss if raw_loss is not None else default_loss).strip().lower()
    if loss_name in {"mse", "mean_squared_error"}:
        return "mse"
    if loss_name in {"esr", "error_to_signal", "error_to_signal_ratio"}:
        return "esr"
    if loss_name in {"preemphasis_mse", "pre_emphasis_mse", "emphasis_mse", "hf_mse"}:
        return "preemphasis_mse"
    if loss_name in {
        "mrstft_preemphasis",
        "mrstft_preemphasis_mse",
        "multi_resolution_stft",
        "multi_resolution_stft_mse",
        "mrstft_mse",
    }:
        return "mrstft_preemphasis"
    raise ValueError(
        "Training loss must be 'mse', 'esr', 'preemphasis_mse', or 'mrstft_preemphasis'."
    )


def build_keras_loss(tf, loss_name: str):  # type: ignore[no-untyped-def]
    if loss_name == "mse":
        return "mse"

    def esr_loss(y_true, y_pred):  # type: ignore[no-untyped-def]
        error = y_true - y_pred
        error_energy = tf.reduce_sum(tf.square(error), axis=[1, 2])
        target_energy = tf.reduce_sum(tf.square(y_true), axis=[1, 2])
        frame_count = tf.cast(tf.shape(y_true)[1] * tf.shape(y_true)[2], y_true.dtype)
        energy_floor = frame_count * tf.cast(1.0e-4, y_true.dtype)
        return tf.reduce_mean(error_energy / tf.maximum(target_energy, energy_floor))

    esr_loss.__name__ = "esr_loss"
    if loss_name == "esr":
        return esr_loss

    def preemphasis_mse_value(y_true, y_pred):  # type: ignore[no-untyped-def]
        base = tf.reduce_mean(tf.square(y_true - y_pred))

        def with_emphasis():  # type: ignore[no-untyped-def]
            coefficient = tf.cast(PREEMPHASIS_COEFFICIENT, y_true.dtype)
            weight = tf.cast(PREEMPHASIS_LOSS_WEIGHT, y_true.dtype)
            target_emphasis = y_true[:, 1:, :] - coefficient * y_true[:, :-1, :]
            prediction_emphasis = y_pred[:, 1:, :] - coefficient * y_pred[:, :-1, :]
            emphasis = tf.reduce_mean(tf.square(target_emphasis - prediction_emphasis))
            return base + weight * emphasis

        return tf.cond(tf.shape(y_true)[1] > 1, with_emphasis, lambda: base)

    def preemphasis_mse_loss(y_true, y_pred):  # type: ignore[no-untyped-def]
        return preemphasis_mse_value(y_true, y_pred)

    preemphasis_mse_loss.__name__ = "preemphasis_mse_loss"
    if loss_name == "preemphasis_mse":
        return preemphasis_mse_loss

    def mrstft_preemphasis_loss(y_true, y_pred):  # type: ignore[no-untyped-def]
        return preemphasis_mse_value(y_true, y_pred) + tf.cast(
            MRSTFT_LOSS_WEIGHT,
            y_true.dtype,
        ) * keras_multi_resolution_stft_loss(tf, y_true, y_pred)

    mrstft_preemphasis_loss.__name__ = "mrstft_preemphasis_loss"
    return mrstft_preemphasis_loss


def build_torch_loss(torch, loss_name: str):  # type: ignore[no-untyped-def]
    if loss_name == "mse":
        return torch.nn.MSELoss()

    def esr_loss(prediction, target):  # type: ignore[no-untyped-def]
        error_energy = torch.sum((target - prediction) ** 2, dim=(1, 2))
        target_energy = torch.sum(target**2, dim=(1, 2))
        frame_count = max(1, int(target.shape[1]) * int(target.shape[2]))
        energy_floor = frame_count * 1.0e-4
        return torch.mean(error_energy / target_energy.clamp_min(energy_floor))

    if loss_name == "esr":
        return esr_loss

    def preemphasis_mse_value(prediction, target):  # type: ignore[no-untyped-def]
        base = torch.mean((target - prediction) ** 2)
        if int(target.shape[1]) <= 1:
            return base
        coefficient = PREEMPHASIS_COEFFICIENT
        target_emphasis = target[:, 1:, :] - coefficient * target[:, :-1, :]
        prediction_emphasis = prediction[:, 1:, :] - coefficient * prediction[:, :-1, :]
        emphasis = torch.mean((target_emphasis - prediction_emphasis) ** 2)
        return base + PREEMPHASIS_LOSS_WEIGHT * emphasis

    def preemphasis_mse_loss(prediction, target):  # type: ignore[no-untyped-def]
        return preemphasis_mse_value(prediction, target)

    if loss_name == "preemphasis_mse":
        return preemphasis_mse_loss

    def mrstft_preemphasis_loss(prediction, target):  # type: ignore[no-untyped-def]
        return preemphasis_mse_value(
            prediction,
            target,
        ) + MRSTFT_LOSS_WEIGHT * torch_multi_resolution_stft_loss(torch, prediction, target)

    return mrstft_preemphasis_loss


def keras_multi_resolution_stft_loss(tf, y_true, y_pred):  # type: ignore[no-untyped-def]
    target = tf.squeeze(y_true, axis=-1)
    prediction = tf.squeeze(y_pred, axis=-1)
    total = tf.cast(0.0, y_true.dtype)
    epsilon = tf.cast(1.0e-5, y_true.dtype)
    log_weight = tf.cast(MRSTFT_LOG_MAG_WEIGHT, y_true.dtype)

    for frame_size in MRSTFT_FRAME_SIZES:
        target_spec = tf.signal.stft(
            target,
            frame_length=frame_size,
            frame_step=max(1, frame_size // 4),
            fft_length=frame_size,
            pad_end=True,
        )
        prediction_spec = tf.signal.stft(
            prediction,
            frame_length=frame_size,
            frame_step=max(1, frame_size // 4),
            fft_length=frame_size,
            pad_end=True,
        )
        target_mag = tf.abs(target_spec)
        prediction_mag = tf.abs(prediction_spec)
        diff = target_mag - prediction_mag
        convergence = tf.norm(diff) / tf.maximum(tf.norm(target_mag), epsilon)
        log_mag = tf.reduce_mean(
            tf.abs(tf.math.log(target_mag + epsilon) - tf.math.log(prediction_mag + epsilon))
        )
        total += convergence + log_weight * log_mag

    return total / tf.cast(len(MRSTFT_FRAME_SIZES), y_true.dtype)


def torch_multi_resolution_stft_loss(torch, prediction, target):  # type: ignore[no-untyped-def]
    prediction_flat = prediction.squeeze(-1)
    target_flat = target.squeeze(-1)
    total = target_flat.new_tensor(0.0)

    for frame_size in MRSTFT_FRAME_SIZES:
        window = torch.hann_window(
            frame_size,
            device=target_flat.device,
            dtype=target_flat.dtype,
        )
        target_spec = torch.stft(
            target_flat,
            n_fft=frame_size,
            hop_length=max(1, frame_size // 4),
            win_length=frame_size,
            window=window,
            center=True,
            return_complex=True,
        )
        prediction_spec = torch.stft(
            prediction_flat,
            n_fft=frame_size,
            hop_length=max(1, frame_size // 4),
            win_length=frame_size,
            window=window,
            center=True,
            return_complex=True,
        )
        target_mag = torch.abs(target_spec)
        prediction_mag = torch.abs(prediction_spec)
        diff = target_mag - prediction_mag
        convergence = torch.linalg.vector_norm(diff) / torch.clamp_min(
            torch.linalg.vector_norm(target_mag),
            1.0e-5,
        )
        log_mag = torch.mean(
            torch.abs(torch.log(target_mag + 1.0e-5) - torch.log(prediction_mag + 1.0e-5))
        )
        total = total + convergence + MRSTFT_LOG_MAG_WEIGHT * log_mag

    return total / len(MRSTFT_FRAME_SIZES)


def fit_keras_context_sequence(
    numpy,
    model,
    input_samples: list[float],
    target_samples: list[float],
    enabled: bool,
) -> float | None:  # type: ignore[no-untyped-def]
    if not enabled or not input_samples or not target_samples:
        return None
    length = min(len(input_samples), len(target_samples))
    context_x = numpy.asarray(input_samples[:length], dtype="float32").reshape(1, length, 1)
    context_y = numpy.asarray(target_samples[:length], dtype="float32").reshape(1, length, 1)
    history = model.fit(
        context_x,
        context_y,
        batch_size=1,
        epochs=1,
        shuffle=False,
        verbose=0,
    )
    return float(history.history.get("loss", [0.0])[-1])


def fit_torch_context_sequence(
    torch,
    model,
    optimizer,
    criterion,
    input_samples: list[float],
    target_samples: list[float],
    device,
    enabled: bool,
) -> float | None:  # type: ignore[no-untyped-def]
    if not enabled or not input_samples or not target_samples:
        return None
    length = min(len(input_samples), len(target_samples))
    model.train()
    context_x = torch.tensor(input_samples[:length], dtype=torch.float32).view(1, length, 1).to(device)
    context_y = torch.tensor(target_samples[:length], dtype=torch.float32).view(1, length, 1).to(device)
    optimizer.zero_grad()
    prediction = model(context_x)
    loss = criterion(prediction, context_y)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def save_keras_model_checkpoint(model, path: Path) -> None:  # type: ignore[no-untyped-def]
    try:
        model.save(path, include_optimizer=False)
    except TypeError:
        model.save(path)


def default_learning_rate_plateau_patience(early_stopping_patience: int) -> int:
    if early_stopping_patience > 0:
        return max(1, min(10, early_stopping_patience // 2))
    return 5


def resolve_learning_rate_schedule(
    manifest: dict[str, Any],
    initial_learning_rate: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> dict[str, bool | float | int | str]:
    factor = finite_float(manifest.get("learning_rate_plateau_factor"), 0.5)
    factor = min(0.99, max(0.01, factor))
    patience = int(
        manifest.get(
            "learning_rate_plateau_patience",
            default_learning_rate_plateau_patience(early_stopping_patience),
        )
    )
    patience = max(1, min(100, patience))
    min_delta = finite_float(
        manifest.get("learning_rate_plateau_min_delta"),
        early_stopping_min_delta,
    )
    min_delta = max(0.0, min_delta)
    min_learning_rate = finite_float(manifest.get("min_learning_rate"), 1e-6)
    min_learning_rate = max(0.0, min(min_learning_rate, initial_learning_rate))
    cooldown = max(0, int(manifest.get("learning_rate_plateau_cooldown", 0)))
    enabled = bool(manifest.get("learning_rate_plateau_enabled", True))
    if initial_learning_rate <= min_learning_rate:
        enabled = False
    return {
        "enabled": enabled,
        "monitor": "validation_score",
        "mode": "min",
        "factor": factor,
        "patience": patience,
        "min_delta": min_delta,
        "cooldown": cooldown,
        "min_learning_rate": min_learning_rate,
    }


def finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def build_keras_lr_scheduler(tf, model, schedule: dict[str, Any]):  # type: ignore[no-untyped-def]
    if not schedule.get("enabled", True):
        return None
    callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=str(schedule["monitor"]),
        factor=float(schedule["factor"]),
        patience=int(schedule["patience"]),
        verbose=0,
        mode=str(schedule["mode"]),
        min_delta=float(schedule["min_delta"]),
        cooldown=int(schedule["cooldown"]),
        min_lr=float(schedule["min_learning_rate"]),
    )
    callback.set_model(model)
    callback.on_train_begin({})
    return callback


def step_keras_lr_scheduler(  # type: ignore[no-untyped-def]
    tf,
    scheduler,
    model,
    epoch: int,
    metric: float,
) -> float:
    if scheduler is not None:
        scheduler.on_epoch_end(epoch - 1, {"validation_score": metric, "val_esr": metric})
    return current_keras_learning_rate(tf, model)


def current_keras_learning_rate(tf, model) -> float:  # type: ignore[no-untyped-def]
    learning_rate = model.optimizer.learning_rate
    try:
        return float(tf.keras.backend.get_value(learning_rate))
    except Exception:
        try:
            return float(learning_rate.numpy())
        except Exception:
            return float(learning_rate)


def build_torch_lr_scheduler(torch, optimizer, schedule: dict[str, Any]):  # type: ignore[no-untyped-def]
    if not schedule.get("enabled", True):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(schedule["mode"]),
        factor=float(schedule["factor"]),
        patience=int(schedule["patience"]),
        threshold=float(schedule["min_delta"]),
        threshold_mode="abs",
        cooldown=int(schedule["cooldown"]),
        min_lr=float(schedule["min_learning_rate"]),
    )


def step_torch_lr_scheduler(scheduler, optimizer, metric: float) -> float:  # type: ignore[no-untyped-def]
    if scheduler is not None:
        scheduler.step(metric)
    return current_torch_learning_rate(optimizer)


def current_torch_learning_rate(optimizer) -> float:  # type: ignore[no-untyped-def]
    return float(optimizer.param_groups[0]["lr"])


def set_torch_learning_rate(optimizer, learning_rate: float) -> None:  # type: ignore[no-untyped-def]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


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
        tf, _numpy = require_tensorflow()
        with tf.device(tensorflow_device_scope(tf, device_preference)):
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


def predict_torch_sequence_chunk_reset(
    torch,
    model,
    samples: list[float],
    device,
    chunk_size: int,
) -> list[float]:  # type: ignore[no-untyped-def]
    chunk_size = max(1, chunk_size)
    prediction: list[float] = []
    for start in range(0, len(samples), chunk_size):
        prediction.extend(
            predict_torch_sequence(torch, model, samples[start : start + chunk_size], device)
        )
    return prediction


def predict_keras_sequence(model, samples: list[float]) -> list[float]:
    _tf, numpy = require_tensorflow()
    tensor = numpy.asarray(samples, dtype="float32").reshape(1, -1, 1)
    prediction = model.predict(tensor, verbose=0)
    return flatten_array(prediction)


def predict_keras_sequence_chunk_reset(
    model,
    samples: list[float],
    chunk_size: int,
) -> list[float]:
    chunk_size = max(1, chunk_size)
    prediction: list[float] = []
    for start in range(0, len(samples), chunk_size):
        prediction.extend(predict_keras_sequence(model, samples[start : start + chunk_size]))
    return prediction


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
    if preset.preset_id == "wavenet_tcn_fast":
        return 8.0
    if preset.preset_id in {"wavenet_tcn", "wavenet_tcn_balanced"}:
        return 3.0
    if preset.preset_id == "wavenet_tcn_quality":
        return 1.5
    return 180.0 if preset.hidden_size <= 12 else 120.0


def quality_assessment(
    metrics: dict[str, float],
    state_diagnostic: dict[str, Any] | None = None,
) -> dict[str, str | float]:
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

    if (state_diagnostic or {}).get("verdict") == "state_drift_suspected":
        verdict = "needs_work"
        summary = "Recurrent state drift suspected."
        action = (
            "Try the Conv1D finite-memory preset, or retrain the recurrent model "
            "with longer sequences before export."
        )

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


def tensorflow_device_scope(tf, preferred: str | None = None) -> str:  # type: ignore[no-untyped-def]
    normalized = normalize_device_preference(preferred)
    if normalized == "cpu":
        return "/CPU:0"
    if normalized in {"mps", "cuda"}:
        if tf.config.list_logical_devices("GPU"):
            return "/GPU:0"
        raise RuntimeError(
            f"{normalized.upper()} was selected, but TensorFlow does not report a GPU device. "
            "On Apple Silicon, install/configure tensorflow-metal for TensorFlow GPU training "
            "or switch to the PyTorch backend for MPS."
        )
    if tf.config.list_logical_devices("GPU"):
        return "/GPU:0"
    return "/CPU:0"


def tensorflow_device_label(  # type: ignore[no-untyped-def]
    tf,
    preferred: str | None = None,
    device_scope: str | None = None,
) -> str:
    normalized = normalize_device_preference(preferred)
    scope = device_scope or tensorflow_device_scope(tf, preferred)
    gpus = tf.config.list_physical_devices("GPU")
    if scope.upper().endswith("GPU:0") and gpus:
        if normalized in {"mps", "cuda"}:
            return f"tensorflow-{normalized}:{gpus[0].name}"
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
