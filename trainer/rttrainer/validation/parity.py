from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rttrainer.data.audio_io import read_wav_mono
from rttrainer.training.runner import load_checkpoint, predict_loaded_sequence
from rttrainer.utils import now


def validate_export_parity(
    *,
    checkpoint_path: Path,
    model_json_path: Path,
    input_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    input_audio = read_wav_mono(input_path)
    backend_output = predict_loaded_sequence(model, checkpoint, input_audio.samples)
    json_output = run_exported_json(model_json_path, input_audio.samples)
    errors = [
        abs(backend_output[index] - json_output[index])
        for index in range(min(len(backend_output), len(json_output)))
    ]
    max_abs_error = max(errors, default=0.0)
    rmse = (sum(error * error for error in errors) / max(1, len(errors))) ** 0.5

    return {
        "schema_version": 1,
        "status": "pass" if max_abs_error <= tolerance else "fail",
        "backend": checkpoint.get("backend", "unknown"),
        "preset": checkpoint["preset"],
        "input_path": str(input_path),
        "model_json_path": str(model_json_path),
        "tolerance": tolerance,
        "sample_count": len(errors),
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "created_at": now(),
    }


def run_exported_json(model_json_path: Path, samples: list[float]) -> list[float]:
    with model_json_path.open("r", encoding="utf-8") as handle:
        model_json = json.load(handle)

    lstm_layer = next(
        (layer for layer in model_json["layers"] if layer.get("type") == "lstm"),
        None,
    )
    dense_layer = next(
        (layer for layer in model_json["layers"] if layer.get("type") == "dense"),
        None,
    )
    if lstm_layer is None or dense_layer is None:
        raise ValueError("Parity simulator currently supports LSTM + Dense exports only.")

    kernel, recurrent_kernel, bias = normalize_lstm_weights(lstm_layer["weights"])
    dense_kernel, dense_bias = normalize_dense_weights(dense_layer["weights"])
    hidden_size = int(lstm_layer.get("hidden_size", len(recurrent_kernel)))
    h = [0.0 for _ in range(hidden_size)]
    c = [0.0 for _ in range(hidden_size)]
    outputs: list[float] = []

    for sample in samples:
        gates = [
            float(bias[index])
            + sum(input_value * kernel[input_index][index] for input_index, input_value in enumerate([sample]))
            + sum(hidden_value * recurrent_kernel[hidden_index][index] for hidden_index, hidden_value in enumerate(h))
            for index in range(hidden_size * 4)
        ]
        i_gate = [sigmoid(value) for value in gates[0:hidden_size]]
        f_gate = [sigmoid(value) for value in gates[hidden_size : hidden_size * 2]]
        g_gate = [math.tanh(value) for value in gates[hidden_size * 2 : hidden_size * 3]]
        o_gate = [sigmoid(value) for value in gates[hidden_size * 3 : hidden_size * 4]]
        c = [
            f_gate[index] * c[index] + i_gate[index] * g_gate[index]
            for index in range(hidden_size)
        ]
        h = [o_gate[index] * math.tanh(c[index]) for index in range(hidden_size)]
        dense = [
            float(dense_bias[output_index])
            + sum(h[index] * dense_kernel[index][output_index] for index in range(hidden_size))
            for output_index in range(len(dense_bias))
        ]
        outputs.append(float(dense[0]))

    return outputs


def normalize_lstm_weights(weights: Any) -> tuple[list[list[float]], list[list[float]], list[float]]:
    if isinstance(weights, dict):
        kernel = transpose(weights["weight_ih"])
        recurrent_kernel = transpose(weights["weight_hh"])
        bias = [
            float(weights["bias_ih"][index]) + float(weights["bias_hh"][index])
            for index in range(len(weights["bias_ih"]))
        ]
        return kernel, recurrent_kernel, bias
    return to_float_matrix(weights[0]), to_float_matrix(weights[1]), to_float_list(weights[2])


def normalize_dense_weights(weights: Any) -> tuple[list[list[float]], list[float]]:
    if isinstance(weights, dict):
        return transpose(weights["weight"]), to_float_list(weights["bias"])
    return to_float_matrix(weights[0]), to_float_list(weights[1])


def transpose(matrix: Any) -> list[list[float]]:
    rows = to_float_matrix(matrix)
    if not rows:
        return []
    return [[row[index] for row in rows] for index in range(len(rows[0]))]


def to_float_matrix(values: Any) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def to_float_list(values: Any) -> list[float]:
    return [float(item) for item in values]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
