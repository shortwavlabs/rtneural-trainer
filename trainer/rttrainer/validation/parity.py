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
    device_preference: str | None = None,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path)
    input_audio = read_wav_mono(input_path)
    backend_output = predict_loaded_sequence(
        model,
        checkpoint,
        input_audio.samples,
        device_preference,
    )
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

    sequence = [[float(sample)] for sample in samples]
    for layer in model_json.get("layers", []):
        layer_type = layer.get("type")
        if layer_type in ("dense", "time-distributed-dense"):
            sequence = run_dense_layer(layer, sequence)
        elif layer_type == "activation":
            sequence = [
                apply_activation_vector(frame, str(layer.get("activation", "")))
                for frame in sequence
            ]
        elif layer_type == "lstm":
            sequence = run_lstm_layer(layer, sequence)
        elif layer_type == "gru":
            sequence = run_gru_layer(layer, sequence)
        elif layer_type == "conv1d":
            sequence = run_conv1d_layer(layer, sequence)
        elif layer_type == "batchnorm":
            sequence = run_batchnorm_layer(layer, sequence)
        elif layer_type == "prelu":
            sequence = run_prelu_layer(layer, sequence)
        else:
            raise ValueError(f"Unsupported RTNeural layer for parity: {layer_type!r}")

    if any(len(frame) != 1 for frame in sequence):
        width = len(sequence[0]) if sequence else 0
        raise ValueError(f"Parity simulator expected single-output JSON, found width {width}.")
    return [float(frame[0]) for frame in sequence]


def run_dense_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    kernel, bias = normalize_dense_weights(layer["weights"])
    outputs: list[list[float]] = []
    for frame in sequence:
        dense = [
            float(bias[output_index])
            + sum(
                frame[input_index] * kernel[input_index][output_index]
                for input_index in range(min(len(frame), len(kernel)))
            )
            for output_index in range(len(bias))
        ]
        outputs.append(apply_activation_vector(dense, str(layer.get("activation", ""))))
    return outputs


def run_lstm_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    kernel, recurrent_kernel, bias = normalize_lstm_weights(layer["weights"])
    hidden_size = int(layer.get("hidden_size", len(recurrent_kernel)))
    h = [0.0 for _ in range(hidden_size)]
    c = [0.0 for _ in range(hidden_size)]
    outputs: list[list[float]] = []

    for frame in sequence:
        gates = [
            float(bias[index])
            + sum(
                frame[input_index] * kernel[input_index][index]
                for input_index in range(min(len(frame), len(kernel)))
            )
            + sum(
                hidden_value * recurrent_kernel[hidden_index][index]
                for hidden_index, hidden_value in enumerate(h)
            )
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
        outputs.append(h.copy())

    return outputs


def run_gru_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    kernel, recurrent_kernel, bias = normalize_gru_weights(layer["weights"])
    hidden_size = int(layer.get("hidden_size", len(recurrent_kernel)))
    h = [0.0 for _ in range(hidden_size)]
    outputs: list[list[float]] = []

    for frame in sequence:
        x_gates = [
            float(bias[0][index])
            + sum(
                frame[input_index] * kernel[input_index][index]
                for input_index in range(min(len(frame), len(kernel)))
            )
            for index in range(hidden_size * 3)
        ]
        recurrent_gates = [
            float(bias[1][index])
            + sum(
                hidden_value * recurrent_kernel[hidden_index][index]
                for hidden_index, hidden_value in enumerate(h)
            )
            for index in range(hidden_size * 3)
        ]
        z_gate = [
            sigmoid(x_gates[index] + recurrent_gates[index])
            for index in range(hidden_size)
        ]
        r_gate = [
            sigmoid(x_gates[index + hidden_size] + recurrent_gates[index + hidden_size])
            for index in range(hidden_size)
        ]
        candidate = [
            math.tanh(
                x_gates[index + hidden_size * 2]
                + r_gate[index] * recurrent_gates[index + hidden_size * 2]
            )
            for index in range(hidden_size)
        ]
        h = [
            (1.0 - z_gate[index]) * candidate[index] + z_gate[index] * h[index]
            for index in range(hidden_size)
        ]
        outputs.append(h.copy())

    return outputs


def run_conv1d_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    kernel, bias = normalize_conv1d_weights(layer["weights"])
    kernel_size = int(first_scalar(layer.get("kernel_size", len(kernel))))
    dilation = int(first_scalar(layer.get("dilation", 1)))
    groups = int(layer.get("groups", 1))
    out_channels = len(bias)
    outputs: list[list[float]] = []

    for time_index in range(len(sequence)):
        frame_out = [float(bias[channel]) for channel in range(out_channels)]
        for kernel_index in range(kernel_size):
            source_index = time_index - (kernel_size - 1 - kernel_index) * dilation
            if source_index < 0:
                continue
            source = sequence[source_index]
            filters_per_group = max(1, len(source) // groups)
            channels_per_group = max(1, out_channels // groups)
            for input_channel in range(min(filters_per_group, len(kernel[kernel_index]))):
                for output_channel in range(out_channels):
                    group_index = output_channel // channels_per_group
                    source_channel = group_index * filters_per_group + input_channel
                    if source_channel >= len(source):
                        continue
                    frame_out[output_channel] += (
                        source[source_channel] * kernel[kernel_index][input_channel][output_channel]
                    )
        outputs.append(apply_activation_vector(frame_out, str(layer.get("activation", ""))))

    return outputs


def run_batchnorm_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    weights = layer["weights"]
    epsilon = float(layer.get("epsilon", 0.001))
    if len(weights) == 4:
        gamma = to_float_list(weights[0])
        beta = to_float_list(weights[1])
        mean = to_float_list(weights[2])
        variance = to_float_list(weights[3])
    elif len(weights) == 2:
        mean = to_float_list(weights[0])
        variance = to_float_list(weights[1])
        gamma = [1.0 for _ in mean]
        beta = [0.0 for _ in mean]
    else:
        raise ValueError("BatchNorm parity expects 2 or 4 weight arrays.")

    return [
        [
            gamma[index] * (value - mean[index]) / math.sqrt(variance[index] + epsilon)
            + beta[index]
            for index, value in enumerate(frame)
        ]
        for frame in sequence
    ]


def run_prelu_layer(layer: dict[str, Any], sequence: list[list[float]]) -> list[list[float]]:
    alpha = flatten_nested_numbers(layer["weights"][0])
    if len(alpha) == 1 and sequence and len(sequence[0]) > 1:
        alpha = alpha * len(sequence[0])
    return [
        [
            value if value >= 0.0 else alpha[min(index, len(alpha) - 1)] * value
            for index, value in enumerate(frame)
        ]
        for frame in sequence
    ]


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


def normalize_gru_weights(
    weights: Any,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    kernel = to_float_matrix(weights[0])
    recurrent_kernel = to_float_matrix(weights[1])
    raw_bias = weights[2]
    if raw_bias and isinstance(raw_bias[0], list):
        bias = [to_float_list(raw_bias[0]), to_float_list(raw_bias[1])]
    else:
        bias = [to_float_list(raw_bias), [0.0 for _ in raw_bias]]
    return kernel, recurrent_kernel, bias


def normalize_dense_weights(weights: Any) -> tuple[list[list[float]], list[float]]:
    if isinstance(weights, dict):
        return transpose(weights["weight"]), to_float_list(weights["bias"])
    kernel = to_float_matrix(weights[0])
    if len(weights) > 1:
        return kernel, to_float_list(weights[1])
    output_size = len(kernel[0]) if kernel else 0
    return kernel, [0.0 for _ in range(output_size)]


def normalize_conv1d_weights(weights: Any) -> tuple[list[list[list[float]]], list[float]]:
    kernel = to_float_3d(weights[0])
    if len(weights) > 1:
        return kernel, to_float_list(weights[1])
    output_size = len(kernel[0][0]) if kernel and kernel[0] else 0
    return kernel, [0.0 for _ in range(output_size)]


def transpose(matrix: Any) -> list[list[float]]:
    rows = to_float_matrix(matrix)
    if not rows:
        return []
    return [[row[index] for row in rows] for index in range(len(rows[0]))]


def to_float_matrix(values: Any) -> list[list[float]]:
    return [[float(item) for item in row] for row in values]


def to_float_3d(values: Any) -> list[list[list[float]]]:
    return [[[float(item) for item in channel] for channel in row] for row in values]


def to_float_list(values: Any) -> list[float]:
    return [float(item) for item in values]


def flatten_nested_numbers(values: Any) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)]
    flattened: list[float] = []
    for value in values:
        flattened.extend(flatten_nested_numbers(value))
    return flattened


def first_scalar(value: Any) -> int | float:
    if isinstance(value, (list, tuple)):
        return first_scalar(value[-1])
    return value


def apply_activation_vector(values: list[float], activation: str) -> list[float]:
    if activation in ("", "linear"):
        return values
    if activation == "tanh":
        return [math.tanh(value) for value in values]
    if activation == "relu":
        return [max(0.0, value) for value in values]
    if activation == "sigmoid":
        return [sigmoid(value) for value in values]
    if activation == "elu":
        return [value if value >= 0.0 else math.expm1(value) for value in values]
    if activation == "softmax":
        maximum = max(values, default=0.0)
        exps = [math.exp(value - maximum) for value in values]
        total = sum(exps)
        return [value / total for value in exps]
    raise ValueError(f"Unsupported RTNeural activation for parity: {activation!r}")


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
