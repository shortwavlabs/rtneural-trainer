from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rttrainer.data.audio_io import read_wav_mono
from rttrainer.training.device import choose_device, require_torch
from rttrainer.training.runner import load_checkpoint, predict_sequence
from rttrainer.utils import now


def validate_export_parity(
    *,
    checkpoint_path: Path,
    model_json_path: Path,
    input_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    torch = require_torch()
    device = choose_device(None)
    model, checkpoint = load_checkpoint(checkpoint_path)
    model = model.to(device)
    input_audio = read_wav_mono(input_path)
    pytorch_output = predict_sequence(torch, model, input_audio.samples, device)
    json_output = run_exported_json(torch, model_json_path, input_audio.samples)
    errors = [
        abs(pytorch_output[index] - json_output[index])
        for index in range(min(len(pytorch_output), len(json_output)))
    ]
    max_abs_error = max(errors, default=0.0)
    rmse = (
        sum(error * error for error in errors) / max(1, len(errors))
    ) ** 0.5

    return {
        "schema_version": 1,
        "status": "pass" if max_abs_error <= tolerance else "fail",
        "preset": checkpoint["preset"],
        "input_path": str(input_path),
        "model_json_path": str(model_json_path),
        "tolerance": tolerance,
        "sample_count": len(errors),
        "max_abs_error": max_abs_error,
        "rmse": rmse,
        "created_at": now(),
    }


def run_exported_json(torch, model_json_path: Path, samples: list[float]) -> list[float]:  # type: ignore[no-untyped-def]
    with model_json_path.open("r", encoding="utf-8") as handle:
        model_json = json.load(handle)
    lstm_layer = model_json["layers"][0]
    dense_layer = model_json["layers"][1]
    weights = lstm_layer["weights"]
    dense = dense_layer["weights"]

    if isinstance(weights, dict):
        weight_ih = torch.tensor(weights["weight_ih"], dtype=torch.float32)
        weight_hh = torch.tensor(weights["weight_hh"], dtype=torch.float32)
        bias = torch.tensor(weights["bias_ih"], dtype=torch.float32) + torch.tensor(
            weights["bias_hh"], dtype=torch.float32
        )
    else:
        weight_ih = torch.tensor(weights[0], dtype=torch.float32).transpose(0, 1)
        weight_hh = torch.tensor(weights[1], dtype=torch.float32).transpose(0, 1)
        bias = torch.tensor(weights[2], dtype=torch.float32)

    if isinstance(dense, dict):
        dense_weight = torch.tensor(dense["weight"], dtype=torch.float32)
        dense_bias = torch.tensor(dense["bias"], dtype=torch.float32)
    else:
        dense_weight = torch.tensor(dense[0], dtype=torch.float32).transpose(0, 1)
        dense_bias = torch.tensor(dense[1], dtype=torch.float32)

    hidden_size = int(lstm_layer.get("hidden_size", weight_hh.shape[1]))
    h = torch.zeros(hidden_size, dtype=torch.float32)
    c = torch.zeros(hidden_size, dtype=torch.float32)
    outputs: list[float] = []

    for sample in samples:
        x = torch.tensor([sample], dtype=torch.float32)
        gates = weight_ih @ x + weight_hh @ h + bias
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c = f_gate * c + i_gate * g_gate
        h = o_gate * torch.tanh(c)
        y = dense_weight @ h + dense_bias
        outputs.append(float(y.squeeze().item()))

    return outputs
