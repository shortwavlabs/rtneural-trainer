from __future__ import annotations

import math


def compute_metrics(target: list[float], prediction: list[float]) -> dict[str, float]:
    length = min(len(target), len(prediction))
    if length == 0:
        raise ValueError("Cannot compute metrics for empty signals.")

    residual = [target[index] - prediction[index] for index in range(length)]
    abs_errors = [abs(value) for value in residual]
    squared_errors = [value * value for value in residual]
    target_energy = sum(sample * sample for sample in target[:length])
    error_energy = sum(squared_errors)

    return {
        "esr": error_energy / max(target_energy, 1e-12),
        "mae": sum(abs_errors) / length,
        "rmse": math.sqrt(error_energy / length),
        "peak_residual": max(abs_errors, default=0.0),
        "rms_residual": math.sqrt(error_energy / length),
        "realtime_factor": 0.0,
    }
