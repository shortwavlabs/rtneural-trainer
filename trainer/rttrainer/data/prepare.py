from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rttrainer.data.audio_io import AudioBuffer, audio_report, read_wav_mono, write_wav_mono
from rttrainer.utils import mkdir, write_json


@dataclass(frozen=True)
class PreparedAudio:
    input_path: Path
    target_path: Path
    report_path: Path
    report: dict


def prepare_audio(input_path: Path, target_path: Path, output_dir: Path) -> PreparedAudio:
    output_dir = mkdir(output_dir)
    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    warnings = validate_audio(input_audio, target_audio)

    latency_samples, confidence = estimate_latency(input_audio.samples, target_audio.samples)
    aligned_input, aligned_target = align_pair(
        input_audio.samples,
        target_audio.samples,
        latency_samples,
    )

    prepared_input_path = output_dir / "input.wav"
    prepared_target_path = output_dir / "target.wav"
    write_wav_mono(prepared_input_path, aligned_input, input_audio.sample_rate)
    write_wav_mono(prepared_target_path, aligned_target, input_audio.sample_rate)

    report = {
        "schema_version": 1,
        "input": audio_report(input_audio),
        "target": audio_report(target_audio),
        "prepared": {
            "input_path": str(prepared_input_path),
            "target_path": str(prepared_target_path),
            "sample_rate": input_audio.sample_rate,
            "samples": len(aligned_input),
            "duration_seconds": len(aligned_input) / input_audio.sample_rate,
        },
        "latency": {
            "estimated_samples": latency_samples,
            "confidence": confidence,
            "method": "cross_correlation",
        },
        "warnings": warnings,
        "status": "ready" if not warnings else "warning",
    }
    report_path = output_dir / "preparation-report.json"
    write_json(report_path, report)
    return PreparedAudio(prepared_input_path, prepared_target_path, report_path, report)


def validate_audio(input_audio: AudioBuffer, target_audio: AudioBuffer) -> list[str]:
    warnings: list[str] = []
    if input_audio.sample_rate != target_audio.sample_rate:
        warnings.append("Input and target sample rates differ; resampling is not implemented yet.")
    if input_audio.sample_rate != 48_000:
        warnings.append("48 kHz WAV is recommended for v1 exports.")
    duration_delta = abs(input_audio.duration_seconds - target_audio.duration_seconds)
    if duration_delta > 0.25:
        warnings.append(f"Input and target durations differ by {duration_delta:.2f} seconds.")
    if len(input_audio.samples) < input_audio.sample_rate:
        warnings.append("Capture is shorter than one second.")
    if active_ratio(input_audio.samples) < 0.05:
        warnings.append("Input appears to contain too much silence.")
    if active_ratio(target_audio.samples) < 0.05:
        warnings.append("Target appears to contain too much silence.")
    if max((abs(sample) for sample in target_audio.samples), default=0.0) >= 0.999:
        warnings.append("Target contains clipped samples.")
    return warnings


def estimate_latency(input_samples: list[float], target_samples: list[float]) -> tuple[int, float]:
    sample_count = min(len(input_samples), len(target_samples), 48_000)
    if sample_count <= 128:
        return 0, 0.0

    input_slice = input_samples[:sample_count]
    target_slice = target_samples[:sample_count]
    max_lag = min(4096, sample_count // 4)
    best_lag = 0
    best_score = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs = input_slice[: sample_count - lag]
            ys = target_slice[lag:sample_count]
        else:
            xs = input_slice[-lag:sample_count]
            ys = target_slice[: sample_count + lag]
        score = normalized_correlation(xs, ys)
        if score > best_score:
            best_lag = lag
            best_score = score

    confidence = max(0.0, min(1.0, best_score))
    return best_lag, confidence


def align_pair(
    input_samples: list[float],
    target_samples: list[float],
    latency_samples: int,
) -> tuple[list[float], list[float]]:
    if latency_samples >= 0:
        input_start = 0
        target_start = latency_samples
    else:
        input_start = -latency_samples
        target_start = 0

    length = min(len(input_samples) - input_start, len(target_samples) - target_start)
    if length <= 0:
        raise ValueError("Latency alignment removed all audio.")
    return (
        input_samples[input_start : input_start + length],
        target_samples[target_start : target_start + length],
    )


def normalized_correlation(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 0.0
    numerator = sum(x * y for x, y in zip(xs, ys))
    x_energy = sum(x * x for x in xs)
    y_energy = sum(y * y for y in ys)
    denominator = math.sqrt(x_energy * y_energy)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def active_ratio(samples: list[float], threshold: float = 0.001) -> float:
    if not samples:
        return 0.0
    active = sum(1 for sample in samples if abs(sample) > threshold)
    return active / len(samples)
