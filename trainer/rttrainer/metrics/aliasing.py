from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rttrainer.utils import now, write_json
from rttrainer.validation.parity import run_exported_json

DEFAULT_ALIASING_FREQUENCIES = (1_250.0, 2_500.0, 5_000.0)
DEFAULT_ANALYSIS_SAMPLES = 4096
DEFAULT_WARMUP_SAMPLES = 2048
DEFAULT_INPUT_AMPLITUDE = 0.5


def analyze_rtneural_json_aliasing(
    *,
    model_json_path: Path,
    sample_rate: int,
    report_path: Path | None = None,
    frequencies: Sequence[float] = DEFAULT_ALIASING_FREQUENCIES,
    analysis_samples: int = DEFAULT_ANALYSIS_SAMPLES,
    warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    input_amplitude: float = DEFAULT_INPUT_AMPLITUDE,
) -> dict[str, Any]:
    """Render deterministic sine probes through RTNeural JSON and estimate ASR.

    The metric is intentionally lightweight and dependency-free so it can run in
    the packaged sidecar. ASR is reported as aliasing energy divided by harmonic
    energy for each sine probe, using FFT bins where the fundamental lands
    exactly on-bin.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if not is_power_of_two(analysis_samples):
        raise ValueError("analysis_samples must be a power of two.")
    if analysis_samples <= 0 or warmup_samples < 0:
        raise ValueError("analysis_samples must be positive and warmup_samples non-negative.")

    tests: list[dict[str, Any]] = []
    for desired_frequency in frequencies:
        fundamental_bin = nearest_signal_bin(
            desired_frequency,
            sample_rate=sample_rate,
            analysis_samples=analysis_samples,
        )
        input_samples = sine_probe(
            fundamental_bin=fundamental_bin,
            analysis_samples=analysis_samples,
            total_samples=warmup_samples + analysis_samples,
            amplitude=input_amplitude,
        )
        rendered = run_exported_json(model_json_path, input_samples)
        analysis_window = rendered[warmup_samples : warmup_samples + analysis_samples]
        test = analyze_signal_aliasing(
            analysis_window,
            sample_rate=sample_rate,
            fundamental_bin=fundamental_bin,
        )
        test["desired_frequency_hz"] = float(desired_frequency)
        tests.append(test)

    worst_asr = max((float(test["asr"]) for test in tests), default=0.0)
    average_asr = sum(float(test["asr"]) for test in tests) / max(1, len(tests))
    aliasing_status, verdict = classify_aliasing(worst_asr)
    report = {
        "schema_version": 1,
        "status": aliasing_status,
        "verdict": verdict,
        "metric": "aliasing_to_signal_ratio",
        "sample_rate": sample_rate,
        "analysis_samples": analysis_samples,
        "warmup_samples": warmup_samples,
        "input_amplitude": input_amplitude,
        "worst_asr": worst_asr,
        "average_asr": average_asr,
        "tests": tests,
        "notes": aliasing_notes(verdict),
        "model_json_path": str(model_json_path),
        "created_at": now(),
    }
    if report_path is not None:
        write_json(report_path, report)
    return report


def analyze_signal_aliasing(
    samples: Sequence[float],
    *,
    sample_rate: int,
    fundamental_bin: int,
) -> dict[str, Any]:
    sample_count = len(samples)
    if sample_count == 0:
        raise ValueError("Cannot analyze aliasing for an empty signal.")
    if not is_power_of_two(sample_count):
        raise ValueError("Aliasing analysis expects a power-of-two sample count.")
    if fundamental_bin <= 0 or fundamental_bin >= sample_count // 2:
        raise ValueError("fundamental_bin must be inside the positive FFT range.")

    mean = sum(float(sample) for sample in samples) / sample_count
    centered = [complex(float(sample) - mean, 0.0) for sample in samples]
    spectrum = fft(centered)
    nyquist = sample_count // 2
    power = [abs(spectrum[index]) ** 2 for index in range(nyquist + 1)]
    total_energy = sum(power[1:])
    harmonic_bins = list(range(fundamental_bin, nyquist + 1, fundamental_bin))
    harmonic_energy = sum(power[index] for index in harmonic_bins)
    aliasing_energy = max(0.0, total_energy - harmonic_energy)
    asr = aliasing_energy / max(harmonic_energy, 1.0e-18)
    alias_fraction = aliasing_energy / max(total_energy, 1.0e-18)

    return {
        "frequency_hz": sample_rate * fundamental_bin / sample_count,
        "fundamental_bin": fundamental_bin,
        "harmonic_bins": harmonic_bins,
        "harmonic_energy": harmonic_energy,
        "aliasing_energy": aliasing_energy,
        "total_energy": total_energy,
        "asr": asr,
        "alias_fraction": alias_fraction,
    }


def classify_aliasing(worst_asr: float) -> tuple[str, str]:
    if worst_asr < 0.02:
        return "pass", "low_aliasing"
    if worst_asr < 0.08:
        return "warning", "review_aliasing"
    return "warning", "high_aliasing"


def aliasing_notes(verdict: str) -> list[str]:
    if verdict == "low_aliasing":
        return [
            "ASR is low for the deterministic sine probes.",
            "Use this as a comparison metric, then confirm with high-note listening tests.",
        ]
    if verdict == "review_aliasing":
        return [
            "ASR is elevated on at least one sine probe.",
            "Compare against the same capture exported from another preset before treating this as a blocker.",
        ]
    return [
        "ASR is high on at least one sine probe.",
        "Listen for foldback grit on sustained high notes and consider a smoothed-tanh or smaller WaveNet candidate.",
    ]


def nearest_signal_bin(
    desired_frequency: float,
    *,
    sample_rate: int,
    analysis_samples: int,
) -> int:
    nyquist = analysis_samples // 2
    raw_bin = round(desired_frequency * analysis_samples / sample_rate)
    return min(max(1, raw_bin), nyquist - 1)


def sine_probe(
    *,
    fundamental_bin: int,
    analysis_samples: int,
    total_samples: int,
    amplitude: float,
) -> list[float]:
    return [
        amplitude * math.sin(2.0 * math.pi * fundamental_bin * index / analysis_samples)
        for index in range(total_samples)
    ]


def fft(values: Sequence[complex]) -> list[complex]:
    count = len(values)
    if not is_power_of_two(count):
        raise ValueError("FFT input length must be a power of two.")

    result = list(values)
    swap_index = 0
    for index in range(1, count):
        bit = count >> 1
        while swap_index & bit:
            swap_index ^= bit
            bit >>= 1
        swap_index ^= bit
        if index < swap_index:
            result[index], result[swap_index] = result[swap_index], result[index]

    length = 2
    while length <= count:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        half_length = length // 2
        for start in range(0, count, length):
            twiddle = 1.0 + 0.0j
            for offset in range(half_length):
                even = result[start + offset]
                odd = result[start + offset + half_length] * twiddle
                result[start + offset] = even + odd
                result[start + offset + half_length] = even - odd
                twiddle *= step
        length *= 2
    return result


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0
