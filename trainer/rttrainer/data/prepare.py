from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from rttrainer.data.audio_io import (
    AudioBuffer,
    audio_report,
    normalize_channel_policy,
    read_wav_mono,
    write_wav_mono,
)
from rttrainer.utils import mkdir, write_json


@dataclass(frozen=True)
class PreparedAudio:
    input_path: Path
    target_path: Path
    report_path: Path
    report: dict


@dataclass(frozen=True)
class LatencyScore:
    lag: int
    score: float
    feature_score: float
    signed_score: float
    window_count: int


@dataclass(frozen=True)
class LatencyAnalysis:
    estimated_samples: int
    confidence: float
    method: str
    search_radius_samples: int
    window_length_samples: int
    analysis_window_count: int
    score_margin: float
    candidates: list[dict[str, int | float]]


LATENCY_MAX_LAG_SAMPLES = 4096
LATENCY_MAX_WINDOWS = 12
LATENCY_FINE_RADIUS_SAMPLES = 48
LATENCY_DISTINCT_CANDIDATE_DISTANCE = 8


def prepare_audio(
    input_path: Path,
    target_path: Path,
    output_dir: Path,
    *,
    target_sample_rate: int | None = None,
    resample: bool = False,
    channel_policy: str = "mixdown",
    manual_latency_adjustment_samples: int = 0,
) -> PreparedAudio:
    output_dir = mkdir(output_dir)
    normalized_channel_policy = normalize_channel_policy(channel_policy)
    preferred_sample_rate = target_sample_rate or 48_000
    if preferred_sample_rate <= 0:
        raise ValueError("target_sample_rate must be a positive integer.")

    source_input_audio = read_wav_mono(input_path, normalized_channel_policy)
    source_target_audio = read_wav_mono(target_path, normalized_channel_policy)
    warning_details = channel_policy_details(
        source_input_audio,
        source_target_audio,
        normalized_channel_policy,
    )

    input_audio = source_input_audio
    target_audio = source_target_audio
    if resample:
        input_audio, input_resample_notice = resample_if_needed(
            input_audio,
            preferred_sample_rate,
            "Dry input",
        )
        target_audio, target_resample_notice = resample_if_needed(
            target_audio,
            preferred_sample_rate,
            "Processed target",
        )
        warning_details.extend(input_resample_notice)
        warning_details.extend(target_resample_notice)

    warning_details.extend(
        validate_audio(
            input_audio,
            target_audio,
            preferred_sample_rate=preferred_sample_rate,
            resample_enabled=resample,
        )
    )
    latency_analysis = analyze_latency(input_audio.samples, target_audio.samples)
    latency_samples = latency_analysis.estimated_samples
    confidence = latency_analysis.confidence
    effective_latency_samples = latency_samples + int(manual_latency_adjustment_samples)
    warning_details.extend(latency_analysis_details(latency_analysis))
    warning_details.extend(capture_profile_details(input_audio, target_audio))
    gain = gain_analysis(input_audio, target_audio)
    warning_details.extend(gain["warnings"])
    warnings = [
        str(item["message"]) for item in warning_details if item.get("severity") == "warning"
    ]

    aligned_input, aligned_target = align_pair(
        input_audio.samples,
        target_audio.samples,
        effective_latency_samples,
    )

    prepared_input_path = output_dir / "input.wav"
    prepared_target_path = output_dir / "target.wav"
    write_wav_mono(prepared_input_path, aligned_input, input_audio.sample_rate)
    write_wav_mono(prepared_target_path, aligned_target, input_audio.sample_rate)

    report = {
        "schema_version": 1,
        "input": audio_report(source_input_audio),
        "target": audio_report(source_target_audio),
        "prepared": {
            "input_path": str(prepared_input_path),
            "target_path": str(prepared_target_path),
            "sample_rate": input_audio.sample_rate,
            "samples": len(aligned_input),
            "duration_seconds": len(aligned_input) / input_audio.sample_rate,
            "channel_policy": normalized_channel_policy,
            "resampled": resample,
        },
        "capture_profile": capture_profile(input_audio, target_audio),
        "gain": {
            key: value for key, value in gain.items() if key != "warnings"
        },
        "options": {
            "target_sample_rate": preferred_sample_rate,
            "resample": resample,
            "channel_policy": normalized_channel_policy,
            "manual_latency_adjustment_samples": int(manual_latency_adjustment_samples),
        },
        "latency": {
            "estimated_samples": effective_latency_samples,
            "auto_estimated_samples": latency_samples,
            "manual_adjustment_samples": int(manual_latency_adjustment_samples),
            "effective_samples": effective_latency_samples,
            "confidence": confidence,
            "method": latency_analysis.method,
            "search_radius_samples": latency_analysis.search_radius_samples,
            "window_length_samples": latency_analysis.window_length_samples,
            "analysis_window_count": latency_analysis.analysis_window_count,
            "score_margin": latency_analysis.score_margin,
            "candidates": latency_analysis.candidates,
        },
        "warnings": warnings,
        "warning_details": warning_details,
        "status": "ready" if not warnings else "warning",
    }
    report_path = output_dir / "preparation-report.json"
    write_json(report_path, report)
    return PreparedAudio(prepared_input_path, prepared_target_path, report_path, report)


def capture_profile(input_audio: AudioBuffer, target_audio: AudioBuffer) -> dict[str, float | int | str]:
    duration = min(input_audio.duration_seconds, target_audio.duration_seconds)
    recommended_max_windows = 512
    if duration >= 120:
        recommended_max_windows = 2048
    elif duration >= 45:
        recommended_max_windows = 1024
    return {
        "duration_seconds": duration,
        "recommended_max_windows": recommended_max_windows,
        "handling": "sampled_windows" if duration >= 45 else "standard_windows",
    }


def capture_profile_details(
    input_audio: AudioBuffer,
    target_audio: AudioBuffer,
) -> list[dict[str, str]]:
    duration = min(input_audio.duration_seconds, target_audio.duration_seconds)
    if duration < 45:
        return []
    return [
        warning_detail(
            "long_capture",
            "info",
            "Long capture detected.",
            f"The prepared pair is {duration:.1f} seconds long.",
            "Training will sample windows across the file; raise the window budget for more coverage.",
        )
    ]


def gain_analysis(input_audio: AudioBuffer, target_audio: AudioBuffer) -> dict:
    input_report = audio_report(input_audio)
    target_report = audio_report(target_audio)
    input_peak = float(input_report["peak_dbfs"])
    target_peak = float(target_report["peak_dbfs"])
    input_rms = float(input_report["rms_dbfs"])
    target_rms = float(target_report["rms_dbfs"])
    rms_delta = target_rms - input_rms
    warnings: list[dict[str, str]] = []

    if input_peak < -24.0 or target_peak < -24.0:
        warnings.append(
            warning_detail(
                "capture_level_low",
                "warning",
                "Capture level is very low.",
                f"Dry peak is {input_peak:.1f} dBFS; target peak is {target_peak:.1f} dBFS.",
                "Recapture closer to -12 to -6 dBFS peak when possible.",
            )
        )
    if max(input_peak, target_peak) > -1.0:
        warnings.append(
            warning_detail(
                "capture_headroom_low",
                "warning",
                "Capture has less than 1 dB of peak headroom.",
                f"Dry peak is {input_peak:.1f} dBFS; target peak is {target_peak:.1f} dBFS.",
                "Leave a little headroom so clipped transients do not dominate training.",
            )
        )
    if abs(rms_delta) > 12.0:
        louder = "processed target" if rms_delta > 0 else "dry input"
        warnings.append(
            warning_detail(
                "rms_mismatch",
                "warning",
                "Dry and processed RMS levels are far apart.",
                f"The {louder} is about {abs(rms_delta):.1f} dB louder on average.",
                "Check capture gain staging; large level offsets can look like model error.",
            )
        )

    if warnings:
        verdict = "fix_gain_before_training"
        guidance = "Recapture or trim/gain-stage before spending a long training run."
    elif -18.0 <= input_rms <= -6.0 and -18.0 <= target_rms <= -6.0:
        verdict = "healthy"
        guidance = "Levels are in a good range for training."
    else:
        verdict = "usable"
        guidance = "Levels are usable; inspect the preview and residual after training."

    return {
        "input_peak_dbfs": input_peak,
        "target_peak_dbfs": target_peak,
        "input_rms_dbfs": input_rms,
        "target_rms_dbfs": target_rms,
        "rms_delta_db": rms_delta,
        "headroom_db": -max(input_peak, target_peak),
        "verdict": verdict,
        "guidance": guidance,
        "warnings": warnings,
    }


def validate_audio(
    input_audio: AudioBuffer,
    target_audio: AudioBuffer,
    *,
    preferred_sample_rate: int,
    resample_enabled: bool,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if input_audio.sample_rate != target_audio.sample_rate:
        warnings.append(
            warning_detail(
                "sample_rate_mismatch",
                "warning",
                "Dry input and processed target have different sample rates.",
                f"Dry input is {input_audio.sample_rate} Hz; processed target is {target_audio.sample_rate} Hz.",
                "Enable resampling or recapture both files at the same sample rate.",
            )
        )
    if input_audio.sample_rate != preferred_sample_rate:
        warnings.append(
            warning_detail(
                "sample_rate_not_target",
                "warning",
                f"Prepared audio is {input_audio.sample_rate} Hz, not {preferred_sample_rate} Hz.",
                "RTNeural Trainer v1 expects a consistent prepared sample rate for training and export.",
                "Enable resampling or choose source files already captured at the target rate.",
            )
        )
    elif not resample_enabled and input_audio.sample_rate != 48_000:
        warnings.append(
            warning_detail(
                "sample_rate_not_48k",
                "warning",
                "Prepared audio is not 48 kHz.",
                f"Current prepared rate is {input_audio.sample_rate} Hz.",
                "48 kHz is the recommended v1 export rate unless your target runtime is fixed to another rate.",
            )
        )
    duration_delta = abs(input_audio.duration_seconds - target_audio.duration_seconds)
    if duration_delta > 0.25:
        warnings.append(
            warning_detail(
                "duration_mismatch",
                "warning",
                f"Dry input and processed target durations differ by {duration_delta:.2f} seconds.",
                "Large duration differences can make latency alignment unreliable.",
                "Trim both captures to the same program material before training.",
            )
        )
    if len(input_audio.samples) < input_audio.sample_rate:
        warnings.append(
            warning_detail(
                "capture_too_short",
                "warning",
                "Capture is shorter than one second.",
                "Very short captures rarely cover enough dynamics for a useful model.",
                "Use at least a few seconds of varied material.",
            )
        )
    if active_ratio(input_audio.samples) < 0.05:
        warnings.append(
            warning_detail(
                "input_too_silent",
                "warning",
                "Dry input appears to contain too much silence.",
                "Most samples are below the activity threshold.",
                "Trim silence or recapture with a stronger dry signal.",
            )
        )
    if active_ratio(target_audio.samples) < 0.05:
        warnings.append(
            warning_detail(
                "target_too_silent",
                "warning",
                "Processed target appears to contain too much silence.",
                "Most samples are below the activity threshold.",
                "Trim silence or recapture the processed signal.",
            )
        )
    if max((abs(sample) for sample in input_audio.samples), default=0.0) >= 0.999:
        warnings.append(
            warning_detail(
                "input_clipped",
                "warning",
                "Dry input contains clipped samples.",
                "Clipping in the dry reference can teach the model the wrong transfer curve.",
                "Lower the capture gain and record again.",
            )
        )
    if max((abs(sample) for sample in target_audio.samples), default=0.0) >= 0.999:
        warnings.append(
            warning_detail(
                "target_clipped",
                "warning",
                "Processed target contains clipped samples.",
                "Clipped target audio can dominate the loss and hide the actual device behavior.",
                "Lower the output gain or use a capture with more headroom.",
            )
        )
    return warnings


def channel_policy_details(
    input_audio: AudioBuffer,
    target_audio: AudioBuffer,
    channel_policy: str,
) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for label, audio in (("Dry input", input_audio), ("Processed target", target_audio)):
        if audio.channels <= 1:
            continue
        if channel_policy == "first":
            details.append(
                warning_detail(
                    "first_channel_selected",
                    "info",
                    f"{label} has {audio.channels} channels; using channel 1 only.",
                    "Prepared audio is mono for the current RTNeural presets.",
                    "Use this only when channel 1 is the intended capture path.",
                )
            )
        else:
            details.append(
                warning_detail(
                    "mixed_to_mono",
                    "info",
                    f"{label} has {audio.channels} channels; mixed to mono.",
                    "Prepared audio averages all source channels before alignment.",
                    "For best repeatability, capture mono when possible.",
                )
            )
    return details


def resample_if_needed(
    audio: AudioBuffer,
    target_sample_rate: int,
    label: str,
) -> tuple[AudioBuffer, list[dict[str, str]]]:
    if audio.sample_rate == target_sample_rate:
        return audio, []
    resampled = resample_audio(audio, target_sample_rate)
    return resampled, [
        warning_detail(
            "resampled",
            "info",
            f"{label} was resampled to {target_sample_rate} Hz.",
            f"Original sample rate was {audio.sample_rate} Hz.",
            "Use high-quality offline resampling before import if this capture is final-critical.",
        )
    ]


def resample_audio(audio: AudioBuffer, target_sample_rate: int) -> AudioBuffer:
    if audio.sample_rate == target_sample_rate or not audio.samples:
        return AudioBuffer(
            samples=list(audio.samples),
            sample_rate=target_sample_rate,
            channels=audio.channels,
            sample_width=audio.sample_width,
            path=audio.path,
        )
    if len(audio.samples) == 1:
        return AudioBuffer(
            samples=[audio.samples[0]],
            sample_rate=target_sample_rate,
            channels=audio.channels,
            sample_width=audio.sample_width,
            path=audio.path,
        )

    output_count = max(1, round(len(audio.samples) * target_sample_rate / audio.sample_rate))
    ratio = audio.sample_rate / target_sample_rate
    resampled: list[float] = []
    for index in range(output_count):
        position = index * ratio
        left = min(int(math.floor(position)), len(audio.samples) - 1)
        right = min(left + 1, len(audio.samples) - 1)
        fraction = position - left
        resampled.append(audio.samples[left] * (1.0 - fraction) + audio.samples[right] * fraction)
    return AudioBuffer(
        samples=resampled,
        sample_rate=target_sample_rate,
        channels=audio.channels,
        sample_width=audio.sample_width,
        path=audio.path,
    )


def warning_detail(
    code: str,
    severity: str,
    message: str,
    detail: str,
    action: str,
) -> dict[str, str]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "detail": detail,
        "action": action,
    }


def estimate_latency(input_samples: list[float], target_samples: list[float]) -> tuple[int, float]:
    analysis = analyze_latency(input_samples, target_samples)
    return analysis.estimated_samples, analysis.confidence


def analyze_latency(input_samples: list[float], target_samples: list[float]) -> LatencyAnalysis:
    length = min(len(input_samples), len(target_samples))
    if length <= 128:
        return LatencyAnalysis(
            estimated_samples=0,
            confidence=0.0,
            method="active_window_correlation",
            search_radius_samples=0,
            window_length_samples=length,
            analysis_window_count=0,
            score_margin=0.0,
            candidates=[],
        )

    window_length = choose_latency_window_length(length)
    search_radius = min(LATENCY_MAX_LAG_SAMPLES, max(1, length // 4), max(1, window_length // 4))
    starts = select_latency_window_starts(
        target_samples[:length],
        window_length=window_length,
        max_windows=LATENCY_MAX_WINDOWS,
    )
    fine_window_length = min(window_length, 32_768)

    if search_radius <= 256:
        fine_lags = list(range(-search_radius, search_radius + 1))
    else:
        coarse_scores = coarse_latency_scores(
            input_samples[:length],
            target_samples[:length],
            starts,
            window_length,
            search_radius,
        )
        coarse_candidates = ranked_latency_scores(
            coarse_scores,
            max_count=8,
            min_distance=max(1, LATENCY_FINE_RADIUS_SAMPLES // 2),
        )
        fine_lags = fine_lag_candidates(
            coarse_candidates,
            search_radius=search_radius,
            radius=LATENCY_FINE_RADIUS_SAMPLES,
        )

    fine_scores = score_latency_lags(
        input_samples[:length],
        target_samples[:length],
        starts,
        fine_window_length,
        fine_lags,
    )
    ranked = ranked_latency_scores(
        fine_scores,
        max_count=8,
        min_distance=LATENCY_DISTINCT_CANDIDATE_DISTANCE,
    )
    if not ranked:
        return LatencyAnalysis(
            estimated_samples=0,
            confidence=0.0,
            method="active_window_correlation",
            search_radius_samples=search_radius,
            window_length_samples=window_length,
            analysis_window_count=len(starts),
            score_margin=0.0,
            candidates=[],
        )

    best = choose_best_latency_score(fine_scores, ranked[0])
    runner_up = next((score for score in ranked if score.lag != best.lag), None)
    score_margin = best.score - runner_up.score if runner_up else best.score
    confidence = latency_confidence(best.score, score_margin)
    candidates = [latency_score_payload(score) for score in ranked[:5]]
    if best.lag not in {score.lag for score in ranked[:5]}:
        candidates.insert(0, latency_score_payload(best))

    return LatencyAnalysis(
        estimated_samples=best.lag,
        confidence=confidence,
        method="active_window_correlation",
        search_radius_samples=search_radius,
        window_length_samples=window_length,
        analysis_window_count=len(starts),
        score_margin=score_margin,
        candidates=candidates[:5],
    )


def latency_analysis_details(analysis: LatencyAnalysis) -> list[dict[str, str]]:
    if not analysis.candidates:
        return [
            warning_detail(
                "latency_estimate_unavailable",
                "info",
                "Latency estimate could not be measured confidently.",
                "The capture did not contain enough active material for alignment analysis.",
                "Inspect the alignment view and use a manual nudge before long training runs.",
            )
        ]

    if analysis.confidence >= 0.65 and analysis.score_margin >= 0.02:
        return []

    candidate_text = ", ".join(
        f"{int(candidate['samples'])} samples"
        for candidate in analysis.candidates[:3]
    )
    return [
        warning_detail(
            "latency_estimate_review",
            "info",
            "Latency estimate should be reviewed.",
            (
                f"Best candidate is {analysis.estimated_samples} samples with "
                f"{analysis.confidence:.2f} confidence; top candidates include {candidate_text}."
            ),
            "Inspect the alignment view or try a manual nudge before long training runs.",
        )
    ]


def choose_latency_window_length(length: int) -> int:
    if length >= 65_536 * 4:
        return 65_536
    if length >= 16_384 * 4:
        return 16_384
    return max(512, min(length, 8_192))


def select_latency_window_starts(
    samples: list[float],
    *,
    window_length: int,
    max_windows: int,
) -> list[int]:
    length = len(samples)
    if length <= window_length:
        return [0]

    step = max(1, window_length // 2)
    max_start = length - window_length
    energies: list[tuple[float, int]] = []
    for start in range(0, max_start + 1, step):
        energies.append((energy_between(samples, start, start + window_length), start))
    if not energies or energies[-1][1] != max_start:
        energies.append((energy_between(samples, max_start, length), max_start))

    selected: list[int] = []
    min_spacing = max(1, window_length // 2)
    for energy, start in sorted(energies, reverse=True):
        if energy <= 1e-12:
            continue
        if all(abs(start - existing) >= min_spacing for existing in selected):
            selected.append(start)
        if len(selected) >= max_windows:
            break
    if not selected:
        selected.append(0)
    return sorted(selected)


def coarse_latency_scores(
    input_samples: list[float],
    target_samples: list[float],
    starts: list[int],
    window_length: int,
    search_radius: int,
) -> list[LatencyScore]:
    block_size = 16 if search_radius > 1024 else 8
    input_feature = downsample_abs(input_samples, block_size)
    target_feature = downsample_abs(target_samples, block_size)
    coarse_starts = [min(start // block_size, max(0, len(target_feature) - 1)) for start in starts]
    coarse_window_length = max(32, window_length // block_size)
    coarse_radius = max(1, search_radius // block_size)
    coarse_lags = list(range(-coarse_radius, coarse_radius + 1))
    coarse_scores = score_latency_lags(
        input_feature,
        target_feature,
        coarse_starts,
        coarse_window_length,
        coarse_lags,
        feature_weight=1.0,
    )
    return [
        LatencyScore(
            lag=score.lag * block_size,
            score=score.score,
            feature_score=score.feature_score,
            signed_score=score.signed_score,
            window_count=score.window_count,
        )
        for score in coarse_scores
    ]


def downsample_abs(samples: list[float], block_size: int) -> list[float]:
    output: list[float] = []
    for start in range(0, len(samples), block_size):
        end = min(len(samples), start + block_size)
        total = 0.0
        for index in range(start, end):
            total += abs(samples[index])
        output.append(total / max(1, end - start))
    return output


def fine_lag_candidates(
    coarse_candidates: list[LatencyScore],
    *,
    search_radius: int,
    radius: int,
) -> list[int]:
    lags: set[int] = {0}
    for candidate in coarse_candidates:
        start = max(-search_radius, candidate.lag - radius)
        end = min(search_radius, candidate.lag + radius)
        lags.update(range(start, end + 1))
    if not coarse_candidates:
        lags.update(range(-min(search_radius, radius), min(search_radius, radius) + 1))
    return sorted(lags)


def score_latency_lags(
    input_samples: list[float],
    target_samples: list[float],
    starts: list[int],
    window_length: int,
    lags: list[int],
    *,
    feature_weight: float = 0.75,
) -> list[LatencyScore]:
    return [
        score_latency_lag(
            input_samples,
            target_samples,
            starts,
            window_length,
            lag,
            feature_weight=feature_weight,
        )
        for lag in lags
    ]


def score_latency_lag(
    input_samples: list[float],
    target_samples: list[float],
    starts: list[int],
    window_length: int,
    lag: int,
    *,
    feature_weight: float,
) -> LatencyScore:
    scores: list[float] = []
    feature_scores: list[float] = []
    signed_scores: list[float] = []
    signed_weight = max(0.0, min(1.0, 1.0 - feature_weight))
    feature_weight = max(0.0, min(1.0, feature_weight))

    for start in starts:
        correlations = latency_window_correlations(
            input_samples,
            target_samples,
            start=start,
            window_length=window_length,
            lag=lag,
        )
        if correlations is None:
            continue
        feature_score, signed_score = correlations
        combined = feature_weight * feature_score + signed_weight * max(0.0, signed_score)
        scores.append(combined)
        feature_scores.append(feature_score)
        signed_scores.append(signed_score)

    return LatencyScore(
        lag=lag,
        score=trimmed_mean(scores),
        feature_score=trimmed_mean(feature_scores),
        signed_score=trimmed_mean(signed_scores),
        window_count=len(scores),
    )


def latency_window_correlations(
    input_samples: list[float],
    target_samples: list[float],
    *,
    start: int,
    window_length: int,
    lag: int,
) -> tuple[float, float] | None:
    if lag >= 0:
        input_start = start
        target_start = start + lag
    else:
        input_start = start - lag
        target_start = start

    length = min(
        window_length,
        len(input_samples) - input_start,
        len(target_samples) - target_start,
    )
    if input_start < 0 or target_start < 0 or length <= 128:
        return None

    feature_num = 0.0
    feature_input_energy = 0.0
    feature_target_energy = 0.0
    signed_num = 0.0
    signed_input_energy = 0.0
    signed_target_energy = 0.0
    for offset in range(length):
        input_value = input_samples[input_start + offset]
        target_value = target_samples[target_start + offset]
        input_feature = abs(input_value)
        target_feature = abs(target_value)
        feature_num += input_feature * target_feature
        feature_input_energy += input_feature * input_feature
        feature_target_energy += target_feature * target_feature
        signed_num += input_value * target_value
        signed_input_energy += input_value * input_value
        signed_target_energy += target_value * target_value

    feature_score = safe_correlation(
        feature_num,
        feature_input_energy,
        feature_target_energy,
    )
    signed_score = safe_correlation(
        signed_num,
        signed_input_energy,
        signed_target_energy,
    )
    return feature_score, signed_score


def safe_correlation(numerator: float, x_energy: float, y_energy: float) -> float:
    denominator = math.sqrt(x_energy * y_energy)
    if denominator <= 1e-12:
        return 0.0
    return numerator / denominator


def trimmed_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) < 5:
        return sum(ordered) / len(ordered)
    trim = max(1, len(ordered) // 5)
    trimmed = ordered[trim:-trim]
    if not trimmed:
        trimmed = ordered
    return sum(trimmed) / len(trimmed)


def ranked_latency_scores(
    scores: list[LatencyScore],
    *,
    max_count: int,
    min_distance: int,
) -> list[LatencyScore]:
    ranked: list[LatencyScore] = []
    for score in sorted(scores, key=lambda item: item.score, reverse=True):
        if score.window_count <= 0:
            continue
        if all(abs(score.lag - existing.lag) >= min_distance for existing in ranked):
            ranked.append(score)
        if len(ranked) >= max_count:
            break
    return ranked


def choose_best_latency_score(
    scores: list[LatencyScore],
    best_score: LatencyScore,
) -> LatencyScore:
    zero_score = next((score for score in scores if score.lag == 0), None)
    if zero_score is not None and zero_score.score >= best_score.score - 0.002:
        return zero_score
    return best_score


def latency_confidence(score: float, margin: float) -> float:
    confidence = max(0.0, min(1.0, score))
    if margin < 0.005:
        return min(confidence, 0.35)
    if margin < 0.02:
        return min(confidence, 0.6)
    return confidence


def latency_score_payload(score: LatencyScore) -> dict[str, int | float]:
    return {
        "samples": score.lag,
        "score": score.score,
        "feature_score": score.feature_score,
        "signed_score": score.signed_score,
        "window_count": score.window_count,
    }


def energy_between(samples: list[float], start: int, end: int) -> float:
    start = max(0, start)
    end = min(len(samples), end)
    total = 0.0
    for index in range(start, end):
        total += samples[index] * samples[index]
    return total


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
