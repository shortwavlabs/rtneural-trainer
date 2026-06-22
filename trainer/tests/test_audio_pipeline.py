from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.data.prepare import analyze_latency, prepare_audio
from rttrainer.metrics.audio_metrics import compute_metrics
from rttrainer.training.dataset import build_windowed_dataset


class AudioPipelineTests(unittest.TestCase):
    def test_wav_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.wav"
            samples = [0.0, 0.25, -0.25, 0.5, -0.5]
            write_wav_mono(path, samples, 48_000)
            audio = read_wav_mono(path)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(len(audio.samples), len(samples))
        self.assertAlmostEqual(audio.samples[1], samples[1], places=3)

    def test_reads_ieee_float_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "float32.wav"
            samples = [0.0, 0.25, -0.25, 0.5, -0.5]
            write_float32_wav(path, samples, 48_000)
            audio = read_wav_mono(path)

        self.assertEqual(audio.sample_rate, 48_000)
        self.assertEqual(audio.sample_width, 4)
        self.assertEqual(len(audio.samples), len(samples))
        self.assertAlmostEqual(audio.samples[3], samples[3], places=6)

    def test_prepare_estimates_and_aligns_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            dry = [0.0] * 4096
            for index in range(200, 900, 31):
                dry[index] = 0.8
            target = [0.0] * 23 + dry
            target = target[: len(dry)]
            write_wav_mono(input_path, dry, 48_000)
            write_wav_mono(target_path, target, 48_000)

            prepared = prepare_audio(input_path, target_path, root / "prepared")
            self.assertEqual(prepared.report["latency"]["estimated_samples"], 23)
            self.assertTrue(prepared.input_path.exists())
            self.assertTrue(prepared.target_path.exists())
            aligned_input = read_wav_mono(prepared.input_path)
            aligned_target = read_wav_mono(prepared.target_path)
            self.assertEqual(len(aligned_input.samples), len(aligned_target.samples))

    def test_latency_estimator_scans_active_regions_beyond_first_second(self) -> None:
        length = 120_000
        dry = [0.0] * length
        target = [0.0] * length

        for index in range(6_000):
            sample = 0.12 * deterministic_sample(index)
            dry[8_000 + 21 + index] = sample
            target[8_000 + index] = sample

        for index in range(14_000):
            sample = 0.55 * deterministic_sample(index)
            dry[70_000 + index] = sample
            target[70_000 + 12 + index] = sample

        analysis = analyze_latency(dry, target)

        self.assertGreaterEqual(analysis.estimated_samples, 10)
        self.assertLessEqual(analysis.estimated_samples, 13)
        self.assertGreater(analysis.confidence, 0.65)
        self.assertGreaterEqual(analysis.analysis_window_count, 2)
        self.assertTrue(
            any(
                10 <= int(candidate["samples"]) <= 13
                for candidate in analysis.candidates
            )
        )

    def test_latency_estimator_handles_nonlinear_targets(self) -> None:
        length = 20_000
        delay = 37
        dry = [0.0] * length
        target = [0.0] * length
        for index in range(1_000, 12_000):
            sample = 0.34 * deterministic_sample(index) + 0.11 * deterministic_sample(index * 3)
            dry[index] = sample
            target[index + delay] = soft_clip(sample * 2.7) * 0.8

        analysis = analyze_latency(dry, target)

        self.assertEqual(analysis.estimated_samples, delay)
        self.assertGreater(analysis.confidence, 0.8)
        self.assertEqual(analysis.candidates[0]["samples"], delay)

    def test_latency_estimator_reports_window_agreement_for_compressed_tones(self) -> None:
        length = 60_000
        delay = 17
        dry = [0.0] * length
        target = [0.0] * length
        for burst_start in (4_000, 16_000, 28_000, 43_000):
            for index in range(3_500):
                envelope = min(1.0, index / 120, (3_500 - index) / 600)
                sample = envelope * (
                    0.34 * deterministic_sample(burst_start + index)
                    + 0.12 * deterministic_sample((burst_start + index) * 5)
                )
                dry[burst_start + index] = sample
                target[burst_start + index + delay] = soft_clip(sample * 5.0) * 0.72

        analysis = analyze_latency(dry, target)
        best = analysis.candidates[0]

        self.assertEqual(analysis.estimated_samples, delay)
        self.assertEqual(best["samples"], delay)
        self.assertGreaterEqual(analysis.agreement, 0.9)
        self.assertGreaterEqual(float(best["agreement"]), 0.9)
        self.assertGreaterEqual(int(best["vote_count"]), 1)
        self.assertIn("preemphasis_score", best)
        self.assertIn("onset_score", best)

    def test_prepare_applies_manual_latency_adjustment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            dry = [0.0] * 4096
            for index in range(200, 900, 31):
                dry[index] = 0.8
            target = ([0.0] * 23 + dry)[: len(dry)]
            write_wav_mono(input_path, dry, 48_000)
            write_wav_mono(target_path, target, 48_000)

            prepared = prepare_audio(
                input_path,
                target_path,
                root / "prepared",
                manual_latency_adjustment_samples=5,
            )

        latency = prepared.report["latency"]
        self.assertEqual(latency["auto_estimated_samples"], 23)
        self.assertEqual(latency["manual_adjustment_samples"], 5)
        self.assertEqual(latency["effective_samples"], 28)
        self.assertEqual(latency["method"], "active_window_correlation")
        self.assertTrue(latency["candidates"])

    def test_prepare_resamples_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            samples = alternating_signal(4_096)
            write_wav_mono(input_path, samples, 44_100)
            write_wav_mono(target_path, samples, 44_100)

            prepared = prepare_audio(
                input_path,
                target_path,
                root / "prepared",
                target_sample_rate=48_000,
                resample=True,
            )
            aligned_input = read_wav_mono(prepared.input_path)

        self.assertEqual(prepared.report["prepared"]["sample_rate"], 48_000)
        self.assertEqual(aligned_input.sample_rate, 48_000)
        self.assertTrue(
            any(item["code"] == "resampled" for item in prepared.report["warning_details"])
        )

    def test_prepare_reports_multichannel_mixdown_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            write_stereo_wav(input_path, [(0.4, 0.2)] * 48_000)
            write_stereo_wav(target_path, [(0.3, 0.1)] * 48_000)

            prepared = prepare_audio(input_path, target_path, root / "prepared")

        self.assertEqual(prepared.report["input"]["channels"], 2)
        self.assertEqual(prepared.report["prepared"]["channel_policy"], "mixdown")
        self.assertEqual(prepared.report["status"], "ready")
        self.assertTrue(
            any(item["code"] == "mixed_to_mono" for item in prepared.report["warning_details"])
        )

    def test_prepare_can_reject_multichannel_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            write_stereo_wav(input_path, [(0.4, 0.2)] * 512)
            write_wav_mono(target_path, [0.2] * 512, 48_000)

            with self.assertRaises(ValueError):
                prepare_audio(
                    input_path,
                    target_path,
                    root / "prepared",
                    channel_policy="reject",
                )

    def test_metrics_zero_for_identical_signals(self) -> None:
        metrics = compute_metrics([0.0, 0.2, -0.2], [0.0, 0.2, -0.2])
        self.assertEqual(metrics["esr"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)

    def test_long_dataset_samples_windows_across_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            samples = alternating_signal(12_000)
            write_wav_mono(input_path, samples, 48_000)
            write_wav_mono(target_path, samples, 48_000)

            dataset = build_windowed_dataset(
                input_path,
                target_path,
                sequence_length=128,
                max_windows=8,
                seed=7,
                backend="list",
            )

        self.assertEqual(dataset.summary["selected_windows"], 8)
        available_windows = int(dataset.summary["available_windows"])
        self.assertGreater(available_windows, 8)
        self.assertEqual(
            dataset.summary["selection"],
            "energy_stratified_sampled_across_capture",
        )
        self.assertGreaterEqual(int(dataset.summary["energy_selected_windows"]), 1)

    def test_dataset_preview_defaults_to_three_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            samples = alternating_signal(240_000)
            write_wav_mono(input_path, samples, 48_000)
            write_wav_mono(target_path, samples, 48_000)

            dataset = build_windowed_dataset(
                input_path,
                target_path,
                sequence_length=512,
                max_windows=16,
                seed=9,
                backend="list",
            )

        self.assertEqual(len(dataset.test_target), 144_000)
        self.assertEqual(len(dataset.stream_val_target), 144_000)
        self.assertEqual(len(dataset.context_train_target), 2_048)
        self.assertEqual(int(dataset.summary["test_samples"]), 144_000)
        self.assertEqual(int(dataset.summary["stream_validation_samples"]), 144_000)
        self.assertEqual(int(dataset.summary["context_training_samples"]), 2_048)
        self.assertAlmostEqual(float(dataset.summary["preview_seconds"]), 3.0)

    def test_dataset_preview_prefers_active_target_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            target_path = root / "target.wav"
            dry = [0.0] * 240_000
            target = [0.0] * 240_000
            for index in range(180_000, 184_000):
                dry[index] = 0.2
                target[index] = 0.4 if index % 2 == 0 else -0.4
            write_wav_mono(input_path, dry, 48_000)
            write_wav_mono(target_path, target, 48_000)

            dataset = build_windowed_dataset(
                input_path,
                target_path,
                sequence_length=512,
                max_windows=16,
                seed=9,
                backend="list",
            )

        self.assertGreater(max(abs(sample) for sample in dataset.test_target), 0.3)
        self.assertEqual(len(dataset.test_target), 144_000)
        self.assertGreaterEqual(int(dataset.summary["test_start_sample"]), 40_000)


def alternating_signal(length: int) -> list[float]:
    return [0.3 if index % 2 == 0 else -0.3 for index in range(length)]


def deterministic_sample(index: int) -> float:
    return (
        0.62 * math.sin(index * 0.047)
        + 0.29 * math.sin(index * 0.173)
        + 0.09 * math.sin(index * 0.011)
    )


def soft_clip(value: float) -> float:
    return math.tanh(value)


def write_stereo_wav(path: Path, frames: list[tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        data = bytearray()
        for left, right in frames:
            for sample in (left, right):
                integer = int(round(max(-1.0, min(1.0, sample)) * 32767.0))
                data.extend(integer.to_bytes(2, "little", signed=True))
        wav.writeframes(bytes(data))


def write_float32_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(struct.pack("<f", sample) for sample in samples)
    byte_rate = sample_rate * 4
    fmt_chunk = struct.pack(
        "<HHIIHH",
        3,
        1,
        sample_rate,
        byte_rate,
        4,
        32,
    )
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write((4 + (8 + len(fmt_chunk)) + (8 + len(payload))).to_bytes(4, "little"))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(len(fmt_chunk).to_bytes(4, "little"))
        handle.write(fmt_chunk)
        handle.write(b"data")
        handle.write(len(payload).to_bytes(4, "little"))
        handle.write(payload)


if __name__ == "__main__":
    unittest.main()
