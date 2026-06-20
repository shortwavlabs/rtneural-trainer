from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.data.prepare import prepare_audio
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
        self.assertGreater(dataset.summary["available_windows"], 8)
        self.assertEqual(dataset.summary["selection"], "sampled_across_capture")


def alternating_signal(length: int) -> list[float]:
    return [0.3 if index % 2 == 0 else -0.3 for index in range(length)]


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


if __name__ == "__main__":
    unittest.main()
