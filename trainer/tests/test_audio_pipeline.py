from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.data.prepare import prepare_audio
from rttrainer.metrics.audio_metrics import compute_metrics


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

    def test_metrics_zero_for_identical_signals(self) -> None:
        metrics = compute_metrics([0.0, 0.2, -0.2], [0.0, 0.2, -0.2])
        self.assertEqual(metrics["esr"], 0.0)
        self.assertEqual(metrics["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
