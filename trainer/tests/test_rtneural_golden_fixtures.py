from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.models.presets import PRESETS
from rttrainer.validation.parity import run_exported_json


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
VALIDATOR = ROOT / "native/rtneural-validator/build" / (
    "rtneural-validator.exe" if os.name == "nt" else "rtneural-validator"
)

try:
    import generate_golden_rtneural_fixtures as golden

    GOLDEN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - reported through skip reason
    golden = None  # type: ignore[assignment]
    GOLDEN_IMPORT_ERROR = exc


def tensorflow_available() -> bool:
    if GOLDEN_IMPORT_ERROR is not None:
        return False
    try:
        golden.require_tensorflow()  # type: ignore[union-attr]
    except Exception:
        return False
    return True


def native_validator_available() -> bool:
    return VALIDATOR.is_file()


@unittest.skipUnless(
    tensorflow_available(),
    "TensorFlow extra is required for golden RTNeural fixture parity tests.",
)
class RTNeuralGoldenFixtureTests(unittest.TestCase):
    def test_golden_fixture_exists_for_every_exported_preset(self) -> None:
        fixture_dir = ROOT / "fixtures/rtneural-json/golden"
        fixture_names = {
            path.name.removesuffix(".rtneural.json")
            for path in fixture_dir.glob("*.rtneural.json")
        }

        self.assertEqual(set(PRESETS), fixture_names)

    def test_golden_fixtures_are_current(self) -> None:
        fixture_dir = ROOT / "fixtures/rtneural-json/golden"
        generated = golden.build_all_fixtures()  # type: ignore[union-attr]

        for preset_id, payload in generated.items():
            with self.subTest(preset=preset_id):
                fixture_path = fixture_dir / f"{preset_id}.rtneural.json"
                self.assertEqual(
                    fixture_path.read_text(encoding="utf-8"),
                    golden.canonical_json(payload),  # type: ignore[union-attr]
                )

    def test_every_exported_preset_matches_keras_backend(self) -> None:
        fixtures = golden.build_fixture_models()  # type: ignore[union-attr]
        input_samples = [
            0.23,
            -0.11,
            0.05,
            0.31,
            -0.18,
            0.0,
            0.12,
            -0.07,
            0.19,
            -0.22,
            0.04,
            0.09,
        ]

        with tempfile.TemporaryDirectory(prefix="rttrainer-golden-parity-") as tmp:
            root = Path(tmp)
            for preset_id, fixture in fixtures.items():
                with self.subTest(preset=preset_id):
                    model_path = root / f"{preset_id}.rtneural.json"
                    model_path.write_text(
                        json.dumps(fixture.payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    expected = predict_keras(fixture.model, input_samples)
                    actual = run_exported_json(model_path, input_samples)
                    self.assertEqual(len(expected), len(actual))
                    self.assertLessEqual(max_abs_error(expected, actual), 1.0e-5)

    @unittest.skipUnless(
        native_validator_available(),
        "Native RTNeural validator is required for golden fixture validation.",
    )
    def test_every_golden_fixture_validates_in_native_rtneural(self) -> None:
        fixtures = golden.build_fixture_models()  # type: ignore[union-attr]
        fixture_dir = ROOT / "fixtures/rtneural-json/golden"
        input_samples = [
            0.19,
            -0.13,
            0.02,
            0.27,
            -0.24,
            0.08,
            0.16,
            -0.05,
            0.11,
            -0.18,
            0.06,
            0.14,
        ]

        with tempfile.TemporaryDirectory(prefix="rttrainer-golden-native-") as tmp:
            root = Path(tmp)
            input_path = root / "input.wav"
            write_wav_mono(input_path, input_samples, 48_000)
            quantized_input = read_wav_mono(input_path).samples

            for preset_id, fixture in fixtures.items():
                with self.subTest(preset=preset_id):
                    reference_path = root / f"{preset_id}-reference.wav"
                    report_path = root / f"{preset_id}-native-report.json"
                    write_wav_mono(reference_path, predict_keras(fixture.model, quantized_input), 48_000)
                    result = subprocess.run(
                        [
                            str(VALIDATOR),
                            "validate",
                            "--model",
                            str(fixture_dir / f"{preset_id}.rtneural.json"),
                            "--input",
                            str(input_path),
                            "--reference",
                            str(reference_path),
                            "--report",
                            str(report_path),
                            "--tolerance",
                            "0.001",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        self.fail(
                            f"Native validator failed for {preset_id}.\n"
                            f"stdout:\n{result.stdout}\n"
                            f"stderr:\n{result.stderr}"
                        )
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    self.assertEqual("pass", report["status"])


def predict_keras(model, samples: list[float]) -> list[float]:  # type: ignore[no-untyped-def]
    import numpy as np

    tensor = np.asarray(samples, dtype=np.float32).reshape(1, len(samples), 1)
    prediction = model(tensor, training=False).numpy()
    return [float(value) for value in prediction.reshape(-1)]


def max_abs_error(left: list[float], right: list[float]) -> float:
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


if __name__ == "__main__":
    unittest.main()
