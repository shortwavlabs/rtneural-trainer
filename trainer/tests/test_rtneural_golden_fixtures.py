from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Protocol, cast

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono
from rttrainer.export_rtneural.json_exporter import default_parity_tolerance
from rttrainer.models.presets import PRESETS, build_keras_model
from rttrainer.training.runner import load_keras_checkpoint, save_keras_model_checkpoint
from rttrainer.validation.parity import run_exported_json


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "native/rtneural-validator/build" / (
    "rtneural-validator.exe" if os.name == "nt" else "rtneural-validator"
)


class GoldenFixture(Protocol):
    payload: dict[str, Any]
    model: Any


class GoldenFixtureModule(Protocol):
    def require_tensorflow(self) -> Any: ...

    def build_all_fixtures(self) -> dict[str, dict[str, Any]]: ...

    def build_fixture_models(self) -> dict[str, GoldenFixture]: ...

    def canonical_json(self, payload: dict[str, Any]) -> str: ...


def load_golden_fixture_module() -> GoldenFixtureModule:
    module_path = ROOT / "scripts" / "generate_golden_rtneural_fixtures.py"
    spec = importlib.util.spec_from_file_location(
        "generate_golden_rtneural_fixtures",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load golden fixture script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(GoldenFixtureModule, module)


try:
    golden: GoldenFixtureModule | None = load_golden_fixture_module()
    GOLDEN_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - reported through skip reason
    golden = None
    GOLDEN_IMPORT_ERROR = exc


def require_golden_fixture_module() -> GoldenFixtureModule:
    if golden is None:
        raise unittest.SkipTest(
            f"Golden RTNeural fixture script could not be loaded: {GOLDEN_IMPORT_ERROR}"
        )
    return golden


def tensorflow_available() -> bool:
    if GOLDEN_IMPORT_ERROR is not None:
        return False
    golden_module = require_golden_fixture_module()
    try:
        golden_module.require_tensorflow()
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
        golden_module = require_golden_fixture_module()
        fixture_dir = ROOT / "fixtures/rtneural-json/golden"
        generated = golden_module.build_all_fixtures()

        for preset_id, payload in generated.items():
            with self.subTest(preset=preset_id):
                fixture_path = fixture_dir / f"{preset_id}.rtneural.json"
                self.assertEqual(
                    fixture_path.read_text(encoding="utf-8"),
                    golden_module.canonical_json(payload),
                )

    def test_every_exported_preset_matches_keras_backend(self) -> None:
        golden_module = require_golden_fixture_module()
        fixtures = golden_module.build_fixture_models()
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
                    self.assertLessEqual(
                        max_abs_error(expected, actual),
                        default_parity_tolerance(preset_id),
                    )

    def test_scaled_tanh_checkpoint_round_trips_through_keras_save(self) -> None:
        import numpy as np

        golden_module = require_golden_fixture_module()
        tf = golden_module.require_tensorflow()
        model = build_keras_model(PRESETS["wavenet_tcn_balanced_tanh18"], tf.keras)
        tensor = np.asarray([0.1, -0.2, 0.3, -0.1], dtype=np.float32).reshape(1, 4, 1)
        expected = model(tensor, training=False).numpy().reshape(-1).tolist()

        with tempfile.TemporaryDirectory(prefix="rttrainer-scaled-tanh-save-") as tmp:
            model_path = Path(tmp) / "checkpoints" / "best-model.keras"
            model_path.parent.mkdir(parents=True)
            save_keras_model_checkpoint(model, model_path)
            reloaded, _checkpoint = load_keras_checkpoint(model_path)
            actual = reloaded(tensor, training=False).numpy().reshape(-1).tolist()

        self.assertLessEqual(max_abs_error(expected, actual), 1.0e-7)

    @unittest.skipUnless(
        native_validator_available(),
        "Native RTNeural validator is required for golden fixture validation.",
    )
    def test_every_golden_fixture_validates_in_native_rtneural(self) -> None:
        golden_module = require_golden_fixture_module()
        fixtures = golden_module.build_fixture_models()
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
