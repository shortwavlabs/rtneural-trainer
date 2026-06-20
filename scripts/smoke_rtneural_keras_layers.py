#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

from rttrainer.data.audio_io import read_wav_mono, write_wav_mono  # noqa: E402
from rttrainer.export_rtneural.json_exporter import build_keras_rtneural_json  # noqa: E402
from rttrainer.utils import write_json  # noqa: E402
from rttrainer.validation.parity import run_exported_json  # noqa: E402

DEFAULT_VALIDATOR = ROOT / "native/rtneural-validator/build/rtneural-validator"
SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class Fixture:
    name: str
    builder: Callable[[Any, int], Any]
    python_tolerance: float = 1.0e-4
    native_tolerance: float = 1.0e-3


def main() -> int:
    fixtures = build_fixtures()
    parser = argparse.ArgumentParser(
        description=(
            "Build Keras RTNeural JSON fixtures for supported layers and validate "
            "them with Python parity plus the native RTNeural validator."
        )
    )
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--fixture",
        action="append",
        choices=[fixture.name for fixture in fixtures],
        help="Run one fixture; can be passed more than once.",
    )
    args = parser.parse_args()

    if args.list:
        for fixture in fixtures:
            print(fixture.name)
        return 0

    selected = (
        [fixture for fixture in fixtures if fixture.name in set(args.fixture)]
        if args.fixture
        else fixtures
    )

    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="rttrainer-keras-layers-"))
        run_smoke(Path(args.validator), root, selected)
        print(f"kept layer smoke project: {root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="rttrainer-keras-layers-") as tmp:
        run_smoke(Path(args.validator), Path(tmp), selected)
    return 0


def run_smoke(validator: Path, root: Path, fixtures: list[Fixture]) -> None:
    if not validator.exists():
        raise FileNotFoundError(
            f"rtneural-validator not found at {validator}. "
            "Build it with: cmake --build native/rtneural-validator/build"
        )

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(2026)

    input_path = root / "input.wav"
    input_samples = [
        0.23 * math.sin(index * 0.043)
        + 0.11 * math.sin(index * 0.017)
        + 0.04 * math.sin(index * 0.131)
        for index in range(768)
    ]
    write_wav_mono(input_path, input_samples, SAMPLE_RATE)
    quantized_input = read_wav_mono(input_path).samples

    for index, fixture in enumerate(fixtures):
        fixture_dir = root / fixture.name
        fixture_dir.mkdir(parents=True, exist_ok=True)

        model = fixture.builder(tf, 10 + index)
        model_json = build_keras_rtneural_json(
            model=model,
            preset_id=fixture.name,
            sample_rate=SAMPLE_RATE,
            latency_samples=0,
            checkpoint_metrics={},
        )
        model_path = fixture_dir / "model.rtneural.json"
        write_json(model_path, model_json)

        reference = predict_keras(model, quantized_input)
        assert_signal_is_safe(fixture.name, reference)
        parity = run_exported_json(model_path, quantized_input)
        parity_error = max_abs_error(reference, parity)
        if parity_error > fixture.python_tolerance:
            raise RuntimeError(
                f"{fixture.name} Python parity failed: "
                f"max_abs_error={parity_error:.8f}, tolerance={fixture.python_tolerance}"
            )

        reference_path = fixture_dir / "reference.wav"
        report_path = fixture_dir / "native-validation-report.json"
        write_wav_mono(reference_path, reference, SAMPLE_RATE)
        run_native_validation(
            validator=validator,
            model_path=model_path,
            input_path=input_path,
            reference_path=reference_path,
            report_path=report_path,
            tolerance=fixture.native_tolerance,
        )
        native_report = json.loads(report_path.read_text(encoding="utf-8"))
        print(
            f"{fixture.name}: python={parity_error:.3e}, "
            f"native={float(native_report['max_abs_error']):.3e}"
        )


def build_fixtures() -> list[Fixture]:
    return [
        Fixture("dense_only", build_dense_only),
        Fixture("activations", build_supported_activations),
        Fixture("gru", build_gru),
        Fixture("conv1d", build_conv1d),
        Fixture("batchnorm_prelu", build_batchnorm_prelu),
    ]


def build_dense_only(tf, seed: int):  # type: ignore[no-untyped-def]
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(None, 1), name="audio"),
            layers.Dense(5, kernel_initializer=uniform(tf, seed, 0.18), name="dense_a"),
            layers.Dense(1, kernel_initializer=uniform(tf, seed + 1, 0.12), name="dense_out"),
        ],
        name="dense_only",
    )
    initialize_model(model)
    return model


def build_supported_activations(tf, seed: int):  # type: ignore[no-untyped-def]
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(None, 1), name="audio"),
            layers.Dense(
                4,
                activation="tanh",
                kernel_initializer=uniform(tf, seed, 0.18),
                name="dense_tanh",
            ),
            layers.Activation("relu", name="relu"),
            layers.Dense(
                4,
                activation="sigmoid",
                kernel_initializer=uniform(tf, seed + 1, 0.14),
                name="dense_sigmoid",
            ),
            layers.Activation("elu", name="elu"),
            layers.Dense(4, kernel_initializer=uniform(tf, seed + 2, 0.10), name="dense_mid"),
            layers.Activation("softmax", name="softmax"),
            layers.Dense(1, kernel_initializer=uniform(tf, seed + 3, 0.10), name="dense_out"),
        ],
        name="activations",
    )
    initialize_model(model)
    return model


def build_gru(tf, seed: int):  # type: ignore[no-untyped-def]
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(None, 1), name="audio"),
            layers.GRU(
                5,
                return_sequences=True,
                activation="tanh",
                recurrent_activation="sigmoid",
                reset_after=True,
                kernel_initializer=uniform(tf, seed, 0.16),
                recurrent_initializer=uniform(tf, seed + 1, 0.10),
                bias_initializer="zeros",
                name="gru",
            ),
            layers.Dense(1, kernel_initializer=uniform(tf, seed + 2, 0.12), name="dense_out"),
        ],
        name="gru",
    )
    initialize_model(model)
    return model


def build_conv1d(tf, seed: int):  # type: ignore[no-untyped-def]
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(None, 1), name="audio"),
            layers.Conv1D(
                4,
                kernel_size=3,
                padding="causal",
                activation="tanh",
                kernel_initializer=uniform(tf, seed, 0.16),
                bias_initializer="zeros",
                name="conv1d",
            ),
            layers.Dense(1, kernel_initializer=uniform(tf, seed + 1, 0.12), name="dense_out"),
        ],
        name="conv1d",
    )
    initialize_model(model)
    return model


def build_batchnorm_prelu(tf, seed: int):  # type: ignore[no-untyped-def]
    layers = tf.keras.layers
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(None, 1), name="audio"),
            layers.Conv1D(
                4,
                kernel_size=3,
                padding="causal",
                kernel_initializer=uniform(tf, seed, 0.16),
                bias_initializer="zeros",
                name="conv1d",
            ),
            layers.BatchNormalization(epsilon=0.01, name="batchnorm"),
            layers.PReLU(shared_axes=[1], name="prelu"),
            layers.Dense(1, kernel_initializer=uniform(tf, seed + 1, 0.12), name="dense_out"),
        ],
        name="batchnorm_prelu",
    )
    initialize_model(model)
    set_batchnorm_weights(model)
    set_prelu_weights(model)
    return model


def initialize_model(model) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    model(np.zeros((1, 16, 1), dtype=np.float32), training=False)


def set_batchnorm_weights(model) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    for layer in model.layers:
        if layer.__class__.__name__ == "BatchNormalization":
            channels = int(layer.gamma.shape[0])
            gamma = np.linspace(0.85, 1.15, channels, dtype=np.float32)
            beta = np.linspace(-0.03, 0.03, channels, dtype=np.float32)
            mean = np.linspace(-0.02, 0.02, channels, dtype=np.float32)
            variance = np.linspace(0.75, 1.10, channels, dtype=np.float32)
            layer.set_weights([gamma, beta, mean, variance])


def set_prelu_weights(model) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    for layer in model.layers:
        if layer.__class__.__name__ == "PReLU":
            alpha = layer.get_weights()[0]
            layer.set_weights([np.full(alpha.shape, 0.22, dtype=np.float32)])


def predict_keras(model, samples: list[float]) -> list[float]:  # type: ignore[no-untyped-def]
    import numpy as np

    tensor = np.asarray(samples, dtype=np.float32).reshape(1, len(samples), 1)
    prediction = model(tensor, training=False).numpy()
    if prediction.shape[-1] != 1:
        raise RuntimeError(f"Expected single-output model, got shape {prediction.shape}")
    return [float(value) for value in prediction.reshape(-1)]


def run_native_validation(
    *,
    validator: Path,
    model_path: Path,
    input_path: Path,
    reference_path: Path,
    report_path: Path,
    tolerance: float,
) -> None:
    command = [
        str(validator),
        "validate",
        "--model",
        str(model_path),
        "--input",
        str(input_path),
        "--reference",
        str(reference_path),
        "--report",
        str(report_path),
        "--tolerance",
        str(tolerance),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Native RTNeural validation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def assert_signal_is_safe(name: str, samples: list[float]) -> None:
    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak >= 0.95:
        raise RuntimeError(f"{name} fixture output peak is too high for PCM smoke WAV: {peak:.4f}")


def max_abs_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"Signal length mismatch: {len(left)} != {len(right)}")
    return max((abs(left[index] - right[index]) for index in range(len(left))), default=0.0)


def uniform(tf, seed: int, limit: float):  # type: ignore[no-untyped-def]
    return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit, seed=seed)


def require_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for layer smoke validation. "
            "Run with: cd trainer && UV_CACHE_DIR=../.uv-cache "
            "uv run --extra tensorflow python ../scripts/smoke_rtneural_keras_layers.py"
        ) from exc
    return tf


if __name__ == "__main__":
    raise SystemExit(main())
