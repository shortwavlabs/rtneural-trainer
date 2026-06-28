from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER = PROJECT_ROOT / "trainer"
DEFAULT_VALIDATOR = PROJECT_ROOT / "native/rtneural-validator/build/rtneural-validator"
sys.path.insert(0, str(TRAINER))

from rttrainer.data.audio_io import write_wav_mono  # noqa: E402
from rttrainer.export_rtneural.json_exporter import export_checkpoint  # noqa: E402
from rttrainer.training.runner import run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a tiny Keras model, export RTNeural JSON, and validate it natively."
    )
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="rttrainer-keras-smoke-"))
        run_smoke(Path(args.validator), root)
        print(f"kept smoke project: {root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="rttrainer-keras-smoke-") as tmp:
        run_smoke(Path(args.validator), Path(tmp))
    return 0


def run_smoke(validator: Path, root: Path) -> None:
    if not validator.exists():
        raise FileNotFoundError(
            f"rtneural-validator not found at {validator}. "
            "Build it with: cmake --build native/rtneural-validator/build"
        )

    prepared_dir = root / "audio/prepared"
    run_dir = root / "runs/keras_smoke"
    export_dir = root / "exports/keras_smoke"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    input_samples = [
        0.28 * math.sin(index * 0.071) + 0.12 * math.sin(index * 0.017)
        for index in range(4096)
    ]
    target_samples = [math.tanh(sample * 1.7) * 0.72 for sample in input_samples]
    write_wav_mono(prepared_dir / "input.wav", input_samples, 48_000)
    write_wav_mono(prepared_dir / "target.wav", target_samples, 48_000)

    run_training(
        {
            "run_id": "keras_smoke",
            "run_dir": str(run_dir),
            "prepared_dir": str(prepared_dir),
            "preset": "wavenet_tcn_fast",
            "backend": "keras",
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.0007,
            "sequence_length": 128,
            "max_windows": 32,
            "seed": 2026,
        }
    )
    export = export_checkpoint(
        {
            "name": "Keras smoke",
            "run_dir": str(run_dir),
            "export_dir": str(export_dir),
            "sample_rate": 48_000,
            "latency_samples": 0,
            "parity_tolerance": 1e-4,
        }
    )
    if export["validation"]["status"] != "pass":
        raise RuntimeError(f"Python parity failed: {export['validation']}")

    validation_report = export_dir / "native-validation-report.json"
    benchmark_report = export_dir / "native-benchmark-report.json"
    subprocess.run(
        [
            str(validator),
            "validate",
            "--model",
            str(export_dir / "model.rtneural.json"),
            "--input",
            str(run_dir / "test-input.wav"),
            "--reference",
            str(run_dir / "previews/prediction.wav"),
            "--report",
            str(validation_report),
            "--tolerance",
            "0.001",
        ],
        check=True,
    )
    subprocess.run(
        [
            str(validator),
            "benchmark",
            "--model",
            str(export_dir / "model.rtneural.json"),
            "--sample-rate",
            "48000",
            "--seconds",
            "1",
            "--report",
            str(benchmark_report),
        ],
        check=True,
    )
    print(f"keras smoke export: {export_dir / 'model.rtneural.json'}")
    print(f"native validation: {validation_report}")
    print(f"native benchmark: {benchmark_report}")


if __name__ == "__main__":
    raise SystemExit(main())
