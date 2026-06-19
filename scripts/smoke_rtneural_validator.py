#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATOR = ROOT / "native/rtneural-validator/build/rtneural-validator"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the native RTNeural validator.")
    parser.add_argument("--validator", default=str(DEFAULT_VALIDATOR))
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    validator = Path(args.validator)
    if not validator.is_file():
        raise SystemExit(f"Validator binary not found: {validator}")

    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="rtneural-validator-"))
        run_smoke(validator, root)
        print(f"Kept smoke files in {root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="rtneural-validator-") as tmp:
        run_smoke(validator, Path(tmp))
    return 0


def run_smoke(validator: Path, root: Path) -> None:
    model_path = root / "identity.rtneural.json"
    input_path = root / "input.wav"
    reference_path = root / "reference.wav"
    validation_report = root / "validation-report.json"
    benchmark_report = root / "benchmark-report.json"

    write_identity_model(model_path)
    write_wav(input_path)
    write_wav(reference_path)

    validate = subprocess.run(
        [
            str(validator),
            "validate",
            "--model",
            str(model_path),
            "--input",
            str(input_path),
            "--reference",
            str(reference_path),
            "--report",
            str(validation_report),
            "--tolerance",
            "0.0001",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    benchmark = subprocess.run(
        [
            str(validator),
            "benchmark",
            "--model",
            str(model_path),
            "--sample-rate",
            "48000",
            "--seconds",
            "1",
            "--report",
            str(benchmark_report),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    validation = json.loads(validation_report.read_text(encoding="utf-8"))
    benchmark_result = json.loads(benchmark_report.read_text(encoding="utf-8"))
    if validation["status"] != "pass":
        raise SystemExit(validate.stdout)
    if benchmark_result["status"] != "pass":
        raise SystemExit(benchmark.stdout)

    print(f"validation: {validation['status']} max_abs_error={validation['max_abs_error']}")
    print(f"benchmark: {benchmark_result['status']} realtime_factor={benchmark_result['realtime_factor']:.2f}")


def write_identity_model(path: Path) -> None:
    model = {
        "in_shape": [None, None, 1],
        "layers": [
            {
                "type": "dense",
                "activation": "",
                "shape": [None, None, 1],
                "weights": [[[1.0]], [0.0]],
            }
        ],
    }
    path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")


def write_wav(path: Path) -> None:
    samples = [0.0, 0.25, -0.25, 0.5, -0.5, 0.125]
    payload = b"".join(
        struct.pack("<h", round(max(-1.0, min(1.0, sample)) * 32767.0))
        for sample in samples
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(payload)


if __name__ == "__main__":
    raise SystemExit(main())
