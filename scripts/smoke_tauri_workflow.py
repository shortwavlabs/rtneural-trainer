#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
TAURI = APP / "src-tauri"
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

from rttrainer.data.audio_io import write_wav_mono  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Tauri sidecar workflow: prepare, train, export, validate."
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--preset", default="wavenet_tcn_fast")
    args = parser.parse_args(strip_pnpm_separator(sys.argv[1:]))

    if not args.skip_build:
        stage_tauri_sidecars()
    verify_tauri_configuration()

    rttrainer = target_debug_binary("rttrainer")
    validator = target_debug_binary("rtneural-validator")
    if args.keep:
        root = Path(tempfile.mkdtemp(prefix="rttrainer-tauri-workflow-"))
        run_workflow(rttrainer, validator, root, args.epochs, preset=args.preset)
        print(f"kept Tauri workflow smoke project: {root}")
        return 0

    with tempfile.TemporaryDirectory(prefix="rttrainer-tauri-workflow-") as tmp:
        run_workflow(rttrainer, validator, Path(tmp), args.epochs, preset=args.preset)
    return 0


def stage_tauri_sidecars() -> None:
    run(["pnpm", "--filter", "rtneural-trainer-app", "package:sidecars:dev"], cwd=ROOT)
    run(["cargo", "check"], cwd=TAURI)


def verify_tauri_configuration() -> None:
    config = read_json(TAURI / "tauri.conf.json")
    external_bins = config.get("bundle", {}).get("externalBin", [])
    expected_bins = {"binaries/rttrainer", "binaries/rtneural-validator"}
    if set(external_bins) != expected_bins:
        raise RuntimeError(f"Unexpected Tauri externalBin entries: {external_bins}")

    capability = read_json(TAURI / "capabilities/default.json")
    permissions = capability.get("permissions", [])
    if "dialog:allow-open" not in permissions:
        raise RuntimeError("Tauri capability is missing dialog:allow-open.")
    shell_permission = next(
        (item for item in permissions if isinstance(item, dict) and item.get("identifier") == "shell:allow-execute"),
        None,
    )
    allowed = {
        item.get("name")
        for item in shell_permission.get("allow", [])  # type: ignore[union-attr]
        if item.get("sidecar") is True
    } if shell_permission else set()
    if allowed != expected_bins:
        raise RuntimeError(f"Unexpected shell sidecar permissions: {sorted(allowed)}")


def run_workflow(
    rttrainer: Path,
    validator: Path,
    root: Path,
    epochs: int,
    *,
    preset: str = "wavenet_tcn_fast",
) -> None:
    if not rttrainer.is_file():
        raise FileNotFoundError(f"rttrainer sidecar not found: {rttrainer}")
    if not validator.is_file():
        raise FileNotFoundError(f"rtneural-validator sidecar not found: {validator}")

    audio_dir = root / "audio"
    prepared_dir = audio_dir / "prepared"
    run_dir = root / "runs/tauri_smoke"
    export_dir = root / "exports/tauri_smoke"
    audio_dir.mkdir(parents=True, exist_ok=True)

    input_samples = [
        0.24 * math.sin(index * 0.071) + 0.10 * math.sin(index * 0.019)
        for index in range(3072)
    ]
    target_samples = [math.tanh(sample * 1.6) * 0.7 for sample in input_samples]
    input_path = audio_dir / "input.wav"
    target_path = audio_dir / "target.wav"
    write_wav_mono(input_path, input_samples, 48_000)
    write_wav_mono(target_path, target_samples, 48_000)

    prepare_manifest = root / "prepare-manifest.json"
    write_json(
        prepare_manifest,
        {
            "input_path": str(input_path),
            "target_path": str(target_path),
            "output_dir": str(prepared_dir),
            "target_sample_rate": 48_000,
            "resample": False,
            "channel_policy": "mixdown",
        },
    )
    run([str(rttrainer), "prepare", "--manifest", str(prepare_manifest)], cwd=ROOT)
    prepare_report = read_json(prepared_dir / "preparation-report.json")
    if prepare_report["status"] not in {"ready", "warning"}:
        raise RuntimeError(f"Prepare failed: {prepare_report}")

    train_manifest = root / "train-manifest.json"
    write_json(
        train_manifest,
        {
            "run_id": "tauri_smoke",
            "run_dir": str(run_dir),
            "prepared_dir": str(prepared_dir),
            "preset": preset,
            "backend": "keras",
            "epochs": epochs,
            "batch_size": 4,
            "learning_rate": 0.0007,
            "sequence_length": 128,
            "max_windows": 24,
            "seed": 2026,
        },
    )
    run([str(rttrainer), "train", "--manifest", str(train_manifest)], cwd=ROOT)
    training_report = read_json(run_dir / "training-report.json")
    if training_report["backend"] != "keras":
        raise RuntimeError(f"Unexpected training backend: {training_report}")

    export_manifest = root / "export-manifest.json"
    write_json(
        export_manifest,
        {
            "name": "Tauri workflow smoke",
            "run_dir": str(run_dir),
            "export_dir": str(export_dir),
            "sample_rate": 48_000,
            "latency_samples": 0,
            "parity_tolerance": 1.0e-4,
        },
    )
    run([str(rttrainer), "export", "--manifest", str(export_manifest)], cwd=ROOT)
    export_package = read_json(export_dir / "package.json")
    if export_package["preset"] != preset:
        raise RuntimeError(f"Unexpected export package: {export_package}")

    native_validation_report = export_dir / "native-validation-report.json"
    native_benchmark_report = export_dir / "native-benchmark-report.json"
    run(
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
            str(native_validation_report),
            "--tolerance",
            "0.001",
        ],
        cwd=ROOT,
    )
    run(
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
            str(native_benchmark_report),
        ],
        cwd=ROOT,
    )
    validation = read_json(native_validation_report)
    benchmark = read_json(native_benchmark_report)
    if validation["status"] != "pass" or benchmark["status"] != "pass":
        raise RuntimeError(f"Native reports failed: {validation} {benchmark}")
    print(
        "tauri workflow smoke passed: "
        f"validation={validation['max_abs_error']:.3e}, "
        f"benchmark={benchmark['realtime_factor']:.2f}x"
    )


def target_debug_binary(stem: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return TAURI / "target" / "debug" / f"{stem}{suffix}"


def strip_pnpm_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], *, cwd: Path) -> None:
    print("+ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with status {result.returncode}: {' '.join(command)}\n"
            f"stdout tail:\n{tail(result.stdout)}\n"
            f"stderr tail:\n{tail(result.stderr)}"
        )


def tail(value: str, *, line_count: int = 80) -> str:
    lines = value.splitlines()
    return "\n".join(lines[-line_count:]) if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
