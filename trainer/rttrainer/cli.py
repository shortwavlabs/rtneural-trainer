from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rttrainer import __version__
from rttrainer.data.prepare import prepare_audio
from rttrainer.export_rtneural.json_exporter import export_checkpoint
from rttrainer.metrics.aliasing import analyze_rtneural_json_aliasing
from rttrainer.training.device import inspect_device
from rttrainer.training.runner import evaluate_checkpoint, run_training
from rttrainer.utils import emit, mkdir, read_json, require_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rttrainer")
    parser.add_argument("--version", action="version", version=f"rttrainer {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-device")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")

    for command in ("prepare", "train", "evaluate", "export"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--manifest", required=True)

    aliasing_parser = subparsers.add_parser("aliasing")
    aliasing_parser.add_argument("--model", required=True)
    aliasing_parser.add_argument("--report")
    aliasing_parser.add_argument("--sample-rate", type=int, default=48_000)
    aliasing_parser.add_argument("--json", action="store_true", dest="as_json")

    args = parser.parse_args(argv)

    try:
        if args.command == "inspect-device":
            payload = inspect_device()
            if args.as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"{payload['selected_device']} ({payload['tensorflow_status']})")
            return 0

        if args.command == "aliasing":
            report = analyze_rtneural_json_aliasing(
                model_json_path=Path(args.model).expanduser(),
                sample_rate=int(args.sample_rate),
                report_path=Path(args.report).expanduser() if args.report else None,
            )
            if args.as_json:
                print(json.dumps(report, indent=2))
            else:
                emit({"type": "aliasing_finished", **report})
            return 0

        manifest = read_json(Path(args.manifest))
        if args.command == "prepare":
            return run_prepare_command(manifest)
        if args.command == "train":
            result = run_training(manifest)
            emit({"type": "train_command_finished", **result})
            return 0
        if args.command == "evaluate":
            result = evaluate_checkpoint(manifest)
            emit({"type": "evaluation_finished", **result})
            return 0
        if args.command == "export":
            result = export_checkpoint(manifest)
            emit({"type": "export_finished", **result})
            return 0
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1

    return 2


def run_prepare_command(manifest: dict) -> int:
    output_dir = mkdir(Path(str(manifest.get("output_dir", "prepared"))).expanduser())
    prepared = prepare_audio(
        input_path=require_path(manifest.get("input_path"), "input_path"),
        target_path=require_path(manifest.get("target_path"), "target_path"),
        output_dir=output_dir,
        target_sample_rate=optional_int(manifest.get("target_sample_rate")),
        resample=bool(manifest.get("resample", False)),
        channel_policy=str(manifest.get("channel_policy", "mixdown")),
        manual_latency_adjustment_samples=optional_int(
            manifest.get("manual_latency_adjustment_samples")
        )
        or 0,
        known_latency_samples=optional_int(manifest.get("known_latency_samples")),
    )
    emit(
        {
            "type": "prepare_finished",
            "input_path": str(prepared.input_path),
            "target_path": str(prepared.target_path),
            "report_path": str(prepared.report_path),
            "status": prepared.report["status"],
            "warnings": prepared.report["warnings"],
        }
    )
    return 0


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Expected an integer value, got boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Expected an integer value, got {value!r}.")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected an integer value, got {type(value).__name__}.")


if __name__ == "__main__":
    raise SystemExit(main())
