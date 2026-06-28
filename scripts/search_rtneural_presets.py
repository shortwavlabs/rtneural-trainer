#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

from rttrainer.models.presets import PRESETS  # noqa: E402

DEFAULT_PRESETS = (
    "wavenet_tcn_fast",
    "wavenet_tcn_clean",
    "wavenet_tcn_edge",
    "wavenet_tcn_edge_detail",
    "wavenet_tcn_balanced",
    "wavenet_tcn_balanced_tanh15",
    "wavenet_tcn_balanced_tanh18",
    "wavenet_tcn_quality",
    "wavenet_tcn_quality_tanh15",
    "wavenet_tcn_quality_tanh18",
    "wavenet_tcn_a2_prelu",
    "wavenet_tcn_separable_fast",
)


@dataclass(frozen=True)
class SearchCandidate:
    preset_id: str
    epochs: int
    sequence_length: int
    max_windows: int
    learning_rate: float

    @property
    def run_id(self) -> str:
        return (
            f"search_{self.preset_id}_"
            f"e{self.epochs}_w{self.max_windows}_lr{self.learning_rate:g}"
        ).replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or run a small RTNeural-aware preset search. "
            "Dry-run mode writes manifests only; --run executes training and export."
        )
    )
    parser.add_argument("--input", required=True, help="Prepared dry input WAV.")
    parser.add_argument("--target", required=True, help="Prepared target WAV.")
    parser.add_argument("--out", default="preset-search", help="Output directory.")
    parser.add_argument("--preset", action="append", dest="presets", help="Preset id to include.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.0007)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--latency-samples", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--limit", type=int, default=0, help="Limit candidates after filtering.")
    parser.add_argument("--run", action="store_true", help="Execute train/export for every candidate.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    target_path = Path(args.target).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"Input WAV not found: {input_path}")
    if not target_path.is_file():
        raise SystemExit(f"Target WAV not found: {target_path}")

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    plan = write_search_plan(
        out_dir=out_dir,
        candidates=candidates,
        input_path=input_path,
        target_path=target_path,
        args=args,
    )
    if not args.run:
        print(f"wrote preset search plan: {plan}")
        return 0

    results = run_search(
        out_dir=out_dir,
        candidates=candidates,
        input_path=input_path,
        target_path=target_path,
        args=args,
    )
    summary_path = out_dir / "preset-search-results.json"
    write_json(summary_path, {"schema_version": 1, "results": results})
    print(f"wrote preset search results: {summary_path}")
    return 0


def build_candidates(args: argparse.Namespace) -> list[SearchCandidate]:
    preset_ids = args.presets or list(DEFAULT_PRESETS)
    missing = [preset_id for preset_id in preset_ids if preset_id not in PRESETS]
    if missing:
        known = ", ".join(sorted(PRESETS))
        raise SystemExit(f"Unknown preset(s): {', '.join(missing)}. Known presets: {known}")

    candidates = [
        SearchCandidate(
            preset_id=preset_id,
            epochs=int(args.epochs),
            sequence_length=int(args.sequence_length),
            max_windows=int(args.max_windows),
            learning_rate=float(args.learning_rate),
        )
        for preset_id in preset_ids
    ]
    return candidates


def write_search_plan(
    *,
    out_dir: Path,
    candidates: list[SearchCandidate],
    input_path: Path,
    target_path: Path,
    args: argparse.Namespace,
) -> Path:
    plan_path = out_dir / "preset-search-plan.json"
    plan = {
        "schema_version": 1,
        "input_path": str(input_path),
        "target_path": str(target_path),
        "sample_rate": int(args.sample_rate),
        "latency_samples": int(args.latency_samples),
        "device": str(args.device),
        "candidates": [
            candidate_manifest_summary(out_dir, candidate)
            for candidate in candidates
        ],
    }
    write_json(plan_path, plan)
    for candidate in candidates:
        write_candidate_manifests(
            out_dir=out_dir,
            candidate=candidate,
            input_path=input_path,
            target_path=target_path,
            args=args,
        )
    return plan_path


def run_search(
    *,
    out_dir: Path,
    candidates: list[SearchCandidate],
    input_path: Path,
    target_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        paths = write_candidate_manifests(
            out_dir=out_dir,
            candidate=candidate,
            input_path=input_path,
            target_path=target_path,
            args=args,
        )
        print(f"training {candidate.preset_id}")
        run_rttrainer("train", paths["train_manifest"])
        print(f"exporting {candidate.preset_id}")
        run_rttrainer("export", paths["export_manifest"])
        results.append(collect_candidate_result(candidate, paths))
    return results


def write_candidate_manifests(
    *,
    out_dir: Path,
    candidate: SearchCandidate,
    input_path: Path,
    target_path: Path,
    args: argparse.Namespace,
) -> dict[str, Path]:
    candidate_dir = out_dir / candidate.run_id
    run_dir = candidate_dir / "run"
    export_dir = candidate_dir / "export"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = candidate_dir / "train-manifest.json"
    export_manifest_path = candidate_dir / "export-manifest.json"

    write_json(
        train_manifest_path,
        {
            "run_id": candidate.run_id,
            "run_dir": str(run_dir),
            "backend": "keras",
            "device": str(args.device),
            "preset": candidate.preset_id,
            "input_path": str(input_path),
            "target_path": str(target_path),
            "epochs": candidate.epochs,
            "batch_size": 16,
            "learning_rate": candidate.learning_rate,
            "sequence_length": candidate.sequence_length,
            "max_windows": candidate.max_windows,
            "resample_training_windows": True,
            "resample_interval_epochs": 1,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 0.00005,
            "preview_seconds": 3.0,
        },
    )
    write_json(
        export_manifest_path,
        {
            "name": candidate.preset_id,
            "run_dir": str(run_dir),
            "export_dir": str(export_dir),
            "backend": "keras",
            "sample_rate": int(args.sample_rate),
            "latency_samples": int(args.latency_samples),
            "parity_tolerance": 0.0001,
        },
    )
    return {
        "candidate_dir": candidate_dir,
        "run_dir": run_dir,
        "export_dir": export_dir,
        "train_manifest": train_manifest_path,
        "export_manifest": export_manifest_path,
    }


def run_rttrainer(command: str, manifest_path: Path) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(TRAINER)
        if not existing_pythonpath
        else f"{TRAINER}{os.pathsep}{existing_pythonpath}"
    )
    subprocess.run(
        [sys.executable, "-m", "rttrainer", command, "--manifest", str(manifest_path)],
        check=True,
        env=env,
    )


def collect_candidate_result(
    candidate: SearchCandidate,
    paths: dict[str, Path],
) -> dict[str, Any]:
    report = read_optional_json(paths["run_dir"] / "training-report.json")
    package = read_optional_json(paths["export_dir"] / "package.json")
    aliasing = read_optional_json(paths["export_dir"] / "aliasing-report.json")
    metrics = report.get("metrics", {}) if report else {}
    return {
        "preset_id": candidate.preset_id,
        "run_id": candidate.run_id,
        "metrics": metrics,
        "esr": metrics.get("esr"),
        "validation_score": metrics.get("validation_score"),
        "aliasing": {
            "status": aliasing.get("status") if aliasing else None,
            "verdict": aliasing.get("verdict") if aliasing else None,
            "worst_asr": aliasing.get("worst_asr") if aliasing else None,
            "average_asr": aliasing.get("average_asr") if aliasing else None,
        },
        "export_package": str(paths["export_dir"] / "package.json"),
        "package_status": package.get("status") if package else None,
    }


def candidate_manifest_summary(out_dir: Path, candidate: SearchCandidate) -> dict[str, Any]:
    candidate_dir = out_dir / candidate.run_id
    return {
        "preset_id": candidate.preset_id,
        "run_id": candidate.run_id,
        "epochs": candidate.epochs,
        "sequence_length": candidate.sequence_length,
        "max_windows": candidate.max_windows,
        "learning_rate": candidate.learning_rate,
        "train_manifest": str(candidate_dir / "train-manifest.json"),
        "export_manifest": str(candidate_dir / "export-manifest.json"),
    }


def read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
