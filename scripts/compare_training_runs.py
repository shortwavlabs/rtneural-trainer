#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
VALIDATOR_BACKENDS = ("eigen", "stl", "xsimd")
WAVENET_RUNTIME_ESTIMATES = {
    "wavenet_tcn_fast": 8.0,
    "wavenet_tcn": 3.0,
    "wavenet_tcn_balanced": 3.0,
    "wavenet_tcn_balanced_tanh15": 3.0,
    "wavenet_tcn_balanced_tanh18": 3.0,
    "wavenet_tcn_quality": 1.5,
    "wavenet_tcn_quality_tanh18": 1.5,
    "wavenet_tcn_high_gain": 1.2,
    "wavenet_tcn_separable_fast": 5.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare trained RTNeural run folders and optionally export/benchmark them."
    )
    parser.add_argument("runs", nargs="+", help="Run directories to compare.")
    parser.add_argument("--label", action="append", dest="labels", help="Friendly label per run.")
    parser.add_argument("--out", default="run-comparison", help="Output directory.")
    parser.add_argument("--export", action="store_true", help="Run rttrainer export for each run.")
    parser.add_argument(
        "--native",
        action="store_true",
        help="Run native RTNeural validation and benchmark matrix when validators are available.",
    )
    parser.add_argument("--force-export", action="store_true", help="Regenerate existing exports.")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--latency-samples", type=int, help="Override export latency.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    run_dirs = [Path(path).expanduser().resolve() for path in args.runs]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")

    labels = list(args.labels or [])
    if labels and len(labels) != len(run_dirs):
        raise SystemExit("--label must be provided once per run when used.")

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    validators = discover_validator_variants() if args.native else []

    comparisons: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs):
        label = labels[index] if labels else run_dir.name
        comparisons.append(compare_run(run_dir, label, out_dir, validators, args))

    summary = {
        "schema_version": 1,
        "runs": comparisons,
        "validator_variants": [
            {"id": variant["id"], "binary": str(variant["binary"])} for variant in validators
        ],
    }
    write_json(out_dir / "comparison.json", summary)
    write_markdown(out_dir / "comparison.md", comparisons, validators)
    print(f"wrote comparison: {out_dir / 'comparison.md'}")
    return 0


def compare_run(
    run_dir: Path,
    label: str,
    out_dir: Path,
    validators: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = read_json(run_dir / "training-report.json")
    history_doc = read_optional_json(run_dir / "history.json")
    prep = read_optional_json(run_dir / "preparation-report.json")
    metrics = as_mapping(report.get("metrics"))
    preset = str(report.get("preset") or metrics.get("preset") or "unknown")
    run_id = str(report.get("run_id") or run_dir.name)
    sample_rate = int(args.sample_rate)
    latency_samples = latency_for_run(prep, args.latency_samples)
    corrected_rtf = corrected_realtime_factor(preset, get_number(metrics, "realtime_factor"))
    export_summary: dict[str, Any] | None = None

    if args.export or args.native:
        export_dir = out_dir / "exports" / run_id
        export_summary = ensure_exported_run(
            run_dir=run_dir,
            report=report,
            export_dir=export_dir,
            sample_rate=sample_rate,
            latency_samples=latency_samples,
            device=str(args.device),
            force=bool(args.force_export),
        )
        if validators:
            export_summary["native"] = run_native_checks(
                run_dir=run_dir,
                export_dir=export_dir,
                sample_rate=sample_rate,
                validators=validators,
            )

    aliasing = read_export_json(export_summary, "aliasing-report.json")
    benchmark = read_export_json(export_summary, "benchmark-report.json")
    native = as_mapping(export_summary.get("native") if export_summary else None)
    best_history = best_history_entry(history_doc)
    final_history = final_history_entry(history_doc)
    validation_score = get_number(best_history, "validation_score")
    window_val_esr = get_number(best_history, "window_val_esr")
    correlation = get_number(metrics, "state_continuous_correlation")
    rmse = get_number(metrics, "rmse")

    return {
        "label": label,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "preset": preset,
        "backend": report.get("backend"),
        "device": report.get("device"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "metrics": {
            "esr": get_number(metrics, "esr"),
            "mae": get_number(metrics, "mae"),
            "rmse": rmse,
            "rmse_dbfs": dbfs(rmse),
            "peak_residual": get_number(metrics, "peak_residual"),
            "correlation": correlation,
            "saved_realtime_factor": get_number(metrics, "realtime_factor"),
            "corrected_realtime_factor": corrected_rtf,
        },
        "history": {
            "best_epoch": best_history.get("epoch"),
            "final_epoch": final_history.get("epoch"),
            "best_validation_score": validation_score,
            "best_window_val_esr": window_val_esr,
            "learning_rate_reductions": count_learning_rate_reductions(history_doc),
        },
        "capture": capture_summary(prep),
        "export": export_summary,
        "export_benchmark": benchmark,
        "aliasing": aliasing,
        "native": native,
        "notes": run_notes(preset, aliasing, corrected_rtf),
    }


def ensure_exported_run(
    *,
    run_dir: Path,
    report: dict[str, Any],
    export_dir: Path,
    sample_rate: int,
    latency_samples: int,
    device: str,
    force: bool,
) -> dict[str, Any]:
    package_path = export_dir / "package.json"
    if package_path.is_file() and not force:
        return {
            "status": "existing",
            "export_dir": str(export_dir),
            "package_path": str(package_path),
        }

    checkpoint_path = Path(str(report.get("best_checkpoint_path", ""))).expanduser()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Best checkpoint not found for {run_dir}: {checkpoint_path}")

    export_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = export_dir / "export-manifest.json"
    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "export_dir": str(export_dir),
        "sample_rate": sample_rate,
        "latency_samples": latency_samples,
        "input_path": str(run_dir / "test-input.wav"),
        "target_path": str(run_dir / "test-target.wav"),
        "device": device,
        "name": f"{report.get('preset', 'model')} {report.get('run_id', run_dir.name)}",
    }
    write_json(manifest_path, manifest)
    completed = run_command(
        [sys.executable, "-m", "rttrainer", "export", "--manifest", str(manifest_path)],
        cwd=ROOT,
        env=python_env(),
    )
    write_text(export_dir / "export.stdout.log", completed.stdout)
    write_text(export_dir / "export.stderr.log", completed.stderr)
    return {
        "status": "exported",
        "export_dir": str(export_dir),
        "package_path": str(package_path),
        "stdout_path": str(export_dir / "export.stdout.log"),
        "stderr_path": str(export_dir / "export.stderr.log"),
    }


def run_native_checks(
    *,
    run_dir: Path,
    export_dir: Path,
    sample_rate: int,
    validators: list[dict[str, Any]],
) -> dict[str, Any]:
    model_path = export_dir / "model.rtneural.json"
    input_path = run_dir / "test-input.wav"
    reference_path = run_dir / "previews" / "prediction.wav"
    validation_report = export_dir / "native-validation-report.json"
    validation: dict[str, Any] | None = None
    if model_path.is_file() and input_path.is_file() and reference_path.is_file() and validators:
        validator = Path(str(validators[0]["binary"]))
        validation = run_validator_report(
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
            validation_report,
        )

    benchmark_reports: list[dict[str, Any]] = []
    for variant in validators:
        report_path = export_dir / f"native-benchmark-{variant['id']}.json"
        report = run_validator_report(
            [
                str(variant["binary"]),
                "benchmark",
                "--model",
                str(model_path),
                "--sample-rate",
                str(sample_rate),
                "--seconds",
                "2",
                "--block-sizes",
                "16,32,64,128,256,512",
                "--channels",
                "1,2",
                "--passes",
                "3",
                "--warmup-blocks",
                "8",
                "--min-realtime-factor",
                "1.0",
                "--report",
                str(report_path),
            ],
            report_path,
        )
        benchmark_reports.append(
            {
                "id": variant["id"],
                "binary": str(variant["binary"]),
                "report_path": str(report_path),
                "report": report,
            }
        )

    return {
        "validation": validation,
        "benchmarks": benchmark_reports,
        "fastest": fastest_passing_benchmark(benchmark_reports),
    }


def run_validator_report(command: list[str], report_path: Path) -> dict[str, Any]:
    try:
        completed = run_command(command, cwd=ROOT, env=os.environ.copy())
    except subprocess.CalledProcessError as exc:
        return {
            "status": "failed",
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
    report = read_optional_json(report_path) or {}
    report["stdout"] = completed.stdout
    report["stderr"] = completed.stderr
    return report


def discover_validator_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for backend in VALIDATOR_BACKENDS:
        add_validator_variant(variants, seen, backend, env_binary(backend, avx=False))
        add_validator_variant(variants, seen, f"{backend}-avx", env_binary(backend, avx=True))
        for suffix in (backend, f"{backend}-avx"):
            for build_dir in (
                ROOT / "native" / "rtneural-validator" / f"build-{suffix}",
                ROOT / "native" / "rtneural-validator" / f"build-release-{suffix}",
            ):
                add_validator_variant(
                    variants,
                    seen,
                    suffix,
                    existing_validator_binary(build_dir),
                )

    add_validator_variant(
        variants,
        seen,
        "default",
        existing_validator_binary(ROOT / "native" / "rtneural-validator" / "build"),
    )
    return variants


def add_validator_variant(
    variants: list[dict[str, Any]],
    seen: set[Path],
    variant_id: str,
    path: Path | None,
) -> None:
    if path is None:
        return
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    variants.append({"id": variant_id, "binary": resolved})


def env_binary(backend: str, *, avx: bool) -> Path | None:
    suffix = "_AVX_BINARY" if avx else "_BINARY"
    value = os.environ.get(f"RTNEURAL_VALIDATOR_{backend.upper()}{suffix}")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_file() else None


def existing_validator_binary(build_dir: Path) -> Path | None:
    for name in ("rtneural-validator", "rtneural-validator.exe"):
        candidate = build_dir / name
        if candidate.is_file():
            return candidate
    return None


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    paths = [str(TRAINER)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def read_export_json(export_summary: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not export_summary:
        return None
    export_dir = Path(str(export_summary.get("export_dir", "")))
    return read_optional_json(export_dir / name)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_markdown(
    path: Path,
    comparisons: list[dict[str, Any]],
    validators: list[dict[str, Any]],
) -> None:
    lines = [
        "# Training Run Comparison",
        "",
        f"- Runs compared: {len(comparisons)}",
        f"- Native validators: {', '.join(str(v['id']) for v in validators) if validators else 'not run'}",
        "",
        "| Run | Preset | ESR | RMSE | Corr | Residual RMS | Est RTF | Val score | Worst ASR | Native RTF | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in comparisons:
        metrics = as_mapping(item.get("metrics"))
        history = as_mapping(item.get("history"))
        aliasing = as_mapping(item.get("aliasing"))
        native = as_mapping(item.get("native"))
        fastest = as_mapping(native.get("fastest"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["label"]),
                    str(item["preset"]),
                    format_float(metrics.get("esr")),
                    format_float(metrics.get("rmse")),
                    format_float(metrics.get("correlation")),
                    format_db(metrics.get("rmse_dbfs")),
                    format_rtf(metrics.get("corrected_realtime_factor")),
                    format_float(history.get("best_validation_score")),
                    format_float(aliasing.get("worst_asr")),
                    format_rtf(fastest.get("realtime_factor")),
                    str(item.get("notes") or ""),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Details", ""])
    for item in comparisons:
        metrics = as_mapping(item.get("metrics"))
        capture = as_mapping(item.get("capture"))
        aliasing = as_mapping(item.get("aliasing"))
        native = as_mapping(item.get("native"))
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- Run: `{item['run_id']}`",
                f"- Directory: `{item['run_dir']}`",
                f"- Preset: `{item['preset']}`",
                f"- Device: `{item.get('device')}`",
                f"- Checkpoint epoch: `{item.get('checkpoint_epoch')}`",
                f"- Saved RTF: `{format_rtf(metrics.get('saved_realtime_factor'))}`; corrected RTF: `{format_rtf(metrics.get('corrected_realtime_factor'))}`",
                f"- Capture latency: `{capture.get('latency_samples', 'unknown')}` samples; confidence: `{format_float(capture.get('latency_confidence'))}`",
                f"- Aliasing: `{aliasing.get('verdict', 'not run')}` worst ASR `{format_float(aliasing.get('worst_asr'))}` average ASR `{format_float(aliasing.get('average_asr'))}`",
                f"- Native fastest: `{format_rtf(as_mapping(native.get('fastest')).get('realtime_factor'))}`",
                "",
            ]
        )
    write_text(path, "\n".join(lines) + "\n")


def latency_for_run(prep: dict[str, Any] | None, override: int | None) -> int:
    if override is not None:
        return override
    latency = as_mapping(prep.get("latency") if prep else None)
    value = latency.get("effective_samples", latency.get("training_latency_samples"))
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def capture_summary(prep: dict[str, Any] | None) -> dict[str, Any]:
    latency = as_mapping(prep.get("latency") if prep else None)
    gain = as_mapping(prep.get("gain") if prep else None)
    return {
        "latency_samples": latency.get("effective_samples"),
        "latency_confidence": latency.get("confidence"),
        "latency_agreement": latency.get("agreement"),
        "target_peak_dbfs": gain.get("target_peak_dbfs"),
        "target_rms_dbfs": gain.get("target_rms_dbfs"),
        "rms_delta_db": gain.get("rms_delta_db"),
    }


def best_history_entry(history_doc: dict[str, Any] | None) -> dict[str, Any]:
    entries = history_entries(history_doc)
    scored = [entry for entry in entries if get_number(entry, "validation_score") is not None]
    if not scored:
        return {}
    return min(scored, key=lambda entry: float(cast(float, get_number(entry, "validation_score"))))


def final_history_entry(history_doc: dict[str, Any] | None) -> dict[str, Any]:
    entries = history_entries(history_doc)
    return entries[-1] if entries else {}


def history_entries(history_doc: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not history_doc:
        return []
    raw = history_doc.get("history")
    if not isinstance(raw, list):
        return []
    return [as_mapping(item) for item in raw]


def count_learning_rate_reductions(history_doc: dict[str, Any] | None) -> int:
    return sum(1 for entry in history_entries(history_doc) if entry.get("learning_rate_reduced"))


def corrected_realtime_factor(preset: str, saved: float | None) -> float | None:
    return WAVENET_RUNTIME_ESTIMATES.get(preset, saved)


def fastest_passing_benchmark(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for item in reports:
        report = as_mapping(item.get("report"))
        status = str(report.get("status", ""))
        realtime_factor = get_number(report, "realtime_factor")
        if status == "pass" and realtime_factor is not None:
            candidates.append(
                {
                    "id": item["id"],
                    "binary": item["binary"],
                    "realtime_factor": realtime_factor,
                    "report_path": item["report_path"],
                }
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(cast(float, item["realtime_factor"])))


def run_notes(
    preset: str,
    aliasing: dict[str, Any] | None,
    corrected_rtf: float | None,
) -> str:
    notes: list[str] = []
    if preset == "wavenet_tcn_balanced":
        notes.append("balanced baseline")
    if preset == "wavenet_tcn_high_gain":
        notes.append("longer receptive-field high-gain probe")
    if "tanh15" in preset:
        notes.append("smoothed tanh quality probe")
    if "tanh18" in preset:
        notes.append("smoothed tanh anti-aliasing probe")
    if aliasing:
        verdict = str(aliasing.get("verdict", ""))
        if verdict == "high_aliasing":
            notes.append("ASR warning; compare by ear")
        elif verdict == "low_aliasing":
            notes.append("lower ASR; still verify previews")
    if corrected_rtf is not None and corrected_rtf < 2.0:
        notes.append("tight native headroom")
    return "; ".join(notes)


def as_mapping(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def get_number(mapping: dict[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def dbfs(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return 20.0 * math.log10(value)


def format_float(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    return f"{float(value):.4f}"


def format_db(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    return f"{float(value):.2f} dBFS"


def format_rtf(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    return f"{float(value):.2f}x"


if __name__ == "__main__":
    raise SystemExit(main())
