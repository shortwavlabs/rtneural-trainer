#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BINARIES_DIR = ROOT / "app" / "src-tauri" / "binaries"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test staged production Tauri sidecars."
    )
    parser.add_argument(
        "--binaries-dir",
        default=str(BINARIES_DIR),
        help="Directory containing target-triple-suffixed sidecars.",
    )
    parser.add_argument(
        "--target-triple",
        default=None,
        help="Target triple suffix. Defaults to `rustc --print host-tuple`.",
    )
    parser.add_argument(
        "--logical-names",
        action="store_true",
        help="Expect packaged logical names instead of target-triple source names.",
    )
    args = parser.parse_args()

    binaries_dir = Path(args.binaries_dir)
    target_triple = args.target_triple or host_triple()
    rttrainer = sidecar_path(
        binaries_dir,
        "rttrainer",
        target_triple,
        logical_name=args.logical_names,
    )
    validator = sidecar_path(
        binaries_dir,
        "rtneural-validator",
        target_triple,
        logical_name=args.logical_names,
    )

    smoke_rttrainer(rttrainer)
    smoke_validator_cli(validator)
    print(f"release sidecar smoke passed: {target_triple}")
    return 0


def host_triple() -> str:
    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    triple = result.stdout.strip()
    if not triple:
        raise RuntimeError("rustc did not return a host tuple")
    return triple


def executable_extension() -> str:
    return ".exe" if os.name == "nt" else ""


def sidecar_path(
    binaries_dir: Path,
    stem: str,
    target_triple: str,
    *,
    logical_name: bool,
) -> Path:
    if logical_name:
        return binaries_dir / f"{stem}{executable_extension()}"
    return binaries_dir / f"{stem}-{target_triple}{executable_extension()}"


def smoke_rttrainer(path: Path) -> None:
    require_file(path, "rttrainer")
    version = run([str(path), "--version"], expected_status=0)
    if "rttrainer" not in version.stdout.lower():
        raise RuntimeError(f"Unexpected rttrainer version output: {version.stdout}")

    inspection = run([str(path), "inspect-device", "--json"], expected_status=0)
    payload = parse_json_payload(inspection.stdout)
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unexpected inspect-device schema: {payload}")
    if not isinstance(payload.get("package_versions"), dict):
        raise RuntimeError("inspect-device payload is missing package_versions")


def smoke_validator_cli(path: Path) -> None:
    require_file(path, "rtneural-validator")
    run([str(path)], expected_status=2)


def parse_json_payload(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"Expected JSON object in stdout:\n{stdout}")
    return json.loads(stdout[start : end + 1])


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} sidecar not found: {path}")


def run(command: list[str], *, expected_status: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != expected_status:
        raise RuntimeError(
            f"{Path(command[0]).name} returned {result.returncode}, expected {expected_status}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
