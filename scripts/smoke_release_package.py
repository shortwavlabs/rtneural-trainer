#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAURI = ROOT / "app" / "src-tauri"
BINARIES_DIR = TAURI / "binaries"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build real sidecars, create a Tauri release bundle, and smoke-test outputs."
    )
    parser.add_argument(
        "--bundles",
        default=None,
        help="Comma-separated Tauri bundle targets, for example app,dmg, deb, or nsis.",
    )
    args = parser.parse_args(strip_pnpm_separator(sys.argv[1:]))

    run(["pnpm", "--filter", "rtneural-trainer-app", "package:sidecars"], cwd=ROOT)
    run(["python3", "scripts/smoke_release_sidecars.py"], cwd=ROOT)

    target_triple = host_triple()
    rttrainer = sidecar_path("rttrainer", target_triple)
    validator = sidecar_path("rtneural-validator", target_triple)

    env = os.environ.copy()
    env["CI"] = "true"
    env["RTTRAINER_SIDECAR_SOURCE"] = str(rttrainer)
    env["RTNEURAL_VALIDATOR_SOURCE"] = str(validator)

    command = ["pnpm", "--filter", "rtneural-trainer-app", "tauri", "build", "--ci", "--no-sign"]
    if platform.system() == "Darwin":
        command.append("--skip-stapling")
    if args.bundles:
        command.extend(["--bundles", args.bundles])
    run(command, cwd=ROOT, env=env)

    app_binary = TAURI / "target" / "release" / f"rtneural-trainer{executable_extension()}"
    if not app_binary.is_file():
        raise FileNotFoundError(f"Tauri release binary was not produced: {app_binary}")

    copied_rttrainer = app_binary.parent / f"rttrainer{executable_extension()}"
    copied_validator = app_binary.parent / f"rtneural-validator{executable_extension()}"
    run(
        [
            "python3",
            "scripts/smoke_release_sidecars.py",
            "--binaries-dir",
            str(app_binary.parent),
            "--logical-names",
        ],
        cwd=ROOT,
    )
    if not copied_rttrainer.is_file() or not copied_validator.is_file():
        raise FileNotFoundError("Tauri release binary exists, but copied sidecars are missing.")

    run(["python3", "scripts/collect_tauri_release_artifacts.py"], cwd=ROOT)
    print(f"release package smoke passed: {app_binary.relative_to(ROOT)}")
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


def sidecar_path(stem: str, target_triple: str) -> Path:
    return BINARIES_DIR / f"{stem}-{target_triple}{executable_extension()}"


def strip_pnpm_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with status {result.returncode}: {' '.join(command)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
