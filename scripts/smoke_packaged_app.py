#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
TAURI = APP / "src-tauri"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the Tauri packaged-app build path with prebuilt sidecars. "
            "Defaults to a debug, no-bundle build for CI speed."
        )
    )
    parser.add_argument("--bundle", action="store_true", help="Generate platform bundles too.")
    parser.add_argument("--release", action="store_true", help="Use a release Tauri build.")
    args = parser.parse_args(strip_pnpm_separator(sys.argv[1:]))

    ensure_prebuilt_sidecars()
    rttrainer = target_binary("rttrainer")
    validator = target_binary("rtneural-validator")
    smoke_sidecar(rttrainer, ["--version"], expected_status=0)
    smoke_sidecar(validator, [], expected_status=2)

    env = os.environ.copy()
    env["CI"] = "true"
    env["RTTRAINER_SIDECAR_SOURCE"] = str(rttrainer)
    env["RTNEURAL_VALIDATOR_SOURCE"] = str(validator)

    command = ["pnpm", "--filter", "rtneural-trainer-app", "tauri", "build", "--ci"]
    if not args.release:
        command.append("--debug")
    if not args.bundle:
        command.append("--no-bundle")
    run(command, cwd=ROOT, env=env)

    app_binary = target_binary("rtneural-trainer", release=args.release)
    if not app_binary.is_file():
        raise FileNotFoundError(f"Tauri app binary was not produced: {app_binary}")

    copied_rttrainer = app_binary.parent / rttrainer.name
    copied_validator = app_binary.parent / validator.name
    if not copied_rttrainer.is_file():
        raise FileNotFoundError(f"Packaged rttrainer sidecar missing: {copied_rttrainer}")
    if not copied_validator.is_file():
        raise FileNotFoundError(f"Packaged rtneural-validator sidecar missing: {copied_validator}")

    smoke_sidecar(copied_rttrainer, ["--version"], expected_status=0)
    smoke_sidecar(copied_validator, [], expected_status=2)
    print(f"packaged app smoke passed: {app_binary}")
    return 0


def ensure_prebuilt_sidecars() -> None:
    run(["pnpm", "--filter", "rtneural-trainer-app", "package:sidecars:dev"], cwd=ROOT)
    run(["cargo", "check"], cwd=TAURI)


def target_binary(stem: str, *, release: bool = False) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    profile = "release" if release else "debug"
    return TAURI / "target" / profile / f"{stem}{suffix}"


def strip_pnpm_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def smoke_sidecar(path: Path, args: list[str], *, expected_status: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Sidecar not found: {path}")
    result = subprocess.run([str(path), *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != expected_status:
        raise RuntimeError(
            f"{path.name} returned {result.returncode}, expected {expected_status}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(command))
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with status {result.returncode}: {' '.join(command)}\n"
            f"stdout tail:\n{tail(result.stdout)}\n"
            f"stderr tail:\n{tail(result.stderr)}"
        )


def tail(value: str, *, line_count: int = 120) -> str:
    lines = value.splitlines()
    return "\n".join(lines[-line_count:]) if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
