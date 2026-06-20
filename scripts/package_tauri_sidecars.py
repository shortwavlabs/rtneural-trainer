#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
TAURI_DIR = APP_DIR / "src-tauri"
BINARIES_DIR = TAURI_DIR / "binaries"
TRAINER_DIR = ROOT / "trainer"
VALIDATOR_DIR = ROOT / "native" / "rtneural-validator"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or stage Tauri sidecars with target-triple filenames."
    )
    parser.add_argument(
        "--target-triple",
        default=None,
        help="Target triple suffix. Defaults to `rustc --print host-tuple`.",
    )
    parser.add_argument(
        "--trainer-source",
        default=os.environ.get("RTTRAINER_SIDECAR_SOURCE"),
        help="Copy a prebuilt rttrainer executable instead of running PyInstaller.",
    )
    parser.add_argument(
        "--validator-source",
        default=os.environ.get("RTNEURAL_VALIDATOR_SOURCE"),
        help="Copy a prebuilt rtneural-validator executable instead of building CMake.",
    )
    parser.add_argument(
        "--skip-trainer",
        action="store_true",
        help="Do not build or copy the rttrainer sidecar.",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Do not build or copy the rtneural-validator sidecar.",
    )
    parser.add_argument(
        "--dev-shims",
        action="store_true",
        help="Create lightweight local-development shims that delegate to uv/CMake outputs.",
    )
    args = parser.parse_args()

    target_triple = args.target_triple or host_triple()
    BINARIES_DIR.mkdir(parents=True, exist_ok=True)

    if args.dev_shims:
        if platform.system() == "Windows":
            raise SystemExit("Dev shims are POSIX shell scripts; use production sidecars on Windows.")
        if not args.skip_trainer:
            write_dev_trainer_shim(sidecar_path("rttrainer", target_triple))
        if not args.skip_validator:
            write_dev_validator_shim(sidecar_path("rtneural-validator", target_triple))
        return 0

    if not args.skip_trainer:
        trainer_source = resolve_user_path(args.trainer_source) if args.trainer_source else build_rttrainer()
        install_sidecar(trainer_source, "rttrainer", target_triple)

    if not args.skip_validator:
        validator_source = (
            resolve_user_path(args.validator_source)
            if args.validator_source
            else build_rtneural_validator()
        )
        install_sidecar(validator_source, "rtneural-validator", target_triple)

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
    return ".exe" if platform.system() == "Windows" else ""


def sidecar_path(stem: str, target_triple: str) -> Path:
    return BINARIES_DIR / f"{stem}-{target_triple}{executable_extension()}"


def resolve_user_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    root_path = (ROOT / path).resolve()
    if root_path.exists():
        return root_path

    return cwd_path


def install_sidecar(source: Path, stem: str, target_triple: str) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"{stem} sidecar source not found: {source}")
    destination = sidecar_path(stem, target_triple)
    if source.resolve() == destination.resolve():
        make_executable(destination)
        print(f"using staged {stem} sidecar: {destination.relative_to(ROOT)}")
        return destination
    shutil.copy2(source, destination)
    make_executable(destination)
    print(f"installed {stem} sidecar: {destination.relative_to(ROOT)}")
    return destination


def build_rttrainer() -> Path:
    run(
        [
            "uv",
            "run",
            "--extra",
            "tensorflow",
            "--with",
            "pyinstaller",
            "pyinstaller",
            "--clean",
            "--noconfirm",
            "--onefile",
            "--name",
            "rttrainer",
            "--paths",
            str(TRAINER_DIR),
            "--collect-all",
            "tensorflow",
            "--collect-all",
            "keras",
            str(TRAINER_DIR / "rttrainer" / "__main__.py"),
        ],
        cwd=TRAINER_DIR,
        env={"UV_CACHE_DIR": str(ROOT / ".uv-cache")},
    )
    output = TRAINER_DIR / "dist" / f"rttrainer{executable_extension()}"
    if not output.is_file():
        raise FileNotFoundError(f"PyInstaller did not create {output}")
    return output


def build_rtneural_validator() -> Path:
    build_dir = VALIDATOR_DIR / "build-release"
    run(
        [
            "cmake",
            "-S",
            str(VALIDATOR_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=ROOT,
    )
    run(["cmake", "--build", str(build_dir), "--config", "Release"], cwd=ROOT)
    candidates = [
        build_dir / f"rtneural-validator{executable_extension()}",
        build_dir / "Release" / f"rtneural-validator{executable_extension()}",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"rtneural-validator build output not found in {build_dir}")


def write_dev_trainer_shim(destination: Path) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shell_quote(str(ROOT))}
cd "$ROOT/trainer"
export UV_CACHE_DIR="${{UV_CACHE_DIR:-$ROOT/.uv-cache}}"
extras="${{RTTRAINER_UV_EXTRAS:-tensorflow training}}"
uv_args=()
for extra in $extras; do
  uv_args+=(--extra "$extra")
done
exec uv run "${{uv_args[@]}}" python -m rttrainer "$@"
"""
    write_executable_text(destination, script)
    print(f"installed development rttrainer shim: {destination.relative_to(ROOT)}")


def write_dev_validator_shim(destination: Path) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shell_quote(str(ROOT))}
VALIDATOR="$ROOT/native/rtneural-validator/build/rtneural-validator"
if [[ ! -x "$VALIDATOR" ]]; then
  echo "rtneural-validator dev binary not found. Build it with: cmake --build native/rtneural-validator/build" >&2
  exit 127
fi
exec "$VALIDATOR" "$@"
"""
    write_executable_text(destination, script)
    print(f"installed development rtneural-validator shim: {destination.relative_to(ROOT)}")


def write_executable_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    make_executable(path)


def make_executable(path: Path) -> None:
    if platform.system() != "Windows":
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    next_env = os.environ.copy()
    if env:
        next_env.update(env)
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=cwd, env=next_env, check=True)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
