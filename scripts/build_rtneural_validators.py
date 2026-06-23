#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_DIR = ROOT / "native" / "rtneural-validator"
VALIDATOR_BACKENDS = ("stl", "eigen", "xsimd")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build rtneural-validator variants for backend benchmark comparisons."
    )
    parser.add_argument(
        "--backends",
        default="eigen,xsimd,stl",
        help="Comma-separated RTNeural backends to build.",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        help="CMake build type.",
    )
    parser.add_argument(
        "--avx",
        action="store_true",
        help="Also build AVX-enabled variants for selected backends.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of skipping unavailable optional backends.",
    )
    args = parser.parse_args()

    backends = parse_backends(args.backends)
    for backend in backends:
        if not backend_available(backend):
            message = (
                f"Skipping {backend}: required vendored headers are not present in "
                f"{rtneural_source_dir()}."
            )
            if args.strict:
                raise SystemExit(message)
            print(message)
            continue
        build_validator(backend, args.build_type, avx=False)
        if args.avx:
            build_validator(backend, args.build_type, avx=True)
    return 0


def parse_backends(value: str) -> list[str]:
    backends = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not backends:
        raise SystemExit("At least one backend is required.")
    invalid = sorted(set(backends) - set(VALIDATOR_BACKENDS))
    if invalid:
        raise SystemExit(
            f"Unsupported backend(s): {', '.join(invalid)}. "
            f"Use: {', '.join(VALIDATOR_BACKENDS)}"
        )
    return backends


def backend_available(backend: str) -> bool:
    if backend != "xsimd":
        return True
    header = rtneural_source_dir() / "modules" / "xsimd" / "include" / "xsimd" / "xsimd.hpp"
    return header.is_file()


def rtneural_source_dir() -> Path:
    import os

    return Path(
        os.environ.get(
            "RTNEURAL_LOCAL_PATH",
            "/Users/shortwavlabs/Workspace/rt-neural/RTNeural",
        )
    ).expanduser()


def build_validator(backend: str, build_type: str, *, avx: bool) -> None:
    suffix = f"{backend}-avx" if avx else backend
    build_dir = VALIDATOR_DIR / f"build-{suffix}"
    configure = [
        "cmake",
        "-S",
        str(VALIDATOR_DIR),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DRTNEURAL_VALIDATOR_BACKEND={backend}",
    ]
    if avx:
        configure.append("-DRTNEURAL_USE_AVX=ON")

    run(configure)
    run(["cmake", "--build", str(build_dir), "--config", build_type])
    print(f"built {backend}{' + AVX' if avx else ''}: {build_dir.relative_to(ROOT)}")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
