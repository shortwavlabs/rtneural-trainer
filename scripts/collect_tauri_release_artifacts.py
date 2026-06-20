#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "app" / "src-tauri" / "target" / "release" / "bundle"
DEFAULT_MANIFEST = ROOT / "app" / "src-tauri" / "target" / "release" / "release-artifacts-manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Tauri release bundle artifacts into a JSON manifest."
    )
    parser.add_argument("--bundle-dir", default=str(BUNDLE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    output = Path(args.output)
    artifacts = collect_artifacts(bundle_dir)
    if not artifacts:
        raise FileNotFoundError(f"No Tauri release artifacts found under {bundle_dir}")

    manifest = {
        "schema_version": 1,
        "bundle_dir": relative_or_absolute(bundle_dir),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"release artifact manifest: {relative_or_absolute(output)}")
    return 0


def collect_artifacts(bundle_dir: Path) -> list[dict[str, object]]:
    if not bundle_dir.exists():
        return []
    artifacts: list[dict[str, object]] = []
    for path in sorted(bundle_dir.rglob("*")):
        if path.is_dir() and path.suffix == ".app":
            artifacts.append(
                {
                    "path": relative_or_absolute(path),
                    "kind": "bundle-directory",
                    "size_bytes": directory_size(path),
                }
            )
            continue
        if not path.is_file():
            continue
        if is_inside_app_bundle(path):
            continue
        artifacts.append(
            {
                "path": relative_or_absolute(path),
                "kind": "file",
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


def is_inside_app_bundle(path: Path) -> bool:
    return any(parent.suffix == ".app" for parent in path.parents)


def directory_size(path: Path) -> int:
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
