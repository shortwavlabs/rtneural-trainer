#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

from rttrainer.export_rtneural.registry import SupportMatrix, support_matrix  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Print RTNeural support matrix.")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args()
    matrix = support_matrix()
    if args.format == "json":
        print(json.dumps(matrix, indent=2))
    else:
        print_markdown(matrix)
    return 0


def print_markdown(matrix: SupportMatrix) -> None:
    print("# RTNeural Layer Support Matrix\n")
    print("Benchmark sizes from RTNeural-compare:", ", ".join(map(str, matrix["benchmark_sizes"])))
    print("\n## Layers\n")
    print("| Key | RTNeural type | Status | Keras | Benchmarked | Priority |")
    print("| --- | --- | --- | --- | --- | --- |")
    for layer in matrix["layers"]:
        print(
            f"| `{layer['key']}` | `{layer['rtneural_type']}` | {layer['status']} | "
            f"`{layer['keras']}` | {yes_no(layer['benchmarked'])} | {layer['priority']} |"
        )
    print("\n## Activations\n")
    print("| Key | RTNeural name | Status | Keras | Benchmarked | Priority |")
    print("| --- | --- | --- | --- | --- | --- |")
    for activation in matrix["activations"]:
        print(
            f"| `{activation['key']}` | `{activation['rtneural_name']}` | {activation['status']} | "
            f"`{activation['keras']}` | {yes_no(activation['benchmarked'])} | {activation['priority']} |"
        )


def yes_no(value: object) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
