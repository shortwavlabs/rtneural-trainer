#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
DEFAULT_OUT = ROOT / "fixtures" / "rtneural-json" / "golden"
sys.path.insert(0, str(TRAINER))

from rttrainer.export_rtneural.json_exporter import build_keras_rtneural_json  # noqa: E402
from rttrainer.models.presets import PRESETS, build_keras_model  # noqa: E402


@dataclass(frozen=True)
class GoldenFixture:
    preset_id: str
    model: Any
    payload: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify deterministic golden RTNeural JSON fixtures."
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    fixtures = build_all_fixtures()
    if args.check:
        verify_fixtures(out_dir, fixtures)
        print(f"golden fixtures verified: {out_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for preset_id, payload in fixtures.items():
        fixture_path = out_dir / f"{preset_id}.rtneural.json"
        fixture_path.write_text(canonical_json(payload), encoding="utf-8")
        print(f"wrote {fixture_path.relative_to(ROOT)}")
    return 0


def verify_fixtures(out_dir: Path, fixtures: dict[str, dict[str, Any]]) -> None:
    missing = [
        preset_id
        for preset_id in sorted(fixtures)
        if not (out_dir / f"{preset_id}.rtneural.json").is_file()
    ]
    if missing:
        raise SystemExit(f"Missing golden fixtures: {', '.join(missing)}")

    for preset_id, payload in fixtures.items():
        fixture_path = out_dir / f"{preset_id}.rtneural.json"
        expected = canonical_json(payload)
        actual = fixture_path.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(
                f"Golden fixture is stale: {fixture_path}. "
                "Regenerate with scripts/generate_golden_rtneural_fixtures.py"
            )


def build_all_fixtures() -> dict[str, dict[str, Any]]:
    return {
        preset_id: fixture.payload
        for preset_id, fixture in build_fixture_models().items()
    }


def build_fixture_models() -> dict[str, GoldenFixture]:
    tf = require_tensorflow()
    fixtures: dict[str, GoldenFixture] = {}
    for index, preset_id in enumerate(sorted(PRESETS)):
        model = build_keras_model(PRESETS[preset_id], tf.keras)
        initialize_model(model)
        assign_deterministic_weights(model, seed=10 + index)
        payload = build_keras_rtneural_json(
            model=model,
            preset_id=preset_id,
            sample_rate=48_000,
            latency_samples=0,
            checkpoint_metrics={
                "esr": 0.0,
                "mae": 0.0,
                "rmse": 0.0,
                "peak_residual": 0.0,
                "rms_residual": 0.0,
                "realtime_factor": 0.0,
            },
        )
        fixtures[preset_id] = GoldenFixture(
            preset_id=preset_id,
            model=model,
            payload=payload,
        )
    return fixtures


def initialize_model(model) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    model(np.zeros((1, 8, 1), dtype=np.float32), training=False)


def assign_deterministic_weights(model, seed: int) -> None:  # type: ignore[no-untyped-def]
    import numpy as np

    for layer_index, layer in enumerate(model.layers):
        class_name = layer.__class__.__name__
        if class_name == "BatchNormalization":
            channels = int(layer.gamma.shape[0])
            gamma = np.linspace(0.82, 1.18, channels, dtype=np.float32)
            beta = np.linspace(-0.025, 0.025, channels, dtype=np.float32)
            mean = np.linspace(-0.04, 0.04, channels, dtype=np.float32)
            variance = np.linspace(0.70, 1.30, channels, dtype=np.float32)
            layer.set_weights([gamma, beta, mean, variance])
            continue
        if class_name == "PReLU":
            alpha = layer.get_weights()[0]
            layer.set_weights([np.full(alpha.shape, 0.18, dtype=np.float32)])
            continue

        weights = layer.get_weights()
        next_weights = []
        for weight_index, weight in enumerate(weights):
            count = int(np.prod(weight.shape))
            if count == 0:
                next_weights.append(weight)
                continue
            scale = 0.018 + 0.004 * ((seed + layer_index + weight_index) % 5)
            values = np.linspace(-scale, scale, count, dtype=np.float32).reshape(weight.shape)
            if "bias" in getattr(layer, "name", "").lower() or weight.ndim == 1:
                values = values * 0.25
            next_weights.append(values)
        if next_weights:
            layer.set_weights(next_weights)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def require_tensorflow():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to generate golden RTNeural fixtures. "
            "Run with: cd trainer && UV_CACHE_DIR=../.uv-cache "
            "uv run --extra tensorflow python ../scripts/generate_golden_rtneural_fixtures.py"
        ) from exc
    return tf


if __name__ == "__main__":
    raise SystemExit(main())
