#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "trainer"
sys.path.insert(0, str(TRAINER))

from rttrainer.export_rtneural.keras_exporter import save_keras_model  # noqa: E402
from rttrainer.export_rtneural.registry import ACTIVATION_SPECS, LAYER_SPECS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Keras Sequential fixtures in RTNeural JSON format."
    )
    parser.add_argument("--out", default="fixtures/rtneural-json")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--include-later", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    selected_layers = [
        key for key, spec in LAYER_SPECS.items()
        if spec.status == "supported" and (args.include_later or spec.priority in {"v1", "v1-plus"})
    ]
    selected_activations = [
        key for key, spec in ACTIVATION_SPECS.items()
        if spec.status == "supported" and (args.include_later or spec.priority == "v1")
    ]

    if args.list:
        print("layers:", ", ".join(selected_layers))
        print("activations:", ", ".join(selected_activations))
        return 0

    keras = require_keras()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        model = build_layer_model(keras, layer, args.size)
        save_keras_model(model, out_dir / f"{layer}_{args.size}.json")
    for activation in selected_activations:
        model = build_activation_model(keras, activation, args.size)
        save_keras_model(model, out_dir / f"activation_{activation}_{args.size}.json")
    print(f"Wrote fixtures to {out_dir}")
    return 0


def build_layer_model(keras, layer: str, size: int):  # type: ignore[no-untyped-def]
    if layer == "dense":
        return keras.Sequential(
            [keras.layers.Input(shape=(size,)), keras.layers.Dense(size, activation="linear")]
        )
    if layer == "gru":
        return keras.Sequential(
            [keras.layers.Input(shape=(None, size)), keras.layers.GRU(size)]
        )
    if layer == "lstm":
        return keras.Sequential(
            [keras.layers.Input(shape=(None, size)), keras.layers.LSTM(size)]
        )
    if layer == "conv1d":
        return keras.Sequential(
            [
                keras.layers.Input(shape=(None, 1)),
                keras.layers.Conv1D(size, kernel_size=max(1, size - 1), activation="linear"),
            ]
        )
    if layer == "conv2d":
        return keras.Sequential(
            [
                keras.layers.Input(shape=(size, size, 1)),
                keras.layers.Conv2D(size, kernel_size=(3, 3), activation="linear"),
            ]
        )
    if layer == "batchnorm1d":
        return keras.Sequential(
            [keras.layers.Input(shape=(None, size)), keras.layers.BatchNormalization()]
        )
    if layer == "batchnorm2d":
        return keras.Sequential(
            [keras.layers.Input(shape=(size, size, 1)), keras.layers.BatchNormalization()]
        )
    if layer == "prelu":
        return keras.Sequential(
            [keras.layers.Input(shape=(None, size)), keras.layers.PReLU()]
        )
    raise ValueError(f"No fixture builder for layer: {layer}")


def build_activation_model(keras, activation: str, size: int):  # type: ignore[no-untyped-def]
    if activation == "prelu":
        return keras.Sequential(
            [keras.layers.Input(shape=(None, size)), keras.layers.PReLU()]
        )
    return keras.Sequential(
        [
            keras.layers.Input(shape=(None, size)),
            keras.layers.Activation(activation),
        ]
    )


def require_keras():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required to generate Keras fixtures. "
            "Install with: cd trainer && uv sync --extra tensorflow"
        ) from exc
    return tf.keras


if __name__ == "__main__":
    raise SystemExit(main())
