from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class LayerSpec:
    key: str
    rtneural_type: str
    status: str
    keras: str
    benchmarked: bool
    priority: str
    notes: str


@dataclass(frozen=True)
class ActivationSpec:
    key: str
    rtneural_name: str
    status: str
    keras: str
    benchmarked: bool
    priority: str
    notes: str


class LayerSpecPayload(TypedDict):
    key: str
    rtneural_type: str
    status: str
    keras: str
    benchmarked: bool
    priority: str
    notes: str


class ActivationSpecPayload(TypedDict):
    key: str
    rtneural_name: str
    status: str
    keras: str
    benchmarked: bool
    priority: str
    notes: str


class SupportMatrix(TypedDict):
    benchmark_sizes: list[int]
    benchmark_engines: list[str]
    layers: list[LayerSpecPayload]
    activations: list[ActivationSpecPayload]


BENCHMARK_SIZES = [4, 8, 16, 32, 64]
BENCHMARK_ENGINES = [
    "RTNeural compile-time",
    "RTNeural run-time",
    "onnxruntime",
    "TensorFlow Lite",
]

LAYER_SPECS: dict[str, LayerSpec] = {
    "dense": LayerSpec(
        key="dense",
        rtneural_type="dense",
        status="supported",
        keras="keras.layers.Dense",
        benchmarked=True,
        priority="v1",
        notes="Canonical feed-forward layer and benchmarked upstream.",
    ),
    "gru": LayerSpec(
        key="gru",
        rtneural_type="gru",
        status="supported",
        keras="keras.layers.GRU",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream; use conservative hidden sizes first.",
    ),
    "lstm": LayerSpec(
        key="lstm",
        rtneural_type="lstm",
        status="supported",
        keras="keras.layers.LSTM",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream and good default for amp/pedal capture.",
    ),
    "conv1d": LayerSpec(
        key="conv1d",
        rtneural_type="conv1d",
        status="supported",
        keras="keras.layers.Conv1D",
        benchmarked=True,
        priority="v1-plus",
        notes="Benchmarked upstream; add after recurrent export is boring.",
    ),
    "conv2d": LayerSpec(
        key="conv2d",
        rtneural_type="conv2d",
        status="supported",
        keras="keras.layers.Conv2D",
        benchmarked=False,
        priority="later",
        notes="Supported by RTNeural JSON exporter, but not in compare README plots.",
    ),
    "batchnorm1d": LayerSpec(
        key="batchnorm1d",
        rtneural_type="batchnorm",
        status="supported",
        keras="keras.layers.BatchNormalization",
        benchmarked=False,
        priority="v1-plus",
        notes="Safe 1D inference path covered by Keras parity and native validation fixtures.",
    ),
    "batchnorm2d": LayerSpec(
        key="batchnorm2d",
        rtneural_type="batchnorm2d",
        status="supported",
        keras="keras.layers.BatchNormalization",
        benchmarked=False,
        priority="later",
        notes="Export after Conv2D parity fixtures exist.",
    ),
    "prelu": LayerSpec(
        key="prelu",
        rtneural_type="prelu",
        status="supported",
        keras="keras.layers.PReLU",
        benchmarked=False,
        priority="v1-plus",
        notes="Safe shared temporal-axis PReLU is covered by Keras parity and native validation fixtures.",
    ),
    "maxpooling": LayerSpec(
        key="maxpooling",
        rtneural_type="maxpooling",
        status="unchecked",
        keras="keras.layers.MaxPooling1D/2D",
        benchmarked=False,
        priority="defer",
        notes="RTNeural README still marks MaxPooling unchecked.",
    ),
}

ACTIVATION_SPECS: dict[str, ActivationSpec] = {
    "tanh": ActivationSpec(
        key="tanh",
        rtneural_name="tanh",
        status="supported",
        keras="keras.activations.tanh",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "relu": ActivationSpec(
        key="relu",
        rtneural_name="relu",
        status="supported",
        keras="keras.activations.relu",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "sigmoid": ActivationSpec(
        key="sigmoid",
        rtneural_name="sigmoid",
        status="supported",
        keras="keras.activations.sigmoid",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "softmax": ActivationSpec(
        key="softmax",
        rtneural_name="softmax",
        status="supported",
        keras="keras.activations.softmax",
        benchmarked=False,
        priority="v1-plus",
        notes="Covered as a hidden activation fixture; not a common audio regression output.",
    ),
    "elu": ActivationSpec(
        key="elu",
        rtneural_name="elu",
        status="supported",
        keras="keras.activations.elu",
        benchmarked=False,
        priority="v1-plus",
        notes="Covered as a hidden activation fixture before UI exposure.",
    ),
    "prelu": ActivationSpec(
        key="prelu",
        rtneural_name="prelu",
        status="supported",
        keras="keras.layers.PReLU",
        benchmarked=False,
        priority="v1-plus",
        notes="Parametric activation represented as its own safe shared-axis layer fixture.",
    ),
}


def support_matrix() -> SupportMatrix:
    return {
        "benchmark_sizes": BENCHMARK_SIZES,
        "benchmark_engines": BENCHMARK_ENGINES,
        "layers": [layer_spec_payload(spec) for spec in LAYER_SPECS.values()],
        "activations": [
            activation_spec_payload(spec)
            for spec in ACTIVATION_SPECS.values()
        ],
    }


def layer_spec_payload(spec: LayerSpec) -> LayerSpecPayload:
    return {
        "key": spec.key,
        "rtneural_type": spec.rtneural_type,
        "status": spec.status,
        "keras": spec.keras,
        "benchmarked": spec.benchmarked,
        "priority": spec.priority,
        "notes": spec.notes,
    }


def activation_spec_payload(spec: ActivationSpec) -> ActivationSpecPayload:
    return {
        "key": spec.key,
        "rtneural_name": spec.rtneural_name,
        "status": spec.status,
        "keras": spec.keras,
        "benchmarked": spec.benchmarked,
        "priority": spec.priority,
        "notes": spec.notes,
    }
