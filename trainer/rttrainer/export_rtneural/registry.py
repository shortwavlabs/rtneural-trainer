from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSpec:
    key: str
    rtneural_type: str
    status: str
    keras: str
    pytorch: str
    benchmarked: bool
    priority: str
    notes: str


@dataclass(frozen=True)
class ActivationSpec:
    key: str
    rtneural_name: str
    status: str
    keras: str
    pytorch: str
    benchmarked: bool
    priority: str
    notes: str


BENCHMARK_SIZES = [4, 8, 16, 32, 64]
BENCHMARK_ENGINES = [
    "RTNeural compile-time",
    "RTNeural run-time",
    "libtorch",
    "onnxruntime",
    "TensorFlow Lite",
]

LAYER_SPECS: dict[str, LayerSpec] = {
    "dense": LayerSpec(
        key="dense",
        rtneural_type="dense",
        status="supported",
        keras="keras.layers.Dense",
        pytorch="torch.nn.Linear",
        benchmarked=True,
        priority="v1",
        notes="Canonical feed-forward layer and benchmarked upstream.",
    ),
    "gru": LayerSpec(
        key="gru",
        rtneural_type="gru",
        status="supported",
        keras="keras.layers.GRU",
        pytorch="torch.nn.GRU",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream; use conservative hidden sizes first.",
    ),
    "lstm": LayerSpec(
        key="lstm",
        rtneural_type="lstm",
        status="supported",
        keras="keras.layers.LSTM",
        pytorch="torch.nn.LSTM",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream and good default for amp/pedal capture.",
    ),
    "conv1d": LayerSpec(
        key="conv1d",
        rtneural_type="conv1d",
        status="supported",
        keras="keras.layers.Conv1D",
        pytorch="torch.nn.Conv1d",
        benchmarked=True,
        priority="v1-plus",
        notes="Benchmarked upstream; add after recurrent export is boring.",
    ),
    "conv2d": LayerSpec(
        key="conv2d",
        rtneural_type="conv2d",
        status="supported",
        keras="keras.layers.Conv2D",
        pytorch="torch.nn.Conv2d",
        benchmarked=False,
        priority="later",
        notes="Supported by RTNeural JSON exporter, but not in compare README plots.",
    ),
    "batchnorm1d": LayerSpec(
        key="batchnorm1d",
        rtneural_type="batchnorm",
        status="supported",
        keras="keras.layers.BatchNormalization",
        pytorch="torch.nn.BatchNorm1d",
        benchmarked=False,
        priority="later",
        notes="Export after parity fixtures exist for running statistics.",
    ),
    "batchnorm2d": LayerSpec(
        key="batchnorm2d",
        rtneural_type="batchnorm2d",
        status="supported",
        keras="keras.layers.BatchNormalization",
        pytorch="torch.nn.BatchNorm2d",
        benchmarked=False,
        priority="later",
        notes="Export after Conv2D parity fixtures exist.",
    ),
    "prelu": LayerSpec(
        key="prelu",
        rtneural_type="prelu",
        status="supported",
        keras="keras.layers.PReLU",
        pytorch="torch.nn.PReLU",
        benchmarked=False,
        priority="later",
        notes="Supported activation layer; keep behind fixture tests.",
    ),
    "maxpooling": LayerSpec(
        key="maxpooling",
        rtneural_type="maxpooling",
        status="unchecked",
        keras="keras.layers.MaxPooling1D/2D",
        pytorch="torch.nn.MaxPool1d/2d",
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
        pytorch="torch.tanh / torch.nn.Tanh",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "relu": ActivationSpec(
        key="relu",
        rtneural_name="relu",
        status="supported",
        keras="keras.activations.relu",
        pytorch="torch.relu / torch.nn.ReLU",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "sigmoid": ActivationSpec(
        key="sigmoid",
        rtneural_name="sigmoid",
        status="supported",
        keras="keras.activations.sigmoid",
        pytorch="torch.sigmoid / torch.nn.Sigmoid",
        benchmarked=True,
        priority="v1",
        notes="Benchmarked upstream.",
    ),
    "softmax": ActivationSpec(
        key="softmax",
        rtneural_name="softmax",
        status="supported",
        keras="keras.activations.softmax",
        pytorch="torch.nn.Softmax",
        benchmarked=False,
        priority="later",
        notes="Supported by RTNeural; not a common audio regression output.",
    ),
    "elu": ActivationSpec(
        key="elu",
        rtneural_name="elu",
        status="supported",
        keras="keras.activations.elu",
        pytorch="torch.nn.ELU",
        benchmarked=False,
        priority="later",
        notes="Supported by RTNeural; add parity fixtures before UI exposure.",
    ),
    "prelu": ActivationSpec(
        key="prelu",
        rtneural_name="prelu",
        status="supported",
        keras="keras.layers.PReLU",
        pytorch="torch.nn.PReLU",
        benchmarked=False,
        priority="later",
        notes="Parametric activation represented as its own layer.",
    ),
}


def support_matrix() -> dict[str, object]:
    return {
        "benchmark_sizes": BENCHMARK_SIZES,
        "benchmark_engines": BENCHMARK_ENGINES,
        "layers": [spec.__dict__ for spec in LAYER_SPECS.values()],
        "activations": [spec.__dict__ for spec in ACTIVATION_SPECS.values()],
    }
