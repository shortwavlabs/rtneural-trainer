from __future__ import annotations

import json
from json import JSONEncoder
from pathlib import Path
from typing import Any


class ArrayEncoder(JSONEncoder):
    def default(self, obj):  # type: ignore[no-untyped-def]
        if hasattr(obj, "numpy"):
            return obj.numpy().tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def save_keras_model_json(model, layers_to_skip: tuple[type, ...] | None = None) -> dict[str, Any]:
    keras = require_keras()
    if layers_to_skip is None:
        layers_to_skip = (keras.layers.InputLayer,)

    return {
        "in_shape": shape_to_json(getattr(model, "input_shape", None)),
        "layers": [
            serialize_keras_layer(layer, keras)
            for layer in model.layers
            if not isinstance(layer, layers_to_skip)
        ],
    }


def save_keras_model(model, filename: str | Path) -> None:
    payload = save_keras_model_json(model)
    with Path(filename).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, cls=ArrayEncoder, indent=2)
        handle.write("\n")


def serialize_keras_layer(layer, keras) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    layer_type = get_layer_type(layer, keras)
    payload: dict[str, Any] = {
        "type": layer_type,
        "activation": get_layer_activation(layer, keras),
        "shape": shape_to_json(get_layer_output_shape(layer)),
    }

    if layer_type == "conv1d":
        payload["kernel_size"] = tuple(layer.kernel_size)
        payload["dilation"] = tuple(layer.dilation_rate)
        payload["groups"] = int(getattr(layer, "groups", 1))
    elif layer_type == "conv2d":
        input_shape = shape_tuple(get_layer_input_shape(layer))
        output_shape = shape_tuple(get_layer_output_shape(layer))
        payload["kernel_size_time"] = int(layer.kernel_size[0])
        payload["kernel_size_feature"] = int(layer.kernel_size[1])
        payload["dilation"] = int(layer.dilation_rate[0])
        payload["strides"] = int(layer.strides[1])
        payload["num_filters_in"] = input_shape[3]
        payload["num_features_in"] = input_shape[2]
        payload["num_filters_out"] = output_shape[3]
        payload["padding"] = str(layer.padding).lower()
    elif layer_type == "batchnorm":
        payload["epsilon"] = float(layer.epsilon)
    elif layer_type == "batchnorm2d":
        input_shape = shape_tuple(get_layer_input_shape(layer))
        payload["epsilon"] = float(layer.epsilon)
        payload["num_filters_in"] = input_shape[3]
        payload["num_features_in"] = input_shape[2]

    payload["weights"] = layer.get_weights()
    return payload


def get_layer_type(layer, keras) -> str:  # type: ignore[no-untyped-def]
    if isinstance(layer, keras.layers.TimeDistributed):
        return "time-distributed-dense"
    if isinstance(layer, keras.layers.GRU):
        return "gru"
    if isinstance(layer, keras.layers.LSTM):
        return "lstm"
    if isinstance(layer, keras.layers.Dense):
        return "dense"
    if isinstance(layer, keras.layers.Conv1D):
        return "conv1d"
    if isinstance(layer, keras.layers.Conv2D):
        return "conv2d"
    if isinstance(layer, keras.layers.PReLU):
        return "prelu"
    if isinstance(layer, keras.layers.BatchNormalization):
        if len(shape_tuple(get_layer_input_shape(layer))) == 4:
            return "batchnorm2d"
        return "batchnorm"
    if isinstance(layer, keras.layers.Activation):
        return "activation"
    return "unknown"


def get_layer_activation(layer, keras) -> str:  # type: ignore[no-untyped-def]
    if isinstance(layer, keras.layers.TimeDistributed):
        return get_layer_activation(layer.layer, keras)
    activation = getattr(layer, "activation", None)
    activations = {
        keras.activations.tanh: "tanh",
        keras.activations.relu: "relu",
        keras.activations.sigmoid: "sigmoid",
        keras.activations.softmax: "softmax",
        keras.activations.elu: "elu",
    }
    return activations.get(activation, "")


def get_layer_output_shape(layer) -> Any:  # type: ignore[no-untyped-def]
    return getattr(layer, "output_shape", None) or getattr(layer.output, "shape", None)


def get_layer_input_shape(layer) -> Any:  # type: ignore[no-untyped-def]
    return getattr(layer, "input_shape", None) or getattr(layer.input, "shape", None)


def shape_to_json(shape: Any) -> list[int | None] | tuple[Any, ...] | None:
    if shape is None:
        return None
    return list(shape_tuple(shape))


def shape_tuple(shape: Any) -> tuple[Any, ...]:
    if hasattr(shape, "as_list"):
        return tuple(shape.as_list())
    if isinstance(shape, (tuple, list)):
        return tuple(shape)
    return tuple(shape)


def require_keras():
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for the canonical Keras exporter. "
            "Install it with: uv sync --extra tensorflow"
        ) from exc
    return tf.keras
