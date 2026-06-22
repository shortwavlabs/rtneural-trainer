from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PresetConfig:
    preset_id: str
    architecture: str
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    kernel_size: int = 3
    conv_filters: int | None = None
    dense_units: int | None = None
    batchnorm: bool = False
    prelu: bool = False
    output_activation: str | None = "tanh"
    conv_dilations: tuple[int, ...] = ()
    default_loss: str = "mse"


PRESETS: dict[str, PresetConfig] = {
    "dense_only": PresetConfig(
        preset_id="dense_only",
        architecture="dense",
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=2,
        dense_units=8,
    ),
    "gru_light": PresetConfig(
        preset_id="gru_light",
        architecture="gru",
        input_size=1,
        hidden_size=10,
        output_size=1,
        num_layers=1,
    ),
    "lstm_light": PresetConfig(
        preset_id="lstm_light",
        architecture="lstm",
        input_size=1,
        hidden_size=12,
        output_size=1,
        num_layers=1,
    ),
    "lstm_standard": PresetConfig(
        preset_id="lstm_standard",
        architecture="lstm",
        input_size=1,
        hidden_size=16,
        output_size=1,
        num_layers=1,
    ),
    "conv1d_light": PresetConfig(
        preset_id="conv1d_light",
        architecture="conv1d",
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=1,
        kernel_size=3,
        conv_filters=8,
    ),
    "conv1d_bn_prelu": PresetConfig(
        preset_id="conv1d_bn_prelu",
        architecture="conv1d",
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=1,
        kernel_size=3,
        conv_filters=8,
        batchnorm=True,
        prelu=True,
    ),
    "conv1d_stack_prelu": PresetConfig(
        preset_id="conv1d_stack_prelu",
        architecture="conv1d",
        input_size=1,
        hidden_size=16,
        output_size=1,
        num_layers=4,
        kernel_size=5,
        conv_filters=16,
        prelu=True,
        conv_dilations=(1, 2, 4, 8),
        default_loss="preemphasis_mse",
    ),
    "wavenet_tcn": PresetConfig(
        preset_id="wavenet_tcn",
        architecture="conv1d",
        input_size=1,
        hidden_size=16,
        output_size=1,
        num_layers=8,
        kernel_size=3,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        default_loss="mrstft_preemphasis",
    ),
    "conv_gru_hybrid": PresetConfig(
        preset_id="conv_gru_hybrid",
        architecture="conv_gru",
        input_size=1,
        hidden_size=10,
        output_size=1,
        num_layers=2,
        kernel_size=3,
        conv_filters=6,
    ),
}


def get_preset(preset_id: str) -> PresetConfig:
    try:
        return PRESETS[preset_id]
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset_id}'. Known presets: {known}") from exc


def build_model(config: PresetConfig):
    import torch

    if config.architecture != "lstm":
        raise ValueError(
            f"PyTorch training/export currently supports only LSTM presets; "
            f"'{config.preset_id}' is a Keras-first {config.architecture} preset."
        )

    class LstmAudioModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
            )
            self.dense = torch.nn.Linear(config.hidden_size, config.output_size)

        def forward(self, x):  # type: ignore[no-untyped-def]
            output, _state = self.lstm(x)
            output = self.dense(output)
            if config.output_activation == "tanh":
                return torch.tanh(output)
            return output

    return LstmAudioModel()


def build_keras_model(config: PresetConfig, keras):
    layers = keras.layers
    model_layers = [keras.Input(shape=(None, config.input_size), name="audio_input")]

    if config.architecture == "dense":
        model_layers.extend(
            [
                layers.Dense(
                    config.dense_units or config.hidden_size,
                    activation="tanh",
                    name="dense_hidden",
                ),
                layers.Dense(
                    config.output_size,
                    activation=config.output_activation,
                    name="dense_out",
                ),
            ]
        )
    elif config.architecture == "gru":
        model_layers.extend(
            [
                layers.GRU(
                    config.hidden_size,
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    reset_after=True,
                    name="gru",
                ),
                layers.Dense(
                    config.output_size,
                    activation=config.output_activation,
                    name="dense_out",
                ),
            ]
        )
    elif config.architecture == "lstm":
        model_layers.extend(
            [
                layers.LSTM(
                    config.hidden_size,
                    return_sequences=True,
                    name="lstm",
                ),
                layers.Dense(
                    config.output_size,
                    activation=config.output_activation,
                    name="dense_out",
                ),
            ]
        )
    elif config.architecture == "conv1d":
        block_count = max(1, config.num_layers)
        dilations = config.conv_dilations or tuple(1 for _index in range(block_count))
        for block_index in range(block_count):
            suffix = "" if block_count == 1 else f"_{block_index + 1}"
            model_layers.append(
                layers.Conv1D(
                    config.conv_filters or config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation_rate=dilations[min(block_index, len(dilations) - 1)],
                    padding="causal",
                    activation=None if config.batchnorm or config.prelu else "tanh",
                    name=f"conv1d{suffix}",
                )
            )
            if config.batchnorm:
                model_layers.append(
                    layers.BatchNormalization(epsilon=0.01, name=f"batchnorm{suffix}")
                )
            if config.prelu:
                model_layers.append(layers.PReLU(shared_axes=[1], name=f"prelu{suffix}"))
        model_layers.append(
            layers.Dense(
                config.output_size,
                activation=config.output_activation,
                name="dense_out",
            )
        )
    elif config.architecture == "conv_gru":
        model_layers.extend(
            [
                layers.Conv1D(
                    config.conv_filters or 6,
                    kernel_size=config.kernel_size,
                    padding="causal",
                    activation="tanh",
                    name="conv1d",
                ),
                layers.GRU(
                    config.hidden_size,
                    return_sequences=True,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    reset_after=True,
                    name="gru",
                ),
                layers.Dense(
                    config.output_size,
                    activation=config.output_activation,
                    name="dense_out",
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported Keras preset architecture: {config.architecture}")

    return keras.Sequential(model_layers, name=config.preset_id)
