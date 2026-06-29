from __future__ import annotations

from dataclasses import dataclass

SCALED_TANH_ALPHAS = (1.5, 1.8, 2.2)


@dataclass(frozen=True)
class PresetConfig:
    preset_id: str
    architecture: str
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    kernel_size: int = 3
    conv_kernel_sizes: tuple[int, ...] = ()
    conv_filters: int | None = None
    dense_units: int | None = None
    batchnorm: bool = False
    prelu: bool = False
    conv_activation: str = "tanh"
    conv_activation_alpha: float = 1.0
    output_activation: str | None = "tanh"
    conv_dilations: tuple[int, ...] = ()
    default_loss: str = "mse"
    default_learning_rate: float = 1.0e-3


def wavenet_tcn_preset(
    preset_id: str,
    *,
    hidden_size: int,
    num_layers: int,
    conv_filters: int,
    conv_dilations: tuple[int, ...],
    conv_activation_alpha: float = 1.0,
    default_learning_rate: float = 1.0e-3,
) -> PresetConfig:
    return PresetConfig(
        preset_id=preset_id,
        architecture="conv1d",
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers,
        kernel_size=3,
        conv_filters=conv_filters,
        conv_dilations=conv_dilations,
        conv_activation_alpha=conv_activation_alpha,
        default_loss="mrstft_preemphasis",
        default_learning_rate=default_learning_rate,
    )


def wavenet_tcn_separable_preset(
    preset_id: str,
    *,
    hidden_size: int,
    num_layers: int,
    conv_filters: int,
    conv_dilations: tuple[int, ...],
) -> PresetConfig:
    return PresetConfig(
        preset_id=preset_id,
        architecture="separable_conv1d",
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        num_layers=num_layers,
        kernel_size=3,
        conv_filters=conv_filters,
        conv_dilations=conv_dilations,
        default_loss="mrstft_preemphasis",
    )


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
    "wavenet_tcn_fast": wavenet_tcn_preset(
        "wavenet_tcn_fast",
        hidden_size=12,
        num_layers=6,
        conv_filters=12,
        conv_dilations=(1, 2, 4, 8, 16, 32),
    ),
    "wavenet_tcn_clean": PresetConfig(
        preset_id="wavenet_tcn_clean",
        architecture="conv1d",
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=10,
        kernel_size=7,
        conv_filters=8,
        conv_activation="linear",
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        default_loss="preemphasis_mse",
        default_learning_rate=2.0e-4,
    ),
    "wavenet_tcn_edge": PresetConfig(
        preset_id="wavenet_tcn_edge",
        architecture="conv1d",
        input_size=1,
        hidden_size=8,
        output_size=1,
        num_layers=10,
        kernel_size=7,
        conv_filters=8,
        conv_activation_alpha=1.8,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        default_loss="preemphasis_mse",
        default_learning_rate=1.5e-4,
    ),
    "wavenet_tcn_edge_detail": PresetConfig(
        preset_id="wavenet_tcn_edge_detail",
        architecture="conv1d",
        input_size=1,
        hidden_size=12,
        output_size=1,
        num_layers=10,
        kernel_size=7,
        conv_filters=12,
        conv_activation_alpha=2.2,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        default_loss="preemphasis_mse",
        default_learning_rate=1.2e-4,
    ),
    "wavenet_tcn": wavenet_tcn_preset(
        "wavenet_tcn",
        hidden_size=16,
        num_layers=8,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    "wavenet_tcn_balanced": wavenet_tcn_preset(
        "wavenet_tcn_balanced",
        hidden_size=16,
        num_layers=8,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
    ),
    "wavenet_tcn_balanced_tanh15": wavenet_tcn_preset(
        "wavenet_tcn_balanced_tanh15",
        hidden_size=16,
        num_layers=8,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        conv_activation_alpha=1.5,
    ),
    "wavenet_tcn_balanced_tanh18": wavenet_tcn_preset(
        "wavenet_tcn_balanced_tanh18",
        hidden_size=16,
        num_layers=8,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        conv_activation_alpha=1.8,
    ),
    "wavenet_tcn_quality": wavenet_tcn_preset(
        "wavenet_tcn_quality",
        hidden_size=20,
        num_layers=10,
        conv_filters=20,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    ),
    "wavenet_tcn_compressor": PresetConfig(
        preset_id="wavenet_tcn_compressor",
        architecture="conv1d",
        input_size=1,
        hidden_size=20,
        output_size=1,
        num_layers=10,
        kernel_size=3,
        conv_filters=20,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        default_loss="compressor_envelope_mrstft",
        default_learning_rate=5.0e-4,
    ),
    "wavenet_tcn_quality_tanh15": wavenet_tcn_preset(
        "wavenet_tcn_quality_tanh15",
        hidden_size=20,
        num_layers=10,
        conv_filters=20,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        conv_activation_alpha=1.5,
    ),
    "wavenet_tcn_high_gain": wavenet_tcn_preset(
        "wavenet_tcn_high_gain",
        hidden_size=20,
        num_layers=11,
        conv_filters=20,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        default_learning_rate=3.5e-4,
    ),
    "wavenet_tcn_quality_tanh18": wavenet_tcn_preset(
        "wavenet_tcn_quality_tanh18",
        hidden_size=20,
        num_layers=10,
        conv_filters=20,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
        conv_activation_alpha=1.8,
    ),
    "wavenet_tcn_a2_prelu": PresetConfig(
        preset_id="wavenet_tcn_a2_prelu",
        architecture="conv1d",
        input_size=1,
        hidden_size=16,
        output_size=1,
        num_layers=12,
        kernel_size=6,
        conv_kernel_sizes=(6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6),
        conv_filters=16,
        prelu=True,
        conv_dilations=(1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41),
        default_loss="mrstft_preemphasis",
        default_learning_rate=3.5e-4,
    ),
    "wavenet_tcn_separable_fast": wavenet_tcn_separable_preset(
        "wavenet_tcn_separable_fast",
        hidden_size=16,
        num_layers=8,
        conv_filters=16,
        conv_dilations=(1, 2, 4, 8, 16, 32, 64, 128),
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


def build_keras_model(config: PresetConfig, keras):
    layers = keras.layers
    model_layers = [keras.Input(shape=(None, config.input_size), name="audio_input")]
    conv_activation = keras_conv_activation(config, keras)

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
        kernel_sizes = config.conv_kernel_sizes or tuple(
            config.kernel_size for _index in range(block_count)
        )
        for block_index in range(block_count):
            suffix = "" if block_count == 1 else f"_{block_index + 1}"
            model_layers.append(
                layers.Conv1D(
                    config.conv_filters or config.hidden_size,
                    kernel_size=kernel_sizes[min(block_index, len(kernel_sizes) - 1)],
                    dilation_rate=dilations[min(block_index, len(dilations) - 1)],
                    padding="causal",
                    activation=None if config.batchnorm or config.prelu else conv_activation,
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
    elif config.architecture == "separable_conv1d":
        block_count = max(1, config.num_layers)
        filters = config.conv_filters or config.hidden_size
        dilations = config.conv_dilations or tuple(1 for _index in range(block_count))
        for block_index in range(block_count):
            suffix = "" if block_count == 1 else f"_{block_index + 1}"
            dilation = dilations[min(block_index, len(dilations) - 1)]
            if block_index == 0:
                model_layers.append(
                    layers.Conv1D(
                        filters,
                        kernel_size=config.kernel_size,
                        dilation_rate=dilation,
                        padding="causal",
                        activation=conv_activation,
                        name=f"conv1d_expand{suffix}",
                    )
                )
                continue

            model_layers.extend(
                [
                    layers.Conv1D(
                        filters,
                        kernel_size=config.kernel_size,
                        dilation_rate=dilation,
                        padding="causal",
                        groups=filters,
                        activation=None,
                        name=f"depthwise_conv1d{suffix}",
                    ),
                    layers.Conv1D(
                        filters,
                        kernel_size=1,
                        dilation_rate=1,
                        padding="causal",
                        activation=conv_activation,
                        name=f"pointwise_conv1d{suffix}",
                    ),
                ]
            )
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


def keras_conv_activation(config: PresetConfig, keras):
    if config.conv_activation != "tanh":
        return config.conv_activation
    if config.conv_activation_alpha == 1.0:
        return "tanh"
    return scaled_tanh_activation(keras, config.conv_activation_alpha)


def scaled_tanh_activation(keras, alpha: float):  # type: ignore[no-untyped-def]
    alpha = float(alpha)

    def scaled_tanh(value):  # type: ignore[no-untyped-def]
        return keras.activations.tanh(value / alpha)

    scaled_tanh.__name__ = scaled_tanh_name(alpha)
    return scaled_tanh


def scaled_tanh_custom_objects(keras) -> dict[str, object]:  # type: ignore[no-untyped-def]
    return {
        scaled_tanh_name(alpha): scaled_tanh_activation(keras, alpha)
        for alpha in SCALED_TANH_ALPHAS
    }


def scaled_tanh_name(alpha: float) -> str:
    return f"scaled_tanh_{str(float(alpha)).replace('.', '_')}"
