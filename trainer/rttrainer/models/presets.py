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


PRESETS: dict[str, PresetConfig] = {
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
        raise ValueError(f"Unsupported PyTorch preset architecture: {config.architecture}")

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
            return self.dense(output)

    return LstmAudioModel()


def build_keras_model(config: PresetConfig, keras):
    if config.architecture != "lstm":
        raise ValueError(f"Unsupported Keras preset architecture: {config.architecture}")

    return keras.Sequential(
        [
            keras.Input(shape=(None, config.input_size), name="audio_input"),
            keras.layers.LSTM(
                config.hidden_size,
                return_sequences=True,
                name="lstm",
            ),
            keras.layers.Dense(config.output_size, name="dense"),
        ],
        name=config.preset_id,
    )
