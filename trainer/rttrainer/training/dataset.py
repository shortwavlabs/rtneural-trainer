from __future__ import annotations

import random
from dataclasses import dataclass

from rttrainer.data.audio_io import read_wav_mono


@dataclass(frozen=True)
class WindowedDataset:
    train_x: object
    train_y: object
    val_x: object
    val_y: object
    test_input: list[float]
    test_target: list[float]
    sample_rate: int


def build_windowed_dataset(
    input_path,
    target_path,
    sequence_length: int,
    max_windows: int,
    seed: int,
    backend: str = "torch",
) -> WindowedDataset:
    input_audio = read_wav_mono(input_path)
    target_audio = read_wav_mono(target_path)
    if input_audio.sample_rate != target_audio.sample_rate:
        raise ValueError("Prepared input and target sample rates differ.")

    length = min(len(input_audio.samples), len(target_audio.samples))
    if length < sequence_length * 4:
        raise ValueError(
            f"Need at least {sequence_length * 4} samples for train/val/test windows."
        )

    input_samples = input_audio.samples[:length]
    target_samples = target_audio.samples[:length]
    windows_x: list[list[float]] = []
    windows_y: list[list[float]] = []
    stride = max(1, sequence_length // 2)
    for start in range(0, length - sequence_length + 1, stride):
        windows_x.append(input_samples[start : start + sequence_length])
        windows_y.append(target_samples[start : start + sequence_length])
        if len(windows_x) >= max_windows:
            break

    if len(windows_x) < 4:
        raise ValueError("Not enough training windows after slicing.")

    permutation = list(range(len(windows_x)))
    random.Random(seed).shuffle(permutation)
    windows_x = [windows_x[index] for index in permutation]
    windows_y = [windows_y[index] for index in permutation]

    train_count = max(1, int(len(windows_x) * 0.8))
    val_count = max(1, int(len(windows_x) * 0.1))
    if train_count + val_count >= len(windows_x):
        train_count = len(windows_x) - 2
        val_count = 1

    train_x = make_backend_array(backend, windows_x[:train_count])
    train_y = make_backend_array(backend, windows_y[:train_count])
    val_x = make_backend_array(backend, windows_x[train_count : train_count + val_count])
    val_y = make_backend_array(backend, windows_y[train_count : train_count + val_count])
    test_start = (train_count + val_count) * stride
    test_end = min(length, test_start + sequence_length * 4)
    if test_end - test_start < sequence_length:
        test_start = max(0, length - sequence_length * 4)
        test_end = length

    return WindowedDataset(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_input=input_samples[test_start:test_end],
        test_target=target_samples[test_start:test_end],
        sample_rate=input_audio.sample_rate,
    )


def make_backend_array(backend: str, windows: list[list[float]]):
    if backend == "numpy":
        numpy = __import__("numpy")
        return numpy.asarray(windows, dtype="float32")[..., None]
    if backend == "torch":
        torch = __import__("torch")
        return make_tensor(torch, windows)
    if backend == "list":
        return [[[sample] for sample in window] for window in windows]
    raise ValueError(f"Unknown dataset backend: {backend}")


def make_tensor(torch, windows: list[list[float]]):  # type: ignore[no-untyped-def]
    return torch.tensor(windows, dtype=torch.float32).unsqueeze(-1)
