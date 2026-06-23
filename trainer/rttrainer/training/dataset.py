from __future__ import annotations

import random
from dataclasses import dataclass

from rttrainer.data.audio_io import read_wav_mono

DEFAULT_PREVIEW_SECONDS = 3.0
MIN_PREVIEW_SEQUENCE_MULTIPLIER = 4


@dataclass(frozen=True)
class WindowedDataset:
    train_x: object
    train_y: object
    val_x: object
    val_y: object
    context_train_input: list[float]
    context_train_target: list[float]
    stream_val_input: list[float]
    stream_val_target: list[float]
    test_input: list[float]
    test_target: list[float]
    sample_rate: int
    summary: dict[str, int | float | str]
    input_samples: list[float]
    target_samples: list[float]
    sequence_length: int
    backend: str
    available_starts: tuple[int, ...]
    train_starts: tuple[int, ...]
    val_starts: tuple[int, ...]


def build_windowed_dataset(
    input_path,
    target_path,
    sequence_length: int,
    max_windows: int,
    seed: int,
    backend: str = "torch",
    preview_seconds: float = DEFAULT_PREVIEW_SECONDS,
    context_multiplier: int = MIN_PREVIEW_SEQUENCE_MULTIPLIER,
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
    stride = max(1, sequence_length // 2)
    starts = list(range(0, length - sequence_length + 1, stride))
    total_windows = len(starts)
    window_budget = max(4, max_windows)
    minimum_preview_samples = sequence_length * MIN_PREVIEW_SEQUENCE_MULTIPLIER
    requested_preview_samples = int(
        round(input_audio.sample_rate * max(0.1, preview_seconds))
    )
    preview_samples = max(minimum_preview_samples, requested_preview_samples)
    excerpt_ranges = choose_active_excerpts(
        target_samples,
        excerpt_length=preview_samples,
        stride=stride,
        count=2,
    )
    stream_val_start, stream_val_end = excerpt_ranges[0]
    test_start, test_end = excerpt_ranges[-1]
    reserved_ranges = list(excerpt_ranges)
    context_samples = min(length, sequence_length * max(1, context_multiplier))
    context_start, context_end = choose_active_excerpt(
        target_samples,
        excerpt_length=context_samples,
        stride=stride,
        reserved_ranges=reserved_ranges,
    )

    available_starts = [
        start
        for start in starts
        if not overlaps_reserved_ranges(
            start,
            start + sequence_length,
            reserved_ranges,
        )
    ]
    if len(available_starts) < 4:
        available_starts = starts
        reserved_ranges = []

    selected_starts, energy_selected_count, random_selected_count = select_training_starts(
        target_samples,
        available_starts,
        sequence_length=sequence_length,
        window_budget=window_budget,
        seed=seed,
    )

    if len(selected_starts) < 4:
        raise ValueError("Not enough training windows after slicing.")

    train_starts, val_starts = split_training_validation_starts(selected_starts, seed)
    train_x, train_y = make_window_arrays(
        backend,
        input_samples,
        target_samples,
        train_starts,
        sequence_length,
    )
    val_x, val_y = make_window_arrays(
        backend,
        input_samples,
        target_samples,
        val_starts,
        sequence_length,
    )

    test_samples = test_end - test_start
    stream_val_samples = stream_val_end - stream_val_start

    return WindowedDataset(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        context_train_input=input_samples[context_start:context_end],
        context_train_target=target_samples[context_start:context_end],
        stream_val_input=input_samples[stream_val_start:stream_val_end],
        stream_val_target=target_samples[stream_val_start:stream_val_end],
        test_input=input_samples[test_start:test_end],
        test_target=target_samples[test_start:test_end],
        sample_rate=input_audio.sample_rate,
        summary={
            "sample_rate": input_audio.sample_rate,
            "duration_seconds": length / input_audio.sample_rate,
            "sequence_length": sequence_length,
            "stride": stride,
            "available_windows": total_windows,
            "available_training_windows": len(available_starts),
            "selected_windows": len(selected_starts),
            "train_windows": len(train_starts),
            "validation_windows": len(val_starts),
            "context_training_samples": context_end - context_start,
            "context_training_start_sample": context_start,
            "stream_validation_samples": stream_val_samples,
            "stream_validation_start_sample": stream_val_start,
            "test_samples": test_samples,
            "test_start_sample": test_start,
            "preview_seconds": test_samples / input_audio.sample_rate,
            "selection": "energy_stratified_sampled_across_capture"
            if total_windows > len(selected_starts)
            else "energy_stratified_all_windows",
            "energy_selected_windows": energy_selected_count,
            "random_selected_windows": random_selected_count,
            "resampled_training_windows": 0,
            "window_resample_seed": seed,
            "reserved_excerpt_count": len(reserved_ranges),
        },
        input_samples=input_samples,
        target_samples=target_samples,
        sequence_length=sequence_length,
        backend=backend,
        available_starts=tuple(available_starts),
        train_starts=tuple(train_starts),
        val_starts=tuple(val_starts),
    )


def resample_windowed_training_data(
    dataset: WindowedDataset,
    *,
    seed: int,
    backend: str | None = None,
) -> WindowedDataset:
    backend = backend or dataset.backend
    validation_starts = set(dataset.val_starts)
    available_starts = [
        start for start in dataset.available_starts if start not in validation_starts
    ]
    train_budget = max(1, len(dataset.train_starts))
    train_starts, energy_selected_count, random_selected_count = select_training_starts(
        dataset.target_samples,
        available_starts,
        sequence_length=dataset.sequence_length,
        window_budget=train_budget,
        seed=seed,
    )
    if not train_starts:
        train_starts = list(dataset.train_starts)

    random.Random(seed).shuffle(train_starts)
    train_x, train_y = make_window_arrays(
        backend,
        dataset.input_samples,
        dataset.target_samples,
        train_starts,
        dataset.sequence_length,
    )
    summary = dict(dataset.summary)
    summary.update(
        {
            "selected_windows": len(train_starts) + len(dataset.val_starts),
            "train_windows": len(train_starts),
            "validation_windows": len(dataset.val_starts),
            "selection": "energy_stratified_resampled_training_windows",
            "energy_selected_windows": energy_selected_count,
            "random_selected_windows": random_selected_count,
            "resampled_training_windows": 1,
            "window_resample_seed": seed,
        }
    )

    return WindowedDataset(
        train_x=train_x,
        train_y=train_y,
        val_x=dataset.val_x,
        val_y=dataset.val_y,
        context_train_input=dataset.context_train_input,
        context_train_target=dataset.context_train_target,
        stream_val_input=dataset.stream_val_input,
        stream_val_target=dataset.stream_val_target,
        test_input=dataset.test_input,
        test_target=dataset.test_target,
        sample_rate=dataset.sample_rate,
        summary=summary,
        input_samples=dataset.input_samples,
        target_samples=dataset.target_samples,
        sequence_length=dataset.sequence_length,
        backend=backend,
        available_starts=dataset.available_starts,
        train_starts=tuple(train_starts),
        val_starts=dataset.val_starts,
    )


def split_training_validation_starts(
    selected_starts: list[int],
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    starts = list(selected_starts)
    random.Random(seed).shuffle(starts)
    train_count = max(1, int(len(starts) * 0.8))
    val_count = max(1, int(len(starts) * 0.1))
    if train_count + val_count >= len(starts):
        train_count = len(starts) - 2
        val_count = 1
    return tuple(starts[:train_count]), tuple(starts[train_count : train_count + val_count])


def select_training_starts(
    target_samples: list[float],
    starts: list[int],
    *,
    sequence_length: int,
    window_budget: int,
    seed: int,
) -> tuple[list[int], int, int]:
    budget = min(len(starts), window_budget)
    if budget <= 0:
        return [], 0, 0

    energy = energy_prefix(target_samples)
    scored = [
        (energy_between_prefix(energy, start, start + sequence_length), start)
        for start in starts
    ]
    scored.sort(reverse=True)
    energy_quota = min(budget, max(1, budget // 4))
    energy_selected = [start for _energy, start in scored[:energy_quota]]
    remaining = [start for _energy, start in scored[energy_quota:]]
    random.Random(seed).shuffle(remaining)
    selected = sorted(energy_selected + remaining[: budget - len(energy_selected)])
    return selected, len(energy_selected), len(selected) - len(energy_selected)


def choose_active_excerpts(
    target_samples: list[float],
    *,
    excerpt_length: int,
    stride: int,
    count: int = 1,
    reserved_ranges: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    length = len(target_samples)
    excerpt_length = min(max(1, excerpt_length), length)
    max_start = max(0, length - excerpt_length)
    step = max(1, stride)
    starts = list(range(0, max_start + 1, step))
    if starts[-1] != max_start:
        starts.append(max_start)

    energy = energy_prefix(target_samples)
    candidates = [
        (energy_between_prefix(energy, start, start + excerpt_length), start)
        for start in starts
    ]
    candidates.sort(reverse=True)
    selected: list[tuple[int, int]] = []
    reserved = reserved_ranges or []
    for _energy, start in candidates:
        candidate = (start, start + excerpt_length)
        if any(ranges_overlap(candidate, existing) for existing in reserved):
            continue
        if any(ranges_overlap(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break

    if not selected:
        selected.append((0, excerpt_length))
    while len(selected) < count:
        selected.append(selected[-1])
    return selected


def choose_active_excerpt(
    target_samples: list[float],
    *,
    excerpt_length: int,
    stride: int,
    reserved_ranges: list[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    return choose_active_excerpts(
        target_samples,
        excerpt_length=excerpt_length,
        stride=stride,
        count=1,
        reserved_ranges=reserved_ranges,
    )[0]


def overlaps_reserved_ranges(
    start: int,
    end: int,
    reserved_ranges: list[tuple[int, int]],
) -> bool:
    return any(ranges_overlap((start, end), reserved) for reserved in reserved_ranges)


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def energy_between(samples: list[float], start: int, end: int) -> float:
    total = 0.0
    for index in range(start, min(end, len(samples))):
        sample = samples[index]
        total += sample * sample
    return total


def energy_prefix(samples: list[float]) -> list[float]:
    prefix = [0.0]
    total = 0.0
    for sample in samples:
        total += sample * sample
        prefix.append(total)
    return prefix


def energy_between_prefix(prefix: list[float], start: int, end: int) -> float:
    bounded_start = max(0, min(start, len(prefix) - 1))
    bounded_end = max(bounded_start, min(end, len(prefix) - 1))
    return prefix[bounded_end] - prefix[bounded_start]


def make_window_arrays(
    backend: str,
    input_samples: list[float],
    target_samples: list[float],
    starts: tuple[int, ...] | list[int],
    sequence_length: int,
) -> tuple[object, object]:
    windows_x: list[list[float]] = []
    windows_y: list[list[float]] = []
    for start in starts:
        windows_x.append(input_samples[start : start + sequence_length])
        windows_y.append(target_samples[start : start + sequence_length])
    return make_backend_array(backend, windows_x), make_backend_array(backend, windows_y)


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
