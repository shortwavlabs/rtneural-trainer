from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioBuffer:
    samples: list[float]
    sample_rate: int
    channels: int
    sample_width: int
    path: str

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate


def read_wav_mono(path: Path) -> AudioBuffer:
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")
    if channels <= 0:
        raise ValueError("WAV must contain at least one channel")

    values = pcm_to_float(raw, sample_width)
    if channels > 1:
        mono = []
        for index in range(0, len(values), channels):
            frame = values[index : index + channels]
            if len(frame) == channels:
                mono.append(sum(frame) / channels)
        values = mono

    return AudioBuffer(
        samples=values,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        path=str(path),
    )


def write_wav_mono(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(float_to_pcm16(samples))


def pcm_to_float(raw: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(byte - 128) / 128.0 for byte in raw]

    values: list[float] = []
    max_int = float(2 ** (sample_width * 8 - 1))
    for offset in range(0, len(raw), sample_width):
        chunk = raw[offset : offset + sample_width]
        if len(chunk) != sample_width:
            continue
        if sample_width == 3:
            sign = b"\xff" if chunk[-1] & 0x80 else b"\x00"
            integer = int.from_bytes(chunk + sign, "little", signed=True)
        else:
            integer = int.from_bytes(chunk, "little", signed=True)
        values.append(clamp(integer / max_int, -1.0, 1.0))
    return values


def float_to_pcm16(samples: list[float]) -> bytes:
    data = bytearray()
    for sample in samples:
        integer = int(round(clamp(sample, -1.0, 1.0) * 32767.0))
        data.extend(integer.to_bytes(2, "little", signed=True))
    return bytes(data)


def audio_report(audio: AudioBuffer) -> dict[str, float | int | str]:
    peak = max((abs(sample) for sample in audio.samples), default=0.0)
    rms = math.sqrt(sum(sample * sample for sample in audio.samples) / max(1, len(audio.samples)))
    clipped = sum(1 for sample in audio.samples if abs(sample) >= 0.999)
    dc_offset = sum(audio.samples) / max(1, len(audio.samples))

    return {
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "duration_seconds": audio.duration_seconds,
        "peak_dbfs": linear_to_dbfs(peak),
        "rms_dbfs": linear_to_dbfs(rms),
        "clipped_samples": clipped,
        "dc_offset": dc_offset,
        "path": audio.path,
    }


def linear_to_dbfs(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
