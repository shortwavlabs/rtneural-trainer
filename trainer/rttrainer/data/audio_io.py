from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path


WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


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


@dataclass(frozen=True)
class WavPayload:
    samples: list[float]
    sample_rate: int
    channels: int
    sample_width: int


def read_wav_mono(path: Path, channel_policy: str = "mixdown") -> AudioBuffer:
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {path}")
    policy = normalize_channel_policy(channel_policy)

    payload = read_wav_payload(path)
    values = payload.samples
    if payload.channels > 1:
        if policy == "reject":
            raise ValueError(
                f"{path} has {payload.channels} channels. Choose mono files or enable a mono channel policy."
            )
        mono = []
        for index in range(0, len(values), payload.channels):
            frame = values[index : index + payload.channels]
            if len(frame) == payload.channels:
                if policy == "first":
                    mono.append(frame[0])
                else:
                    mono.append(sum(frame) / payload.channels)
        values = mono

    return AudioBuffer(
        samples=values,
        sample_rate=payload.sample_rate,
        channels=payload.channels,
        sample_width=payload.sample_width,
        path=str(path),
    )


def read_wav_payload(path: Path) -> WavPayload:
    try:
        return read_wav_payload_with_wave_module(path)
    except wave.Error as exc:
        try:
            return read_wav_payload_fallback(path)
        except ValueError:
            raise ValueError(f"Unsupported WAV encoding in {path}: {exc}") from exc


def read_wav_payload_with_wave_module(path: Path) -> WavPayload:
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
    return WavPayload(
        samples=values,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def read_wav_payload_fallback(path: Path) -> WavPayload:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    fmt: bytes | None = None
    raw: bytes | None = None
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            raise ValueError(f"Malformed WAV chunk in {path}")
        if chunk_id == b"fmt ":
            fmt = data[chunk_start:chunk_end]
        elif chunk_id == b"data":
            raw = data[chunk_start:chunk_end]
        offset = chunk_end + (chunk_size % 2)

    if fmt is None or raw is None:
        raise ValueError(f"WAV file is missing fmt or data chunk: {path}")
    if len(fmt) < 16:
        raise ValueError(f"WAV fmt chunk is too short: {path}")

    format_tag, channels, sample_rate, _byte_rate, _block_align, bits_per_sample = (
        struct.unpack("<HHIIHH", fmt[:16])
    )
    sample_width = bits_per_sample // 8
    if channels <= 0:
        raise ValueError("WAV must contain at least one channel")
    if format_tag == WAVE_FORMAT_PCM:
        samples = pcm_to_float(raw, sample_width)
    elif format_tag == WAVE_FORMAT_IEEE_FLOAT:
        samples = ieee_float_to_float(raw, sample_width)
    else:
        raise ValueError(f"Unsupported WAV format tag: {format_tag}")

    return WavPayload(
        samples=samples,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def normalize_channel_policy(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"mixdown", "mono_mixdown", "mix_to_mono"}:
        return "mixdown"
    if normalized in {"first", "first_channel", "left"}:
        return "first"
    if normalized in {"reject", "reject_multichannel", "mono_only"}:
        return "reject"
    raise ValueError("Channel policy must be 'mixdown', 'first', or 'reject'.")


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


def ieee_float_to_float(raw: bytes, sample_width: int) -> list[float]:
    if sample_width not in (4, 8):
        raise ValueError(f"Unsupported float WAV sample width: {sample_width} bytes")
    values: list[float] = []
    format_char = "f" if sample_width == 4 else "d"
    for offset in range(0, len(raw), sample_width):
        chunk = raw[offset : offset + sample_width]
        if len(chunk) != sample_width:
            continue
        value = float(struct.unpack(f"<{format_char}", chunk)[0])
        if math.isfinite(value):
            values.append(clamp(value, -1.0, 1.0))
        else:
            values.append(0.0)
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
