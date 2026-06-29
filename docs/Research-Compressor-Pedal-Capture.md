# Compressor Pedal Capture Findings

Date: 2026-06-29

Project:

```text
/Users/shortwavlabs/Library/Application Support/labs.shortwav.rtneural-trainer/projects/project_f6d971fbc8d3462a9cb29fb17bc6c867
```

Capture files:

```text
/Users/shortwavlabs/Music/CAPTURES/cap DI MONO.wav
/Users/shortwavlabs/Music/CAPTURES/cap COMP.wav
```

## Summary

The compressor pedal capture does not look like a bad capture or a gross
alignment failure. The prep report is strong:

- mono 48 kHz input and target;
- no clipping;
- prepared WAVs preserved as float32;
- input peak `-6.30 dBFS`, target peak `-9.73 dBFS`;
- input RMS `-27.28 dBFS`, target RMS `-26.87 dBFS`;
- RMS delta only `+0.41 dB`;
- latency estimate `86 samples` with `0.940` confidence;
- all six active analysis windows voted for the same latency;
- polarity confidence is `1.0`.

The best existing run was `wavenet_tcn_quality` at about `0.012` ESR. That is
close, but still short of the desired sub-`0.01` target and the ideal
`~0.004` range.

The diagnosis is that this capture is a dynamics-modeling problem, not a simple
amp-tone problem. The existing high-gain/edge presets can learn the broad sound,
but they do not explicitly prioritize compressor attack/release behavior and
micro-transient gain shaping.

## Existing Runs

| Run | Preset | Loss | Epochs | Best Epoch | ESR | RMSE | MAE | Peak Residual | Correlation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_9a2788ddfca5443ab08053e5843f1074` | `wavenet_tcn_clean` | `preemphasis_mse` | 61 | 41 | 0.0389 | 0.01135 | 0.00762 | 0.1618 | 0.9823 |
| `run_8421fbbbff9c48eaad981c2447245188` | `wavenet_tcn_edge` | `preemphasis_mse` | 88 | 64 | 0.0287 | 0.00975 | 0.00712 | 0.1259 | 0.9861 |
| `run_13f0e6267a72421596b5d30fcbcd6a12` | `wavenet_tcn_edge_detail` | `preemphasis_mse` | 119 | 89 | 0.0280 | 0.00964 | 0.00697 | 0.1252 | 0.9867 |
| `run_55b492b9fb694570a26fddcb6032a42b` | `wavenet_tcn_a2_prelu` | `mrstft_preemphasis` | 41 | 21 | 0.6097 | 0.04493 | 0.03371 | 0.2715 | 0.6631 |
| `run_8b25f257e2174679b669aeba5cea4bad` | `wavenet_tcn_quality` | `mrstft_preemphasis` | 180 | 168 | 0.0120 | 0.00630 | 0.00424 | 0.0849 | 0.9944 |

`wavenet_tcn_quality` was the clear winner among the existing presets. It
continued improving slowly deep into the run, with learning-rate drops at epochs
`45`, `144`, and `178`.

## Alignment Check

The trained preview from the best run does not improve if nudged:

| Prediction Lag | Gain Correction | ESR | Correlation |
| ---: | ---: | ---: | ---: |
| `0 samples` | `+0.245 dB` | `0.01123` | `0.99442` |
| `-1 sample` | `+0.231 dB` | `0.01425` | `0.99290` |
| `+1 sample` | `+0.228 dB` | `0.01487` | `0.99259` |
| `-2 samples` | `+0.189 dB` | `0.02385` | `0.98805` |
| `+2 samples` | `+0.183 dB` | `0.02509` | `0.98743` |

That makes a hidden one- or two-sample alignment correction unlikely. The
current 86-sample prep alignment should be trusted for the next run.

## Signal Diagnostics

The raw DI to compressor target relationship is strongly dynamic:

- simple identity input-to-target ESR: `0.202`;
- best scalar input-to-target ESR: `0.198`;
- static polynomial-ish input map ESR: `0.196`;
- trained WaveNet Quality ESR: `0.012`.

So the trainer is learning real behavior that a static level/EQ mapping cannot
capture.

Envelope diagnostics show the trained model is already close on slow compressor
movement:

| Envelope Window | Target/Prediction ESR | Target/Prediction Correlation |
| ---: | ---: | ---: |
| `2 ms` | `0.00731` | `0.99019` |
| `5 ms` | `0.00521` | `0.98924` |
| `10 ms` | `0.00348` | `0.98756` |
| `25 ms` | `0.00164` | `0.99200` |
| `50 ms` | `0.00135` | `0.99329` |
| `100 ms` | `0.00115` | `0.99400` |

The remaining residual is disproportionately transient/upper-mid energy:

- target spectral centroid around `635 Hz`;
- prediction spectral centroid around `681 Hz`;
- residual spectral centroid around `1299 Hz`;
- residual band energy is concentrated in `120-500 Hz`, `500-1500 Hz`, and
  `1500-4000 Hz`, not ultrasonic aliasing.

This points to attack/release and pick-transient gain-shaping error more than
tonal EQ mismatch or latency.

## Code Change Implemented

Added `wavenet_tcn_compressor`, a compressor/dynamics-pedal preset:

- quality-compatible Conv1D WaveNet-style stack;
- 10 dilated causal layers, matching `wavenet_tcn_quality`;
- `20` filters;
- kernel size `3`;
- dilations `1, 2, 4, ..., 512`;
- receptive field approximately `2047` samples, or `43 ms` at 48 kHz;
- default learning rate `5e-4`;
- default loss `compressor_envelope_mrstft`.

The new loss keeps the existing MR-STFT/pre-emphasis loss and adds a multi-rate
envelope term. The envelope term compares smoothed absolute amplitude at
`64`, `256`, and `1024` samples, plus a lightly weighted envelope slope term.
The goal is to give compressor attack, release, and gain-recovery errors more
gradient without abandoning waveform fidelity.

The desktop app now exposes a built-in `WaveNet compressor` recipe:

- preset `wavenet_tcn_compressor`;
- 220 epochs;
- batch size 16;
- learning rate `5e-4`;
- sequence length `8192`;
- max windows `8192`;
- resampled training windows enabled;
- early-stop patience 32;
- min delta `3e-5`.

## First Compressor Preset Run

The first compressor preset run was:

```text
/Users/shortwavlabs/Library/Application Support/labs.shortwav.rtneural-trainer/projects/project_f6d971fbc8d3462a9cb29fb17bc6c867/runs/run_3243ee77390840d8a3b1be62523dea26
```

It used the initial experimental compressor design: kernel size `7`, smoothed
`tanh(x / 1.5)`, sequence length `16384`, learning rate `2.5e-4`, and a heavier
envelope/slope loss. The result was useful, but it did not beat the prior
`wavenet_tcn_quality` baseline:

| Run | Preset | ESR | RMSE | MAE | Peak Residual | Correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_8b25f257e2174679b669aeba5cea4bad` | `wavenet_tcn_quality` | `0.01199` | `0.00630` | `0.00424` | `0.0849` | `0.9944` |
| `run_3243ee77390840d8a3b1be62523dea26` | initial `wavenet_tcn_compressor` | `0.02048` | `0.00821` | `0.00541` | `0.0935` | `0.9901` |

The first compressor run also had worse envelope diagnostics than the quality
baseline. For example, the `2 ms` envelope ESR rose from about `0.0062` to
`0.0106`, and the `10 ms` envelope ESR rose from about `0.0024` to `0.0037`.
This means the first compressor preset did not merely trade waveform ESR for
better compressor movement; it was broadly worse.

The lesson is that the quality architecture is already the best foundation for
this capture. The compressor preset was revised to be quality-compatible and to
use the envelope term only as a light refinement signal. The learning-rate
plateau default was also relaxed for long-patience recipes so dynamics runs do
not fall to near-idle learning rates too early.

## Next Test

Run the new `WaveNet compressor` recipe on the existing prepared capture without
changing latency.

Suggested interpretation:

- If it drops below `0.01`, continue from the best checkpoint at about `8e-5`
  learning rate.
- If it lands around `0.012` again but sounds better, inspect peak residual and
  envelope movement before judging by ESR alone.
- If it underperforms quality, try continuing the existing
  `wavenet_tcn_quality` run at `8e-5` before changing the capture.
- If both quality and compressor plateau above `0.01`, the next model-side
  experiment should be a true dynamics-aware input feature path or an explicit
  sidechain/envelope auxiliary branch. That would require RTNeural runtime
  design work and is not an MVP-safe preset tweak.

## Capture Guidance

No immediate recapture is required based on the current evidence.

For future compressor captures, include:

- strong isolated pick attacks;
- sustained notes that force release behavior;
- palm mutes and staccato parts;
- soft-to-hard dynamic ramps;
- long decays into noise floor;
- single notes and chords.

Avoid changing the pedal settings between DI and target capture. Compression is
memoryful, so the DI performance must be identical and the beginning preamble
should be preserved.
