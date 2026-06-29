# Aliasing-Aware RTNeural-Compatible WaveNet Modeling Of Guitar Amplifier And Pedal Captures

Status: internal scientific draft  
Date: 2026-06-29  
Repository: `shortwavlabs/rtneural-trainer`  
Peer review: [Peer-Review-RTNeural-WaveNet-DSP-Research.md](Peer-Review-RTNeural-WaveNet-DSP-Research.md)

## Abstract

This paper summarizes the DSP and machine-learning research performed during
development of RTNeural Trainer, a desktop system for learning real-time
RTNeural-compatible models of guitar amplifiers and pedals from paired dry and
processed audio captures. The work began with broad support for dense,
recurrent, and convolutional architectures, but converged on finite-memory
WaveNet-style temporal convolutional networks as the best-tested product path
inside the current 48 kHz mono paired-capture dataset and RTNeural JSON
constraints. The core engineering constraint was not merely training accuracy;
the exported model had to run as a causal real-time DSP block through RTNeural
JSON, validate against Python parity, benchmark above real time in a native C++
runtime on the reference system, and avoid obvious aliasing or latency
artifacts in engineering diagnostics and listening checks.

Across clean, crunch, rhythm, lead, edge-of-breakup, and overdrive-pedal
captures, WaveNet-family presets provided the most reliable end-to-end export
results in our internal comparisons. The strongest high-gain result came from
an A2-inspired, RTNeural-safe sequential WaveNet variant using mixed kernel
sizes, non-power-of-two dilations, and PReLU nonlinearities. On the RHYTHM4
high-gain case study, this preset improved preview ESR from 0.0646 for the
previous quality smoothed-tanh model to 0.0381 after continuation, while
reducing probe-measured average aliasing-to-signal ratio from 0.0419 to 0.0145
and benchmarking at a native Eigen worst-case real-time factor of approximately
6.62x on the reference system. Real hardware captures later produced even
stronger export packages: clean/edge, overdrive pedal, and rhythm exports
reached ESR values of 0.00217, 0.00043, and 0.00355 respectively, all passing
native RTNeural validation with low probe ASR.

The results support four scoped technical conclusions. First, within the
current internal captures and export constraints, WaveNet-family causal Conv1D
presets are the strongest product path tested so far; this paper does not claim
a general architecture theorem over all recurrent or convolutional alternatives.
Second, latency estimation, polarity handling, capture headroom, and source
material diversity are first-order model-quality variables rather than
preprocessing details. Third, ESR is necessary but insufficient:
probe-measured aliasing, residual spectra, prediction level, native runtime, and
listening tests must all participate in the export gate. Fourth, NAM A2-style
topology is a promising direction, but exact A2 support requires RTNeural
runtime extensions for residual/skip/head structure; the current A2-inspired
sequential preset is a practical MVP compromise.

## Keywords

Neural amplifier modeling; WaveNet; temporal convolutional networks; RTNeural;
JUCE; guitar DSP; aliasing; real-time audio; causal Conv1D; model export;
latency alignment; neural distortion.

## 1. Introduction

The goal of RTNeural Trainer is to capture a fixed guitar amplifier or pedal
setting as a real-time neural DSP model. The user supplies a dry direct-input
recording and a corresponding processed target recording. The system aligns and
prepares the pair, trains a model, exports RTNeural JSON, validates native
inference against Python/Keras, benchmarks the exact exported model, and finally
loads the package in a JUCE Audio Unit test plugin.

This sounds like a familiar supervised regression problem:

```text
dry guitar x[n] -> unknown amplifier or pedal H -> target y[n]
train f_theta(x[n]) ~= y[n]
```

In practice, the problem is less forgiving. Guitar amplification is nonlinear,
level-dependent, memory-bearing, and extremely sensitive to small timing errors.
High-gain amp tones amplify upper-band residuals and aliasing artifacts; clean
or edge-of-breakup tones may punish overpowered nonlinear models that learn
window fragments but fail full-stream validation. Meanwhile, the final model is
not allowed to be an arbitrary Python graph. It must be causal, stateful,
allocation-free at audio time, exportable to RTNeural, and fast enough at small
DAW buffer sizes.

The project therefore became a combined DSP, ML, and systems-engineering study.
We did not merely ask which architecture minimized validation loss. We asked:

1. Which architectures train reliably across real guitar captures?
2. Which architectures can be expressed as RTNeural JSON today?
3. Which exports survive native C++ parity and real-time benchmarking?
4. Which metrics catch the problems guitarists actually hear?
5. Which capture habits reduce ambiguity before training begins?

The answer that emerged is centered on WaveNet-style temporal convolution:
causal dilated Conv1D stacks with finite receptive fields, spectral pressure in
the loss, and export-time validation. Early product plans included Dense, GRU,
LSTM, Conv1D, BatchNorm/PReLU, and hybrid presets. Those remain useful as
internal RTNeural layer-coverage fixtures, but the user-facing trainer has moved
to WaveNet-only recipes. That simplification is an empirical conclusion, not an
aesthetic one.

## 2. Related Work And Project Context

### 2.1 Neural amp modeling and finite-memory nonlinear systems

A guitar amplifier or pedal can be viewed as a discrete-time nonlinear system
with memory:

```text
y[n] = H(x[n], x[n - 1], ..., x[n - M], internal_state)
```

For a static control setting, a finite-memory model is attractive. It avoids
requiring an explicit circuit model, but it can still run as a deterministic
causal DSP block. Earlier neural amp-modeling work explored recurrent networks,
WaveNet/TCN models, pre-emphasis losses, multi-resolution STFT losses, and
parametric conditioning. The local research notes reviewed PANAMA and related
papers and found that their conclusions matched our experiments: dilated causal
convolutions are a practical fit for amp modeling, spectral loss matters, and
alignment errors of only a few samples can visibly damage results.

### 2.2 RTNeural as the inference target

RTNeural provides real-time neural-network layers for C++ audio applications.
The project uses dynamic RTNeural JSON exports as the current runtime contract.
That choice has consequences:

- Sequential models are straightforward.
- Causal Conv1D, Dense, GRU, LSTM, PReLU, BatchNorm, and activations are within
  the supported coverage envelope.
- Arbitrary computational graphs, residual skip accumulation, and true NAM A2
  topology are not directly expressible in the current dynamic JSON path.
- Native validation must compare the exported JSON model against the Keras
  source model, because export compatibility is not assumed.

The user also maintains an editable RTNeural fork at
`/Users/shortwavlabs/Workspace/rt-neural/RTNeural`, published as
`shortwavlabs/rtneural-extended`. That makes future runtime work possible, but
we treated RTNeural changes as day-2 infrastructure rather than an MVP
dependency.

### 2.3 NAM as a reference system

Neural Amp Modeler influenced the project in three ways. First, it reinforced
the separation between training/export tooling and real-time plugin playback.
Second, it demonstrated the importance of model metadata, snapshots, native
benchmarking, and package discipline. Third, NAM A2 models provided a concrete
architecture target: residual/skip WaveNet-style stacks with carefully chosen
kernel sizes, dilations, and runtime specializations.

We did not adopt the `.nam` format. We instead used NAM as a design reference.
The current best RTNeural Trainer high-gain preset, `wavenet_tcn_a2_prelu`, is
A2-inspired but not A2-equivalent. It borrows mixed kernels, non-power-of-two
dilations, and leaky/PReLU-like activations while staying sequential and
RTNeural JSON compatible.

### 2.4 Aliasing and activation smoothing

Nonlinear audio models generate harmonics. If those harmonics exceed Nyquist,
they fold back as aliasing. High-gain amp modeling is especially sensitive
because the model is asked to reproduce dense saturation, fizz, and pick attack
without an explicit oversampling stage.

The DAFx 2025 paper on smoothing activations motivated direct aliasing
measurement and smoothed-tanh experiments. We implemented deterministic sine
probe analysis and added WaveNet variants trained with `tanh(x / alpha)`. For
RTNeural export, this smoothing can be represented without a custom activation:
the scale factor is folded into the preceding Conv1D weights and bias before a
standard RTNeural `tanh`.

Empirically, smoothing was not a universal improvement. On one Rhythm2 balanced
test, `alpha = 1.8` reduced ASR but worsened ESR, while `alpha = 1.5` worsened
ASR. On RHYTHM4 at quality depth, `alpha = 1.5` improved both ESR and ASR
relative to the plain quality model. This result is important: activation
smoothing is a controllable trade-off, not a universally beneficial
anti-aliasing transform.

## 3. Problem Formulation

Let `x[n]` be the prepared dry input signal and `y[n]` the prepared target
signal. Both are mono 48 kHz floating-point sequences after channel policy,
resampling policy, polarity handling, and latency alignment have been applied.
The trained model `f_theta` produces:

```text
y_hat[n] = f_theta(x[n], x[n - 1], ..., x[n - R + 1])
```

where `R` is the effective receptive field of the causal model.

For a sequential stack of causal Conv1D layers with kernel sizes `k_l` and
dilations `d_l`, and no striding, the receptive field is:

```text
R = 1 + sum_l ((k_l - 1) * d_l)
```

The residual is:

```text
e[n] = y[n] - y_hat[n]
```

The primary waveform metric is error-to-signal ratio:

```text
ESR = sum_n e[n]^2 / max(sum_n y[n]^2, epsilon)
```

We also compute:

```text
MAE  = mean_n |e[n]|
RMSE = sqrt(mean_n e[n]^2)
peak_residual = max_n |e[n]|
```

In the app reports, residual RMS is the same quantity as RMSE, often expressed
in dBFS for audio interpretation:

```text
residual_rms_dbfs = 20 * log10(max(RMSE, epsilon))
```

Correlation is used as a shape agreement diagnostic:

```text
corr = cov(y, y_hat) / (std(y) * std(y_hat))
```

The report also tracks prediction RMS ratio:

```text
prediction_rms_ratio = rms(y_hat) / max(rms(y), epsilon)
```

This is not a fidelity metric by itself. It is a guardrail for models that
reduce error by becoming underpowered or otherwise mismatching target level.

Several metric spans appear in the project notes:

| Term | Definition | Use |
| --- | --- | --- |
| Window validation | ESR/MAE/RMSE on held-out training windows. | Stable epoch-to-epoch checkpoint signal. |
| Stream validation | ESR/MAE/RMSE on a longer continuous validation stream. | Catches state and continuity failures missed by isolated windows. |
| Composite validation score | `stream_esr + 0.25 * window_esr + underpowered_output_penalty`. | Checkpoint selection and LR/early-stop monitor. |
| Preview or state-continuous metrics | Metrics computed on rendered preview WAVs from a saved checkpoint/export. | User-facing quality comparison and listening aid. |

The main result tables use preview/state-continuous ESR unless explicitly
identified otherwise. This means they are directly useful for product
comparison inside the app, but they are not identical to training-window loss or
stream-validation score.

These metrics are useful but incomplete. A model can have low ESR while still
producing objectionable aliasing or level-dependent artifacts. Conversely, a
model may have a warning-level ASR but sound acceptable in a dense mix or after
cabinet filtering. Therefore, the export system treats ASR as warning-oriented
rather than as a hard rejection criterion.

## 4. Methods

### 4.0 Reproducibility envelope

This paper is an internal scientific report over iterative product experiments,
not a fully controlled benchmark suite. The table below records the
reproducibility envelope of the reported runs and separates fixed implementation
details from remaining experimental limitations.

| Item | Current value or protocol | Reproducibility note |
| --- | --- | --- |
| Audio format | 48 kHz mono prepared WAV, preferably float32. | Stereo/dual-mono captures are mixed to mono. |
| Pairing | Dry DI and processed target from the same performance. | Mismatched performances are outside scope. |
| Latency | Auto-estimated or known sample offset; top candidates exposed for review. | Low-confidence high-gain offsets remain a threat to validity. |
| Polarity | Preparation is polarity-aware for captures that invert target polarity. | Polarity metadata should be preserved per project. |
| Sequence length | Default 1024 samples unless overridden by recipe/user. | Some long-field presets have receptive fields larger than one sequence window, so stream validation remains important. |
| Window budget | Default `max_windows = 512`; larger runs often used 2048+ windows. | Training windows may be resampled between epochs when enabled. |
| Validation windows | Held fixed while training windows may resample. | Supports stable checkpoint comparisons. |
| Default seed | `1337` unless overridden. | Most reported comparisons are single-seed product experiments, not repeated-seed statistics. |
| Batch size | Default 16 unless overridden. | User may override in custom recipes. |
| Optimizer | TensorFlow/Keras Adam. | PyTorch support has been removed from the product trainer. |
| Early stopping | Composite validation score, default patience 5, min delta `1e-4`. | Longer product runs often use larger patience. |
| LR plateau | `ReduceLROnPlateau`, factor 0.5, patience derived from early-stop patience, min LR `1e-6`. | Resume runs cap LR unless explicitly allowed to increase. |
| Pre-emphasis | Coefficient 0.95; pre-emphasis loss weight 0.35. | Used by `preemphasis_mse` and WaveNet losses. |
| MR-STFT | Frame sizes 256, 1024, 2048; MRSTFT weight 0.02; log-mag weight 0.05. | Exact psychoacoustic weighting remains research territory. |
| ASR probes | 1250, 2500, 5000 Hz requested frequencies; amplitude 0.5; 2048-sample warmup; 4096-sample analysis. | Probe-specific diagnostic, not a perceptual aliasing model. |
| Native benchmark | RTNeural validator block-size/channel matrix, usually Eigen backend on Apple Silicon. | Release/plugin benchmarks across weaker machines remain future work. |
| Plugin smoke | JUCE AU debug loader in Logic Pro on the reference MacBook Pro M5 Max. | Demonstrates viability on one strong machine, not production portability. |

### 4.1 Capture preparation

The capture contract is simple:

1. Record a dry DI file.
2. Reamp or render the same performance through the target amp/pedal chain.
3. Keep the pair sample-rate consistent, preferably 48 kHz float32 WAV.
4. Avoid clipping and preserve headroom.
5. Use the same exact source performance for input and target.
6. Include a short transient preamble before the musical pass.

The project converged on a practical preamble recommendation: start the DI with
2-3 seconds of clear, dry, varied attacks before the musical pass. Palm mutes,
single-note plucks, hard/soft pick attacks, and a few different registers help
the latency estimator distinguish true delay from repeated musical structure.

The preparation stage computes level, duration, channel count, clipping,
latency candidates, confidence, and warnings. Stereo or multichannel files are
mixed to mono for current model training. Float32 prepared WAVs are preserved;
this became important after earlier preparation paths risked unnecessary format
conversion.

Latency is the most fragile preprocessing variable. If the target is shifted by
`L` samples, the training pair is effectively:

```text
x_aligned[n] = x[n]
y_aligned[n] = y[n + L]
```

For clean signals, cross-correlation-like methods are often stable. For heavy
distortion, the processed waveform can have a weak linear relationship to the
DI, and multiple candidate offsets separated by a few samples can score
similarly. The UI therefore exposes top candidate offsets, confidence, agreement
across active windows, and a manual nudge workflow.

### 4.2 Training window selection

Training does not use one fixed chunk every epoch. The prepared audio is divided
into candidate windows, validation windows are held fixed, and training windows
are sampled across the file. Energy-stratified selection was added to avoid a
long capture being dominated by silence, decays, or low-information passages.
This matters for guitar because the model must learn attacks, palm mutes,
sustain, noise, harmonics, and level-dependent breakup.

Long captures are possible but not always efficient. The experiments suggest
that approximately 2.5-4 minutes of focused, varied material is often enough for
a fixed setting. Very long captures slow training and can include redundant or
low-information material. Shorter captures train faster if they preserve
variety and avoid excessive silence.

### 4.3 Loss functions

The WaveNet path uses TensorFlow/Keras and has moved away from PyTorch support.
The relevant loss families are:

- ESR-style waveform loss.
- Pre-emphasis MSE to pressure high-frequency detail.
- Multi-resolution STFT plus pre-emphasis for high-gain tones.

The basic pre-emphasis idea is to compare a high-passed or differenced version
of target and prediction:

```text
p_y[n] = y[n] - a * y[n - 1]
p_hat[n] = y_hat[n] - a * y_hat[n - 1]
```

where `a` is a pre-emphasis coefficient. This increases the cost of upper-band
mistakes, which are perceptually important in distorted guitar. Multi-resolution
STFT loss adds time-frequency pressure at several window sizes, improving
spectral behavior relative to pure pointwise waveform loss.

### 4.4 Architectures tested

The project initially included Dense, GRU, LSTM, Conv1D, BatchNorm/PReLU, and
hybrid presets. After repeated experiments, only WaveNet-family presets remain
product-facing.

The current product/research WaveNet family includes:

| Preset family | Layers | Filters | Kernel(s) | Dilations | Activation | Default loss | Purpose |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `wavenet_tcn_fast` | 6 | 12 | 3 | 1, 2, 4, 8, 16, 32 | tanh | MR-STFT pre-emphasis | Quick sanity probe. |
| `wavenet_tcn_balanced` | 8 | 16 | 3 | 1, 2, 4, ..., 128 | tanh | MR-STFT pre-emphasis | Practical first quality run. |
| `wavenet_tcn_quality` | 10 | 20 | 3 | 1, 2, 4, ..., 512 | tanh | MR-STFT pre-emphasis | High-gain or maximum-fidelity run. |
| `wavenet_tcn_quality_tanh15` | 10 | 20 | 3 | 1, 2, 4, ..., 512 | `tanh(x / 1.5)` during training | MR-STFT pre-emphasis | Anti-alias/tone research candidate. |
| `wavenet_tcn_a2_prelu` | 12 | 16 | 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6 | 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41 | PReLU | MR-STFT pre-emphasis | Strong high-gain candidate. |
| `wavenet_tcn_clean` | 10 | 8 | 7 | 1, 2, 4, ..., 512 | linear hidden Conv1D | pre-emphasis MSE | Clean amp transfer. |
| `wavenet_tcn_edge` | 10 | 8 | 7 | 1, 2, 4, ..., 512 | `tanh(x / 1.8)` | pre-emphasis MSE | Edge-of-breakup captures. |

The standard sequential WaveNet graph is intentionally RTNeural-safe:

```text
x -> [causal Conv1D -> activation] * N -> output layer -> y_hat
```

It is not a full WaveNet with residual and skip connections. That limitation
keeps export simple but also creates optimization and capacity limits. The
failed `wavenet_tcn_high_gain` experiment showed that simply adding another
plain tanh dilation stage can cause optimization collapse. Hidden activations
stayed tiny relative to successful quality checkpoints, suggesting that deeper
non-residual sequential stacks are not the right way to increase receptive
field.

The A2-inspired preset was the strongest response to this. It does not
implement true NAM A2 residual topology, but it introduces:

- Mixed kernel sizes, especially 6 and 15 samples.
- Non-power-of-two dilations inspired by A2 patterns.
- PReLU/leaky hidden nonlinearities instead of only tanh.
- A longer but still manageable receptive field.

### 4.5 Export and validation

Every serious result must pass a multi-stage export gate:

1. Keras model exports to RTNeural JSON.
2. Python parity checks compare Keras inference with the exported JSON path.
3. Native `rtneural-validator` loads the JSON and compares output snapshots.
4. The native validator benchmarks realistic block sizes and mono/stereo cases.
5. The trainer writes reports and package metadata.
6. The desktop app surfaces validation, benchmark, ASR, package metadata, and
   open-folder controls.

The native validator benchmark matrix includes block sizes such as 16, 32, 64,
128, 256, and 512 samples, across mono and stereo where applicable. The package
records the conservative worst-case real-time factor, per-case timings, model
size, latency, sample rate, architecture metadata, and receptive-field
information.

The real-time factor is conceptually:

```text
RTF = audio_duration_processed / wall_time_elapsed
```

An RTF above 1.0 is faster than real time. In practice, we want comfortable
headroom because DAW plugin execution has additional constraints, especially at
small buffers and across multiple instances.

### 4.6 Aliasing-to-signal ratio

The aliasing report renders deterministic sine probes through the exported
RTNeural JSON. The current default probes are approximately 1.25 kHz, 2.5 kHz,
and 5 kHz with a 4096-sample analysis window and a warmup period. Each probe is
adjusted to the nearest FFT bin so the input sine is on-bin.

For each rendered output, the analyzer subtracts the mean, computes FFT power,
identifies harmonic bins of the probe frequency up to Nyquist, and defines:

```text
harmonic_energy = sum power at harmonic bins
aliasing_energy = total_energy - harmonic_energy
ASR = aliasing_energy / max(harmonic_energy, epsilon)
alias_fraction = aliasing_energy / max(total_energy, epsilon)
```

The current classification is deliberately conservative:

| Worst ASR | Status | Meaning |
| ---: | --- | --- |
| `< 0.02` | pass | Low aliasing in current probes. |
| `< 0.08` | warning | Review aliasing by ear. |
| `>= 0.08` | warning | High aliasing risk; inspect carefully. |

These thresholds are not psychoacoustic truth. They are engineering gates to
force listening attention on potential foldback. The thresholds need more
calibration across guitars, registers, cabinets, and mixes.

### 4.7 JUCE plugin smoke path

The test plugin was built as a runtime harness, not a finished product. It can
load exported RTNeural package folders, show package metadata, apply input and
output gain, provide simple EQ, load an optional pedal model before the amp,
load an optional impulse response after the amp, bypass stages, and restore
paths across DAW sessions.

The plugin work followed standard hard real-time constraints:

- No file IO in `processBlock()`.
- No JSON parsing in the audio callback.
- No heap allocation in the audio callback.
- No locks or UI calls in the audio callback.
- Use prepared state, atomics, and pointer swaps.
- Keep one stateful model instance per channel.
- Treat receptive field as model history, not plugin-reported latency.
- Use denormal prevention.

Logic Pro smoke tests on the reference MacBook Pro M5 Max showed that the
continued A2 PReLU high-gain export could run in four tracks at a 32-sample
buffer with barely visible CPU increase. This result is encouraging, but it is
not a substitute for pluginval, weaker-machine tests, release builds, or
systematic CPU profiling.

## 5. Experimental Evidence

### 5.1 Capture families

The main capture families are summarized below.

| Family | Duration | Sample rate | Notes |
| --- | ---: | ---: | --- |
| DI2 family | 151.6 s | 48 kHz | Same dry DI, outputs CLEAN2, CRUNCH2, RHYTHM2, EDGE2, LEAD2, DRIVE2. |
| DI3 / CLEAN3 | 613.08 s | 48 kHz | Long clean capture, known 13-sample DAW latency. |
| DI3-B / RHYTHM3-B | 444.25 s | 48 kHz | Trimmed high-gain rhythm, estimated 10-sample latency with low confidence. |
| DI4 / RHYTHM4 | 158.8 s | 48 kHz | Shorter high-gain rhythm control point, estimated 9-sample latency. |
| Real hardware trio | focused captures | 48 kHz | Real amp clean/edge, real overdrive pedal, real amp rhythm. |

The DI2 family established tone-dependent difficulty. The later DI3/DI4 and
hardware experiments refined capture workflow, latency handling, anti-aliasing
analysis, and architecture choices.

### 5.2 DI2 tone baseline

WaveNet won every major DI2 amp/pedal comparison. Preview ESR values are shown
below; lower is better.

| Tone | Best or notable preset | Preview ESR | Correlation | Interpretation |
| --- | --- | ---: | ---: | --- |
| Clean | `wavenet_tcn_quality` | 0.0052 | 0.9974 | Quality best; balanced nearly tied at 0.0058. |
| Crunch | `wavenet_tcn_quality` | 0.0115 | 0.9943 | Clear quality win. |
| Rhythm | `wavenet_tcn_quality` | 0.1100 | 0.9438 | Harder high-gain case; continuation justified. |
| Edge | `wavenet_tcn_quality` | 0.0045 | 0.9978 | Excellent result. |
| Lead | `wavenet_tcn_balanced` | 0.0810 | 0.9587 | Balanced narrowly beat quality. |
| Overdrive pedal | `wavenet_tcn_quality` | 0.0021 | 0.9990 | Excellent result. |

The baseline also showed that `conv1d_stack_prelu`, while fast and useful for
internal RTNeural coverage, underfit product captures. It trailed WaveNet on
clean, crunch, rhythm, edge, lead, and overdrive pedal.

### 5.3 Clean and edge-of-breakup behavior

The real clean amp project exposed an important architecture-specific failure.
The A2/PReLU preset initially improved but then diverged from full-stream
validation:

| Epoch | Stream ESR | Window ESR | Validation score | Prediction RMS ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.416 | 1.422 | 1.772 | 0.638 |
| 8 | 0.725 | 0.655 | 0.889 | 0.874 |
| 28 | 1.090 | 0.803 | 1.291 | 0.920 |

This was not primarily a latency issue. Preparation had stable 130-sample
alignment, inverted target polarity, 0.743 confidence, and unanimous candidate
agreement across active windows. The more plausible interpretation was
architecture/loss mismatch: a saturation-oriented A2/PReLU path was too
nonlinear for a clean amp-head transfer.

The `wavenet_tcn_clean` preset improved the situation, reaching ESR 0.0675,
MAE 0.0095, RMSE 0.0133, and correlation 0.9663. However, the specific capture
was only clean for single coils and closer to edge-of-breakup with humbuckers.
The `wavenet_tcn_edge` preset, which adds gentle nonlinearity to the clean
long-field design, then reached ESR 0.00362, MAE 0.00206, RMSE 0.00308, peak
residual 0.0896, and correlation 0.9982. A heavier `edge_detail` refinement did
not improve the result, confirming that the missing ingredient was light
nonlinearity rather than simply more channels.

### 5.4 High-gain RHYTHM4 architecture progression

The RHYTHM4 project is the clearest high-gain architecture study.

| Preset | ESR | RMSE | Correlation | Worst ASR | Avg ASR | Native RTF | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | 0.6309 | 0.0740 | 0.6109 | not measured in export study | not measured in export study | not measured in export study | Plateaued immediately. |
| `wavenet_tcn_quality` | 0.0713 | 0.0249 | 0.9640 | 0.290 | 0.100 | 12.17x | Strong waveform result; ASR warning. |
| `wavenet_tcn_quality_tanh15` | 0.0646 | 0.0237 | 0.9674 | 0.0670 | 0.0419 | 11.74x | Improved fit and probe ASR. |
| `wavenet_tcn_a2_prelu` | 0.0440 | 0.0196 | 0.9778 | 0.0354 | 0.0205 | 6.54x | Best fresh-run result. |
| `wavenet_tcn_a2_prelu` continued | 0.0381 | 0.0182 | 0.9808 | 0.0201 | 0.0145 | 6.62x | Best high-gain result. |

This sequence is scientifically useful because it generates several concrete
hypotheses, although it does not isolate them with matched repeated-seed
ablations:

- More capacity within the existing quality graph improves high-gain modeling.
- Smoothed tanh can help, but its effect is capture- and depth-dependent.
- The A2-inspired combination of mixed kernels, non-power dilations, and PReLU
  nonlinearities is associated with better waveform fit and better probe ASR in
  this case.
- The strongest model is slower and larger, but still real-time on the
  reference machine's native benchmark and debug-plugin smoke path.

### 5.5 RHYTHM3-B quality export

RHYTHM3-B provided a second high-gain case with a longer trimmed capture and
low-confidence latency. The successful quality export reported:

| Field | Value |
| --- | ---: |
| Preview/state-continuous ESR | 0.11670 |
| Stream validation ESR | 0.11075 |
| Window validation ESR | 0.10639 |
| RMSE | 0.03575 |
| Correlation | 0.94049 |
| Native worst-case RTF | 11.78x |
| Worst ASR | 0.06779 |
| Average ASR | 0.02516 |

The ASR warning was driven mainly by the approximately 5 kHz probe. Lower
probes were low. An amplitude sweep from 0.05 to 0.75 input amplitude kept
worst ASR in a similar warning band, suggesting the warning was not merely
caused by overdriving the default sine probe.

### 5.6 Real hardware exports

The strongest practical evidence came from the real hardware export trio.

| Export | Source | Preset | ESR | Worst ASR | Avg ASR | Native RTF | Latency | RF |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `export_clean` | Real amp clean/edge | `wavenet_tcn_edge` | 0.00217 | 0.00673 | 0.00251 | 20.72x | 130 samples | 6139 samples |
| `export_drive` | Real overdrive pedal | `wavenet_tcn_a2_prelu` | 0.00043 | 0.00195 | 0.00080 | 6.49x | 86 samples | 2481 samples |
| `export_rhythm` | Real amp rhythm | `wavenet_tcn_a2_prelu` | 0.00355 | 0.00913 | 0.00359 | 6.63x | 136 samples | 2481 samples |

All three passed native RTNeural validation and reported low probe ASR. This is
a major result: the real amp and pedal captures trained more reliably than many
of the earlier DAW amp-simulation renders. We should not overgeneralize from
one hardware set. The plausible contributors below are hypotheses for follow-up
experiments, not isolated causal effects:

- The capture workflow had matured by then.
- Float32 preparation was preserved.
- Source material was shorter and more focused.
- Real analog output plus ADC naturally band-limited some extreme digital
  artifacts.
- DAW amp-simulation renders may include hidden latency, oversampling filters,
  downsampling filters, and digital nonlinearities that are harder for a
  compact student model to match exactly.

## 6. Discussion

### 6.1 Why WaveNet won

Guitar amp and pedal modeling needs nonlinear memory. A dense model sees too
little temporal context. A shallow Conv model can be extremely fast but left
upper-band residuals and compression behavior behind in our internal
comparisons. Recurrent models are expressive, but they did not become the best
validated export path in this product workflow, and the current paper does not
present a matched recurrent-versus-TCN ablation. The supported claim is narrower
and more practical: under the current RTNeural JSON constraints and capture set,
WaveNet-family TCNs produced the strongest end-to-end product evidence.

WaveNet-style TCNs occupy the right compromise:

- They have explicit finite receptive field.
- They are causal and streamable.
- Dilation expands memory without requiring very deep sample-by-sample
  recurrence.
- Conv1D maps cleanly to RTNeural JSON.
- Width/depth/receptive field can be tuned by preset.
- Native validation and benchmarking are tractable.

The result is not full academic WaveNet. It is an RTNeural-compatible temporal
convolutional model shaped by plugin constraints. That constraint is a feature:
the system optimizes models that can actually be validated as real-time DSP on
the reference runtime before being considered for plugin use.

### 6.2 Latency as a bottleneck

Latency was the most persistent non-modeling difficulty. Heavy tones reduce the
linear correlation between DI and target, especially when the target is clipped,
compressed, and rich in high harmonics. The estimator can find several plausible
offsets separated by only a few samples. A one-sample error at 48 kHz is only
20.8 microseconds, but in a high-frequency residual it matters.

The project therefore moved from a single hidden estimate to a workflow:

- Estimate latency.
- Expose confidence and agreement.
- Show top candidate offsets.
- Let the user visually inspect waveform movement.
- Allow manual nudge.
- Encourage transient preamble capture.
- Use known latency when rendering from a deterministic DAW path.

This is not just UI polish. It is experimental control. Without reliable
alignment, architecture comparisons can become meaningless.

### 6.3 ESR, ASR, and listening

ESR is a good first metric because it is simple and energy-normalized. However,
for neural distortion it misses several questions:

- Is the prediction underpowered or overpowered?
- Is the residual mostly transient, broadband, or upper-band?
- Does the model generate foldback aliasing on high notes?
- Does the model sound good under a cabinet, EQ, or mix context?
- Does a lower ESR model produce worse artifacts?

ASR gives us a targeted stress test for nonlinear aliasing. It is not a final
audibility score. A model with warning ASR can still sound good, especially if
the warning is narrow, if the profile is intended to feed a cabinet IR, or if
the residual is below musical masking thresholds. A model with low ASR can still
fail musically if its waveform fit is poor. The correct export gate is therefore
multi-objective:

```text
export readiness =
  Python parity
  AND native RTNeural validation
  AND native benchmark headroom
  AND acceptable preview metrics
  AND ASR/listening review
  AND user judgment
```

### 6.4 Real hardware versus amp-simulation renders

One surprising pattern was that real hardware captures trained better than
several DAW amp-simulation renders. This should not be interpreted as evidence
that analog targets are categorically easier to model. A more disciplined
interpretation is that the whole measurement chain matters. A real pedal or amp-head capture may be easier for the model if
the analog chain and converter create a consistent, band-limited target. A DAW
plugin render may be internally oversampled, latency-compensated, filtered, or
phase-shifted in ways that are opaque to the trainer. If the target contains
high-frequency digital artifacts, a compact 48 kHz student model may reproduce
the overall tone while struggling with exact residuals.

The immediate practical conclusion is capture-specific: our current real
hardware captures are strong evidence that the workflow is viable. The research
conclusion is broader but cautious: future experiments should record and report
the full capture chain, including interface, reamp path, sample rate, buffer,
DAW/plugin latency compensation, and any hidden oversampling.

### 6.5 Runtime is currently acceptable but not solved

The A2-inspired high-gain model ran surprisingly well in the debug AU on the M5
Max reference machine. Four instances at a 32-sample buffer showed minimal
apparent CPU increase. This is encouraging, especially because the native
validator reported 6.6x worst-case RTF for the continued A2 export.

However, performance work is not finished:

- The reference machine is unusually powerful.
- Debug plugin smoke is not a release benchmark.
- Multi-instance tests need controlled CPU measurement.
- Pedal + amp + IR chains multiply cost.
- Dynamic JSON is not the RTNeural performance ceiling.
- Static `ModelT`, fused layers, and backend-specific kernels may become
  necessary for weaker machines.

The correct conclusion is that WaveNet/A2 is viable enough to continue, not
that optimization can be ignored.

## 7. Threats To Validity

### 7.1 Dataset size and diversity

The experiments used meaningful real captures, but they are still a small
internal dataset. The results are strongest for the user's guitars, interfaces,
amp/pedal settings, and source material. Generalization to other players,
interfaces, sample rates, pickups, cabinets, and gain structures remains to be
measured.

### 7.2 Metric calibration

ESR, RMSE, correlation, and ASR are useful engineering metrics, but they are not
substitutes for controlled listening tests. ASR thresholds are especially early.
They are deliberately warning-oriented and need calibration against audibility,
playing style, cabinet filtering, and mix context.

### 7.3 Alignment ambiguity

Some high-gain and lead captures had low latency confidence. Preview shift
checks often found no simple post-training offset, but ambiguous alignment can
still affect model ranking. Future reports should preserve the exact candidate
offset used for each run and encourage A/B training on top offsets when
confidence is low.

### 7.4 Hardware bias

Native benchmarks and DAW smoke tests were mostly on Apple Silicon, including a
MacBook Pro M5 Max with large memory. Runtime conclusions must be retested on
less powerful systems, Intel machines, and release builds.

### 7.5 Architecture search limitations

The current architecture search is not a full neural architecture search. It is
a curated set of RTNeural-safe presets. This is appropriate for product
stability, but it means untested architectures may outperform the current best
model.

## 8. Future Work

### 8.1 Better latency detection

The next latency work should combine multiple estimators:

- Transient/preamble alignment.
- Band-limited correlation.
- Envelope correlation.
- GCC-PHAT-like phase methods.
- Polarity-aware scoring.
- Multi-window agreement.
- Post-training residual shift diagnostics.

The UI should continue to show candidate offsets and waveform movement, because
human review is valuable when the target is heavily nonlinear.

### 8.2 Listening-calibrated ASR

ASR should become a calibrated perceptual aid. A useful study would render sine
probes, musical high-note passages, harmonics, palm mutes, and sustained bends
through several models, then compare ASR against blind listening judgments. The
goal is not to replace listening, but to set better warning thresholds.

### 8.3 True RTNeural A2 or residual TCN layer

Exact NAM A2 support requires topology that current dynamic RTNeural JSON does
not expose:

- Residual adds.
- Per-layer skip accumulation.
- Layer-array head convolution.
- Input mix-in paths.
- Optional slimmable submodel selection.

The recommended day-2 path is not a generic graph runtime first. It is a fused
`a2_wavenet` or `residual_tcn` layer in `shortwavlabs/rtneural-extended`, with
preallocated state and a constrained JSON schema. This mirrors NAM Core's own
fast-path approach and preserves RTNeural's sequential outer model.

### 8.4 Static and fused plugin runtimes

Dynamic JSON is excellent for flexibility and validation. Product plugins may
eventually need:

- Static `ModelT` specializations for built-in shapes.
- Weight loading into known static graphs.
- Fused Conv1D/activation blocks.
- Backend-specific Eigen/xsimd/Accelerate paths.
- Plugin-side oversampling or higher-rate model support for high-gain aliasing.
- Controlled pluginval and DAW matrix tests.

### 8.5 Parametric and active-learning captures

PANAMA-style parametric modeling is a natural future direction if the plugin
exposes controls such as gain or EQ. For MVP, one project models one fixed
setting. Later, active learning could choose knob positions and playing
material to efficiently cover a controllable amp surface.

### 8.6 NAM distillation and interoperability

Direct `.nam` to RTNeural conversion is hard for A2 models because the topology
does not match current RTNeural dynamic JSON. A practical near-term route is
distillation: render a NAM model on rich DI material and train an
RTNeural-compatible student. This would not be lossless conversion, but it
could provide useful interoperability and architecture research data.

## 9. Conclusion

The research converged on a scoped engineering thesis: for the current internal
48 kHz mono paired-capture workflow, RTNeural-compatible WaveNet/TCN exports
are the best-tested product path. The evidence is end-to-end rather than purely
loss-based. WaveNet-family models trained well on the tested captures, exported
cleanly, passed native validation, benchmarked above real time on the reference
runtime, and worked in a JUCE/Logic smoke path on the reference machine.

The current strongest high-gain model is `wavenet_tcn_a2_prelu`, an A2-inspired
sequential architecture that substantially improved RHYTHM4 ESR and probe ASR
while remaining RTNeural-compatible. The strongest real hardware evidence is the
clean/drive/rhythm export trio, where all three models passed validation with
low probe ASR and strong reference-system runtime headroom.

The most important lesson is that model architecture is only one part of the
DSP problem. Capture design, level discipline, latency alignment, spectral
loss, aliasing diagnostics, native runtime validation, and DAW playback behavior
all determine whether a model is musically useful. The successful path is
therefore not "train a neural network"; it is a full measurement and real-time
DSP pipeline with scientific controls at each stage.

## References

Project research notes:

- [Clean, Crunch, Rhythm, Edge, Lead, And Pedal Capture Baseline](Research-Clean-Crunch-Rhythm-Capture-Baseline.md)
- [WaveNet Amp Simulation Paper Review](Research-WaveNet-Amp-Simulation-Papers-2026-06-24.md)
- [RTNeural A2 Runtime Feasibility](Research-RTNeural-A2-Runtime-Feasibility.md)
- [RTNeural WaveNet And JUCE Plugin Performance Notes](Research-RTNeural-WaveNet-JUCE-Performance.md)
- [NAM Performance Notes For RTNeural Trainer](Research-NAM-Performance-And-WaveNet.md)
- [PANAMA / WaveNet Amp Modeling Findings](Research-PANAMA-WaveNet-Active-Learning.md)
- [Audio Capture Guidelines](Audio-Capture-Guidelines.md)

External references reviewed during project research:

- Neural Amp Modeler: <https://github.com/sdatkinson/neural-amp-modeler>
- NeuralAmpModelerPlugin: <https://github.com/sdatkinson/NeuralAmpModelerPlugin>
- NeuralAmpModelerCore: <https://github.com/sdatkinson/NeuralAmpModelerCore>
- RTNeural: <https://github.com/jatinchowdhury18/RTNeural>
- Shortwav Labs RTNeural fork: <https://github.com/shortwavlabs/rtneural-extended>
- PANAMA paper: <https://arxiv.org/html/2507.02109v1>
- PANAMA implementation: <https://github.com/ETH-DISCO/PANAMA>
- Aliasing Reduction in Neural Amp Modeling by Smoothing Activations: <https://arxiv.org/abs/2505.04082>
- Improving Unsupervised Clean-to-Rendered Guitar Tone Transformation Using GANs and Integrated Unaligned Clean Data: <https://arxiv.org/abs/2406.15751>
- Distortion Recovery: A Two-Stage Method for Guitar Effect Removal: <https://arxiv.org/abs/2407.16639>
- End-to-End Amp Modeling: From Data to Controllable Guitar Amplifier Models: <https://arxiv.org/abs/2403.08559>
- Hardware-aware black-box audio effects NAS, DAFx 2024: <https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_44.pdf>
- JUCE AudioProcessor documentation: <https://docs.juce.com/master/classjuce_1_1AudioProcessor.html>
- pluginval: <https://github.com/Tracktion/pluginval>

## Appendix A: Subagent Evidence Synthesis

This draft integrates three delegated reviews:

1. A capture/experiment evidence review covering DI2, DI3, DI4, RHYTHM3-B,
   RHYTHM4, and real hardware exports.
2. An architecture/literature review covering WaveNet, NAM, PANAMA, A2 topology,
   smoothed activations, and RTNeural compatibility.
3. A native runtime/plugin review covering RTNeural backends, export validation,
   benchmark matrices, and JUCE real-time constraints.

The subagent findings were used as evidence summaries, then cross-checked
against the repository's markdown research notes and metric implementation
files before this paper was written.
