# WaveNet Amp Simulation Paper Review

Reviewed: 2026-06-25

Input file: `/Users/shortwavlabs/Downloads/extract-data-2026-06-24.json`

This note reviews the WaveNet/neural amp simulation papers from the extracted
link set and maps them to RTNeural Trainer. It intentionally avoids repeating
the existing NAM and PANAMA notes except where these papers add a new action.

## Papers Reviewed

- [Improving Unsupervised Clean-to-Rendered Guitar Tone Transformation Using GANs and Integrated Unaligned Clean Data](https://arxiv.org/abs/2406.15751), DAFx 2024.
- [Distortion Recovery: A Two-Stage Method for Guitar Effect Removal](https://arxiv.org/abs/2407.16639), DAFx 2024.
- [Aliasing Reduction in Neural Amp Modeling by Smoothing Activations](https://arxiv.org/abs/2505.04082), DAFx 2025.
- [Parametric Neural Amp Modeling with Active Learning](https://arxiv.org/abs/2509.26564), ISMIR 2025.
- [Balancing Error and Latency of Black-Box Models for Audio Effects Using Hardware-Aware Neural Architecture Search](https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_44.pdf), DAFx 2024.
- [PANAMA implementation](https://github.com/ETH-DISCO/PANAMA), referenced by the PANAMA paper.

## Short Answer

Yes, this set is useful. The most actionable paper for the current app is
**Aliasing Reduction in Neural Amp Modeling by Smoothing Activations**. It
suggests that our WaveNet work should stop optimizing only for ESR/MRSTFT and
start measuring aliasing directly. It also gives a safe experiment for
RTNeural-compatible WaveNet variants: train with smoothed tanh activations and
export them by folding the stretch factor into the preceding Conv1D weights.

The second most useful paper is the **hardware-aware NAS** paper. It supports a
small, app-specific architecture search loop that ranks candidates by both
training error and native RTNeural benchmark results. This fits our current
export benchmark matrix nicely.

The GAN and distortion-recovery papers are useful directionally, especially for
perceptual loss and representative training data, but they are not immediate V1
implementation targets. PANAMA remains a future path for parametric captures,
especially if the eventual plugin exposes amp/EQ controls.

## What Is Already Aligned With Our App

The papers reinforce decisions already made in the project:

- WaveNet/TCN is a valid high-gain quality lane.
- Causal dilated Conv1D is the right finite-memory structure for RTNeural-safe
  exports.
- MR-STFT or spectral pressure matters for distorted guitar, not just waveform
  MSE.
- ESR alone can disagree with perceived audio quality.
- Runtime must be benchmarked on native inference, not inferred from Python
  model size.
- Capture amplitude consistency is not optional. The GAN paper specifically
  warns that amplitude differences can destabilize training, matching what we
  observed in our capture experiments.

Our current WaveNet presets already use `mrstft_preemphasis` as the default
loss. That is good. The next gap is anti-alias evaluation and activation
experimentation.

## Implementation Status

Implemented in the trainer app after this review:

- `rttrainer aliasing --model model.rtneural.json --report aliasing-report.json`
  renders deterministic sine probes through RTNeural JSON and computes ASR.
- `rttrainer export` now writes `aliasing-report.json`, includes it in
  `package.json`, and the Tauri export metadata rewrite preserves it beside
  validation, benchmark, and native backend matrix reports.
- The desktop export UI now surfaces an `Aliasing` report pill with status,
  verdict, worst ASR, average ASR, and probe count.
- Added RTNeural-safe smoothed-tanh WaveNet research presets:
  - `wavenet_tcn_balanced_tanh15`
  - `wavenet_tcn_balanced_tanh18`
  - `wavenet_tcn_quality_tanh18`
- These presets train with `tanh(x / alpha)` but export as ordinary RTNeural
  `tanh` by folding `1 / alpha` into the preceding Conv1D kernel and bias.
- Golden RTNeural JSON fixtures now cover the scaled-tanh presets with Python
  parity and native validator parity.
- Added `scripts/search_rtneural_presets.py` to generate a repeatable search
  plan or run train/export comparisons across the WaveNet, smoothed-tanh
  WaveNet, separable WaveNet, and stacked Conv presets.

Still deferred:

- Listening-calibrated ASR thresholds. ASR is warning-only for now.
- Plugin-side anti-aliasing or oversampling. The app can measure/report ASR,
  but the plugin runtime still needs its own performance and aliasing review.
- Full architecture NAS with newly generated architectures. The first script
  searches registered RTNeural-safe presets rather than inventing new presets
  dynamically.

## First Smoothed-Tanh Result

Rhythm2 follow-up runs on `project_98f406e8108d423ab624bc8ca5b1fcb7` gave the
first useful check on the anti-aliasing idea:

| Preset | Preview ESR | RMSE | Corr | Worst ASR | Average ASR | Corrected Est. RTF | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | `0.1463` | `0.0610` | `0.9245` | `0.1463` | `0.0643` | `3.0x` | Best waveform fit in this rerun. |
| `wavenet_tcn_balanced_tanh15` | `0.1873` | `0.0690` | `0.9025` | `0.4302` | `0.1803` | `3.0x` | Better smoothed-tanh fit, but ASR worsened. |
| `wavenet_tcn_balanced_tanh18` | `0.2058` | `0.0724` | `0.8934` | `0.1104` | `0.0498` | `3.0x` | Lower ASR than baseline, weaker waveform fit. |

The result supports the paper's main warning: smoothing activation functions is
not a free quality improvement. On this capture, `alpha = 1.5` gave the better
prediction metrics among smoothed-tanh variants, but it also produced the worst
ASR. `alpha = 1.8` did what the paper suggested on aliasing, but at a visible
ESR/RMSE cost. Treat `tanh18` as the actual anti-aliasing probe and `tanh15` as
an intermediate tone-quality experiment.

The smoothed-tanh presets do not reduce runtime by themselves. They export to
the same RTNeural layer structure as balanced WaveNet, with Conv1D weights
scaled before a normal `tanh`, so their runtime class is the balanced WaveNet
class. A `120x` display from the first run review was a runtime-estimator bug,
not a real native benchmark result.

## RHYTHM3B ASR Calibration Note

The second-generation RHYTHM3B quality export gives the first useful
listening-calibration target for ASR warnings on a successful high-gain model:

| Field | Value |
| --- | --- |
| Project | `project_f94c77d3aefe4f5e8abbaf3a86cfcf6a` |
| Run | `run_9a920dd9be4347369519547ada5d9395` |
| Export | `export_0459ae977d3e4e38b891718b94ec3305` |
| Preset | `wavenet_tcn_quality` |
| Preview/state-continuous ESR | `0.11670` |
| Stream validation ESR | `0.11075` |
| Native RTNeural parity | pass, max abs error `0.00002556` |
| Native Eigen worst RTF | `11.78x` |
| Worst ASR | `0.06779` |
| Average ASR | `0.02516` |
| ASR verdict | `review_aliasing` |

The warning was driven almost entirely by the ~`5 kHz` sine probe:

| Probe | ASR | Interpretation |
| ---: | ---: | --- |
| ~`1.25 kHz` | `0.00177` | Low. |
| ~`2.5 kHz` | `0.00593` | Low. |
| ~`5 kHz` | `0.06779` | Review by ear. |

An amplitude sweep showed the same basic behavior:

| Sine amplitude | Worst ASR |
| ---: | ---: |
| `0.05` | `0.05131` |
| `0.10` | `0.05891` |
| `0.20` | `0.06672` |
| `0.30` | `0.06968` |
| `0.50` | `0.06779` |
| `0.75` | `0.06593` |

This means the warning is not just an artifact of the default `0.5` probe being
too hot. The model likely has real high-frequency foldback risk on raw
high-gain material. It is still below the current `high_aliasing` threshold of
`0.08`, so the correct behavior is to surface the warning and listen for
metallic foldback on sustained high notes, bends, and harmonics.

Product implication:

- Keep ASR warning-only for now.
- Treat `0.02-0.08` as "listen carefully", not "reject".
- Treat raw high-gain amp-head exports as the main ASR stress case because no
  cabinet filter hides the upper harmonics.
- Plugin-side oversampling or a higher-sample-rate model path remains the most
  likely long-term fix when ASR is audible.
- Smoothed-tanh presets remain research candidates, not proven replacements:
  previous rhythm2 tests lowered ASR for `tanh18` but hurt ESR/RMSE.

## Actionable Findings

### 1. Add An Aliasing-To-Signal Ratio Gate

The Sato/Smith paper introduces ASR, an Aliasing-to-Signal Ratio metric, to
quantify aliasing energy in neural amp models. Their central finding is that
smoother activation functions reduce aliasing, and that the trade-off can be
measured against ESR.

Why it matters for us:

- High-gain WaveNet can sound good while still creating foldback artifacts.
- Native RTNeural validation currently checks parity and speed, not aliasing.
- ESR and preview residuals do not isolate aliasing from ordinary model error.

Recommended implementation:

- Add an offline `rttrainer` metric that renders deterministic sine tests
  through the exported model and computes ASR.
- Start with one or more high-ish fundamentals where amp nonlinearities create
  many harmonics, for example around 1.25 kHz, 2 kHz, and 4 kHz.
- Use a DFT setup where the test sine lands exactly on a bin and harmonic bins
  are known. The paper uses a prime-length one-second setup at 48,017 Hz; for
  our 48 kHz workflow we can either reproduce that in an analysis-only path or
  choose a practical coprime-bin FFT design at the project sample rate.
- Report ASR alongside ESR, MR-STFT, native parity, and native benchmark
  results in export metadata.
- Use ASR as a warning/report metric first, not a hard export blocker, until we
  have listening-calibrated thresholds from our own amp captures.

Suggested report language:

- `low aliasing`: ASR near or below the current WaveNet baseline.
- `review aliasing`: ASR noticeably worse than baseline, especially for high
  fundamentals.
- `quality trade-off`: ESR improved but ASR worsened, so listen for high-note
  grit or foldback artifacts.

### 2. Try Smoothed Tanh WaveNet Presets

The same paper shows CustomTanh with a stretch factor as a flexible
accuracy/aliasing trade-off. The notable balanced point reported is
CustomTanh around `alpha = 1.8`: lower aliasing while keeping ESR acceptable.

This is especially attractive because our current RTNeural JSON support already
handles `tanh`.

RTNeural-safe export approach:

```text
tanh(z / alpha) where z = Conv1D(x; W, b)
is equivalent at inference to:
tanh(Conv1D(x; W / alpha, b / alpha))
```

So we can train a Keras WaveNet with a custom `tanh(x / alpha)` activation, then
export as ordinary `tanh` by scaling the preceding Conv1D weights and bias
before writing RTNeural JSON. That keeps the runtime graph compatible with the
existing native validator.

Recommended experiment:

- Add research presets:
  - `wavenet_tcn_balanced_tanh15`
  - `wavenet_tcn_balanced_tanh18`
  - `wavenet_tcn_quality_tanh18`
- Keep the architecture identical to current balanced/quality WaveNet.
- Train on the captures where WaveNet already worked well: rhythm, lead,
  overdrive pedal.
- Compare ESR, MR-STFT, ASR, prediction preview, and native benchmark.
- Keep these labeled as research presets until listening tests show a repeatable
  benefit over the current balanced/quality presets.

Do not jump to gated WaveNet yet. The anti-alias paper notes that gated variants
can minimize ESR well but tend to introduce more aliasing. Gated blocks also
need exporter/runtime work beyond our current sequential RTNeural-safe graph.

### 3. Add A Small RTNeural-Aware Architecture Search

The hardware-aware NAS paper is directly relevant because it optimizes model
error and inference latency together. It also searched WaveNet/TCN-style
families with variable stack size, dilation growth, channels, kernels,
activations, residual convolutions, and skip/mixing choices.

We do not need a full evolutionary NAS system yet. A smaller project-specific
search would already help.

Recommended implementation:

- Add a script such as `scripts/search_rtneural_presets.py`.
- Generate RTNeural-safe candidates only:
  - architecture: current sequential Conv1D WaveNet
  - layers: 5, 6, 8, 10
  - filters: 12, 16, 20, 24
  - kernel: 3, 5, maybe 7
  - dilation patterns: powers of two, repeated powers, and shallower growth
  - activation: tanh, scaled tanh variants, maybe PReLU for non-WaveNet conv
- Train each candidate with short early-stopped runs.
- Export and run native validation/benchmark matrix for every candidate.
- Rank by Pareto frontier:
  - validation score
  - MR-STFT
  - ASR
  - native worst-case realtime factor
  - model size
  - receptive field

This would turn our current hand-picked presets into measured presets. The
app-side UI can stay simple while the underlying preset definitions become
better justified.

### 4. Keep GAN Training As A Research Branch, Not V1

The clean-to-rendered GAN paper uses a causal feed-forward WaveNet generator
and multi-scale/multi-period discriminators inspired by neural vocoders. The
important product idea is that discriminators are training-only: the exported
generator can stay lightweight.

What is useful now:

- Multi-period discriminators are designed to notice periodic/high-frequency
  structure, which is relevant to distorted guitar.
- Adversarial training can use unpaired clean DI data, which could help when
  paired captures are scarce or imperfect.
- The paper's amplitude-normalization warning supports our capture guidelines.

Why to defer:

- GAN training adds instability, extra model code, and new failure modes.
- Our current paired workflow is finally producing strong WaveNet results.
- We need ASR and preset search before adding a discriminator stack.

Possible later experiment:

- Keep the same RTNeural-safe WaveNet generator.
- Add a training-only discriminator branch in Keras/TensorFlow.
- Start with paired feature matching or MR-STFT-assisted adversarial loss before
  going fully unpaired.
- Treat this as an offline research mode, not the default desktop workflow.

### 5. Learn From Distortion Recovery, But Do Not Productize It Yet

The distortion recovery paper solves the inverse task: wet/effected guitar to
dry guitar. That is not our export target, but it provides two useful warnings:

- Objective metrics can disagree. Their results discuss ESR disagreeing with
  other perceptual/quality measures.
- Training on representative VST-derived data beat synthetic distortion for the
  real-world task.

Product implication:

- Keep displaying audio previews and residuals prominently.
- Add richer report language that explains metric conflicts.
- Consider SI-SDR and MR-STFT as secondary diagnostics, but avoid adding FAD
  unless we have a strong embedding dependency and enough reference audio to
  make it meaningful.

### 6. PANAMA Is Still V2, But More Relevant If The Plugin Has Controls

PANAMA trains parametric models conditioned on knob settings, using active
learning to choose which amp settings to record. The public repo is based on
NAM and provides WaveNet/LSTM configs, active-learning scripts, and inference
tools.

For the current trainer:

- Keep fixed-setting captures as V1.
- Do not ask users to record 75 knob states yet.

For the future plugin:

- If we add a 3-band EQ and input gain after the model, fixed-setting captures
  are still enough.
- If we want the neural model itself to respond like amp gain/tone knobs, then
  PANAMA-style conditioning and active capture guidance become relevant.
- A future "parametric project" would need:
  - a fixed DI stimulus
  - a list of target knob vectors
  - per-setting wet captures
  - active-learning suggestions for the next setting to record
  - a conditioned RTNeural-compatible export format

## Recommended Next Steps

### Near Term

1. Continue testing scaled tanh against current `wavenet_tcn_balanced` and
   `wavenet_tcn_quality` on lead/pedal captures, with rhythm2 as the first data
   point.
2. Use `scripts/search_rtneural_presets.py` when launching a small preset sweep,
   then run `scripts/compare_training_runs.py --export --native` on the
   completed run folders to compare ESR, ASR, native parity, and native
   benchmark metadata in one report.
3. Calibrate ASR report language against listening notes before turning it into
   a hard export gate.

### Medium Term

1. Convert the best search results into product presets only after native
   parity, ASR, benchmark, and listening checks agree.
2. Add a report section that distinguishes "low error" from "low aliasing."
3. Add optional SI-SDR and richer MR-STFT diagnostics for metric conflict cases.
4. Explore training-only adversarial loss only after the above metrics are
   stable.

### Deferred

- Full gated WaveNet residual/skip architecture.
- Full GAN training as a default desktop mode.
- Unpaired clean-data training.
- PANAMA-style parametric capture projects.
- Conditioned RTNeural exports for amp knob controls.

## Bottom Line For This Project

The main lesson is not "make WaveNet bigger." It is:

1. Keep WaveNet as the high-gain quality path.
2. Measure aliasing explicitly.
3. Try smoother activations before widening the model.
4. Rank presets by native runtime and audio quality together.

That gives us a realistic way to improve high-gain sound quality without
blindly increasing layer count, model size, and plugin CPU cost.
