# WaveNet Amp Simulation Paper Review

Reviewed: 2026-06-24

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

- Add hidden/research presets:
  - `wavenet_tcn_balanced_tanh15`
  - `wavenet_tcn_balanced_tanh18`
  - `wavenet_tcn_quality_tanh18`
- Keep the architecture identical to current balanced/quality WaveNet.
- Train on the captures where WaveNet already worked well: rhythm, lead,
  overdrive pedal.
- Compare ESR, MR-STFT, ASR, prediction preview, and native benchmark.
- Only expose these in the main UI if they beat or clearly complement the
  current balanced/quality presets.

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

1. Add ASR calculation as a standalone metric script or trainer command.
2. Add ASR fields to export/report metadata, initially behind a warning-only
   UI.
3. Add scaled-tanh WaveNet research presets using export-time weight folding.
4. Test scaled tanh against current `wavenet_tcn_balanced` and
   `wavenet_tcn_quality` on the known rhythm/lead/pedal captures.
5. Add a small preset-search script that can train/export/benchmark a grid of
   RTNeural-safe WaveNet candidates.

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
