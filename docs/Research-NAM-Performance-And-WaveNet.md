# NAM Performance Notes for RTNeural Trainer

Reviewed: 2026-06-22

Implementation status: the app now exposes `wavenet_tcn_fast`,
`wavenet_tcn_balanced`, and `wavenet_tcn_quality`; keeps legacy `wavenet_tcn`
for existing checkpoints; recommends WaveNet as the amp quality lane; includes
a one-click "Continue best WaveNet" refinement helper; and surfaces candidate
latency offsets for low-confidence captures. Export now runs a native RTNeural
benchmark matrix across realistic block sizes and mono/stereo channel counts,
writes parity snapshot input/output artifacts, and uses report language
calibrated from the clean/crunch/rhythm listening and metric results.

This note captures what we can learn from the Neural Amp Modeler ecosystem after the continued
`wavenet_tcn` run on the current high-gain capture. It focuses on product and engineering choices
for RTNeural Trainer, not on adopting NAM's file format or copying its code.

## Short Answer

Yes: optimize around WaveNet-style temporal convolution for high-gain tones.

The continued run `run_68e6fb8ab0b64c86b39a133a2333dea6` confirms the direction:

- Preset: `wavenet_tcn`
- Resume source: `run_c3525ffc63634b9ca6a136ab3ccd40de`
- Backend/device: Keras on TensorFlow GPU
- Total epoch reached: 240
- Preview ESR: `0.1202`
- Preview RMSE: `0.0553`
- Continuous correlation: `0.9385`
- Best checkpoint epoch: `240`
- Stream validation ESR at epoch 240: `0.2710`
- Window validation ESR at epoch 240: `0.0896`
- Prediction RMS ratio: `0.9601`

The important trend is that epoch 240 was still marked as a new best. That means the preset had not
obviously saturated yet. More epochs, a lower final learning rate schedule, or a staged quality run
are justified for high-gain captures.

For product defaults, this suggests:

- High gain: default to WaveNet/TCN quality presets.
- Medium and low gain amp captures: start with WaveNet balanced; use Stacked
  Conv as the fast CPU fallback until a different capture family proves it can
  serve as the quality default.
- Fast CPU preview/export checks: keep Dense, GRU, and smaller Conv presets as sanity and baseline
  paths, not as the main high-gain quality lane.

## NAM Repositories Reviewed

- [sdatkinson/neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler)
- [sdatkinson/NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin)
- [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore), used as a
  submodule by the plugin repository.

## What NAM Does Well

### Separate training/export from real-time inference

The Python repository is explicitly for training models and exporting `.nam` files, while the plugin
repository is the real-time playback target. This is a useful architectural boundary for us:

- `rttrainer` should remain the training/export/report generator.
- The native validator should evolve into a reliable RTNeural runtime confidence gate.
- Future plugin examples should consume exported artifacts, not training internals.

### Treat the model package as a runtime contract

NAM's export path writes a model dictionary with:

- format/version metadata
- non-user metadata such as export date
- architecture name
- architecture config
- flattened weights
- optional snapshot input/output arrays

For us, richer RTNeural package metadata should become first-class. The export bundle should include:

- app/trainer/export schema versions
- preset id and architecture family
- sample rate and expected channel policy
- latency and receptive field
- input/output gain normalization data
- training recipe and resume source
- preview metrics
- native validator result
- native benchmark result
- optional parity snapshot data

This is more useful than only writing the RTNeural JSON and a short report.

### Benchmark the actual native model

NeuralAmpModelerCore includes separate tools for:

- loading a model
- running tests
- benchmarking a model's real-time speed

That maps directly to a missing product gate in our app: the export report should not just say
"validation passed"; it should show the benchmark for the exact exported model on the user's machine.
For WaveNet-quality presets, this matters more than for small LSTM/GRU models.

### Optimize the C++ runtime around known audio patterns

The NAM core uses a dedicated C++ DSP layer instead of asking the plugin to interpret training graphs.
Notable patterns worth adopting conceptually:

- prewarm stateful models after reset
- set max buffer sizes up front
- avoid model parsing/loading on the audio thread
- stage new model/IR objects, then swap them into the DSP path
- keep input history/receptive-field buffers inside the model runtime
- benchmark small-buffer behavior, not only offline throughput
- use fast activation paths where quality permits

These ideas line up with real-time plugin practice: no heap allocation, blocking IO, locks, or parser
work in the audio callback.

### Slimmable/packed models are worth tracking

NAM has active support for slimmable and packed WaveNet variants. The product idea is powerful:
one capture can expose a quality/performance control at runtime. We should not chase this immediately,
but it is relevant to a future "quality vs CPU" export mode.

Near-term equivalent for us:

- `wavenet_tcn_fast`
- `wavenet_tcn_balanced`
- `wavenet_tcn_quality`

Each should be separately benchmarked and documented. Later, if RTNeural or our native sidecar can
support a true slimmable envelope, revisit a single model with selectable cost.

## What Applies To RTNeural Trainer Now

### 1. Promote WaveNet as the high-gain recommendation

The current run is the first one where the subjective result and metrics both strongly support the
same direction. The app should recommend WaveNet when:

- gain is high or the target has dense clipping/saturation
- capture length is long enough
- user is willing to spend more training time
- native benchmark passes for the target machine

The UI language should be clear: "WaveNet quality" is not just another preset; it is the recommended
high-gain path.

### 2. Add a staged WaveNet training recipe

A good next recipe:

- Phase 1: train or resume with current `wavenet_tcn` settings until improvement slows.
- Phase 2: lower learning rate by 2x to 4x, continue from best checkpoint.
- Phase 3: run a final validation/export pass with longer preview windows and native benchmark.

The current run had no learning-rate reduction near the end, while it continued improving. That
suggests room for an automatic "continue quality run" mode.

### 3. Make RTNeural runtime performance visible

WaveNet can sound right and still fail as a real-time plugin model if the exported runtime is too slow.
For every WaveNet export we should show:

- realtime factor
- worst tested block size
- sample rate
- CPU model/backend when available
- validator parity error
- model size
- receptive field
- export latency

The existing `rtneural-validator` now covers the first version of this gate: it writes a
block-size/channel matrix, preserves the conservative worst-case real-time factor, and includes
model size, latency, architecture, and inferred Conv1D receptive-field metadata. The next layer is to
calibrate preset-specific pass/fail language from more captures rather than relying on one universal
runtime threshold.

### 4. Keep Stacked Conv as the fast fallback

On the clean/crunch/rhythm test set, Stacked Conv improved over the baseline
Conv preset but lagged WaveNet on every amp capture. That does not make it a bad
preset. It remains valuable when:

- the tone is cleaner or less saturated
- training speed matters
- CPU budget is limited
- user wants a quick first model before a quality run

We should keep validating this with lead, edge-of-breakup, pedal, and line-level
captures, but the current app should not describe it as the default amp-quality
path.

### 5. Improve package compatibility rather than adopt `.nam`

NAM's `.nam` format is a strong example of a model package, but RTNeural Trainer's target remains
RTNeural JSON. We should learn from the metadata and testing discipline, not switch formats.

Deferred idea: a future `.aidax` envelope can carry RTNeural JSON plus reports, snapshots, and
license metadata after format and licensing review.

## Implementation Plan And Status

1. Add WaveNet recommendation logic: implemented.

   Promote `wavenet_tcn` when the capture is high gain or when the user chooses "quality". Add clear
   UI copy that it is slower but better for dense distortion.

2. Add WaveNet recipe variants: implemented.

   Add `wavenet_tcn_fast`, `wavenet_tcn_balanced`, and `wavenet_tcn_quality`. Keep the current preset
   as balanced unless benchmark results suggest otherwise.

3. Add a "continue quality run" action: implemented.

   Let the user resume from the best checkpoint with a lower learning rate and more epochs without
   manually reconstructing settings.

4. Expand native validator benchmarking: implemented for local v1.

   Add a benchmark mode that runs the exported RTNeural JSON at realistic sample rates and block sizes,
   then writes results into the export package. The current report includes worst-case and per-case
   timing across block-size/channel combinations, plus model size, latency, architecture, and
   receptive-field metadata.

5. Add parity snapshots to exports: implemented.

   Each export now saves a deterministic short input/output pair plus
   `parity-snapshot.json`. The final package preserves these artifacts after
   native validation and benchmark reports are written.

6. Make export quality gates architecture-aware: partially implemented.

   Dense/GRU/LSTM/Stacked Conv/WaveNet should have different benchmark expectations. A small Dense
   model and a WaveNet quality model should not be judged with identical runtime language. The first
   runtime gate now uses `>= 1x` native real-time factor for exported models, and the WaveNet report
   language is calibrated around the clean/crunch/rhythm baseline. Those captures
   drove the current `excellent`/`good`/`usable` pass; edge-of-breakup, lead,
   pedal, and bad-capture fixtures should drive the next threshold pass.

7. Track slimmable models as a later research item: deferred.

   Do not build this now. First prove separate fast/balanced/quality WaveNet presets and native
   benchmark reporting.

## Risks And Caveats

- The current conclusion is strongest for high-gain rhythm captures. Clean and edge-of-breakup tones
  still need a preset shootout.
- Preview ESR and stream validation ESR differ because they measure different spans/conditions. The
  preview result is the better proxy for what the user auditioned; stream validation is still useful
  for training consistency.
- Native RTNeural performance must be measured before we claim WaveNet is plugin-ready.
- NAM code is MIT-licensed, but we should avoid copying implementation details unless we perform a
  dedicated dependency and license review.

## Sources

- NAM training/export repo:
  [sdatkinson/neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler)
- NAM plugin repo:
  [sdatkinson/NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin)
- NAM core C++ DSP repo:
  [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
- NAM WaveNet implementation:
  [nam/models/wavenet](https://github.com/sdatkinson/neural-amp-modeler/tree/main/nam/models/wavenet)
- NAM export interface:
  [nam/models/exportable.py](https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam/models/exportable.py)
- NAM Plugin main processor:
  [NeuralAmpModeler.cpp](https://github.com/sdatkinson/NeuralAmpModelerPlugin/blob/main/NeuralAmpModeler/NeuralAmpModeler.cpp)
- NAM Core benchmarking/build tools:
  [tools/CMakeLists.txt](https://github.com/sdatkinson/NeuralAmpModelerCore/blob/main/tools/CMakeLists.txt)
