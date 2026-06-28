# RTNeural WaveNet And JUCE Plugin Performance Notes

Reviewed: 2026-06-22
Updated: 2026-06-25

## Short Answer

WaveNet is the right quality lane, but the product should treat WaveNet export
as a native runtime problem, not only a training problem.

The highest-leverage performance actions are:

1. Benchmark every exported WaveNet model with multiple RTNeural backends:
   STL, Eigen, xsimd, and AVX where available.
2. Stop treating any single dynamic-validator result as the final plugin
   answer.
3. Add a compile-time `ModelT` validation/benchmark path for known preset
   shapes.
4. Benchmark grouped or depthwise-style dilated Conv1D blocks while staying
   inside RTNeural JSON support.
5. Build the eventual JUCE plugin around real-time constraints from day one:
   no parsing, allocation, locking, file IO, UI calls, or model swaps in
   `processBlock()`.

The current app is already doing one important thing correctly: it benchmarks
the exported native RTNeural JSON. The app-side gap around the STL-only
validator has now been closed: local/dev and packaged validator builds can use
Eigen by default, with STL and xsimd available as explicit build targets when
their dependencies are present.

## Implementation Update

Implemented in the trainer app on 2026-06-22:

- `native/rtneural-validator` now supports
  `RTNEURAL_VALIDATOR_BACKEND=stl|eigen|xsimd`.
- Tauri sidecar packaging defaults the native validator to Eigen and accepts
  `--validator-backend` plus optional AVX flags.
- `pnpm --filter rtneural-trainer-app build:validators` builds local backend
  variants for benchmark comparisons.
- The dev validator shim can select a local backend with
  `RTNEURAL_VALIDATOR_BACKEND=eigen|stl|xsimd`.
- An experimental `wavenet_tcn_separable_fast` preset was added with grouped
  dilated Conv1D plus 1x1 pointwise mixing.
- Golden JSON fixtures, Python parity, and native RTNeural validation now cover
  the grouped Conv1D export path.

Local benchmark snapshot on Apple Silicon with dynamic RTNeural JSON:

| Preset | Eigen Worst RTF | STL Worst RTF | Notes |
| --- | ---: | ---: | --- |
| `wavenet_tcn_fast` | `35.54x` | `22.45x` | Fastest current WaveNet runtime lane. |
| `wavenet_tcn_balanced` | `21.35x` | `10.36x` | Eigen is the clear app-side win. |
| `wavenet_tcn_quality` | `12.41x` | `5.10x` | Still viable dynamically on this machine, but plugin headroom remains unproven. |
| `wavenet_tcn_separable_fast` | `9.24x` | `8.50x` | Parity-safe, but not faster than balanced in dynamic JSON. Keep experimental. |

The separable result is important: reducing theoretical MACs did not beat the
extra dynamic-layer overhead in the current validator path. It may still become
useful with static `ModelT`, xsimd, or a fused plugin-side implementation, but
it should not replace `wavenet_tcn_fast` or `wavenet_tcn_balanced` yet.

Follow-up trained lead-capture result:

| Preset | Test ESR | RMSE | Correlation | Eigen Worst RTF | Model Size | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | `0.08098` | `0.05532` | `0.95874` | `18.35x` | `211 KB` | Best metric/runtime balance. |
| `wavenet_tcn_quality` | `0.08210` | `0.05570` | `0.95822` | `11.14x` | `414 KB` | Similar quality, slower and larger. |
| `wavenet_tcn_separable_fast` | `0.08332` | `0.05612` | `0.95768` | `8.90x` | `96 KB` | Quality is on par; dynamic runtime is still slower. |

This makes the separable preset more interesting than the fixture benchmark
alone suggested. It can reach comparable quality with a much smaller JSON model,
but the current dynamic RTNeural path still pays for the extra layer count.

Second-generation rhythm quality export:

| Preset | Test ESR | RMSE | Correlation | Eigen Worst RTF | Model Size | ASR | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality` | `0.11670` | `0.03575` | `0.94049` | `11.78x` | `414 KB` | `0.0678` | Good high-gain rhythm candidate; ASR review warning at ~5 kHz. |

This is the first quality WaveNet export from the trimmed RHYTHM3B capture that
both sounded promising and passed native validation. The result supports keeping
`wavenet_tcn_quality` as the production lane for hard raw amp-head rhythm
profiles. Runtime is not the immediate blocker on this machine; aliasing review
is. The plugin-side roadmap should therefore keep oversampling or higher-rate
model support in scope for high-gain WaveNet, even when native RTNeural
benchmark headroom is comfortable.

DI4/RHYTHM4 adds a shorter-capture control point. At 158.8 seconds, the file
trained faster than the longer DI3/RHYTHM3B captures while still preserving
enough variation to separate presets. `wavenet_tcn_balanced` plateaued early
at ESR `0.6309`, while `wavenet_tcn_quality` continued to epoch 120 and reached
ESR `0.1369` with correlation `0.9295`. Its export passed native parity with
max error `2.4e-5` and benchmarked at roughly `12x` worst-case Eigen realtime,
but the aliasing probe warned most strongly around 5 kHz.

That result motivated `wavenet_tcn_high_gain`: one extra dilation stage
(`1024`, receptive field `4095` samples / `85.3 ms`) and a lower default
learning rate (`3.5e-4`). The first DI4/RHYTHM4 run did not validate the idea:
it stopped at epoch 37 with ESR `0.6310`, correlation `0.6142`, and essentially
the same failure mode as `wavenet_tcn_balanced`. Preview analysis put the best
target/prediction lag at `0 samples`, and optimal gain scaling only improved
ESR from `0.6310` to about `0.6228`, so this was not a simple alignment or
level miss. Layer probes showed the hidden Conv1D activations staying around
`0.015-0.020` RMS, while the successful quality checkpoint reached roughly
`0.47-0.64` RMS in deeper layers. The current conclusion is that adding a
plain eleventh tanh Conv1D layer creates an optimization wall. Longer receptive
field work should move toward residual/skip/gated blocks or a stronger warmup
schedule instead of another sequential dilation layer.

Rhythm2 smoothed-tanh follow-up:

| Preset | Preview ESR | RMSE | Worst ASR | Est. RTF | Runtime Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_balanced` | `0.1463` | `0.0610` | `0.1463` | `3.0x` | Baseline balanced graph. |
| `wavenet_tcn_balanced_tanh15` | `0.1873` | `0.0690` | `0.4302` | `3.0x` | Same graph cost after export-time activation folding. |
| `wavenet_tcn_balanced_tanh18` | `0.2058` | `0.0724` | `0.1104` | `3.0x` | Same graph cost, lower ASR than baseline. |

The smoothed-tanh presets are not a performance optimization. They train with a
different activation slope, then export by scaling Conv1D weights before a
standard RTNeural `tanh`. That changes model behavior and ASR, but it does not
remove layers, filters, or activation calls. Any earlier `120x` estimate for
these presets was the trainer falling through to the generic Conv1D estimate;
the corrected estimate is balanced WaveNet class.
That keeps the main recommendation unchanged: use balanced as the practical
default, keep separable for static/fused/plugin-side experiments.

The local RTNeural checkout currently lacks vendored xsimd headers at
`modules/xsimd/include/xsimd/xsimd.hpp`, so xsimd builds are skipped by the
matrix helper unless that dependency is initialized.

## Sources Reviewed

Local RTNeural repositories:

- `/Users/shortwavlabs/Workspace/rt-neural/RTNeural`
- `/Users/shortwavlabs/Workspace/rt-neural/RTNeural-example`
- `/Users/shortwavlabs/Workspace/rt-neural/RTNeural-compare`

The local RTNeural checkout is the Shortwav Labs fork
[`shortwavlabs/rtneural-extended`](https://github.com/shortwavlabs/rtneural-extended),
so future work can modify RTNeural itself when the app/plugin needs graph,
layer, or fused-kernel support that upstream dynamic JSON does not expose.

Project files reviewed:

- `trainer/rttrainer/models/presets.py`
- `trainer/rttrainer/export_rtneural/keras_exporter.py`
- `native/rtneural-validator/CMakeLists.txt`
- `native/rtneural-validator/src/main.cpp`

External docs:

- [JUCE AudioProcessor docs](https://docs.juce.com/master/classjuce_1_1AudioProcessor.html)
- [JUCE ScopedNoDenormals docs](https://docs.juce.com/master/classjuce_1_1ScopedNoDenormals.html)
- [JUCE dsp::ProcessSpec docs](https://docs.juce.com/master/structjuce_1_1dsp_1_1ProcessSpec.html)
- [pluginval](https://github.com/Tracktion/pluginval)

## Current WaveNet Runtime Shape

The app's WaveNet presets are sequential Keras models exported to RTNeural JSON:

| Preset | Conv1D Layers | Filters | Kernel | Dilations | Receptive Field |
| --- | ---: | ---: | ---: | --- | ---: |
| `wavenet_tcn_fast` | 6 | 12 | 3 | `1..32` | 127 samples |
| `wavenet_tcn_balanced` | 8 | 16 | 3 | `1..128` | 511 samples |
| `wavenet_tcn_quality` | 10 | 20 | 3 | `1..512` | 2047 samples |
| `wavenet_tcn_high_gain` | 11 | 20 | 3 | `1..1024` | 4095 samples |
| `wavenet_tcn_separable_fast` | 15 Conv1D ops | 16 | 3 plus 1 | `1..128` | 511 samples |

The model graph is intentionally simple:

- Causal dilated `Conv1D`
- Tanh activation on each Conv1D block
- Final dense output with tanh

This is not full WaveNet with gated residual/skip blocks. That is good for
RTNeural compatibility, but it also means performance is dominated by repeated
full-channel Conv1D layers.

For a full Conv1D block, rough multiply-add cost per sample is:

```text
in_channels * out_channels * kernel_size
```

Once the first layer expands from 1 channel to N channels, later blocks are
roughly `N * N * 3`. That makes width expensive:

| Width | Approx Conv MACs Per Middle Block |
| ---: | ---: |
| 12 | 432 |
| 16 | 768 |
| 20 | 1200 |

This explains why `wavenet_tcn_quality` can sound better but needs much more
runtime care than balanced.

## RTNeural Findings

### 1. Use RTNeural Backends Strategically

RTNeural supports STL, Eigen, and xsimd backends. The RTNeural README says Eigen
is generally best for larger networks, while smaller networks may perform better
with xsimd. The comparison repo confirms the answer is shape-dependent.

The original validator did this:

```cmake
set(RTNEURAL_STL ON CACHE BOOL "Use RTNeural STL backend" FORCE)
```

That made the original benchmark conservative but incomplete. It was useful as
a baseline, not as the final product runtime result.

Implemented app-side:

- Added `RTNEURAL_VALIDATOR_BACKEND=stl|eigen|xsimd`.
- Added sidecar packaging flags for backend and AVX selection.
- Added a local backend build helper.
- Export now writes `native-benchmark-matrix.json`, embeds it in
  `package.json` as `benchmark_matrix`, and surfaces fastest backend/headroom
  in the Export UI.

Still pending:

- Package every backend-specific validator sidecar in signed release bundles if
  we want production apps to benchmark all variants offline.
- Build xsimd once the RTNeural xsimd headers/submodule are present on CI and
  release builders.

Expected outcome:

- Balanced and quality WaveNet exports move from conservative STL numbers to
  more realistic Eigen numbers on supported machines.
- Backend choice can become architecture-specific instead of global.

### 2. Dynamic JSON Models Are Convenient, Not The Performance Ceiling

RTNeural has two model APIs:

- Dynamic `RTNeural::Model<float>`, created from JSON at runtime.
- Static `RTNeural::ModelT<float, ...>`, with layer shapes known at compile time.

RTNeural's own README says the compile-time API can significantly improve
performance. The example plugin embeds one runtime model path and one
compile-time `ModelT` path, then lets the user switch between them. The
comparison repo also benchmarks static and dynamic paths separately.

The dynamic path is valuable for:

- Loading arbitrary user-exported JSON.
- Native validation.
- Development and compatibility checks.

The plugin-ready path should be:

- Precompiled `ModelT` specializations for known built-in presets.
- JSON weight loading into the static shape.
- Runtime fallback only for custom or unsupported shapes.

Action:

- Generate a C++ `ModelT` type from exported JSON metadata for each built-in
  preset shape.
- Add a `static-benchmark` validator mode that compares:
  dynamic RTNeural JSON vs static `ModelT` with the same weights.
- Add static parity snapshots before using static inference in plugin code.

### 3. Conv1D Is The Right Primitive, But Dilated WaveNet Is The Hot Path

RTNeural's Conv1D implementation is temporal and stateful. It stores past inputs
in a ring buffer and supports dilation and groups. That maps well to our
WaveNet/TCN presets.

Important implementation details:

- Dynamic Conv1D uses virtual layer dispatch through `Layer<T>`.
- Dynamic Conv1D allocates state and weight containers when the model loads, not
  during `forward()`.
- Static Conv1D removes virtual dispatch and makes dimensions compile-time.
- The xsimd static Conv1D has special cases for `kernel_size == 1` and
  `dilation == 1`.
- Our WaveNet middle layers are mostly `kernel_size == 3` and `dilation > 1`, so
  they hit the dilated temporal path.

Action:

- Benchmark exact exported WaveNet shapes rather than generic Conv1D layers.
- Include fast, balanced, and quality presets in the native benchmark matrix.
- Add a local micro-benchmark for just the dilated Conv1D stack, separate from
  JSON parsing and report generation.

### 4. Grouped Conv1D Is A Promising RTNeural-Compatible Experiment

RTNeural's Conv1D loader supports a `groups` field, and our Keras exporter
already writes `groups` for Conv1D layers.

This gives us a strategic performance experiment: a separable WaveNet-style TCN.

Instead of each block being:

```text
full dilated Conv1D: C in -> C out, kernel 3
```

Try:

```text
grouped/depthwise dilated Conv1D: C in -> C out, kernel 3, groups C
pointwise Conv1D or Dense: C -> C, kernel 1
tanh
```

For width 16:

- Full Conv1D is roughly `16 * 16 * 3 = 768` MACs per middle block.
- Depthwise plus pointwise is roughly `(16 * 3) + (16 * 16) = 304` MACs.

For width 20:

- Full Conv1D is roughly `20 * 20 * 3 = 1200` MACs.
- Depthwise plus pointwise is roughly `(20 * 3) + (20 * 20) = 460` MACs.

That is around a 60 percent reduction in the dominant block cost before
activation overhead.

Caveats:

- This changes model capacity and may hurt quality.
- RTNeural grouped Conv1D parity needs golden fixtures before product use.
- Keras grouped Conv1D must export exactly as RTNeural expects.
- The pointwise layer must be represented by supported RTNeural layers.

Implemented app-side:

- Added a `wavenet_tcn_separable_fast` experimental preset.
- Added Keras export parity and native RTNeural parity fixtures for grouped
  Conv1D.

Finding:

- The separable preset was not faster than `wavenet_tcn_balanced` in the current
  dynamic JSON validator. Keep it as an experiment, not as the recommended
  performance path.

Still pending:

- Test against clean, crunch, rhythm, lead, and pedal captures.
- Re-test with static `ModelT` and xsimd before deciding whether to keep or
  remove the separable family.

### 5. Activation Cost Matters, But Do Not Optimize It First

Each current WaveNet Conv1D block has a tanh activation. RTNeural represents that
as a separate activation layer after Conv1D. xsimd and Eigen both provide
vectorized activation paths.

Possible future optimizations:

- Try PReLU or ReLU variants for lower activation cost.
- Try a custom fast-tanh layer.
- Fuse Conv1D plus activation in a custom layer.

Recommendation:

Do not start here. Backend selection, static `ModelT`, and grouped Conv1D are
more likely to produce large wins without changing the sound. Activation
changes should be tested only after we have benchmark data for the current
architecture.

### 6. Float Is The Correct Plugin Runtime Type

The validator uses `RTNeural::Model<float>`, and the example plugin uses
`float`. Keep plugin inference in float unless a specific model proves unstable.
Double precision will be slower and is unnecessary for this audio use case.

## JUCE Plugin Performance Findings

### 1. Treat `processBlock()` As A Hard Real-Time Function

JUCE's `AudioProcessor::processBlock()` is called by the audio thread. It must
handle variable block sizes, including possible zero-sample buffers, and it
should not interact with the UI.

For an RTNeural WaveNet plugin, `processBlock()` should only:

- Read cached atomic parameter values.
- Iterate channels and samples.
- Call preloaded model instances.
- Apply simple gain/utility DSP.
- Update lightweight atomics for meters.

It must not:

- Parse JSON.
- Open files.
- Allocate or resize vectors.
- Lock mutexes.
- Update UI components.
- Rebuild models.
- Log or format strings in production.

### 2. Load And Swap Models Off The Audio Thread

The RTNeural example plugin parses embedded JSON in the constructor and resets
models in `prepareToPlay()`. That is fine for a fixed model.

For a user-loadable plugin:

- Load JSON on the message thread or a background worker.
- Build the model into an immutable object.
- Warm/reset it.
- Atomically swap a pointer to the new model between blocks.
- Keep the old model alive until the audio thread is done with it.

For stereo, stateful temporal models need independent model state per channel.
Do not share one stateful model instance across channels.

### 3. Use `prepareToPlay()` For Allocation And Reset

Use `prepareToPlay(sampleRate, maximumExpectedSamplesPerBlock)` to:

- Allocate any buffers.
- Construct per-channel model state if the model is already loaded.
- Reset RTNeural model state.
- Prepare JUCE DSP utilities.
- Store sample rate and max block size for diagnostics.

JUCE's `dsp::ProcessSpec` exists for this exact setup pattern: sample rate,
maximum block size, and channel count.

### 4. Prevent Denormals

Use `juce::ScopedNoDenormals` at the top of `processBlock()`. Dense distortion
models can decay into very small floating-point states, and denormals can cause
large CPU spikes on some processors.

### 5. Cache Parameter Atomics

The RTNeural example plugin caches APVTS raw parameter pointers in the
constructor and reads them in `processBlock()`. Keep that pattern.

Avoid repeated parameter lookup by string inside the audio callback.

### 6. Receptive Field Is Not Plugin Latency

WaveNet's causal Conv1D receptive field is memory, not lookahead. The balanced
preset's 511 samples and quality preset's 2047 samples describe how much past
context the model uses.

Do not report that as JUCE plugin latency unless the plugin deliberately delays
audio. The plugin should report zero added latency for causal inference.

Capture alignment latency in the export metadata is also not necessarily plugin
latency. It describes how the recorded target was aligned for training. The
plugin should not blindly call `setLatencySamples()` from that field.

### 7. Validate With pluginval And Small Buffers

A model that benchmarks at `1.1x` realtime offline is not plugin-ready. DAWs need
headroom for the host, UI, automation, other plugins, OS scheduling, and small
buffer sizes.

Use pluginval and DAW smoke tests at:

- 32 samples
- 64 samples
- 128 samples
- 48 kHz and 96 kHz where possible
- mono and stereo

Recommended export language:

| Worst Native RTF | Suggested Label |
| ---: | --- |
| `< 1x` | Not realtime |
| `1x - 2x` | Risky, offline/export only |
| `2x - 4x` | Usable with caution |
| `4x - 8x` | Plugin-ready |
| `> 8x` | Comfortable |

These thresholds should be calibrated after real plugin tests. They are more
honest than a single `>= 1x` gate.

## Recommended Implementation Roadmap

### Phase 1: Backend Benchmark Matrix

Status: implemented app-side for available native validator binaries.

Build and package native validator variants:

- `rtneural-validator-stl`
- `rtneural-validator-eigen`
- `rtneural-validator-xsimd`
- `rtneural-validator-xsimd-avx` on Intel platforms where legal

For each export, the app now runs the benchmark matrix with:

- block sizes: `16,32,64,128,256,512`
- channels: `1,2`
- sample rate: export sample rate, usually `48000`
- passes: `3`
- warmup blocks: `8`

Write a report that includes:

- backend
- build type
- compiler
- CPU architecture
- worst RTF
- median RTF
- best RTF
- worst block/channel case

### Phase 2: Static `ModelT` Benchmarks

Status: deferred to plugin/native runtime work.

For built-in presets, generate static RTNeural model types:

- `wavenet_tcn_fast`
- `wavenet_tcn_balanced`
- `wavenet_tcn_balanced_tanh15`
- `wavenet_tcn_balanced_tanh18`
- `wavenet_tcn_quality`
- `wavenet_tcn_quality_tanh15`
- `wavenet_tcn_quality_tanh18`

Then compare:

- dynamic JSON runtime
- static `ModelT` runtime
- parity snapshot error
- compile time and binary size

This tells us whether the eventual plugin should ship precompiled preset
specializations or stay fully dynamic.

### Phase 3: Separable WaveNet Experiment

Status: parity-safe in the trainer app, not promoted.

Added experimental preset:

- `wavenet_tcn_separable_fast`

Use grouped dilated Conv1D plus pointwise mixing.

Success criteria:

- Native parity passes.
- RTF improves by at least 1.5x over the comparable full Conv1D WaveNet.
- Preview quality stays close to current WaveNet balanced on clean, crunch,
  rhythm, lead, and pedal captures.

Current result:

- Native parity passes.
- Dynamic JSON RTF did not improve over `wavenet_tcn_balanced`.
- Next meaningful test is static/plugin-side, not more trainer-side copy.

### Phase 3B: Quality Smoothed-Tanh Experiment

Status: implemented in the trainer app as a safe architecture probe.

Added research preset:

- `wavenet_tcn_quality_tanh15`

This keeps the proven sequential quality Conv1D graph and only changes the
training activation to `tanh(x / 1.5)`. Export folds the scale into the Conv1D
weights and writes ordinary RTNeural `tanh`, so native validation and plugin
runtime compatibility stay in the same class as `wavenet_tcn_quality`.

Reason for trying it:

- The best RHYTHM4 quality export plateaued with most remaining residual energy
  in upper bands.
- Earlier balanced smoothed-tanh tests showed `alpha = 1.5` was the better
  waveform-fit compromise than `alpha = 1.8`, though not a guaranteed aliasing
  win.
- It avoids the deeper-stack activation collapse seen in the hidden
  `wavenet_tcn_high_gain` preset.

RHYTHM4 result:

- `wavenet_tcn_quality_tanh15` became the best run so far on
  `project_ab40008405d546398afff4a8d6a8dde7`: ESR `0.0646`, correlation
  `0.9674`, and selected checkpoint epoch `671`.
- Native validation passed with `2.73e-5` max abs error.
- Native Eigen benchmark stayed comfortable at `11.74x` worst-case realtime.
- Worst ASR fell from `0.290` on the prior plain quality export to `0.067`;
  average ASR fell from `0.100` to `0.042`.
- Residual RMS improved only modestly and remains upper-band weighted, so this
  is a strong preset candidate rather than the end of the residual/aliasing
  work.

### Phase 3C: A2-Inspired Architecture Direction

Status: trainer-safe A2 probe implemented and validated on RHYTHM4.

The inspected NAM A2 samples point to a more important direction than another
larger sequential preset:

- residual/skip WaveNet graph support,
- long receptive fields with small channel counts,
- non-power-of-two dilation schedules,
- mixed kernel sizes,
- LeakyReLU/PReLU-style nonlinearities,
- and bundled lite/full quality modes.

The current RTNeural JSON path cannot represent the full A2 graph losslessly, so
these ideas should be split into two lanes:

1. Trainer-safe experiments: PReLU/LeakyReLU WaveNet-style sequential presets,
   non-power-of-two dilations, and maybe mixed kernels where the exporter can
   still emit ordinary Conv1D layers.
2. Plugin/native work: residual/skip graph support, custom fused WaveNet
   blocks, or native NAM/NeuralAudio support. Because the local RTNeural checkout
   is our editable fork, this can happen inside RTNeural itself instead of only
   in the downstream plugin.

Implemented trainer-safe probe:

- `wavenet_tcn_a2_prelu`: 12 sequential Conv1D/PReLU blocks, mixed `6`/`15`
  sample kernels, A2-style dilations
  `[1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41]`, `16` filters, MR-STFT
  pre-emphasis loss, and `3.5e-4` default learning rate. This is intended for
  RHYTHM4 comparison against `wavenet_tcn_quality_tanh15`; it is not a lossless
  A2 graph replacement.

First RHYTHM4 export result:

| Preset | Run | Export ESR | Worst ASR | Average ASR | Eigen Worst RTF | Model Size | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `wavenet_tcn_quality_tanh15` | `run_cc3dc9235cf7426b8529c546003e0e75` | `0.0646` | `0.0670` | `0.0419` | `11.74x` | `416 KB` | Best prior high-gain export after long continuation chain. |
| `wavenet_tcn_a2_prelu` | `run_e61c249debfa4f04a140cf0ff9d7f4ff` | `0.0440` | `0.0354` | `0.0205` | `6.54x` | `832 KB` | One 180-epoch run, native parity pass, still plugin-ready on this workstation. |
| `wavenet_tcn_a2_prelu` continued | `run_0c18cca414014233bf5cd3824768021a` | `0.0381` | `0.0201` | `0.0145` | `6.62x` | `833 KB` | Best current export; Logic AU smoke sounded good, including four instances at a 32-sample buffer on the M5 Max test machine. |

This changes the architecture ranking: A2 PReLU is now the strongest high-gain
candidate by both ESR and ASR, while dynamic RTNeural runtime cost is the main
trade-off. The plugin-side opportunity is to preserve this quality while
reducing cost through the editable `rtneural-extended` fork: fused Conv1D/PReLU
blocks, static model generation, residual/skip graph support, or a custom
A2-style runtime block.

First DAW smoke:

- Built `plugin/rtneural-loader`, a minimal JUCE AU/VST3/Standalone loader with
  a file picker for RTNeural JSON.
- Installed the AU to `~/Library/Audio/Plug-Ins/Components` and validated it
  with `auval -v aufx RtL1 SwLv`.
- Logic Pro loaded the AU and the continued A2 PReLU export. In the initial
  single-instance test, CPU use appeared minimal and the live model sounded
  good.
- Follow-up Logic smoke ran four plugin instances at a `32` sample buffer with
  barely visible CPU increase on the MacBook Pro M5 Max test machine. That does
  not remove the need for modest-machine testing, but it proves the dynamic
  Eigen loader has far more practical headroom than the early offline RTF fear
  suggested.
- The hardened loader now persists the selected model path, reloads it when the
  host restores state, accepts package folders, displays export metadata, and
  exposes input/output gain, bypass, low/mid/high EQ, and an output peak
  indicator.
- Next plugin-side performance checks: saved-session reload in Logic,
  64/128-sample buffers, 48/96 kHz sessions, and modest-machine testing.

### Phase 4: Product Runtime Gate

Update the export report and UI:

- Show native backend matrix.
- Highlight fastest passing backend.
- Use architecture-aware labels.
- Warn when quality WaveNet passes accuracy but has weak runtime headroom.
- Recommend balanced over quality when quality only improves metrics slightly
  but costs too much CPU.

### Phase 5: JUCE Plugin Prototype

Status: debug prototype implemented; product plugin remains open.

Build a minimal RTNeural playback plugin:

- Loads one exported RTNeural package. Done in the debug loader.
- Preloads model off the audio thread. Done for editor-driven loads; restored
  session loads happen during state restore, outside `processBlock()`.
- Keeps one model instance per channel. Done.
- Uses `ScopedNoDenormals`. Done.
- Handles zero-sample and variable-size blocks. Covered by `auval`; keep this
  in pluginval coverage later.
- Runs pluginval.
- Benchmarks 32/64/128 sample buffers in a real host.

This plugin is now a runtime test harness. The product plugin still needs a
designed UI, robust model/package management, explicit version compatibility,
preset browsing, input/output metering, optional EQ design, installer/signing,
and broader host validation.

## Deferred Ideas

These are promising but should wait:

- Quantization: RTNeural is float-oriented today, so this is not a quick win.
- Full gated residual WaveNet: likely quality-positive, but it expands exporter
  and native graph complexity.
- Custom fused Conv1D+activation layer: useful only after backend/static/grouped
  experiments.
- Fast approximate tanh: possible, but it changes model numerics and needs
  strict parity/listening tests.
- GPU inference in plugin: not appropriate for low-latency JUCE plugin runtime.
- Oversampling inside the plugin: expensive and not needed for the current
  capture/export contract.

## Concrete Next Actions

Done in the trainer app:

1. Change `native/rtneural-validator` so the RTNeural backend is configurable.
2. Add local build/packaging support for Eigen, STL, and optional xsimd
   validator variants.
3. Run local backend benchmarks on WaveNet fast, balanced, quality, and
   separable fixtures.
4. Add grouped Conv1D golden fixtures.
5. Add an experimental separable WaveNet preset and compare it against balanced.

Remaining trainer-app work:

1. Initialize or vendor xsimd for local/CI builds if we want xsimd data.
2. Add signed-release packaging for backend-specific validator variants if
   production packages should run the full matrix without local build folders.
3. Calibrate the current headroom labels against a real JUCE plugin prototype.

Remaining plugin-side work:

1. Add generated/static `ModelT` benchmark paths for built-in presets.
2. Build a minimal JUCE RTNeural playback plugin.
3. Load/exported models off the audio thread and atomically swap prepared model
   instances.
4. Keep `processBlock()` allocation-free, lock-free, file-IO-free, and
   denormal-safe.
5. Run pluginval plus small-buffer DAW smoke tests at 32/64/128 sample buffers.
6. Decide whether static `ModelT`, xsimd, or a custom fused Conv1D path is
   needed for quality WaveNet in a shipping plugin.

## What This Means For The Project

WaveNet is still the quality direction. The next major project risk is not
"will WaveNet learn the tone?" We have enough evidence that it can.

The risk is:

```text
Can the exact exported WaveNet model run comfortably inside a real plugin at
small buffer sizes on normal user machines?
```

Everything above is aimed at answering that question with measurements and then
shrinking the model only where the measurements say we need to.
