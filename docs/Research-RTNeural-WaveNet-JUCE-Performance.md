# RTNeural WaveNet And JUCE Plugin Performance Notes

Reviewed: 2026-06-22
Updated: 2026-06-22

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

Still pending:

- Record a full backend matrix in each export package instead of one selected
  backend report.
- Let the UI label the fastest passing backend.
- Build xsimd once the RTNeural xsimd headers/submodule are present.

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

Status: partly implemented in the trainer app.

Build and package native validator variants:

- `rtneural-validator-stl`
- `rtneural-validator-eigen`
- `rtneural-validator-xsimd`
- `rtneural-validator-xsimd-avx` on Intel platforms where legal

For each export, run the benchmark matrix with:

- block sizes: `32,64,128,256`
- channels: `1,2`
- sample rate: `48000`
- passes: at least `3`
- warmup blocks: at least `8`

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
- `wavenet_tcn_quality`
- `conv1d_stack_prelu`

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

### Phase 4: Product Runtime Gate

Update the export report and UI:

- Show native backend matrix.
- Highlight fastest passing backend.
- Use architecture-aware labels.
- Warn when quality WaveNet passes accuracy but has weak runtime headroom.
- Recommend balanced over quality when quality only improves metrics slightly
  but costs too much CPU.

### Phase 5: JUCE Plugin Prototype

Status: remaining plugin-side work.

Build a minimal RTNeural playback plugin:

- Loads one exported RTNeural package.
- Preloads model off the audio thread.
- Keeps one model instance per channel.
- Uses `ScopedNoDenormals`.
- Handles zero-sample and variable-size blocks.
- Runs pluginval.
- Benchmarks 32/64/128 sample buffers in a real host.

This plugin should be a runtime test harness before it becomes a product.

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

1. Store a full backend benchmark matrix in export packages.
2. Surface fastest passing backend and plugin-headroom warnings in the export
   UI.
3. Initialize or vendor xsimd for local/CI builds if we want xsimd data.

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
