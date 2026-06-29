# DSP Plugin Handoff: RTNeural Player

Date: 2026-06-29

Audience: DSP/plugin engineer building the production amp-model player from the
current JUCE proof-of-concept.

## Executive Summary

The trainer has converged on TensorFlow/Keras WaveNet-style RTNeural exports for
product use. The production plugin should be a real-time RTNeural JSON player
with front-of-amp pedal support, amp model loading, simple gain/EQ controls,
optional cabinet IR convolution, DAW state restore, and robust export metadata
validation.

The current test plugin at `plugin/rtneural-loader` is no longer just a trivial
loader. It already proves the main runtime shape:

- AU/VST3/Standalone JUCE target.
- RTNeural JSON package loading.
- Optional pedal model before the amp model.
- Optional cabinet IR after the model/EQ path.
- Input gain, pedal output gain, output gain, low/mid/high EQ, bypass toggles,
  output peak display, and DAW path restore.
- Successful Logic Pro smoke tests, including four instances at a 32-sample
  buffer on the MacBook Pro M5 Max test machine.

The production plugin should harden this approach, not restart from scratch.

## Repositories And Local Context

Primary project:

```text
/Users/shortwavlabs/Workspace/shortwavlabs/rtneural-trainer
```

Current JUCE test plugin:

```text
plugin/rtneural-loader
```

Local RTNeural checkout used by the plugin CMake build:

```text
/Users/shortwavlabs/Workspace/rt-neural/RTNeural
```

The plugin currently prefers that local checkout and can be pointed elsewhere
with `RTNEURAL_LOCAL_PATH=/path/to/RTNeural`. Before release, decide whether the
production plugin vendors, submodules, fetches, or expects RTNeural through a
managed dependency path.

Useful reference documents in this repo:

- `docs/Research-RTNeural-WaveNet-JUCE-Performance.md`
- `docs/Research-RTNeural-A2-Runtime-Feasibility.md`
- `docs/Research-Clean-Crunch-Rhythm-Capture-Baseline.md`
- `docs/Research-NAM-Performance-And-WaveNet.md`
- `docs/Research-NAM-To-RTNeural-Conversion.md`
- `docs/Audio-Capture-Guidelines.md`
- `plugin/rtneural-loader/README.md`

## Current Build Commands

Build the plugin:

```bash
cmake -S plugin/rtneural-loader -B plugin/rtneural-loader/build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin/rtneural-loader/build --config Release
```

Disable bundle copying after build:

```bash
cmake -S plugin/rtneural-loader -B plugin/rtneural-loader/build \
  -DRTNEURAL_LOADER_COPY_PLUGIN_AFTER_BUILD=OFF
```

Validate the Audio Unit currently used in Logic smoke tests:

```bash
auval -v aufx RtL1 SwLv
```

## Product Scope For MVP

Build a production-quality RTNeural amp-model player. MVP scope:

1. Load an amp RTNeural export package.
2. Load an optional pedal RTNeural export package before the amp.
3. Enable/disable pedal processing.
4. Control pedal output level when a pedal is loaded and enabled.
5. Control input gain before the neural chain.
6. Control output gain after the full chain.
7. Provide simple low/mid/high EQ.
8. Load and enable/disable a cabinet impulse response.
9. Persist all parameters and loaded asset paths in DAW state.
10. Display export safety metadata: sample rate, preset, ESR, ASR, validation,
    native runtime headroom, latency metadata, and receptive field metadata.
11. Fail safely to passthrough if a model cannot be loaded.
12. Keep `processBlock()` hard real-time safe.

Non-goals for MVP:

- Training inside the plugin.
- GPU inference.
- Direct `.nam` model loading or NAM-to-RTNeural conversion.
- True NAM A2 graph execution.
- AIDA-X envelope support until format and license review are complete.
- Built-in cabinet library, model marketplace, preset cloud sync, or account
  features.

## Trainer And Export Assumptions

The trainer app is now TensorFlow/Keras-only for product training. Do not build
new plugin behavior around Torch checkpoints or Torch-side metadata.

The product-facing model family is WaveNet-style RTNeural JSON. Older Dense,
GRU, LSTM, and basic Conv1D presets still exist internally for export/parity
coverage, but the plugin should optimize around WaveNet exports. Known WaveNet
package names the plugin may encounter include:

- `wavenet_tcn`
- `wavenet_tcn_clean`
- `wavenet_tcn_edge`
- `wavenet_tcn_edge_detail`
- `wavenet_tcn_fast`
- `wavenet_tcn_balanced`
- `wavenet_tcn_balanced_tanh15`
- `wavenet_tcn_balanced_tanh18`
- `wavenet_tcn_quality`
- `wavenet_tcn_quality_tanh15`
- `wavenet_tcn_quality_tanh18`
- `wavenet_tcn_high_gain`
- `wavenet_tcn_a2_prelu`
- `wavenet_tcn_separable_fast`

Current practical favorites from experiments:

- Clean or nearly clean hardware captures: `wavenet_tcn_clean` or
  `wavenet_tcn_edge`.
- Edge-of-breakup and dynamic clean/dirty captures: `wavenet_tcn_edge` first,
  `wavenet_tcn_edge_detail` only if it proves useful.
- Medium to high gain amp captures: `wavenet_tcn_a2_prelu`.
- Fast auditioning or CPU-constrained trials: `wavenet_tcn_fast` or
  `wavenet_tcn_balanced`.
- `wavenet_tcn_high_gain` is a research preset that has underperformed so far;
  support loading it as metadata, but do not optimize product decisions around
  it yet.

## Export Package Contract

The production plugin should prefer loading an export package folder, not just a
raw JSON file. The package root currently contains:

```text
model.rtneural.json
package.json
validation-report.json
benchmark-report.json
native-benchmark-matrix.json
aliasing-report.json
export-manifest.json
export-events.jsonl
parity-snapshot.json
parity-snapshot-input.wav
parity-snapshot-expected.wav
stderr.log
```

Minimum files required for audio playback:

- `model.rtneural.json`

Minimum files required for product-quality safety display:

- `package.json`
- `validation-report.json`
- `benchmark-report.json`
- `aliasing-report.json`
- `native-benchmark-matrix.json`

The current loader reads metadata opportunistically from:

- `model.rtneural.json`
- `package.json`
- `validation-report.json`
- `benchmark-report.json`
- `aliasing-report.json`

Important fields:

- `sample_rate`
- `preset`
- `quality.esr`
- `validation.status`
- `validation.max_abs_error`
- `benchmark.summary.realtime_factor_worst`
- `benchmark.model_info.architecture`
- `benchmark.model_info.latency_samples`
- `benchmark.model_info.receptive_field_samples`
- `benchmark.model_info.conv1d_layers`
- `aliasing.status`
- `aliasing.worst_asr`
- `aliasing.average_asr`

The package can also be used for offline regression tests because
`parity-snapshot-input.wav` and `parity-snapshot-expected.wav` provide a known
input and expected RTNeural output.

## Current Best Export Examples

Recent real-hardware exports are good smoke-test assets:

```text
/Users/shortwavlabs/Desktop/export_clean
/Users/shortwavlabs/Desktop/export_drive
/Users/shortwavlabs/Desktop/export_rhythm
```

Observed metrics from the real hardware set:

| Export | Preset | ESR | Worst ASR | Avg ASR | Native RTF | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `export_clean` | `wavenet_tcn_edge` | 0.00217 | 0.00673 | 0.00251 | 20.72x | Clean/edge amp head, very strong result. |
| `export_drive` | `wavenet_tcn_a2_prelu` | 0.00043 | 0.00195 | 0.00080 | 6.49x | Pedal capture, excellent result. |
| `export_rhythm` | `wavenet_tcn_a2_prelu` | 0.00355 | 0.00913 | 0.00359 | 6.63x | High-gain amp head, strong result. |

The `export_rhythm` benchmark matrix shows the A2 PReLU-style preset still has
usable stereo headroom at small block sizes:

- 32-sample block, mono: about 14.50x realtime worst case.
- 32-sample block, stereo: about 6.66x realtime worst case.
- 512-sample block, stereo: about 6.63x realtime worst case.

These numbers are native validator measurements on the development machine, not
a universal performance guarantee. They are good acceptance-test references for
regression tracking.

## Signal Chain

The current test plugin chain is:

```text
input gain
  -> optional pedal RTNeural model
  -> pedal output gain
  -> amp RTNeural model
  -> low/mid/high EQ
  -> optional cabinet IR
  -> output gain
```

This is a useful MVP chain. Product decisions still needed:

1. EQ placement:
   - Current test harness places EQ after the amp model and before cab IR.
   - If the product labels this as a post-amp EQ, consider moving it after IR.
   - If the product labels this as a simple tone-shaping stage before the cab,
     keep the current order.

2. Pedal/amp gain staging:
   - Input gain is pre-pedal and pre-amp.
   - Pedal output gain is only meaningful when a pedal is loaded and enabled.
   - Output gain is post-model, post-EQ, post-IR.
   - All knobs should avoid zippering. Use smoothed values before release.

3. Cabinet IR:
   - The IR stage should remain optional because current captures are amp-head
     captures, not amp-plus-cab captures.
   - The user wants to add cabinet response in the plugin.

## Real-Time Safety Rules

`AudioProcessor::processBlock()` is the hard real-time boundary. The following
must never happen inside `processBlock()`:

- File I/O.
- JSON parsing.
- Model construction.
- IR file loading.
- Heap allocation.
- Locks or blocking waits.
- UI calls.
- Logging.
- Host queries that can block.
- Destroying a model object that another audio callback might still read.

Allowed work inside `processBlock()`:

- Read cached atomic parameter values.
- Read atomically published model pointers.
- Process samples through already constructed RTNeural models.
- Process already prepared EQ filters.
- Process already loaded JUCE convolution state.
- Update lightweight atomics such as a held peak meter.

The current loader already follows several good patterns:

- `juce::ScopedNoDenormals` at the top of `processBlock()`.
- Raw parameter pointers cached from APVTS.
- One RTNeural model instance per channel.
- Atomic publication of the current amp and pedal `ModelSet`.
- Loaded model objects retained for the processor lifetime so a pointer swap
  cannot immediately delete an object the audio thread may still read.
- Audio thread reads model pointers, atomics, prepared convolution, EQ state,
  and sample data only.

Keep those patterns.

## Model Instance Policy

RTNeural recurrent/stateful models and causal Conv1D models carry internal
state. Do not share a single RTNeural `Model<float>` instance across stereo
channels.

Current loader policy:

- Parse the same JSON twice.
- Store two independent model instances.
- Use channel 0 model for channel 0 and channel 1 model for channel 1.
- For more than two channels, use channel index clamped to 1.

Production policy:

- Support mono and stereo for MVP.
- Reject or downmix unsupported bus layouts deliberately.
- Keep independent model states per audio channel.
- Reset model states in `prepareToPlay()`.
- Reset model states when a new model is loaded.
- Consider a host transport reset policy later, but avoid surprising resets
  during continuous playback.

## Latency And Receptive Field Semantics

Do not confuse three separate concepts:

1. Export alignment latency:
   - Stored in package metadata as `latency_samples`.
   - Describes how the target capture was aligned during training.
   - It is not automatically plugin latency.

2. WaveNet receptive field:
   - Stored as `receptive_field_samples`.
   - This is model memory, not lookahead.
   - Causal inference can run with zero added host latency.

3. Plugin processing latency:
   - Only report nonzero JUCE latency if the plugin deliberately delays audio,
     oversamples with a latency-introducing filter, or uses a convolution mode
     that adds latency.

For current causal RTNeural inference, report zero plugin latency unless a
future oversampling or convolution mode changes that.

## Sample Rate Policy

Current captures and exports are 48 kHz. The plugin currently warns when the
host sample rate differs from model metadata.

Production MVP should do one of these, in order of preference:

1. Require matching sample rate and display a clear warning if mismatched.
2. Offer explicit resampled/oversampled runtime only after measurement.
3. Support separate model exports per sample rate.

Do not silently run a 48 kHz model in a 96 kHz session and call it correct. The
model's learned frequency response, nonlinear behavior, receptive-field time
scale, and aliasing profile are tied to the training sample rate.

## Aliasing And Oversampling

Aliasing is no longer catastrophic in the best real hardware exports, but it is
still a real product concern for high-gain tones.

Observed pattern:

- Real amp and pedal captures trained much better than many amp-sim renders.
- Current A2 PReLU-style exports can have low ASR and good subjective sound.
- Some high-gain experiments produced warning-level ASR, often around upper
  mid/high-frequency residual content.

MVP policy:

- Surface the export aliasing report to the user.
- Warn on `aliasing-report.json.status != "pass"`.
- Do not block loading solely on ASR warnings.
- Keep the cab IR and EQ post stages stable so the user can audition safely.

Post-MVP options:

- 2x oversampling around the neural model.
- 96 kHz exports for high-gain models.
- Anti-aliasing-aware training loss or validation gate.
- RTNeural/runtime support for more efficient high-rate inference.

Oversampling must be measured. It can improve aliasing but also increases CPU,
adds filter design choices, and may add latency depending on implementation.

## Current RTNeural Runtime Shape

The current plugin uses dynamic RTNeural JSON parsing:

```cpp
RTNeural::json_parser::parseJson<float>(stream)
```

That is the right MVP path because the trainer exports RTNeural JSON packages
and the model architecture varies between presets.

Current A2-inspired preset status:

- `wavenet_tcn_a2_prelu` is a sequential RTNeural-safe approximation inspired by
  NAM A2 ideas.
- It is not true NAM A2.
- It does not require RTNeural graph/fan-out changes.
- It has performed extremely well in testing.

True NAM A2-style runtime is a day-2 item. It would require RTNeural support for
residual/skip topology or a fused custom layer, because the current dynamic
sequential model cannot express:

- residual layer outputs,
- per-layer skip accumulation,
- input mix-in paths,
- layer-array head convolution,
- slimmable model selection,
- direct NAM parser compatibility.

## Performance Targets

Minimum MVP targets:

- Pass `auval` on macOS AU.
- Pass plugin validation on VST3 where available.
- No allocations or blocking work in `processBlock()`.
- No denormal CPU spikes during silence or decays.
- Stable playback at 48 kHz with 32, 64, 128, 256, and 512 sample buffers.
- Stable mono and stereo operation.
- Stable amp-only, pedal-only, amp-plus-pedal, amp-plus-IR, and full chain.
- Warning display for sample-rate mismatch and low native runtime headroom.

Practical performance target:

- One stereo high-gain A2 PReLU model plus optional IR should run comfortably at
  48 kHz / 64 samples on the development machine.
- Four instances at 48 kHz / 32 samples should remain usable on the development
  machine.
- The product should be tested on a less powerful machine before promising
  broader compatibility.

Suggested runtime headroom thresholds:

- `>= 6x` native validator worst-case RTF: comfortable.
- `>= 3x`: likely usable but test in DAW with UI and IR.
- `>= 2x`: caution; warn.
- `< 2x`: high risk; warn prominently.
- `< 1x`: do not recommend for realtime use.

## State Restore

State restore matters. The test plugin now restores:

- amp model path,
- amp package path,
- amp display name,
- pedal model path,
- pedal package path,
- pedal display name,
- IR path,
- IR display name,
- APVTS parameters.

Production expectations:

- A saved DAW project should reopen with the same amp, pedal, IR, and knob
  values.
- Missing files should not crash or stall the audio thread.
- Missing files should produce a visible warning and safe passthrough for the
  missing stage.
- Long-term product packaging should decide whether loaded exports are copied
  into a managed library or referenced by absolute path.

## File Loading

Model and IR loading should happen off the audio thread.

Current test harness loads from editor actions on the message thread. This is
acceptable for a debug plugin, but production should avoid UI stalls for large
models or slow storage:

1. Start load from UI.
2. Parse JSON and construct model instances on a background worker.
3. Validate input size, sample rate, metadata, and model health.
4. Reset the new model instances.
5. Atomically publish the new model set.
6. Retire the old model set safely.

Do not hold a lock in `processBlock()` while waiting for a background load.

## Cabinet IR Implementation Notes

The current test plugin uses JUCE `dsp::Convolution`:

- stereo enabled,
- trim enabled,
- normalise enabled,
- loaded outside `processBlock()`,
- processed after amp inference and EQ.

Production decisions:

- Decide whether IR should be before or after post-EQ.
- Decide whether IR normalisation is desired for user-loaded third-party IRs.
- Display IR sample rate and length.
- Test mono IR into stereo output.
- Test long IRs and very short IRs.
- Confirm tail length reporting does not surprise hosts.
- Confirm convolution latency behavior and report it if nonzero.

## EQ Implementation Notes

Current EQ is intentionally simple:

- low shelf at 120 Hz,
- mid peak at 750 Hz, Q 0.85,
- high shelf at 4 kHz,
- +/- 12 dB range.

Production options:

- Keep it simple for MVP.
- Add smoothing to avoid zipper noise.
- Consider moving to a named "Post EQ" section if the chain remains after the
  amp model.
- If it becomes a tone stack, redesign frequency points and placement.

## Gain And Headroom

Training captures are level-sensitive. The plugin should make gain staging
visible and predictable.

Recommended controls:

- Input gain: pre-pedal and pre-amp.
- Pedal output gain: after pedal, before amp.
- Output gain: final trim.
- Output peak or clip indicator.

Recommended behavior:

- Use parameter smoothing.
- Avoid hidden automatic normalization after model inference.
- Preserve user gain decisions in DAW state.
- Make bypass behavior clear:
  - full plugin bypass should bypass neural models, EQ, and IR;
  - host bypass is separate from the plugin's own model bypass toggle.

## Safety Warnings To Surface

Surface warnings in the plugin UI when possible:

- No amp or pedal model loaded.
- Amp/pedal sample rate differs from host sample rate.
- Validation status is not `pass`.
- Aliasing report status is not `pass`.
- Native realtime headroom is low.
- Cab IR is enabled but no IR is loaded.
- Loaded file is a raw JSON without package metadata.
- Model input size is not one.

Warnings should not allocate or parse in the audio callback. Cache them after
load and update UI from the message thread.

## Testing Matrix

Core automated tests:

1. Load package folder.
2. Load raw `model.rtneural.json`.
3. Reject a folder without `model.rtneural.json`.
4. Reject a model with unsupported input size.
5. Restore DAW state with valid paths.
6. Restore DAW state with missing paths.
7. Render amp-only audio.
8. Render pedal-only audio.
9. Render pedal-plus-amp audio.
10. Render amp-plus-IR audio.
11. Toggle bypass while processing.
12. Toggle pedal enable while processing.
13. Toggle IR enable while processing.
14. Compare plugin offline render against `parity-snapshot-expected.wav` for a
    package with no EQ, no IR, unity gains, and reset state.
15. Confirm silence does not cause denormal CPU spikes.
16. Confirm no allocations in `processBlock()` under instrumentation.

Manual DAW smoke tests:

- Logic Pro AU at 48 kHz, 32 sample buffer.
- Logic Pro AU at 48 kHz, 64 and 128 sample buffers.
- 96 kHz session mismatch warning.
- Four stereo instances of a high-gain model.
- Session save, close, reopen, and playback without reloading files.
- Load `export_clean`, `export_drive`, and `export_rhythm`.
- Full chain: drive pedal into rhythm amp into a cab IR.
- Host bypass and plugin bypass.
- Automation of input gain, output gain, EQ, pedal enable, and IR enable.

Validation tools:

- `auval` for Audio Unit.
- pluginval for VST3/AU when added to CI.
- Native validator benchmark and package reports from trainer exports.
- DAW real-time CPU observation at small buffers.

## Suggested Implementation Phases

### Phase 1: Harden The Existing Loader

- Move model loading from synchronous editor calls to an asynchronous loader.
- Keep atomic publication and per-channel model instances.
- Add parameter smoothing for gain/EQ.
- Add clearer sample-rate mismatch and validation/ASR warnings.
- Add package-folder library path handling.
- Add pluginval smoke.
- Add offline render parity test using package snapshots.

### Phase 2: Production UX

- Replace debug UI with a clean amp-player interface.
- Keep details available behind an info panel.
- Add explicit Amp, Pedal, Cab, EQ, and Output sections.
- Show loaded package names and quality badges.
- Make missing/mismatched assets obvious without blocking audio.
- Add a managed model folder or asset browser if needed.

### Phase 3: Runtime Optimization

- Benchmark dynamic JSON RTNeural versus any static/fused runtime path.
- Investigate `ModelT` or generated C++ for fixed architecture exports.
- Benchmark Eigen/xsimd/STL backends on Apple Silicon and Intel.
- Consider a fused RTNeural residual/TCN layer for day-2 A2-style runtime.
- Revisit oversampling after CPU and ASR data justify it.

### Phase 4: Release Engineering

- Sign/notarize macOS builds.
- Produce AU/VST3 artifacts.
- Add CI build matrix.
- Add plugin validation gates.
- Add installer/package layout.
- Document supported DAWs, sample rates, and operating systems.

## Current Risks

1. Sample-rate mismatch:
   - Current models are trained at 48 kHz. Production needs a clear policy for
     44.1, 88.2, and 96 kHz sessions.

2. High-gain aliasing:
   - Best current exports are good, but high-gain aliasing remains a thing to
     monitor.

3. CPU on weaker machines:
   - The M5 Max results are encouraging but not representative of everyone.

4. Async loading:
   - Current debug flow can synchronously parse/load from UI actions. Production
     should avoid UI stalls and must never risk audio-thread stalls.

5. Asset lifetime:
   - Absolute path restore is fine for a test plugin. A commercial plugin may
     need managed copies, model libraries, or missing-file relinking.

6. True A2 support:
   - Current `wavenet_tcn_a2_prelu` is excellent and RTNeural-safe, but true NAM
     A2 topology is not representable by the current sequential dynamic JSON
     model. Treat true A2 as a separate runtime project.

7. IR licensing and normalization:
   - User-loaded IRs are easy. Bundled IRs require license review. Normalizing
     third-party IRs changes gain staging and should be intentional.

## Engineering Checklist

Before changing DSP/runtime code:

- Read `plugin/rtneural-loader/README.md`.
- Read `docs/Research-RTNeural-WaveNet-JUCE-Performance.md`.
- Read `docs/Research-RTNeural-A2-Runtime-Feasibility.md`.
- Confirm whether you are modifying the debug loader or production plugin.
- Confirm the test export package and sample rate.

When changing model loading:

- Keep parsing off the audio thread.
- Construct independent channel model instances.
- Validate input size.
- Reset models before publishing.
- Publish with atomics or another real-time-safe ownership scheme.
- Keep old model objects alive until no callback can read them.

When changing processing:

- Avoid allocation in `processBlock()`.
- Avoid locks in `processBlock()`.
- Use `juce::ScopedNoDenormals`.
- Keep sample loops straightforward and branch-light.
- Smooth gain and EQ parameters.
- Test mono and stereo.
- Test silence.

When changing UI/state:

- Persist all plugin parameters through APVTS.
- Persist selected amp, pedal, and IR paths.
- Restore safely when files are missing.
- Do not parse heavy metadata repeatedly from the timer callback.
- Cache display strings after load.

When changing package metadata:

- Update the trainer export package first.
- Keep backward compatibility where possible.
- Update plugin metadata parsing.
- Update package-loading tests.
- Update this handoff or the plugin README if the contract changes.

## Acceptance Criteria For Production MVP

The production plugin is ready for wider testing when:

- AU passes `auval`.
- VST3 passes plugin validation.
- No model, missing model, missing IR, and corrupt package all fail safely.
- DAW state restore works for amp, pedal, IR, and parameters.
- `export_clean`, `export_drive`, and `export_rhythm` load and play.
- 48 kHz / 32 sample buffer works in Logic with at least one full-chain
  instance.
- Four stereo instances of a strong model are usable on the development
  machine.
- Offline render parity matches package snapshots within a documented
  tolerance for amp-only unity-gain playback.
- Sample-rate mismatch warning is visible.
- Aliasing and validation warnings are visible.
- No allocations, locks, file I/O, or parsing occur in `processBlock()`.

## One-Line Mental Model

The plugin is a hard-real-time, package-aware RTNeural player: load slowly and
safely off the audio thread, publish immutable model state atomically, process
causal WaveNet samples with zero added neural latency, and let the export
metadata tell the user whether the model is safe, fast, and trustworthy.
