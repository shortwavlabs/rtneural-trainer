# Implementation Guide: RTNeural Training Desktop App

Date: 2026-06-20

Source reviewed: [Research-RTNeural-Training-Desktop-App.md](Research-RTNeural-Training-Desktop-App.md)

This guide turns the research note into a step-by-step build plan for a desktop
application that trains neural audio models locally and exports RTNeural-ready
artifacts. The core product promise is intentionally narrow:

> Give the app a dry input file and a matching processed target file. It trains a
> real-time-safe model, proves the exported RTNeural JSON matches the trained
> model, proves RTNeural can load it, and reports runtime cost before the user
> ships the model.

## Current Implementation Status

This document started as a build plan. As of June 20, 2026, the repository has
crossed from architecture planning into a working desktop prototype. Treat this
guide as both the historical implementation map and the current engineering
checklist.

Implemented:

1. Tauri v2 + React desktop app with project creation, project rename/delete,
   runtime inspection, capture import, preparation, alignment, training,
   evaluation, export, notes, and progress streaming.
2. `uv`-managed Python `rttrainer` sidecar with `prepare`, `train`, `evaluate`,
   `export`, and `inspect-device` commands.
3. TensorFlow/Keras-first training and RTNeural JSON export path, with optional
   PyTorch support only for curated LSTM presets.
4. Dense-only, GRU, LSTM, causal Conv1D, safe BatchNorm/PReLU, and Conv+GRU
   hybrid Keras presets with golden JSON and native RTNeural parity coverage.
5. Native C++ `rtneural-validator` sidecar that validates exported JSON against
   WAV fixtures and benchmarks runtime cost.
6. SQLite-backed project/job store with durable job events, project rename/delete,
   status recovery, cancellation, resume-from-checkpoint, failed/interrupted
   states, and job locking.
7. Native file pickers, capture validation, optional resampling, stereo policy,
   manual alignment override, sampled-window handling for long captures,
   gain/headroom guidance, validation curves, early stopping controls, preset
   recommendations, good/usable/needs-work report language, and focused error
   recovery copy.
8. Offline preview playback for target, prediction, and residual WAVs, plus
   peak-envelope waveform comparison.
9. Rich export package metadata, validation/benchmark report display, and
   open-export-folder support.
10. Development sidecar shims, production sidecar packaging, release sidecar
    smoke tests, Tauri bundle smoke tests, and release artifact manifests.
11. First-run generated sample project, setup/empty states, visible focus
    styling, and reduced-motion support.

Deferred or still productization work:

1. Signed/notarized release distribution and final release publishing policy.
2. Installer metadata polish for macOS, Windows, and Linux release channels.
3. Real-world capture threshold tuning for preset recommendations, gain
   guidance, and good/usable/needs-work quality language.
4. Deeper waveform/spectrum inspection for target/prediction/residual.
5. `.aidax` envelope pending format/license review.
6. Generated JUCE/player integration and compile-time RTNeural model generation.
7. Full accessibility audit, UI smoke automation, and broader product polish
   after real-world use.

## 1. Non-Negotiable Product Decisions

Start by locking these constraints into the implementation. They keep the first
version buildable and protect the app from becoming a generic ML IDE.

1. Treat RTNeural as the inference target, not the training framework.
2. Use TensorFlow/Keras as the canonical RTNeural JSON export path.
3. Export only app-owned, curated architectures.
4. Keep PyTorch as an optional backend only for presets with proven parity.
5. Do not support arbitrary PyTorch, TensorFlow/Keras, or ONNX import in v1.
6. Validate every export twice: backend parity and native RTNeural parity.
7. Use dynamic RTNeural JSON loading for v1.
8. Keep live low-latency monitoring out of v1 unless the MVP is already stable.
9. Prefer 48 kHz WAV workflows for v1, while still recording conversion metadata.
10. Benchmark every exported model before letting the user publish it.
11. Avoid copying trainer/exporter code from reference projects until licenses
    have been reviewed.

## 2. Target Architecture

Use a Tauri v2 desktop shell with a React/TypeScript UI, a Rust orchestration
layer, a `uv`-managed Python trainer/export sidecar, and a native C++ RTNeural
validator sidecar. The Python sidecar makes TensorFlow/Keras the canonical
RTNeural JSON path while keeping PyTorch isolated behind optional extras and
known-good parity tests.

```text
rtneural-trainer/
  app/
    package.json
    src/
      App.tsx
      lib/api.ts
      styles.css
      types.ts
    src-tauri/
      Cargo.toml
      tauri.conf.json
      capabilities/
      binaries/
      src/
        lib.rs
        main.rs
  trainer/
    pyproject.toml
    uv.lock
    rttrainer/
      __init__.py
      cli.py
      data/
      export_rtneural/
      metrics/
      models/
      training/
      validation/
    tests/
  native/
    rtneural-validator/
      CMakeLists.txt
      src/
  scripts/
  projects/
    .gitkeep
  docs/
```

The app stores structured state in SQLite and large artifacts on disk.

```text
projects/
  <project-id>/
    audio/
      original/
      prepared/
    runs/
      <run-id>/
        train-manifest.json
        checkpoints/
        events.jsonl
        metrics.json
        history.json
        training-report.json
        previews/
    exports/
      <export-id>/
        model.rtneural.json
        package.json
        validation-report.json
        benchmark-report.json
        native-validation-report.json
        native-benchmark-report.json
        export-events.jsonl
```

## 3. Milestone Map

The original build map used four phases. The repo is now in late Phase 2: the
core desktop workflow works locally, while release distribution and runtime
integrations remain productization work.

| Phase | Name | Current Status | Notes |
| --- | --- | --- | --- |
| 0 | Export spike | Complete | Dense/LSTM/GRU/Conv1D/activation/BatchNorm/PReLU fixtures export, run through Python parity, and validate in native RTNeural. |
| 1 | CLI trainer | Complete for local v1 | Paired WAVs produce metrics, checkpoints, preview WAVs, RTNeural JSON, package metadata, and validation/benchmark reports. |
| 2 | Desktop MVP | Implemented prototype | Project creation, import, align, train, evaluate, export, progress, cancel/resume, runtime inspection, previews, and report display are wired. |
| 3 | Runtime integrations | Deferred | `.aidax`, generated player, compile-time model generation, and cloud training remain later work. |

## 4. Prerequisites And Pins

Before writing product code, create a dependency and toolchain document in the
repo. Pinning early matters because parity failures are often version-sensitive.

1. Choose supported operating systems:
   - macOS Apple Silicon
   - macOS Intel if needed
   - Windows x64
   - Linux x64 if the product scope includes it
2. Install local development tools:
   - Rust stable toolchain
   - Node.js LTS with `pnpm` 11
   - Python 3.11 or 3.12 managed by `uv`
   - CMake
   - A C++17-capable compiler
   - Platform audio libraries needed by Python packages
3. Pin RTNeural to the commit from the research note:
   - `1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d`
   - On the current workstation, prefer the local checkout at
     `/Users/shortwavlabs/Workspace/rt-neural/RTNeural` and fall back to fetching
     only when the local clone is absent.
4. Record expected Python package versions:
   - `numpy`
   - `tensorflow>=2.16,<2.19` for canonical Keras export fixtures and
     compatible presets
   - `tensorflow-metal>=1.2` on Darwin arm64 for Apple Silicon Metal GPU
     training
   - `torch` only for the optional PyTorch training/export backend
   - `pyinstaller`, injected by the packaging script when building production
     sidecars
   - WAV IO currently uses Python's standard library `wave` module
   - CLI parsing currently uses `argparse`
5. Record frontend/runtime versions:
   - Tauri v2
   - React
   - TypeScript
   - Vite or the selected frontend build tool
6. Current packaging decision:
   - production `rttrainer` sidecars include the TensorFlow/Keras path
   - PyTorch remains optional and limited to supported LSTM presets
   - the desktop app exposes an external Python path for advanced environments

Recommended local commands:

```bash
cd trainer
uv lock
uv sync --extra tensorflow
uv sync --extra training
uv run --extra tensorflow rttrainer inspect-device --json
```

Acceptance criteria:

- A new developer can install the toolchains and run skeleton tests.
- RTNeural commit, Python version, and C++ compiler requirements are documented.
- The decision about packaged TensorFlow/PyTorch versus external Python is
  explicit.

## 5. Phase 0: Export Spike

Phase 0 is the most important technical milestone. Keep it small and ruthless:
one synthetic dataset, one Keras Dense/LSTM/GRU export path, the optional PyTorch
presets kept behind parity tests, and a native parity test.

### Step 5.1 Python Package

Current package:

```text
trainer/rttrainer/
  cli.py
  models/presets.py
  export_rtneural/keras_exporter.py
  export_rtneural/json_exporter.py
  export_rtneural/registry.py
  validation/parity.py
  training/runner.py
  training/dataset.py
  data/audio_io.py
  data/prepare.py
  metrics/audio_metrics.py
```

Implemented CLI commands:

```bash
rttrainer inspect-device --json
rttrainer prepare --manifest prepare-manifest.json
rttrainer train --manifest train-manifest.json
rttrainer evaluate --manifest evaluate-manifest.json
rttrainer export --manifest export-manifest.json
```

Current status:

- `trainer/pyproject.toml` defines the package and optional `tensorflow` and
  `training` extras.
- Keras/TensorFlow is the canonical training/export path.
- PyTorch is retained only where parity is proven and worth supporting.
- Deterministic fixture generation and parity tests now cover the export surface
  instead of a separate `spike-train` command.

### Step 5.2 Implement First Model Presets And Support Matrix

Implement presets as code-owned architecture factories. Do not serialize
arbitrary model definitions.

Implemented presets:

| Preset ID | Architecture | Use |
| --- | --- | --- |
| `dense_only` | Dense stack | Fast memoryless baseline |
| `gru_light` | 1 GRU, hidden 10, dense output | Compact recurrent model |
| `lstm_light` | 1 LSTM, hidden 12, dense output | Lowest-risk recurrent export |
| `lstm_standard` | 1 LSTM, hidden 16, dense output | Default recurrent target |
| `conv1d_light` | Causal Conv1D | Fast temporal front-end |
| `conv1d_bn_prelu` | Causal Conv1D with BatchNorm/PReLU | Safe BatchNorm/PReLU coverage |
| `conv_gru_hybrid` | Conv1D front-end + GRU | Richer Keras temporal preset |

Model rules:

1. Use mono input and mono output first.
2. Keep tensor shape conventions documented in the model file.
3. BatchNorm/PReLU is exposed only in the safe Conv1D preset with parity
   coverage.
4. Conv1D and Conv+GRU are exposed because golden JSON plus native parity now
   cover those paths.
5. Add each preset to a known-good compatibility matrix.
6. Keep the RTNeural layer/activation support matrix in code and generate docs
   from it instead of hand-maintaining scattered tables.

Benchmark-driven v1 support:

| Kind | v1 | v1-plus | Later | Defer |
| --- | --- | --- | --- | --- |
| Layers | Dense, GRU, LSTM, Conv1D | BatchNorm/PReLU only in safe cases | Conv2D, broader BatchNorm variants | MaxPooling |
| Activations | tanh, ReLU, sigmoid, softmax, ELU, PReLU |  |  |  |

Acceptance criteria:

- Presets can be created by stable string IDs.
- Each preset declares input channels, output channels, hidden size, layer count,
  estimated CPU tier, and export support.
- `scripts/rtneural_support_matrix.py --format markdown` prints the current
  layer and activation plan.
- `scripts/generate_rtneural_keras_fixtures.py --list` reports the default
  fixture scope without importing TensorFlow.

### Step 5.3 Build The RTNeural JSON Exporters

Implemented in `trainer/rttrainer/export_rtneural/keras_exporter.py` for the
canonical TensorFlow/Keras path and
`trainer/rttrainer/export_rtneural/json_exporter.py` for the optional PyTorch
path plus shared package metadata helpers.

Exporter responsibilities:

1. Accept a known Keras Sequential model and emit RTNeural's `in_shape` and
   `layers` JSON structure.
2. For optional PyTorch presets, accept a known preset instance and a
   `state_dict`.
3. Convert tensor layouts to RTNeural's expected format when the source backend
   does not already match RTNeural's JSON conventions.
4. Emit the RTNeural JSON layer list.
5. Attach metadata required by the desktop app.
6. Reject unsupported layers with clear errors.

Important conversion checks:

- Keras Sequential export is the reference JSON shape.
- Dense and recurrent PyTorch weights may need transposition.
- GRU gate order must be verified against RTNeural expectations.
- Conv1D kernel reversal matters for PyTorch Conv1D exports.
- Recurrent activations must be explicit because RTNeural recurrent
  post-activation handling has known caveats.

Example export envelope:

```json
{
  "in_shape": [null, null, 1],
  "layers": [
    {
      "type": "lstm",
      "activation": "",
      "shape": [null, null, 16],
      "weights": []
    },
    {
      "type": "dense",
      "activation": "",
      "shape": [null, null, 1],
      "weights": []
    }
  ],
  "metadata": {
    "schema_version": 1,
    "sample_rate": 48000,
    "latency_samples": 0,
    "architecture": "lstm_standard",
    "trainer_version": "0.1.0",
    "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d"
  }
}
```

Acceptance criteria:

- Exporter produces deterministic JSON for deterministic weights.
- Unsupported modules fail before writing an invalid export.
- Unit tests assert layer order, shapes, and key tensor transforms.

### Step 5.4 Write A Python Parity Validator

Implemented in `trainer/rttrainer/validation/parity.py`.

The Python parity validator:

1. Load the original backend checkpoint or Keras model.
2. Load the exported JSON.
3. Run the same test input through both paths.
4. Compare output arrays sample by sample.
5. Save a tolerance report.

The native validator remains the runtime source of truth, while Python parity
tests keep backend/export tensor layout changes honest before the native smoke
stage runs.

Acceptance criteria:

- Reports include max absolute error, mean absolute error, RMSE, and failing
  sample indices when tolerance is exceeded.
- Parity thresholds are preset-specific and stored in code.

### Step 5.5 Build The Native RTNeural Validator

Implemented in `native/rtneural-validator`.

CLI shape:

```bash
rtneural-validator validate \
  --model model.rtneural.json \
  --input test-input.wav \
  --reference reference-output.wav \
  --report validation-report.json

rtneural-validator benchmark \
  --model model.rtneural.json \
  --sample-rate 48000 \
  --seconds 30 \
  --report benchmark-report.json
```

Current responsibilities:

1. Add a CMake project.
2. Fetch or vendor RTNeural at the pinned commit.
3. Load JSON with RTNeural's dynamic parser.
4. Feed deterministic sample buffers.
5. Compare output to reference audio or `.npy` fixtures.
6. Measure real-time factor at 48 kHz.
7. Emit machine-readable JSON reports.

Benchmark report fields:

```json
{
  "schema_version": 1,
  "backend": "eigen",
  "sample_rate": 48000,
  "frames_processed": 1440000,
  "elapsed_ms": 100.0,
  "realtime_factor": 300.0,
  "max_abs_output": 0.98,
  "status": "pass"
}
```

Acceptance criteria:

- Native validator exits non-zero on parse errors, NaN output, tolerance failure,
  or runtime factor below the configured threshold.
- CI runs the validator against the Phase 0 exported model.
- Reports are stable enough for the desktop app to display directly.

## 6. Phase 1: CLI Trainer

Phase 1 is implemented as the `rttrainer` Python sidecar. The Tauri app calls
these commands through saved manifest JSON files and streams JSONL progress
events into the desktop UI and SQLite job event store.

### Step 6.1 Define CLI Commands And JSON Contracts

Implement these commands:

```bash
rttrainer prepare --manifest prepare-manifest.json
rttrainer train --manifest train-manifest.json
rttrainer evaluate --manifest evaluate-manifest.json
rttrainer export --manifest export-manifest.json
rttrainer inspect-device --json
```

General CLI rules:

1. All commands accept manifest JSON files instead of long argument lists.
2. All commands write progress events as JSON lines to stdout.
3. Human-readable logs go to stderr and run log files.
4. Final command output is written as files, not only printed.
5. Every command exits with a meaningful non-zero code on failure.

Progress event example:

```json
{"type":"run_started","run_id":"run_123","timestamp":"2026-06-19T12:00:00Z"}
{"type":"epoch","epoch":10,"total_epochs":200,"train_loss":0.04,"val_esr":0.08}
{"type":"checkpoint","path":"checkpoints/epoch-010.pt","is_best":true}
{"type":"run_finished","status":"completed","metrics_path":"metrics.json"}
```

Acceptance criteria:

- Rust/Tauri can treat stdout as a stream of structured events.
- Commands are restartable from manifests saved inside the project folder.
- Invalid manifests fail with a clear validation error.

### Step 6.2 Implement Audio Import And Preparation

Implemented in `rttrainer/data/audio_io.py` and `rttrainer/data/prepare.py`.

Input contract:

- `input.wav`: dry/reference signal.
- `target.wav`: wet/processed signal from hardware, plugin, or signal chain.

Preparation tasks:

1. Decode WAV files.
2. Validate sample rate, channel count, duration, and bit depth.
3. Convert to the project target sample rate, defaulting to 48 kHz.
4. Convert mono/stereo deterministically according to project settings.
5. Detect hard clipping.
6. Detect DC offset.
7. Detect excessive silence.
8. Detect mismatched active duration.
9. Estimate latency in samples.
10. Trim and align prepared files.
11. Create reproducible train/validation/test splits.
12. Save a preparation report.
13. Apply and persist a manual latency adjustment when the user overrides the
    automatic estimate.
14. Report long-capture handling, recommended training window budget, and
    gain/headroom guidance.

Validation report fields:

```json
{
  "schema_version": 1,
  "input": {
    "sample_rate": 48000,
    "channels": 1,
    "duration_seconds": 180.0,
    "peak_dbfs": -1.2,
    "rms_dbfs": -18.0,
    "clipped_samples": 0,
    "dc_offset": 0.0001
  },
  "target": {
    "sample_rate": 48000,
    "channels": 1,
    "duration_seconds": 180.0,
    "peak_dbfs": -0.8,
    "rms_dbfs": -15.5,
    "clipped_samples": 0,
    "dc_offset": 0.0002
  },
  "latency": {
    "estimated_samples": 128,
    "auto_estimated_samples": 123,
    "manual_adjustment_samples": 5,
    "effective_samples": 128,
    "confidence": 0.94,
    "method": "cross_correlation"
  },
  "capture_profile": {
    "duration_seconds": 120.0,
    "recommended_max_windows": 2048,
    "handling": "sampled_windows"
  },
  "gain": {
    "input_peak_dbfs": -8.2,
    "target_peak_dbfs": -7.1,
    "rms_delta_db": 1.4,
    "headroom_db": 7.1,
    "verdict": "healthy",
    "guidance": "Levels are in a good range for training."
  },
  "warnings": []
}
```

Latency alignment strategy:

1. Prefer known impulse markers in the capture signal.
2. Fall back to cross-correlation on active regions.
3. Store latency in samples, milliseconds, and confidence.
4. Preserve enough pre/post-roll so the user can manually nudge in the UI.
5. Never silently discard large unmatched regions.

Acceptance criteria:

- Prepared files have matching sample rate, channel count, and effective length.
- The report contains enough data to drive UI warnings.
- Alignment can be manually overridden from the desktop app; applying the
  override regenerates prepared audio with the saved manual adjustment.

### Step 6.3 Implement Dataset Windowing

Implemented in `rttrainer/training/dataset.py`.

Training examples should be generated from aligned audio with reproducible
windowing.

Recommended first approach:

1. Normalize or scale according to a documented policy.
2. Slice sequences into fixed windows.
3. Keep recurrent hidden state handling explicit.
4. Use deterministic train/validation/test splits.
5. Avoid training on long silence-dominated regions.
6. Save split metadata and random seed.
7. Sample windows across long captures instead of only taking the first windows.

Acceptance criteria:

- Re-running training with the same seed produces stable sampled windows.
- Dataset summaries record available windows, selected windows, train/validation
  counts, stride, sample rate, duration, and selection mode.
- Unit tests cover WAV IO, resampling, stereo policy, manual alignment metadata,
  and long-capture window sampling.

### Step 6.4 Implement Training Loop

Implemented in `rttrainer/training/runner.py`.

The default runner trains Keras models whose layer graph is known to export
through `rttrainer.export_rtneural.keras_exporter`. Backend selection remains
explicit in the run manifest: `keras` for the canonical path and `pytorch` only
for presets that have matching parity fixtures.

Device selection is reported, but the first reliable requirement is
reproducibility. The run records TensorFlow-visible accelerators for Keras runs
and PyTorch CUDA/MPS availability for optional PyTorch runs.

Training loop requirements:

1. Build the model from a stable preset ID rather than deserializing arbitrary
   user graphs.
2. Save Keras model weights/config for the canonical path, or
   `model.state_dict()` for optional PyTorch runs.
3. Save optimizer state and scheduler state for resume.
4. Checkpoint every configured N epochs.
5. Track best checkpoint by validation ESR or selected quality metric.
6. Support cancellation by signal and by a cancellation file in the run folder.
7. Emit JSONL progress events after every epoch.
8. Save exact package versions, device name, seed, and preset metadata.
9. Support early stopping by validation ESR plateau.
10. Save per-epoch validation history for the desktop validation curve.
11. Save good/usable/needs-work quality assessment language for the report UI.

Validation/inference preview requirements:

1. Run the backend in inference mode.
2. For PyTorch, call `model.eval()` and use `torch.no_grad()` or
   `torch.inference_mode()`.
3. Restore training mode only if the training loop continues afterward.
4. Render target, prediction, and residual WAV files.
5. Compute metrics on the same aligned test segment used for preview audio.

Acceptance criteria:

- Training can resume from a checkpoint.
- Cancelling leaves a valid interrupted state and latest checkpoint.
- Validation previews do not allocate training graphs.
- Run metadata is sufficient to reproduce the run.
- Desktop controls now expose epochs, early-stop patience, minimum ESR
  improvement, and training window budget.

### Step 6.5 Implement Losses And Metrics

Implemented in `rttrainer/metrics/audio_metrics.py`.

Minimum v1 metrics:

1. ESR or normalized error.
2. MAE.
3. RMSE.
4. Peak residual.
5. RMS residual.
6. Real-time factor from the native benchmark.

Recommended additional metrics:

1. A-weighted or task-filtered ESR.
2. Multi-resolution STFT loss for training or reporting.
3. Loudness-aware segment reporting.

Expose user-facing summaries as:

- Quality
- CPU
- Latency

Raw loss is not the only quality indicator. Users can A/B/C target, prediction,
and residual audio in the desktop app.

Acceptance criteria:

- Metrics are saved as `metrics.json`.
- Metric names and units are stable.
- The UI can display metrics without interpreting training internals.

### Step 6.6 Implement Export Packaging

Implemented across `trainer/rttrainer/export_rtneural/json_exporter.py`, the
`rttrainer export` CLI command, and the Rust `export_run` orchestration command.

Export folder contents:

```text
exports/<export-id>/
  model.rtneural.json
  package.json
  training-config.json
  metrics.json
  validation-report.json
  benchmark-report.json
  native-validation-report.json
  native-benchmark-report.json
  export-events.jsonl
  stderr.log
```

Package metadata:

```json
{
  "schema_version": 1,
  "name": "My Amp Capture",
  "project_id": "project_123",
  "run_id": "run_456",
  "preset": "lstm_standard",
  "sample_rate": 48000,
  "latency_samples": 123,
  "quality": {
    "esr": 0.05,
    "rmse": 0.01
  },
  "runtime": {
    "realtime_factor": 120.0,
    "backend": "eigen"
  },
  "compatibility": {
    "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d",
    "dynamic_json": true
  }
}
```

Acceptance criteria:

- Export fails if parity validation or native validation fails.
- Export fails if benchmark does not meet the selected target tier.
- Export can be regenerated from the saved run without retraining.
- Export package metadata includes validation/benchmark report summaries,
  artifact metadata, compatibility flags, and `.aidax` status marked deferred.

## 7. Phase 2: Desktop MVP

Only begin Phase 2 after the CLI can complete the core workflow.

### Step 7.1 Scaffold The Tauri App

Implemented with Tauri v2, React, TypeScript, Vite,
`@tauri-apps/plugin-shell`, and `@tauri-apps/plugin-dialog`.

Current implementation:

1. `app/` contains the Tauri v2 desktop app.
2. Tauri v2 dependencies, `@tauri-apps/plugin-shell`, and
   `@tauri-apps/plugin-dialog` are installed.
3. The shell and dialog plugins are registered in the Rust app.
4. `app/src/App.tsx` owns the current screen state system.
5. Rust owns project and app data directory paths.
6. `app/src/types.ts` mirrors the DTOs and events used by Tauri commands.

Acceptance criteria:

- `app` runs in development mode.
- A Tauri command can return app version and app data path.
- The frontend does not spawn arbitrary commands directly.
- Runtime inspection, external Python path selection, backend selection, CPU/MPS
  /CUDA availability, and package versions are exposed in the desktop UI.

### Step 7.2 Configure Sidecars

Tauri v2 sidecars should be bundled through `bundle.externalBin` and executed
through `tauri-plugin-shell`. Capability files must explicitly allow the sidecar
and arguments. Keep permissions narrow.

Expected sidecars:

```text
app/src-tauri/binaries/
  rttrainer-<target-triple>
  rtneural-validator-<target-triple>
```

The `externalBin` paths intentionally omit the target triple. Tauri resolves the
configured logical path to the platform-specific source file during bundling and
copies the resulting executable into the app bundle under the logical stem.

Tauri config shape:

```json
{
  "bundle": {
    "externalBin": [
      "binaries/rttrainer",
      "binaries/rtneural-validator"
    ]
  }
}
```

Capability shape:

```json
{
  "permissions": [
    "core:default",
    {
      "identifier": "shell:allow-execute",
      "allow": [
        {
          "name": "binaries/rttrainer",
          "sidecar": true,
          "args": true
        },
        {
          "name": "binaries/rtneural-validator",
          "sidecar": true,
          "args": true
        }
      ]
    }
  ]
}
```

Development staging command:

```bash
pnpm --filter rtneural-trainer-app package:sidecars:dev
```

This creates ignored POSIX shims for local work. The `rttrainer` shim delegates
to `uv run --extra tensorflow python -m rttrainer`; the validator shim delegates
to `native/rtneural-validator/build/rtneural-validator`.

Production staging command:

```bash
pnpm --filter rtneural-trainer-app package:sidecars
```

This builds or copies real executables into `app/src-tauri/binaries/`.
Support prebuilt override paths for release automation:

```bash
RTTRAINER_SIDECAR_SOURCE=/path/to/rttrainer \
RTNEURAL_VALIDATOR_SOURCE=/path/to/rtneural-validator \
pnpm --filter rtneural-trainer-app package:sidecars
```

Rust invocation uses `tauri_plugin_shell::ShellExt` and streams events from the
child process into the app event bus.

Acceptance criteria:

- Development builds can find local sidecar binaries.
- Production builds include platform-specific sidecar binaries.
- The app cannot execute arbitrary shell commands.
- Argument validation is tightened before public release.

### Step 7.3 Implement SQLite Project Store

Implemented as a Rust-side SQLite store in `app/src-tauri/src/lib.rs`. Large
artifacts remain on disk and SQLite stores relative paths, statuses, metadata,
job events, and project settings.

Tables:

```text
projects
audio_files
training_runs
checkpoints
exports
metrics
hardware_profiles
app_settings
job_events
```

The current schema is embedded in Rust migrations and includes the durable job
state needed for restart recovery. Artifact hashing and separate hardware
profile tables are still future polish.

Recommended status enums:

```text
project: active, archived
audio_file: imported, prepared, warning, invalid
training_run: queued, preparing, running, cancelling, interrupted, failed, completed
export: pending, validating, failed, ready
```

Implementation tasks:

1. Add migrations.
2. Add Rust repository methods.
3. Add TypeScript DTOs.
4. Persist every job event.
5. Store relative artifact paths from the project root.
6. Hash imported source audio for traceability.
7. Keep project lifecycle actions in Rust so SQLite rows and managed project
   folders stay consistent.

Acceptance criteria:

- Restarting the app reconstructs project state from SQLite and project folders.
- Interrupted runs remain visible and resumable when possible.
- Missing artifact files are detected and surfaced as repairable errors.
- Running jobs are marked interrupted after restart and exports with missing
  artifacts are audited into failed status.
- Renaming a project updates SQLite, refreshes the selected detail, and refreshes
  the sidebar list.
- Deleting a project cascades SQLite rows for audio reports, runs, exports, jobs,
  and events, then removes only the app-managed project folder. It must refuse
  unmanaged folders and active jobs.

### Step 7.4 Implement Job Orchestration

Rust should own job orchestration. Python owns training. The frontend owns
display and user intent.

Rust commands:

```text
create_project
rename_project
delete_project
update_project_audio
update_project_alignment
start_training
cancel_training_run
resume_training_run
get_run_preview
export_run
open_export_folder
inspect_device
get_runtime_settings
update_runtime_settings
```

Job runner responsibilities:

1. Write manifest JSON into the run folder.
2. Spawn the appropriate sidecar.
3. Stream JSONL stdout into `job_events`.
4. Store stderr logs.
5. Update SQLite statuses transactionally.
6. Send progress updates to the frontend.
7. Handle cancellation.
8. Prevent two training jobs from corrupting the same run folder.

Acceptance criteria:

- Training progress survives app refresh.
- Cancelling from the UI reaches the Python process.
- Failed jobs keep enough logs for diagnosis.
- A project can have multiple runs and exports.
- Sidecar stdout/stderr is streamed to the frontend as `sidecar-progress` events
  and persisted as job events.

### Step 7.5 Build The Main Screens

The app should feel like a guided audio workbench, not an ML dashboard.

Current screen: Projects and runtime sidebar

- Recent projects
- Project selection from the sidebar
- Create project
- Rename project from the selected project header
- Delete project with two-step confirmation
- Last run status
- Last quality score
- Runtime source, backend, CPU/MPS/CUDA availability, package versions
- Export readiness
- Search/filter by tags is deferred

Project lifecycle behavior:

- Rename opens an inline editor in the selected project header. Names are
  trimmed, required, limited to 120 characters, and blocked while project jobs or
  other mutations are active.
- Delete uses a confirmation click before the destructive action. It is blocked
  while project jobs or other mutations are active.
- Delete removes SQLite metadata and the managed project folder under app data.
  It does not delete arbitrary external source WAVs outside the app-managed
  project directory.

Current screen: Capture

- Import dry input WAV with native file picker
- Import processed target WAV with native file picker
- Show sample rate, duration, channel count, peak, RMS, clipping, DC offset
- Show warnings before the user trains
- Optional resampling and stereo/multichannel policy controls
- Gain/headroom and long-capture guidance

Current screen: Align

- Display latency estimate
- Show waveform-style alignment overlay
- Manual sample nudge with saved apply action
- Trim controls are deferred
- Confidence indicator
- Re-run alignment through saved manual adjustment

Current screen: Train

- Preset picker: Dense, GRU, Light LSTM, Standard LSTM, Conv1D, Conv1D
  BatchNorm/PReLU, Hybrid
- Hardware indicator: CUDA, MPS, or CPU
- Runtime warning for heavy presets
- Validation ESR curve
- Current best metric
- Cancel/resume
- Early stopping controls
- Preset recommendation from target type, backend, capture duration, alignment,
  and gain warnings
- "Train again with same settings" and "train three seeds" are deferred

Current screen: Evaluate

- A/B/C playback: target, prediction, residual
- Mini peak waveforms for preview artifacts
- Spectrum or residual plot is deferred
- Metrics table
- Training report display with good/usable/needs-work language
- Notes field
- Mark run as favorite is deferred

Current screen: Export

- RTNeural JSON export status
- Python parity status
- Native RTNeural validation status
- Benchmark table
- `.aidax` compatibility marked deferred after license/format review
- Open export folder

Deferred screen: Library

- Local model collection
- Tags
- Notes
- Quality and CPU columns
- Re-export or duplicate project

Acceptance criteria:

- A non-developer can complete the primary workflow without seeing a terminal.
- Warnings are surfaced before expensive training starts.
- Export is disabled until validation passes.

### Step 7.6 Implement Audio Preview

Implemented as offline previews. Live processing remains deferred.

Current behavior:

1. Render preview WAV files during `evaluate`.
2. Load mini peaks for display.
3. Support target, prediction, and residual playback.
4. Keep playback latency non-critical.
5. Keep playback and preview metadata tied to the run artifacts.

Acceptance criteria:

- The user can hear target versus prediction without exporting.
- Residual audio is easy to access.
- Large WAV files do not freeze the UI.

### Step 7.7 Implement Export Gate

Before an export is marked ready, the app verifies:

1. Audio preparation report has no blocking errors.
2. Training run completed.
3. Best checkpoint exists.
4. RTNeural JSON was generated.
5. Python parity passed.
6. Native RTNeural validator passed.
7. Benchmark passed the selected CPU tier.
8. Package metadata includes sample rate, latency, architecture, metrics, and
   RTNeural commit.

Acceptance criteria:

- The UI clearly says why an export is blocked.
- The exported folder is complete and self-describing.
- Re-running validation updates the report without retraining.

## 8. Phase 3: Runtime Integrations

Only add these after the core v1 workflow is reliable.

### Option 8.1 `.aidax`-Style Export

Tasks:

1. Review the format and license obligations.
2. Define a metadata envelope owned by this app if direct compatibility is not
   legally or technically appropriate.
3. Include model JSON, sample rate, latency, architecture, and notes.
4. Add import/export tests with representative players if allowed.

Exit criteria:

- The package loads in the intended target or clearly documents compatibility
  limits.

### Option 8.2 Generated JUCE Player

Tasks:

1. Generate a small JUCE standalone or plugin project.
2. Use RTNeural-example as the implementation reference for the generated
   project shape.
3. Embed the RTNeural JSON with JUCE `BinaryData` for fixed exports, or load it
   from disk for user-swappable models.
4. Link RTNeural with CMake and choose the backend explicitly:
   `RTNEURAL_STL`, `RTNEURAL_XSIMD`, or `RTNEURAL_EIGEN`.
5. Parse dynamic JSON with `RTNeural::json_parser::parseJson<float>()`.
6. Keep one model instance per audio channel unless the exported architecture is
   explicitly multi-channel.
7. Reset models in `prepareToPlay` and call `forward()` in the sample loop.
8. Add gain, bypass, meters, and a benchmark or smoke test.

Exit criteria:

- A user can immediately audition the model in a simple native player.
- The generated player can load this app's exported JSON without manual edits.

### Option 8.3 Compile-Time RTNeural Model Generation

Tasks:

1. Limit generation to known architecture presets.
2. Emit C++ model type and weight arrays.
3. Compare compile-time output to dynamic JSON output.
4. Benchmark compile-time versus dynamic loading.
5. Use RTNeural-example's compile-time path as a reference, but keep dynamic JSON
   as the canonical v1 path.

Exit criteria:

- Generated compile-time models are faster or smaller enough to justify the
  additional complexity.

### Option 8.4 Cloud Training

Tasks:

1. Keep local training as the default.
2. Add upload consent and project packaging.
3. Run the same CLI pipeline remotely.
4. Download the same export package shape.

Exit criteria:

- Cloud and local runs produce compatible reports and exports.

## 9. Testing Strategy

Testing must focus on parity, audio edge cases, and packaging. A beautiful UI is
not useful if exported weights are subtly wrong.

Current local verification commands:

```bash
# Python unit, audio, registry, golden fixture, and parity tests
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m unittest discover -s tests -v)

# Golden fixture freshness
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/generate_golden_rtneural_fixtures.py --check)

# Native validator build and smoke
cmake -S native/rtneural-validator -B native/rtneural-validator/build
cmake --build native/rtneural-validator/build
python3 scripts/smoke_rtneural_validator.py

# Supported Keras layer export matrix through native RTNeural
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/smoke_rtneural_keras_layers.py)

# Frontend and Rust
pnpm --filter rtneural-trainer-app build
(cd app/src-tauri && cargo test)

# Tauri workflow and debug packaged-app smoke
pnpm --filter rtneural-trainer-app smoke:tauri-workflow
pnpm --filter rtneural-trainer-app smoke:packaged-app
```

### Python Tests

Current coverage includes:

1. Audio decode and validation.
2. Sample-rate conversion metadata.
3. Mono/stereo conversion.
4. Manual latency adjustment metadata.
5. Latency estimation.
6. Long-capture sampled window selection.
7. Preset construction.
8. Metrics.
9. Golden RTNeural JSON fixture freshness.
10. Python parity for every exported preset.
11. Native RTNeural validation for every golden fixture.
12. Support matrix registry checks.

Still worth adding:

- Training checkpoint save/resume unit tests at the Python layer.
- More real-world capture edge cases.
- Stronger schema validation around manifests and reports.

### Native Tests

Current native coverage is mostly smoke/golden driven:

1. Valid RTNeural JSON loads.
2. Known fixture output matches tolerance.
3. Benchmark output schema is stable enough for desktop display.

Still worth adding:

- Invalid JSON failure fixtures.
- NaN/Inf output failure fixtures.
- More benchmark threshold fixtures by preset tier.

### Desktop Tests

Current desktop coverage includes Rust unit tests and smoke scripts for:

1. SQLite migrations.
2. Project folder creation.
3. Manifest generation.
4. Job event parsing.
5. Cancellation status transitions.
6. Runtime settings persistence.
7. Missing artifact audits and restart recovery.
8. Tauri workflow smoke.
9. Debug packaged-app smoke.

Still worth adding:

- UI screen smoke with a real Tauri window.
- Packaged-app smoke that runs a tiny end-to-end training/export workflow.
- Export gate edge-case tests around failed native validation and failed
  benchmarks.

### Golden Fixture Tests

Golden fixtures are generated and checked by script rather than hand-maintained
by editing individual JSON files:

```text
fixtures/rtneural-json/golden/
  dense_only.rtneural.json
  gru_light.rtneural.json
  lstm_light.rtneural.json
  lstm_standard.rtneural.json
  conv1d_light.rtneural.json
  conv1d_bn_prelu.rtneural.json
  conv_gru_hybrid.rtneural.json
```

Acceptance criteria:

- CI can run a small end-to-end parity workflow.
- Any exporter change must update or pass golden tests.
- Native validation runs in CI on at least one platform.

## 10. Packaging Strategy

Packaging is one of the highest-risk areas because TensorFlow and PyTorch are
large and platform-specific.

Recommended v1 approach:

1. Package `rttrainer` as a sidecar binary per platform.
2. Package `rtneural-validator` as a native sidecar per platform.
3. Put sidecars in `app/src-tauri/binaries/` using Tauri's expected target-triple
   naming convention.
4. Keep a developer mode that can call an unpackaged Python module.
5. Keep an advanced setting for an external Python environment if packaging size
   becomes unacceptable.

Current packaging tasks:

1. Build Python sidecar with PyInstaller.
2. Build native validator with CMake release settings.
3. Copy sidecars into Tauri binary folder.
4. Name sidecars as `rttrainer-<target-triple>` and
   `rtneural-validator-<target-triple>`.
5. Configure Tauri `bundle.externalBin` with unsuffixed logical paths.
6. Execute packaged sidecars through `tauri-plugin-shell`, not through arbitrary
   shell command spawning.
7. Run a packaged-app smoke test.
8. Verify sidecar execution permissions.
9. Verify code signing and notarization requirements on macOS.
10. Verify Windows signing and antivirus false-positive risk.

Current release smoke command:

```bash
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles app,dmg
```

Use `--bundles deb` on Linux and `--bundles nsis` on Windows. The release smoke
builds real sidecars, runs `rttrainer --version`, runs
`rttrainer inspect-device --json`, checks the native validator CLI, builds the
Tauri release bundle with `--ci --no-sign`, validates the copied sidecars beside
the release binary, and writes
`app/src-tauri/target/release/release-artifacts-manifest.json`.

Current release smoke gate:

- Packaged sidecars report their versions.
- Packaged sidecars can inspect hardware.
- Native validator CLI is present and executable.
- Tauri release bundles can be produced with `--ci --no-sign`.
- Copied sidecars are found beside the release binary.
- Release bundles produce uploadable artifacts and a manifest on every target OS.

Remaining release acceptance:

- Packaged app can prepare audio, train a tiny model, validate, and export from
  inside the installed bundle.
- The app can surface a clear error when a sidecar is missing or incompatible.
- macOS and Windows signing/notarization policy is implemented.

## 11. CI And Release Gates

CI is split into a fast desktop workflow and a slower release-packaging
workflow.

Current `.github/workflows/ci.yml` jobs:

1. Install Tauri Linux dependencies.
2. Setup pnpm 11, Node 22, uv, Python 3.12, and Rust stable.
3. Sync trainer dependencies with the TensorFlow extra.
4. Build the native RTNeural validator.
5. Smoke-test the native validator.
6. Run Python tests.
7. Check golden RTNeural JSON fixtures.
8. Build the frontend.
9. Stage Tauri development sidecars.
10. Run Rust tests.
11. Run Tauri workflow smoke.
12. Run debug packaged-app smoke.

Current `.github/workflows/release-packaging.yml` jobs:

1. Build real PyInstaller `rttrainer` and CMake `rtneural-validator` sidecars.
2. Smoke-test staged release sidecars with `rttrainer --version`,
   `rttrainer inspect-device --json`, and native validator CLI checks.
3. Run Tauri release bundle smoke with `tauri build --ci --no-sign`.
4. Validate copied packaged sidecars beside the release binary.
5. Collect `release-artifacts-manifest.json`.
6. Upload bundle outputs and staged sidecars for Linux deb, macOS app+dmg, and
   Windows NSIS matrix jobs.

Release gate checklist:

- Export parity passes.
- Native validation passes.
- Benchmark report is generated.
- Known-good preset matrix is current.
- Licenses are reviewed.
- Packaged sidecars run on each target OS.
- Tauri release bundles are produced on Linux, macOS, and Windows.
- Release artifacts include the platform bundle, sidecars, and artifact manifest.
- Project migration test passes from the previous app version.

Known CI/release gaps:

- Release bundles are unsigned; signing and notarization require credentials and
  product policy decisions.
- Packaged-app smoke currently proves bundle shape and sidecar execution, not a
  full tiny train/export workflow inside the installed app.
- UI smoke tests with real Tauri windows are still missing.

## 12. Known-Good Preset Matrix

Maintain this table in code and docs. Update it whenever RTNeural, TensorFlow,
PyTorch, or export logic changes.

| Preset | Keras Train/Build | PyTorch Train | JSON Export | Python Parity | Native Validate | Benchmark | Release |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dense_only` | Required | Later | Required | Required | Required | Required | v1 |
| `gru_light` | Required | Later | Required | Required | Required | Required | v1 |
| `lstm_light` | Required | Optional | Required | Required | Required | Required | v1 |
| `lstm_standard` | Required | Optional | Required | Required | Required | Required | v1 |
| `conv1d_light` | Required | Later | Required | Required | Required | Required | v1-plus |
| `conv1d_bn_prelu` | Required | Later | Required | Required | Required | Required | v1-plus |
| `conv_gru_hybrid` | Required | Later | Required | Required | Required | Required | v1-plus |
| `heavy_recurrent` | Later | Later | Required before exposure | Required | Required | Required | Warned v1 or later |
| `conv1d_tcn` | Later | Later | Later | Later | Later | Later | Defer |

Do not expose a preset in the UI unless its row is green for the target release.

## 13. Data And Metadata Schemas

The UI, Python sidecar, Rust orchestrator, and reports all need a shared
contract. Today those schemas are encoded in TypeScript DTOs, Rust structs and
SQLite migrations, Python manifest/report writers, and the golden fixture tests.
Dedicated JSON Schema files are deferred until the contracts stabilize further.

Deferred schema file layout:

```text
schemas/
  prepare-manifest.schema.json
  train-manifest.schema.json
  evaluate-manifest.schema.json
  export-manifest.schema.json
  progress-event.schema.json
  metrics.schema.json
  validation-report.schema.json
  benchmark-report.schema.json
  package.schema.json
```

Schema rules:

1. Include `schema_version` in every manifest and report.
2. Store paths relative to the project root when possible.
3. Store absolute paths only for imported sources outside the project.
4. Include UTC timestamps.
5. Include app, trainer, validator, TensorFlow, PyTorch, and RTNeural versions.
6. Include hardware and device selection metadata for training runs.

Current acceptance:

- Rust and Python both write and read versioned manifests/reports.
- Reports are readable without opening the SQLite database.
- SQLite migrations are covered by Rust tests.

Remaining work:

- Add dedicated JSON Schema files for external integrations.
- Add stricter manifest validation errors at the Python boundary.
- Require explicit migration notes for breaking schema changes.

## 14. Implementation Order

The original order below is now mostly complete:

1. Repo skeleton and dependency pins: complete.
2. Python trainer/export package: complete.
3. LSTM, GRU, Dense, Conv1D, BatchNorm/PReLU, and hybrid presets: complete for
   the supported Keras path.
4. RTNeural JSON export and Python parity: complete for every exposed preset.
5. Native RTNeural validator/benchmark sidecar: complete.
6. WAV preparation, resampling, stereo policy, latency estimation, manual
   alignment override, and preparation reports: complete.
7. Paired-WAV training with metrics, checkpoints, preview WAVs, validation
   history, early stopping, and quality language: complete for local v1.
8. SQLite project/job store, recovery, cancellation, resume, and event
   persistence: complete.
9. Tauri/React UI for project creation/selection/rename/delete, Capture, Align,
   Train, Evaluate, Export, notes, runtime inspection, progress, and preview
   playback: complete.
10. Development and production sidecar packaging plus debug/release smoke
    scripts: complete.

Current next implementation order:

1. Run a real capture project through the full app and tune capture/gain/preset
   recommendation thresholds.
2. Add UI smoke tests with real Tauri windows.
3. Add packaged-app smoke that exercises a tiny train/export workflow inside the
   packaged app, not only bundle shape and sidecar execution.
4. Decide macOS signing/notarization, Windows signing, artifact retention, and
   release-publishing policy.
5. Add deeper waveform/spectrum inspection for target, prediction, and residual.
6. Complete a full accessibility audit and tune error/report copy from
   real-world captures.
7. Revisit `.aidax`, generated JUCE/player, and compile-time RTNeural only after
   release distribution is boring.

## 15. First End-To-End Demo Target

The first demo should be deliberately modest.

Input:

- 48 kHz mono dry WAV
- 48 kHz mono wet WAV
- 30 to 180 seconds
- No severe clipping

Model:

- `lstm_light`
- Hidden size 8 or 12
- One recurrent layer
- Dense output

Output:

- `model.rtneural.json`
- `package.json`
- `validation-report.json`
- `benchmark-report.json`
- target/prediction/residual preview WAVs

Demo script:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer prepare \
  --manifest ../projects/demo/prepare-manifest.json
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer train \
  --manifest ../projects/demo/runs/run_001/train-manifest.json
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer evaluate \
  --manifest ../projects/demo/runs/run_001/evaluate-manifest.json
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer export \
  --manifest ../projects/demo/runs/run_001/export-manifest.json

cd ..
native/rtneural-validator/build/rtneural-validator validate \
  --model projects/demo/exports/export_001/model.rtneural.json \
  --input projects/demo/runs/run_001/test-input.wav \
  --reference projects/demo/runs/run_001/previews/prediction.wav \
  --report projects/demo/exports/export_001/validation-report.json
```

Success means:

- Training completes.
- Preview audio is generated.
- RTNeural JSON loads.
- Native output matches expected tolerance.
- Benchmark says the model is safe for the selected target tier.

## 16. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Python ML sidecar is too large | Large installers and slow updates | Use `uv` lockfiles, keep TensorFlow/PyTorch extras explicit, support external Python env as advanced path |
| Export weight layouts are wrong | Bad audio or crashes | Golden parity tests for every preset and layer |
| Latency alignment is wrong | Model learns delay instead of tone | Impulse alignment, cross-correlation fallback, manual nudge UI |
| User capture data is poor | Bad trained models | Preflight warnings, capture checklist, clipping/silence detection |
| Heavy models miss real-time budgets | Poor downstream experience | Benchmark before export and gate heavy presets |
| RTNeural issue affects preset | Crash or mismatch | Pin commit, keep known-good matrix, stay conservative |
| Reference project license contamination | Legal/product risk | Reimplement exporter logic and review licenses before compatibility features |
| UI hides important warnings | User mistrust | Block export on validation failures and explain the blocker |

## 17. Definition Of Done For V1

Local desktop V1 is effectively done when all of the following are true:

| Requirement | Status |
| --- | --- |
| A user can create a project from paired WAV files. | Implemented |
| A user can select, rename, and delete local projects from the desktop UI. | Implemented |
| The app validates audio and shows actionable warnings. | Implemented |
| The app estimates, stores, and manually overrides latency. | Implemented |
| The user can train curated Keras presets locally. | Implemented |
| Training progress, logs, checkpoints, metrics, validation curves, and quality language persist. | Implemented |
| The app renders and plays target, prediction, and residual preview audio. | Implemented |
| The app exports RTNeural JSON. | Implemented |
| Python/export parity passes. | Implemented in tests and golden fixtures |
| Native RTNeural validation passes. | Implemented in validator and CI smoke |
| Benchmark results are shown before export is marked ready. | Implemented |
| The package contains metadata, reports, previews, and model JSON. | Implemented |
| The debug packaged desktop app can complete sidecar discovery and launch checks without a terminal. | Implemented |
| Signed/notarized release packages are ready for end users. | Deferred |
| Packaged-app smoke runs a tiny full train/export workflow inside the installed bundle. | Deferred |
| Real capture thresholds are tuned against representative material. | Deferred |

## 18. Documentation To Keep Updated

Current documents:

1. `README.md`
2. `docs/Research-RTNeural-Training-Desktop-App.md`
3. `docs/Implementation-Guide-RTNeural-Training-Desktop-App.md`

The README should stay task-oriented for users and developers. The research note
should preserve source findings and RTNeural reference context. This
implementation guide should remain the engineering map: what is built, what is
deferred, and which smoke/CI gates define confidence today.

Deferred documents worth splitting out when the project hardens:

1. `docs/RTNeural-Export-Schema.md`
2. `docs/Preset-Compatibility-Matrix.md`
3. `docs/Audio-Capture-Guidelines.md`
4. `docs/Packaging-And-Sidecars.md`
5. `docs/Troubleshooting.md`

Until those exist, keep the authoritative preset matrix in code and generated
script output, keep export/package contract details in this guide, and mirror
operator-facing setup in `README.md`.
