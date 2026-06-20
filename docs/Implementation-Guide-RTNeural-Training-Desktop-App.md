# Implementation Guide: RTNeural Training Desktop App

Date: 2026-06-19

Source reviewed: [Research-RTNeural-Training-Desktop-App.md](Research-RTNeural-Training-Desktop-App.md)

This guide turns the research note into a step-by-step build plan for a desktop
application that trains neural audio models locally and exports RTNeural-ready
artifacts. The core product promise is intentionally narrow:

> Give the app a dry input file and a matching processed target file. It trains a
> real-time-safe model, proves the exported RTNeural JSON matches the trained
> model, proves RTNeural can load it, and reports runtime cost before the user
> ships the model.

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
validator sidecar. The Python sidecar should make TensorFlow/Keras the canonical
RTNeural JSON path while keeping PyTorch isolated behind optional extras and
known-good parity tests.

```text
rtneural-trainer/
  app/
    package.json
    src/
      app/
      components/
      features/
      lib/
    src-tauri/
      Cargo.toml
      tauri.conf.json
      capabilities/
      binaries/
      src/
  trainer/
    pyproject.toml
    rttrainer/
      __init__.py
      cli.py
      config.py
      data/
      export_rtneural/
      metrics/
      models/
      reports/
      training/
      validation/
    tests/
  native/
    rtneural-validator/
      CMakeLists.txt
      src/
      tests/
  fixtures/
    audio/
    models/
  projects/
    .gitkeep
  docs/
```

The app should store structured state in SQLite and large artifacts on disk.

```text
projects/
  <project-id>/
    audio/
      original/
      prepared/
    runs/
      <run-id>/
        manifest.json
        checkpoints/
        events.jsonl
        metrics.json
        previews/
        plots/
    exports/
      <export-id>/
        model.rtneural.json
        package.json
        validation-report.json
        benchmark-report.json
        preview-target.wav
        preview-prediction.wav
        preview-residual.wav
```

## 3. Milestone Map

Build the product in four phases. Do not start the desktop UI until the CLI can
train, export, and validate one boring LSTM preset.

| Phase | Name | Goal | Exit Criteria |
| --- | --- | --- | --- |
| 0 | Export spike | Prove Keras to RTNeural JSON parity | Tiny Dense/LSTM/GRU/Conv1D/activation/BatchNorm/PReLU exports load in native RTNeural and match within tolerance |
| 1 | CLI trainer | Build product core without UI | Paired WAVs produce validated RTNeural JSON, metrics, and preview audio |
| 2 | Desktop MVP | Make the workflow usable by non-developers | Project creation, import, align, train, evaluate, export all work without a terminal |
| 3 | Runtime integrations | Make exports immediately useful | `.aidax` envelope, generated player, compile-time model, or cloud training path |

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
   - `tensorflow` for canonical Keras export fixtures and compatible presets
   - `torch` only for the optional PyTorch training/export backend
   - `torchaudio` if used for PyTorch-side audio IO/resampling
   - `scipy`
   - `soundfile`
   - `pydantic`
   - `typer` or `argparse`
   - `matplotlib` or another plot renderer
5. Record frontend/runtime versions:
   - Tauri v2
   - React
   - TypeScript
   - Vite or the selected frontend build tool
6. Decide whether the default packaged trainer includes TensorFlow and/or
   PyTorch, or whether the app also supports an advanced external Python
   environment.

Recommended local commands:

```bash
cd trainer
uv lock
uv sync
uv sync --extra tensorflow
uv sync --extra training
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

### Step 5.1 Create The Minimal Python Package

Create `trainer/pyproject.toml` and a package named `rttrainer`.

Initial modules:

```text
trainer/rttrainer/
  cli.py
  models/presets.py
  export_rtneural/keras_exporter.py
  export_rtneural/json_exporter.py
  export_rtneural/registry.py
  validation/parity.py
  training/synthetic.py
```

Implementation tasks:

1. Add a CLI entry point named `rttrainer`.
2. Add `rttrainer spike-train --preset lstm-light --out <dir>`.
3. Generate deterministic synthetic data:
   - sine sweep
   - stepped gain
   - simple nonlinear saturation target
4. Train a tiny recurrent model quickly enough for CI.
5. Save:
   - backend checkpoint
   - `training-config.json`
   - `backend-output.npy`
   - `test-input.npy`

Acceptance criteria:

- The command trains in a few seconds on CPU.
- The same seed produces stable outputs.
- The model uses only layers planned for RTNeural export.

### Step 5.2 Implement First Model Presets And Support Matrix

Implement presets as code-owned architecture factories. Do not serialize
arbitrary model definitions.

Start with:

| Preset ID | Architecture | Use |
| --- | --- | --- |
| `lstm_light` | 1 LSTM, hidden 8 or 12, dense output | Lowest-risk recurrent export |
| `lstm_standard` | 1 LSTM, hidden 16 or 20, dense output | Default v1 target |
| `gru_light` | 1 GRU, hidden 8 or 12, dense output | GRU parity coverage |
| `dense_memoryless` | Small dense stack | Simple gain/EQ/saturation targets |

Model rules:

1. Use mono input and mono output first.
2. Keep tensor shape conventions documented in the model file.
3. Avoid BatchNorm in v1 unless export parity is already covered.
4. Avoid Conv1D/TCN in the UI until recurrent export and validation are boring,
   but keep Conv1D in the fixture/support scripts because RTNeural-compare
   benchmarks it.
5. Add each preset to a known-good compatibility matrix.
6. Keep the RTNeural layer/activation support matrix in code and generate docs
   from it instead of hand-maintaining scattered tables.

Benchmark-driven v1 support:

| Kind | v1 | v1-plus | Later | Defer |
| --- | --- | --- | --- | --- |
| Layers | Dense, GRU, LSTM | Conv1D | Conv2D, BatchNorm1D, BatchNorm2D, PReLU | MaxPooling |
| Activations | tanh, ReLU, sigmoid |  | softmax, ELU, PReLU |  |

Acceptance criteria:

- Presets can be created by stable string IDs.
- Each preset declares input channels, output channels, hidden size, layer count,
  estimated CPU tier, and export support.
- `scripts/rtneural_support_matrix.py --format markdown` prints the current
  layer and activation plan.
- `scripts/generate_rtneural_keras_fixtures.py --list` reports the default
  fixture scope without importing TensorFlow.

### Step 5.3 Build The RTNeural JSON Exporters

Create `trainer/rttrainer/export_rtneural/keras_exporter.py` first, then keep
`trainer/rttrainer/export_rtneural/json_exporter.py` for the optional PyTorch
path.

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

Create `trainer/rttrainer/validation/parity.py`.

The Python parity validator should:

1. Load the original backend checkpoint or Keras model.
2. Load the exported JSON.
3. Run the same test input through both paths.
4. Compare output arrays sample by sample.
5. Save a tolerance report.

If a full Python RTNeural JSON simulator is too much for Phase 0, start with
exporter-level tensor layout tests and let the native validator be the runtime
source of truth. Keep the Python validator interface anyway so the UI and CI
contract does not change later.

Acceptance criteria:

- Reports include max absolute error, mean absolute error, RMSE, and failing
  sample indices when tolerance is exceeded.
- Parity thresholds are preset-specific and stored in code.

### Step 5.5 Build The Native RTNeural Validator

Create `native/rtneural-validator`.

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

Implementation tasks:

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

Phase 1 turns the spike into the real product core. The app UI will eventually
call these commands, so design them as stable sidecar contracts.

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

Create `rttrainer/data/audio_io.py`, `rttrainer/data/prepare.py`, and
`rttrainer/data/analysis.py`.

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
    "estimated_samples": 123,
    "confidence": 0.94,
    "method": "impulse"
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
- Alignment can be manually overridden later without rerunning decode.

### Step 6.3 Implement Dataset Windowing

Create `rttrainer/data/dataset.py`.

Training examples should be generated from aligned audio with reproducible
windowing.

Recommended first approach:

1. Normalize or scale according to a documented policy.
2. Slice sequences into fixed windows.
3. Keep recurrent hidden state handling explicit.
4. Use deterministic train/validation/test splits.
5. Avoid training on long silence-dominated regions.
6. Save split indices and random seed.

Acceptance criteria:

- Re-running `prepare` with the same seed produces the same splits.
- Dataset code can stream from disk if files become too large for memory.
- Unit tests cover short files, stereo-to-mono conversion, and silence trimming.

### Step 6.4 Implement Training Loop

Create `rttrainer/training/runner.py`.

The default runner should train Keras models whose layer graph is known to export
through `rttrainer.export_rtneural.keras_exporter`. Keep backend selection
explicit in the run manifest: `keras` for the canonical path and `pytorch` only
for presets that have matching parity fixtures.

Device selection should be reported, but the first reliable requirement is
reproducibility. Record TensorFlow-visible accelerators for Keras runs and
PyTorch CUDA/MPS availability for optional PyTorch runs.

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

### Step 6.5 Implement Losses And Metrics

Create `rttrainer/metrics`.

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

Do not make raw loss the only quality indicator. Users should be able to A/B/C
target, prediction, and residual audio in the desktop app.

Acceptance criteria:

- Metrics are saved as `metrics.json`.
- Metric names and units are stable.
- The UI can display metrics without interpreting training internals.

### Step 6.6 Implement Export Packaging

Create `rttrainer/export_rtneural/package.py`.

Export folder contents:

```text
exports/<export-id>/
  model.rtneural.json
  package.json
  training-config.json
  metrics.json
  validation-report.json
  benchmark-report.json
  preview-target.wav
  preview-prediction.wav
  preview-residual.wav
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

## 7. Phase 2: Desktop MVP

Only begin Phase 2 after the CLI can complete the core workflow.

### Step 7.1 Scaffold The Tauri App

Create the app with Tauri v2, React, and TypeScript.

Implementation tasks:

1. Create `app/`.
2. Install Tauri v2 dependencies and `@tauri-apps/plugin-shell`.
3. Add the shell plugin to the Rust app.
4. Create a minimal React router or screen state system.
5. Add a local app data directory abstraction.
6. Add TypeScript types that mirror CLI manifest and event schemas.

Acceptance criteria:

- `app` runs in development mode.
- A Tauri command can return app version and app data path.
- The frontend does not spawn arbitrary commands directly.

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

This should create ignored POSIX shims for local work. The `rttrainer` shim can
delegate to `uv run --extra tensorflow python -m rttrainer`; the validator shim
can delegate to `native/rtneural-validator/build/rtneural-validator`.

Production staging command:

```bash
pnpm --filter rtneural-trainer-app package:sidecars
```

This should build or copy real executables into `app/src-tauri/binaries/`.
Support prebuilt override paths for release automation:

```bash
RTTRAINER_SIDECAR_SOURCE=/path/to/rttrainer \
RTNEURAL_VALIDATOR_SOURCE=/path/to/rtneural-validator \
pnpm --filter rtneural-trainer-app package:sidecars
```

Rust invocation should use `tauri_plugin_shell::ShellExt` and stream events from
the child process into the app event bus.

Acceptance criteria:

- Development builds can find local sidecar binaries.
- Production builds include platform-specific sidecar binaries.
- The app cannot execute arbitrary shell commands.
- Argument validation is tightened before public release.

### Step 7.3 Implement SQLite Project Store

Create a Rust-side SQLite store. Keep large artifacts on disk and store paths,
hashes, status, and metadata in SQLite.

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

Acceptance criteria:

- Restarting the app reconstructs project state from SQLite and project folders.
- Interrupted runs remain visible and resumable when possible.
- Missing artifact files are detected and surfaced as repairable errors.

### Step 7.4 Implement Job Orchestration

Rust should own job orchestration. Python owns training. The frontend owns
display and user intent.

Rust commands:

```text
create_project
import_audio
prepare_audio
start_training_run
cancel_training_run
resume_training_run
evaluate_run
export_run
open_export_folder
list_hardware
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

### Step 7.5 Build The Main Screens

The app should feel like a guided audio workbench, not an ML dashboard.

Screen 1: Projects

- Recent projects
- Last run status
- Last quality score
- Hardware used
- Export readiness
- Search/filter by tags

Screen 2: Capture

- Create or choose capture signal
- Import dry input WAV
- Import wet target WAV
- Show sample rate, duration, channel count, peak, RMS, clipping, DC offset
- Show warnings before the user trains

Screen 3: Align

- Display latency estimate
- Show waveform overlay near impulse or active transient
- Manual sample nudge
- Trim controls
- Confidence indicator
- Re-run alignment

Screen 4: Train

- Preset picker: Light, Standard, Heavy, Dense
- Hardware indicator: CUDA, MPS, or CPU
- Runtime warning for heavy presets
- Progress chart
- Current best metric
- Cancel/resume
- "Train again with same settings"
- "Train three seeds and keep best" after basic training is stable

Screen 5: Evaluate

- A/B/C playback: target, prediction, residual
- Waveform overlay
- Spectrum or residual plot
- Metrics table
- Notes field
- Mark run as favorite

Screen 6: Export

- RTNeural JSON export status
- Python parity status
- Native RTNeural validation status
- Benchmark table
- Optional `.aidax`-style package after license/format review
- Open export folder

Screen 7: Library

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

For v1, use offline previews. Live processing can wait.

Implementation tasks:

1. Render preview WAV files during `evaluate`.
2. Load peaks or downsampled waveforms for display.
3. Support target, prediction, and residual playback.
4. Keep playback latency non-critical.
5. Cache waveform data in the project folder.

Acceptance criteria:

- The user can hear target versus prediction without exporting.
- Residual audio is easy to access.
- Large WAV files do not freeze the UI.

### Step 7.7 Implement Export Gate

Before an export is marked ready, the app must verify:

1. Audio preparation report has no blocking errors.
2. Training run completed.
3. Best checkpoint exists.
4. RTNeural JSON was generated.
5. Python parity passed or was intentionally waived for a documented reason.
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

### Python Tests

Add tests for:

1. Audio decode and validation.
2. Sample-rate conversion metadata.
3. Mono/stereo conversion.
4. Clipping and DC offset detection.
5. Latency estimation.
6. Dataset split reproducibility.
7. Preset construction.
8. Training checkpoint save/resume.
9. Metrics.
10. Export JSON schema.
11. Tensor layout conversion.
12. Python parity reports.

### Native Tests

Add tests for:

1. Valid RTNeural JSON loads.
2. Invalid JSON fails clearly.
3. Known fixture output matches tolerance.
4. Benchmark output schema is stable.
5. NaN/Inf detection fails validation.

### Desktop Tests

Add tests for:

1. SQLite migrations.
2. Project folder creation.
3. Manifest generation.
4. Job event parsing.
5. Cancellation status transitions.
6. Export gate rules.
7. UI screen smoke tests.

### Golden Fixture Tests

Keep tiny fixtures in `fixtures/`:

```text
fixtures/
  audio/
    dry-48k-mono-short.wav
    wet-48k-mono-short.wav
  models/
    lstm-light-checkpoint.pt
    lstm-light.rtneural.json
    lstm-light-reference-output.wav
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

Packaging tasks:

1. Build Python sidecar with PyInstaller or equivalent.
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

Acceptance criteria:

- Packaged app can inspect hardware, prepare audio, train a tiny model, validate,
  and export.
- Packaged sidecars report their versions.
- The app can surface a clear error when a sidecar is missing or incompatible.

## 11. CI And Release Gates

Set up CI before Phase 2 becomes large.

Suggested jobs:

1. Python lint and unit tests.
2. Python tiny training smoke test.
3. RTNeural JSON export test.
4. Native CMake build.
5. Native validator fixture test.
6. Rust unit tests.
7. TypeScript typecheck.
8. Frontend test/build.
9. Tauri development build smoke test.
10. Packaged app smoke test on release branches.

Release gate checklist:

- Export parity passes.
- Native validation passes.
- Benchmark report is generated.
- Known-good preset matrix is current.
- Licenses are reviewed.
- Packaged sidecars run on each target OS.
- Project migration test passes from the previous app version.

## 12. Known-Good Preset Matrix

Maintain this table in code and docs. Update it whenever RTNeural, TensorFlow,
PyTorch, or export logic changes.

| Preset | Keras Train/Build | PyTorch Train | JSON Export | Python Parity | Native Validate | Benchmark | Release |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lstm_light` | Required | Optional | Required | Required | Required | Required | v1 |
| `lstm_standard` | Required | Optional | Required | Required | Required | Required | v1 |
| `gru_light` | Required | Optional | Required | Required | Required | Required | v1 if stable |
| `dense_memoryless` | Required | Optional | Required | Required | Required | Required | v1 if useful |
| `conv1d_fixture` | Required | Optional | Required | Required | Required | Required | v1-plus |
| `activations_fixture` | Required | Optional | Required | Required | Required | Required | v1-plus |
| `batchnorm_prelu_fixture` | Required | Optional | Required | Required | Required | Required | v1-plus |
| `heavy_recurrent` | Later | Later | Required before exposure | Required | Required | Required | Warned v1 or later |
| `conv1d_tcn` | Later | Later | Later | Later | Later | Later | Defer |

Do not expose a preset in the UI unless its row is green for the target release.

## 13. Data And Metadata Schemas

Define schemas early. The UI, Python sidecar, Rust orchestrator, and reports all
need a shared contract.

Recommended schema files:

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

Acceptance criteria:

- Rust and Python both validate manifests.
- Breaking schema changes require a migration plan.
- Reports are readable without opening the SQLite database.

## 14. Implementation Order

Use this order to avoid UI-first churn.

1. Create repo skeleton and dependency pins.
2. Build Python synthetic training spike.
3. Add LSTM light preset.
4. Export LSTM light to RTNeural JSON.
5. Build native RTNeural validator.
6. Make native parity pass for the synthetic fixture.
7. Add GRU light or dense memoryless parity coverage.
8. Build real WAV preparation command.
9. Add latency estimation and preparation reports.
10. Train from paired WAVs through CLI.
11. Export and validate a real paired-WAV run.
12. Add metrics and preview WAV rendering.
13. Add SQLite schema and Rust project store.
14. Scaffold Tauri/React UI.
15. Wire sidecar execution and event streaming.
16. Build Projects, Capture, Align, and Train screens.
17. Build Evaluate and Export screens.
18. Package sidecars.
19. Run packaged smoke tests.
20. Add optional runtime integration only after v1 gates pass, starting with an
    RTNeural-example-style JUCE player if immediate plugin auditioning is the
    highest-value path.

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
rttrainer prepare --manifest projects/demo/prepare-manifest.json
rttrainer train --manifest projects/demo/runs/run_001/train-manifest.json
rttrainer evaluate --manifest projects/demo/runs/run_001/evaluate-manifest.json
rttrainer export --manifest projects/demo/runs/run_001/export-manifest.json
rtneural-validator validate \
  --model projects/demo/exports/export_001/model.rtneural.json \
  --input projects/demo/audio/prepared/test-input.wav \
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

V1 is done when all of the following are true:

1. A user can create a project from paired WAV files.
2. The app validates audio and shows actionable warnings.
3. The app estimates and stores latency.
4. The user can train at least one Keras LSTM preset locally.
5. Training progress, logs, checkpoints, and metrics persist.
6. The app renders target, prediction, and residual preview audio.
7. The app exports RTNeural JSON.
8. Python/export parity passes.
9. Native RTNeural validation passes.
10. Benchmark results are shown before export is marked ready.
11. The package contains metadata, reports, previews, and model JSON.
12. The packaged desktop app can complete the workflow without a terminal.

## 18. Documentation To Keep Updated

Maintain these documents as the build progresses:

1. `docs/Research-RTNeural-Training-Desktop-App.md`
2. `docs/Implementation-Guide-RTNeural-Training-Desktop-App.md`
3. `docs/RTNeural-Export-Schema.md`
4. `docs/Preset-Compatibility-Matrix.md`
5. `docs/Audio-Capture-Guidelines.md`
6. `docs/Packaging-And-Sidecars.md`
7. `docs/Troubleshooting.md`

The implementation guide should remain the build map. The export schema and
preset matrix should become stricter engineering references once code exists.
