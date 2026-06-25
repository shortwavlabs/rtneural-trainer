# RTNeural Trainer

RTNeural Trainer is a local desktop workbench for preparing paired audio,
training a small neural audio model, exporting RTNeural-compatible JSON, and
validating/benchmarking the result before use in a real-time target.

The repo currently contains:

- `app/`: Tauri v2 + React desktop shell.
- `trainer/`: `uv`-managed Python sidecar and CLI.
- `native/rtneural-validator/`: CMake-built validator/benchmark sidecar.
- `scripts/`: RTNeural support and Keras fixture helper scripts.
- `docs/`: research notes and the current implementation guide.

The implementation is now a working local desktop prototype. The desktop app
calls the real Python `prepare`, `train`, `evaluate`, and `export` commands, and
the export path invokes the native RTNeural validator/benchmark sidecar. The
commands resolve after the job finishes, while stdout/stderr stream to the UI as
`sidecar-progress` events for live prepare, training, export, validation, and
benchmark updates. The current train/evaluate/export CLI uses TensorFlow/Keras
as the canonical RTNeural JSON path, with PyTorch retained as an optional
compatibility backend for curated LSTM presets.

Current local v1 coverage includes SQLite-backed project/job state, project
rename/delete actions, native file pickers, optional resampling and stereo
policy, manual latency override, cancel/resume/recovery, validation curves,
streaming validation checkpoints, early stopping controls, learning-rate
plateau decay, recurrent state-drift diagnostics, runtime inspection,
target/prediction/residual playback, transient-aware latency candidate review
with window agreement, golden RTNeural JSON fixtures, native parity checks,
block-size/channel native benchmark reports, export-time ASR aliasing reports,
smoothed-tanh WaveNet research presets, and debug/release smoke scripts.

Still deferred: signed/notarized release distribution, richer waveform/spectrum
inspection, a full tiny train/export smoke inside an installed bundle, UI smoke
tests with a real Tauri window, and any `.aidax` or generated player envelope
until format/license review is complete.

## Requirements

- Node.js LTS with `pnpm` 11
- Rust stable and Cargo
- CMake 3.20+
- Python 3.11 or 3.12
- `uv`

Recommended package-manager setup:

```bash
corepack enable
corepack prepare pnpm@11.5.2 --activate
```

## Install

From the repo root:

```bash
pnpm install

cd trainer
UV_CACHE_DIR=../.uv-cache uv sync
```

Install Python extras as needed:

```bash
# Canonical Keras/TensorFlow training and RTNeural export path
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra tensorflow

# Optional PyTorch compatibility backend
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra training
```

On Apple Silicon, the `tensorflow` extra also installs `tensorflow-metal` for
Metal GPU training. TensorFlow is pinned below `2.19` because newer TensorFlow
builds can fail to load Apple's current Metal plugin. Verify the active runtime
with:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow rttrainer inspect-device --json
```

## Run The Desktop App

For the web UI during development:

```bash
pnpm --filter rtneural-trainer-app dev
```

For the Tauri desktop shell:

```bash
pnpm --filter rtneural-trainer-app tauri dev
```

`tauri dev` runs `pnpm dev:tauri-assets` first. That creates local development
sidecar shims in `app/src-tauri/binaries/` using Tauri's required target-triple
filenames. The shims keep development fast by delegating to:

- `uv run --extra tensorflow python -m rttrainer`
- `native/rtneural-validator/build/rtneural-validator`

Build the frontend:

```bash
pnpm --filter rtneural-trainer-app build
```

Check the Rust side:

```bash
cd app/src-tauri
cargo test
```

## Desktop Project Management

The desktop sidebar is the project switcher. Select a project row to load its
current capture, runs, exports, notes, and progress history from SQLite.

Use the project header actions to manage the selected project:

- `Rename project` opens an inline editor. Names are trimmed, must be non-empty,
  and are limited to 120 characters. Saving refreshes the header and sidebar.
- `Delete project` uses a two-step confirmation. It removes the SQLite project
  record, cascades related audio reports, runs, exports, jobs, and job events,
  and deletes the app-managed project folder from the local app data directory.

Rename and delete are disabled while training/export jobs or other project
mutations are active. Delete only removes files inside the app-managed project
folder; it does not delete arbitrary external source WAVs elsewhere on disk.

## Package Tauri Sidecars

Tauri's `bundle.externalBin` entries use logical names without a target triple:

```json
{
  "externalBin": [
    "binaries/rttrainer",
    "binaries/rtneural-validator"
  ]
}
```

The source files on disk must include the target triple suffix:

```text
app/src-tauri/binaries/
  rttrainer-<target-triple>
  rtneural-validator-<target-triple>
```

For local development, generate ignored shim binaries:

```bash
pnpm --filter rtneural-trainer-app package:sidecars:dev
```

For production packaging, build or copy real sidecars:

```bash
pnpm --filter rtneural-trainer-app package:sidecars
```

The production script packages `rttrainer` with PyInstaller through `uv` and
builds the native validator with CMake release settings. The validator uses the
Eigen RTNeural backend by default; choose another backend with:

```bash
pnpm --filter rtneural-trainer-app package:sidecars -- --validator-backend stl
```

For local backend comparison builds:

```bash
pnpm --filter rtneural-trainer-app build:validators
```

`xsimd` builds require the RTNeural xsimd headers/submodule to be present in the
local RTNeural checkout. You can also provide prebuilt executables:

```bash
RTTRAINER_SIDECAR_SOURCE=/path/to/rttrainer \
RTNEURAL_VALIDATOR_SOURCE=/path/to/rtneural-validator \
pnpm --filter rtneural-trainer-app package:sidecars
```

`tauri build` runs `pnpm build:tauri-assets`, which stages these sidecars before
building the frontend and app bundle.

To exercise the real release path locally, build production sidecars, validate
the PyInstaller and native binaries, run a CI-mode Tauri release bundle build,
and collect a release artifact manifest:

```bash
# macOS
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles app,dmg

# Linux
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles deb

# Windows
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles nsis
```

The release smoke uses `tauri build --ci --no-sign`, so it proves bundle shape
and sidecar execution but does not replace macOS signing/notarization or Windows
code-signing checks.

## Build The Native Validator

If the local RTNeural clones exist under `/Users/shortwavlabs/Workspace/rt-neural`,
the validator CMake project uses `/Users/shortwavlabs/Workspace/rt-neural/RTNeural`
by default. Override with `RTNEURAL_LOCAL_PATH=/path/to/RTNeural` if needed. If
no local checkout is found, CMake fetches the pinned RTNeural commit.

```bash
cmake -S native/rtneural-validator -B native/rtneural-validator/build
cmake --build native/rtneural-validator/build
```

The default backend is Eigen. To build an explicit backend:

```bash
cmake -S native/rtneural-validator -B native/rtneural-validator/build-stl \
  -DRTNEURAL_VALIDATOR_BACKEND=stl
cmake --build native/rtneural-validator/build-stl
```

Example validator commands:

```bash
native/rtneural-validator/build/rtneural-validator validate \
  --model projects/demo/exports/export_001/model.rtneural.json \
  --input projects/demo/runs/run_001/test-input.wav \
  --reference projects/demo/runs/run_001/test-target.wav \
  --report projects/demo/exports/export_001/native-validation-report.json

native/rtneural-validator/build/rtneural-validator benchmark \
  --model projects/demo/exports/export_001/model.rtneural.json \
  --sample-rate 48000 \
  --seconds 2 \
  --block-sizes 16,32,64,128,256,512 \
  --channels 1,2 \
  --passes 3 \
  --warmup-blocks 8 \
  --min-realtime-factor 1.0 \
  --report projects/demo/exports/export_001/native-benchmark-report.json
```

The native validator loads RTNeural dynamic JSON, runs mono WAV input through the
model, compares against mono reference audio, and writes structured validation
and benchmark reports. The benchmark report includes a worst-case real-time
factor across the requested block-size/channel matrix, per-case timings, model
size, architecture metadata, latency, and inferred Conv1D receptive field.
Desktop exports also write `native-benchmark-matrix.json` and embed it in
`package.json` as `benchmark_matrix`; when backend-specific validator builds are
available, this compares Eigen, STL, xsimd, and optional AVX variants and marks
the fastest passing backend. Run
`pnpm --filter rtneural-trainer-app build:validators` before export when you
want the local matrix to include every available native backend.

Python export also writes `aliasing-report.json`, a warning-only ASR
diagnostic that renders deterministic sine probes through the exported RTNeural
JSON. The desktop package metadata surfaces it as the export `aliasing` report
beside validation and benchmark results.

## Use The Trainer CLI

Run CLI commands from `trainer/` with `uv run`:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer inspect-device --json
```

### 1. Prepare Paired WAV Files

Before recording a profile set, read the
[audio capture guidelines](docs/Audio-Capture-Guidelines.md) for input/target
pairing, output level consistency, headroom, latency, and source material
recommendations.

Create a manifest, for example `projects/demo/prepare.json`:

```json
{
  "input_path": "projects/demo/audio/input.wav",
  "target_path": "projects/demo/audio/target.wav",
  "output_dir": "projects/demo/audio/prepared"
}
```

Run:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer prepare \
  --manifest ../projects/demo/prepare.json
```

This writes aligned `input.wav`, `target.wav`, and
`preparation-report.json` into the manifest's `output_dir`.

### 2. Train

Install the TensorFlow extra first:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra tensorflow
```

Create `projects/demo/train.json`:

```json
{
  "run_id": "run_001",
  "run_dir": "projects/demo/runs/run_001",
  "prepared_dir": "projects/demo/audio/prepared",
  "preset": "lstm_standard",
  "backend": "keras",
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 0.001,
  "sequence_length": 8192,
  "max_windows": 2048,
  "resample_training_windows": true,
  "resample_interval_epochs": 1,
  "seed": 1337
}
```

Run:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer train \
  --manifest ../projects/demo/train.json
```

The run folder receives checkpoints, metrics, preview WAVs, and test fixtures.
Keras runs save `checkpoints/best-model.keras` plus checkpoint metadata. To use
the optional PyTorch path, set `"backend": "pytorch"` and install the
`training` extra.

Training samples windows with an energy-stratified pass and reserves long
excerpts for streaming validation and preview audio. Recurrent presets also get
one longer active context excerpt per epoch so hidden state sees more continuous
audio than a single short window. Checkpoints use a validation score that is
anchored by streaming ESR, with short-window ESR and an underpowered-output
penalty to avoid selecting near-silent early checkpoints. If that validation
score plateaus, the trainer lowers the learning rate before early stopping has a
chance to stop the run. When `resample_training_windows` is enabled, validation
and preview excerpts stay fixed, but the training windows rotate at
`resample_interval_epochs` so long captures get broader coverage across long
runs. By default, plateau patience is half the early-stop patience, the decay
factor is `0.5`, and the floor is `1e-6`. Progress events and `history.json`
record streaming ESR, short-window diagnostics, validation score, output level
ratio, learning rate, context-training loss, window rotation state, and any
reductions.

Final reports compare normal continuous inference against a reset-per-chunk
diagnostic render for recurrent presets. If the reset render has much better ESR
and correlation, the report flags recurrent state drift and writes extra
`chunk-reset-prediction.wav` / `chunk-reset-residual.wav` previews. Treat the
normal continuous prediction as the export truth; the chunk-reset audio is a
debugging aid that usually means the next run should try a finite-memory Conv1D
baseline or longer recurrent context.

Current Keras-first presets are:

- `dense_only`: memoryless Dense baseline for very fast checks.
- `gru_light`: compact GRU recurrent model.
- `lstm_light`: low-CPU LSTM recurrent model.
- `lstm_standard`: default LSTM recurrent model.
- `conv1d_light`: causal Conv1D model.
- `conv1d_bn_prelu`: causal Conv1D with safe BatchNorm/PReLU; this is the
  compact finite-memory baseline for capture sanity checks.
- `conv1d_stack_prelu`: stacked causal Conv1D/PReLU with dilations and a
  pre-emphasis MSE default loss. This is now the fast CPU fallback and sanity
  check for amp/pedal captures.
- `wavenet_tcn_fast`: smaller RTNeural-safe WaveNet-style TCN for a faster
  quality probe.
- `wavenet_tcn_balanced`: the current default amp quality path, matching the
  proven legacy `wavenet_tcn` architecture.
- `wavenet_tcn_balanced_tanh15`: research balanced WaveNet with smoothed
  `tanh(x / 1.5)` training, exported as standard RTNeural `tanh`.
- `wavenet_tcn_balanced_tanh18`: research balanced WaveNet with smoothed
  `tanh(x / 1.8)` training for ASR comparisons.
- `wavenet_tcn_quality`: wider/deeper WaveNet-style TCN for slower refinement
  runs, especially crunch/rhythm/high-gain tones. Benchmark before treating
  quality exports as plugin-ready.
- `wavenet_tcn_quality_tanh15`: research quality WaveNet with smoothed
  `tanh(x / 1.5)` training. This keeps the proven quality receptive field while
  probing whether a gentler nonlinearity reduces high-band residual and aliasing.
- `wavenet_tcn_high_gain`: experimental rhythm/high-gain WaveNet with a
  4095-sample receptive field and `3.5e-4` default learning rate. The first
  DI4/RHYTHM4 check underperformed `wavenet_tcn_quality`, so it is hidden from
  normal UI recommendations and kept only for architecture research.
- `wavenet_tcn_quality_tanh18`: research quality WaveNet with smoothed
  `tanh(x / 1.8)` training.
- `wavenet_tcn_separable_fast`: experimental grouped/dilated Conv1D plus 1x1
  pointwise WaveNet variant. It has Python/native parity coverage, but current
  dynamic RTNeural benchmarks do not beat `wavenet_tcn_balanced`; use it only
  for runtime research.
- `wavenet_tcn`: legacy balanced WaveNet preset kept for existing runs and
  checkpoint compatibility.
- `conv_gru_hybrid`: causal Conv1D front-end feeding a compact GRU.

Newly initialized presets use a bounded `tanh` output layer so long streaming
previews cannot run away into clipped full-scale prediction WAVs.

The PyTorch compatibility backend is currently limited to the LSTM presets.

Research notes from the PANAMA paper and related WaveNet amp-modeling work are
captured in
[docs/Research-PANAMA-WaveNet-Active-Learning.md](docs/Research-PANAMA-WaveNet-Active-Learning.md)
and
[docs/Research-WaveNet-Amp-Simulation-Papers-2026-06-24.md](docs/Research-WaveNet-Amp-Simulation-Papers-2026-06-24.md).

### 3. Evaluate

Create `projects/demo/evaluate.json`:

```json
{
  "run_dir": "projects/demo/runs/run_001",
  "output_dir": "projects/demo/runs/run_001/evaluation"
}
```

Run:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer evaluate \
  --manifest ../projects/demo/evaluate.json
```

### 4. Export RTNeural JSON

Create `projects/demo/export.json`:

```json
{
  "name": "Demo RTNeural Model",
  "run_dir": "projects/demo/runs/run_001",
  "export_dir": "projects/demo/exports/export_001",
  "sample_rate": 48000,
  "latency_samples": 0,
  "parity_tolerance": 0.00001
}
```

Run:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m rttrainer export \
  --manifest ../projects/demo/export.json
```

The export folder receives:

- `model.rtneural.json`
- `validation-report.json`
- `benchmark-report.json`
- `aliasing-report.json`
- `native-benchmark-matrix.json` when exported from the desktop app
- `parity-snapshot.json`
- `parity-snapshot-input.wav`
- `parity-snapshot-expected.wav`
- `package.json`

## RTNeural Support Scripts

Print the current benchmark-informed layer/activation plan:

```bash
python3 scripts/rtneural_support_matrix.py --format markdown
python3 scripts/rtneural_support_matrix.py --format json
```

List Keras fixture coverage without importing TensorFlow:

```bash
python3 scripts/generate_rtneural_keras_fixtures.py --list
python3 scripts/generate_rtneural_keras_fixtures.py --list --include-later
```

Generate Keras RTNeural JSON fixtures after installing the TensorFlow extra:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra tensorflow
cd ..
python3 scripts/generate_rtneural_keras_fixtures.py \
  --out fixtures/rtneural-json \
  --size 8
```

Golden RTNeural JSON fixtures for exported presets live in
`fixtures/rtneural-json/golden/`. Regenerate them when preset architecture or
export serialization changes:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/generate_golden_rtneural_fixtures.py
```

Compare completed training runs and optionally re-export them with ASR and
native RTNeural benchmark checks:

```bash
UV_CACHE_DIR=.uv-cache uv run --project trainer --extra tensorflow python \
  scripts/compare_training_runs.py \
  --out /tmp/rttrainer-run-comparison \
  --export \
  --native \
  /path/to/run_a \
  /path/to/run_b
```

The script writes `comparison.md` and `comparison.json`, keeping generated
exports under the comparison output folder instead of modifying the original
project.

## Downstream Plugin Reference

[RTNeural-example](https://github.com/jatinchowdhury18/RTNeural-example) is a
useful reference for what happens after this tool exports a model. It is a JUCE
audio plugin that embeds an RTNeural JSON as binary data, parses it at run time,
keeps one RTNeural model per audio channel, resets the models in the audio
prepare step, and calls `forward()` per sample in the processing block.

Useful patterns from that repo:

- Use CMake to add RTNeural and choose the backend with `RTNEURAL_STL`,
  `RTNEURAL_XSIMD`, or `RTNEURAL_EIGEN`.
- Use JUCE `BinaryData` for a bundled JSON model, or adapt the same lifecycle for
  user-selected model files.
- Parse dynamic JSON with `RTNeural::json_parser::parseJson<float>()`.
- Keep stereo processing as two independent model instances unless the exported
  architecture is explicitly multi-channel.
- Treat compile-time RTNeural models as a later optimization path once dynamic
  JSON validation is solid.

The example also reinforces the Keras-first exporter direction: its Python model
script builds a TensorFlow/Keras Sequential network and exports with RTNeural's
Python `model_utils`.

## Test And Verify

Python:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run python -m unittest discover -s tests -v
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m unittest discover -s tests -v
python3 -m compileall rttrainer tests
```

Golden RTNeural preset fixtures and parity:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/generate_golden_rtneural_fixtures.py --check
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  -m unittest tests.test_rtneural_golden_fixtures -v
```

Frontend:

```bash
pnpm --filter rtneural-trainer-app build
```

Rust:

```bash
cd app/src-tauri
cargo test
```

Native validator:

```bash
cmake -S native/rtneural-validator -B native/rtneural-validator/build
cmake --build native/rtneural-validator/build
python3 scripts/smoke_rtneural_validator.py
```

Keras training/export through native RTNeural:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/smoke_keras_training_export.py
```

Supported Keras layer export matrix through native RTNeural:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python \
  ../scripts/smoke_rtneural_keras_layers.py
```

That smoke covers Dense-only, GRU, causal Conv1D, supported activations
(`tanh`, `relu`, `sigmoid`, `softmax`, `elu`), and the safe 1D
BatchNorm/PReLU path.

Tauri sidecar workflow smoke:

```bash
pnpm --filter rtneural-trainer-app smoke:tauri-workflow
```

Tauri UI smoke:

```bash
pnpm --filter rtneural-trainer-app smoke:tauri-ui
```

This Vitest/jsdom smoke runs the full React app with mocked Tauri commands. It
covers first-run onboarding, generated sample projects, project switching,
Capture, Align, Train, Evaluate, Export, Runtime, rename/delete, and a
regression for per-project WAV path state. Tauri's desktop WebDriver path is
limited to Linux/Windows; macOS uses this mocked UI smoke instead.

Packaged-app smoke:

```bash
pnpm --filter rtneural-trainer-app smoke:packaged-app
```

The packaged-app smoke defaults to a debug, no-bundle Tauri build and reuses
prebuilt sidecar binaries so it can run quickly in CI. Pass `-- --bundle` to the
script if you need to exercise platform bundle creation too.

Release package smoke:

```bash
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles app,dmg
```

## CI Gates

`.github/workflows/ci.yml` runs the fast desktop gate on Ubuntu: dependency
setup, TensorFlow trainer sync, native validator build/smoke, Python tests,
golden fixture freshness, frontend build, development sidecar staging, Rust
tests, Tauri UI smoke, Tauri workflow smoke, and debug packaged-app smoke.

GitHub Actions has a separate `Release Packaging` workflow for the slow path. It
runs the release package smoke on Linux, macOS, and Windows, then uploads the
Tauri bundle outputs, staged sidecars, and
`app/src-tauri/target/release/release-artifacts-manifest.json`.

## Useful Docs

- [Research note](docs/Research-RTNeural-Training-Desktop-App.md)
- [PANAMA / WaveNet findings](docs/Research-PANAMA-WaveNet-Active-Learning.md)
- [NAM / WaveNet performance findings](docs/Research-NAM-Performance-And-WaveNet.md)
- [Clean/crunch/rhythm capture baseline](docs/Research-Clean-Crunch-Rhythm-Capture-Baseline.md)
- [WaveNet amp simulation paper review](docs/Research-WaveNet-Amp-Simulation-Papers-2026-06-24.md)
- [Implementation guide](docs/Implementation-Guide-RTNeural-Training-Desktop-App.md)
- [Audio capture guidelines](docs/Audio-Capture-Guidelines.md)
- [RTNeural upstream](https://github.com/jatinchowdhury18/RTNeural)
- [RTNeural Python examples](https://github.com/jatinchowdhury18/RTNeural/tree/main/python)
- [RTNeural-compare benchmarks](https://github.com/jatinchowdhury18/RTNeural-compare)
- [RTNeural-example plugin](https://github.com/jatinchowdhury18/RTNeural-example)
