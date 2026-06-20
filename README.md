# RTNeural Trainer

RTNeural Trainer is an early desktop workbench for preparing paired audio,
training a small neural audio model, exporting RTNeural-compatible JSON, and
validating/benchmarking the result before use in a real-time target.

The repo currently contains:

- `app/`: Tauri v2 + React desktop shell.
- `trainer/`: `uv`-managed Python sidecar and CLI.
- `native/rtneural-validator/`: CMake-built validator/benchmark sidecar.
- `scripts/`: RTNeural support and Keras fixture helper scripts.
- `docs/`: research notes and implementation plan.

The implementation is still a prototype. The desktop app now calls the real
Python `prepare`, `train`, and `export` commands, and the export path invokes the
native RTNeural validator/benchmark sidecar. The commands resolve after the job
finishes, while stdout/stderr stream to the UI as `sidecar-progress` events for
live prepare, training, export, validation, and benchmark updates. The current
train/evaluate/export CLI uses TensorFlow/Keras as the canonical RTNeural JSON
path, with PyTorch retained as an optional compatibility backend for curated
presets.

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
cargo check
```

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
builds the native validator with CMake release settings. You can also provide
prebuilt executables:

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
  --seconds 30 \
  --report projects/demo/exports/export_001/native-benchmark-report.json
```

The native validator loads RTNeural dynamic JSON, runs mono WAV input through the
model, compares against mono reference audio, and writes structured validation
and benchmark reports.

## Use The Trainer CLI

Run CLI commands from `trainer/` with `uv run`:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer inspect-device --json
```

### 1. Prepare Paired WAV Files

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
  "preset": "lstm_light",
  "backend": "keras",
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 0.001,
  "sequence_length": 1024,
  "max_windows": 512,
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

Current Keras-first presets are:

- `dense_only`: memoryless Dense baseline for very fast checks.
- `gru_light`: compact GRU recurrent model.
- `lstm_light`: low-CPU LSTM recurrent model.
- `lstm_standard`: default LSTM recurrent model.
- `conv1d_light`: causal Conv1D model.
- `conv1d_bn_prelu`: causal Conv1D with safe BatchNorm/PReLU.
- `conv_gru_hybrid`: causal Conv1D front-end feeding a compact GRU.

The PyTorch compatibility backend is currently limited to the LSTM presets.

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
cargo check
```

Native validator:

```bash
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

GitHub Actions has a separate `Release Packaging` workflow for the slow path. It
runs the release package smoke on Linux, macOS, and Windows, then uploads the
Tauri bundle outputs, staged sidecars, and
`app/src-tauri/target/release/release-artifacts-manifest.json`.

## Useful Docs

- [Research note](docs/Research-RTNeural-Training-Desktop-App.md)
- [Implementation guide](docs/Implementation-Guide-RTNeural-Training-Desktop-App.md)
- [RTNeural upstream](https://github.com/jatinchowdhury18/RTNeural)
- [RTNeural Python examples](https://github.com/jatinchowdhury18/RTNeural/tree/main/python)
- [RTNeural-compare benchmarks](https://github.com/jatinchowdhury18/RTNeural-compare)
- [RTNeural-example plugin](https://github.com/jatinchowdhury18/RTNeural-example)
