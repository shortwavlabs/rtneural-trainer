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

The implementation is still a prototype. The current train/evaluate/export CLI
uses the PyTorch training extra, while the RTNeural JSON strategy and fixture
scripts are moving toward TensorFlow/Keras as the canonical exporter path.

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
# Current PyTorch-based training CLI
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra training

# Keras/TensorFlow RTNeural fixture generation
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra tensorflow
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

Build the frontend:

```bash
pnpm --filter rtneural-trainer-app build
```

Check the Rust side:

```bash
cd app/src-tauri
cargo check
```

## Build The Native Validator

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

The native validator is currently a sidecar stub that writes structured reports.
Wiring it to RTNeural's real dynamic JSON parser is still on the roadmap.

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

Install the training extra first:

```bash
cd trainer
UV_CACHE_DIR=../.uv-cache uv sync --extra training
```

Create `projects/demo/train.json`:

```json
{
  "run_id": "run_001",
  "run_dir": "projects/demo/runs/run_001",
  "prepared_dir": "projects/demo/audio/prepared",
  "preset": "lstm_light",
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
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer train \
  --manifest ../projects/demo/train.json
```

The run folder receives checkpoints, metrics, preview WAVs, and test fixtures.

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
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer evaluate \
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
UV_CACHE_DIR=../.uv-cache uv run python -m rttrainer export \
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
python3 -m compileall rttrainer tests
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
```

## Useful Docs

- [Research note](docs/Research-RTNeural-Training-Desktop-App.md)
- [Implementation guide](docs/Implementation-Guide-RTNeural-Training-Desktop-App.md)
- [RTNeural upstream](https://github.com/jatinchowdhury18/RTNeural)
- [RTNeural Python examples](https://github.com/jatinchowdhury18/RTNeural/tree/main/python)
- [RTNeural-compare benchmarks](https://github.com/jatinchowdhury18/RTNeural-compare)
- [RTNeural-example plugin](https://github.com/jatinchowdhury18/RTNeural-example)
