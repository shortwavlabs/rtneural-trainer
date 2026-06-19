# Research: RTNeural Training Desktop App

Date: 2026-06-19

Scope: research how Shortwav Labs could build a desktop application that lets
users train neural audio models that are easy to deploy with
[RTNeural](https://github.com/jatinchowdhury18/RTNeural).

RTNeural repo inspected at commit `1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d`
from 2025-11-08.

## Executive Summary

RTNeural should be treated as the real-time inference target, not the training
engine. The desktop app should provide a guided local training workflow around
PyTorch, then export RTNeural-compatible JSON and run a native RTNeural
validation harness before the user can publish or use the model.

Recommended v1: a desktop app for black-box audio effect capture, starting with
guitar/bass amps, pedals, and simple line-level effects. Users import or record
paired audio: a dry/input signal and the processed/target signal. The app aligns
the files, trains one of several safe model presets, previews prediction quality,
exports RTNeural JSON, and optionally exports an `.aidax`-style envelope with
metadata for AIDA-X-like workflows.

The best technical shape is:

1. Tauri v2 desktop shell with a React/TypeScript UI.
2. Python/PyTorch training sidecar managed by the app.
3. Native C++ RTNeural validator/benchmark sidecar built with CMake.
4. SQLite project/job store plus filesystem-backed project folders.
5. A small set of curated model architectures instead of arbitrary neural-net
   graph import.

The hardest product problems are not the UI shell. They are audio data quality,
latency alignment, export correctness, PyTorch environment packaging, and making
users understand CPU/runtime tradeoffs before they train a model that cannot run
comfortably in real time.

## Key RTNeural Findings

RTNeural is a lightweight C++ inferencing library designed for real-time systems,
especially real-time audio. Its paper describes the design emphasis as speed,
flexibility, size, and convenience under hard real-time constraints.

RTNeural's README says it can load weights from an already-trained network and
run inference. Training is expected to happen in Python libraries such as
TensorFlow or PyTorch, then weights are exported to JSON for RTNeural.

Supported layer families in the current repo:

- Dense
- GRU
- LSTM
- Conv1D
- Conv2D
- BatchNorm1D
- BatchNorm2D
- PReLU and common activations: tanh, ReLU, sigmoid, softmax, ELU

Important caveats:

- MaxPooling is still marked unchecked in the local README, and there is an open
  issue asking about MaxPool support.
- The JSON parser does not currently add post-activation layers for GRU/LSTM
  layers, which is tracked in open issue #124. Exporters should set recurrent
  layer activations intentionally and validate parity against PyTorch.
- There is an open large-GRU segmentation fault issue around a 512-hidden-size,
  multi-layer GRU static model. v1 presets should stay conservative and benchmark
  every exported model.
- RTNeural has dynamic JSON loading and a compile-time API. Dynamic loading is
  ideal for a training app's first usable export. Compile-time models are a later
  optimization path for generated plugins or embedded targets.
- RTNeural supports Eigen, xsimd, and STL backends. The README recommends
  measuring backend performance for the target model/platform because Eigen is
  often better for larger networks while xsimd may win for smaller networks.

## Product Opportunity

Existing RTNeural-facing workflows are powerful but developer-shaped:

- RTNeural provides examples and exporters, but not a productized training UI.
- AIDA-X is a real RTNeural-based amp model player with model loading, IR support,
  meters, and standalone/plugin builds. Its public workflow sends users to a Colab
  notebook for training and export.
- NAM has a more polished training ecosystem with local GUI, Colab, calibration
  guidance, and `.nam` export, but NAM is a separate model format and runtime
  ecosystem.

This leaves room for a focused desktop app that gives RTNeural users the "NAM
trainer experience" while producing RTNeural-compatible output.

## Recommended MVP

Build a local desktop "RTNeural Trainer" with one primary workflow:

1. Create project.
2. Choose target: amp, pedal, line effect, or generic audio mapping.
3. Download/generate a capture signal, or import a known dry/input file.
4. Record or import the target/wet output.
5. Let the app validate sample rate, bit depth, length, clipping, DC offset, and
   silence.
6. Auto-align input/output latency and let the user inspect/adjust it.
7. Pick a model preset: light, standard, heavy.
8. Train locally with PyTorch on `mps`, `cuda`, or CPU.
9. Monitor loss, ESR, validation audio, waveform overlay, and residual error.
10. Export RTNeural JSON.
11. Run RTNeural C++ validation and CPU benchmark.
12. Save model package with metadata, plots, test audio, and benchmark results.

The MVP should not try to support arbitrary PyTorch, ONNX, or TensorFlow models.
That path creates an unbounded compatibility problem. Users should choose from
app-curated architectures that we know can be exported and loaded by RTNeural.

## Training Pipeline

### Data Capture And Import

Use paired supervised training data:

- `input.wav`: the dry/reference signal.
- `target.wav`: the output of the hardware, plugin, or signal chain.

The app should prefer 48 kHz WAV for v1, because AIDA-X and NAM training guides
both normalize their beginner flows around a 48 kHz capture signal. Internally we
can resample with libsamplerate/r8brain/soxr, but the UI should make mismatches
visible.

Required validation:

- Same sample rate after conversion.
- Same channel count or deterministic mono/stereo conversion.
- Same effective duration after trimming.
- No hard clipping unless the user accepts it.
- Enough active signal for training.
- Input and output latency estimated and stored in samples.
- Train/validation/test split generated reproducibly.

Borrow the NAM/AIDA-X product pattern: use impulses near the beginning of the
capture signal to estimate round-trip latency, then show the user an overlay
before training continues.

### Model Presets

Start with recurrent models because RTNeural has mature GRU/LSTM support and
AIDA-X uses LSTM-size presets successfully.

Suggested v1 presets:

| Preset | Architecture | Intended use | Notes |
| --- | --- | --- | --- |
| Light | 1x LSTM or GRU, hidden 8-12, dense out | pedals, low CPU targets | Similar shape to AIDA-X light presets |
| Standard | 1x LSTM, hidden 16-20, dense out | default amp/pedal capture | Good first target for RTNeural JSON export |
| Heavy | 1-2 recurrent layers, hidden 24-32 | desktop/plugin only | Gate behind benchmark warning |
| Dense | small dense stack | memoryless tone/EQ/saturation | Good for very low latency if target is simple |

Defer TCN/WaveNet-style models until after v1. RTNeural has Conv1D and examples
for micro-TCN-like blocks, but open issue #120 asks about TCN support and real
TCN/WaveNet export quickly runs into custom block, dilation, residual, and
streaming-state details.

### Training Runtime

Use PyTorch for the training sidecar. Current PyTorch docs support runtime device
selection across CUDA, MPS, and CPU. The app should select in this order:

1. CUDA when available.
2. Apple MPS when available.
3. CPU fallback.

For Apple Silicon, PyTorch's MPS backend is checked via
`torch.backends.mps.is_available()` / `torch.backends.mps.is_built()` and models
move to `torch.device("mps")`.

The training process should run outside the UI process:

- One job per project version.
- Checkpoint every N epochs.
- Stream progress events to the app.
- Allow cancel/resume from checkpoint.
- Persist stdout/stderr logs.
- Save exact package versions and hardware device in metadata.

Use `model.eval()` and `torch.no_grad()` for validation/inference previews. This
matters for memory use and for layers such as BatchNorm.

### Loss And Metrics

Minimum metrics:

- ESR or normalized error as the main simple metric.
- MAE/RMSE for sanity.
- Peak and RMS residual.
- Model CPU cost in real-time factor at 48 kHz.

Better audio losses:

- ESR pre-filtered by A-weighting or a task-specific filter.
- Multi-resolution STFT loss for perceptual frequency balance.
- Loudness-aware reporting so low-level silence does not dominate the story.

Expose the final result as "Quality", "CPU", and "Latency" rather than raw loss
alone. Musicians will trust the app more if they can immediately A/B dry, target,
prediction, and residual audio.

## RTNeural Export Strategy

The app should own a dedicated exporter rather than asking users to run ad hoc
notebook code.

Export artifact:

```json
{
  "in_shape": [null, null, 1],
  "layers": [
    {
      "type": "lstm",
      "activation": "",
      "shape": [null, null, 16],
      "weights": [...]
    },
    {
      "type": "dense",
      "activation": "",
      "shape": [null, null, 1],
      "weights": [...]
    }
  ],
  "metadata": {
    "sample_rate": 48000,
    "latency_samples": 123,
    "architecture": "lstm-16-1",
    "loss": {...},
    "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d"
  }
}
```

Exporter details:

- For TensorFlow/Keras sequential models, RTNeural's `python/model_utils.py`
  already demonstrates the expected `in_shape` and `layers` JSON structure.
- For PyTorch, export from `state_dict()` and transform weight layouts to match
  RTNeural. RTNeural's `torch_helpers.h` documents key differences such as
  transposed dense/recurrent weights, Conv1D kernel reversal, and GRU reset/update
  gate swapping.
- AIDA-X's `modelToRTNeural.py` is a useful reference for converting PyTorch
  SimpleRNN/LSTM/GRU style models into RTNeural JSON, but it should not be copied
  into a proprietary product without a license review. Reimplement the logic and
  write parity tests.

Every export should run two validators:

1. Python parity validator: compare PyTorch prediction to exported JSON loaded by
   a small Python-side simulator or C++ harness.
2. RTNeural native validator: parse the JSON with RTNeural, run known input
   batches, compare output tolerance, and benchmark Eigen/xsimd/STL where
   practical.

## Desktop Architecture

Recommended stack:

```text
rtneural-trainer/
  app/                         # Tauri + React UI
    src/
    src-tauri/
  trainer/                     # Python package
    rttrainer/
      data/
      models/
      training/
      export_rtneural/
      metrics/
  native/
    rtneural-validator/         # C++ CMake target linking RTNeural
  projects/
    <project-id>/
      input.wav
      target.wav
      runs/
      exports/
      reports/
```

Why Tauri v2:

- It gives a small native desktop shell with a web UI.
- Tauri sidecars are designed for bundled external binaries, including Python CLI
  apps packaged with PyInstaller or a similar tool.
- The shell plugin/permission model is a better fit than letting a browser UI
  execute arbitrary commands.
- Rust is a good place for job orchestration, file validation, and controlled IPC.

Alternative stacks:

- Electron: easiest Node ecosystem and Python process spawning, but much heavier.
- Electrobun: attractive for macOS-first apps and already appears in this repo's
  other research, but the Python/PyTorch packaging story is less established than
  Tauri sidecars.
- Qt/PySide: simplest if everything is Python, but less appealing for a polished
  modern product UI and native code signing/updating.

## Core App Services

### Project Store

Use SQLite for metadata:

- projects
- audio_files
- training_runs
- checkpoints
- exports
- metrics
- hardware_profiles
- app_settings

Store large artifacts on disk in project folders:

- source audio
- processed/aligned audio
- model checkpoints
- RTNeural JSON exports
- rendered plots
- preview WAV files
- validation reports

### Job Runner

Training must be cancellable and resilient:

- Rust/Tauri starts the Python sidecar with a job manifest JSON.
- Python writes machine-readable progress events as JSON lines.
- App stores progress in SQLite.
- Cancellation sends a signal, Python writes a final interrupted state, and the
  latest checkpoint remains resumable.

### Preview And Validation

The app should have two preview modes:

- Offline preview: render model output for selected clips and show waveform/error
  overlays.
- Live preview: later, pipe audio through the RTNeural validator or a small JUCE
  standalone engine. This is a phase 2 feature unless live monitoring is essential
  for launch.

For v1, offline preview is enough and much safer.

## UX Shape

The app should be a guided workbench, not a generic ML IDE.

Main screens:

- Projects: recent model projects, status, hardware, last quality score.
- Capture: download test signal, record/import input and target, validate audio.
- Align: show latency estimate, impulses, manual nudge, trim controls.
- Train: preset picker, hardware indicator, quality/runtime estimate, progress.
- Evaluate: A/B/C listening for target/prediction/residual, waveform, spectrum,
  error metrics.
- Export: RTNeural JSON, optional `.aidax`, validation status, benchmark table.
- Library: searchable local model collection with tags and notes.

Useful defaults:

- Hide hyperparameters behind an "Advanced" drawer.
- Explain CPU tradeoffs before training heavy presets.
- Surface warnings early: clipping, mismatched length, low signal, sample-rate
  conversion, latency uncertainty.
- Make "Train again with same settings" and "Train three seeds and keep best" a
  first-class action.

## Development Plan

### Phase 0: Export Spike

Goal: prove PyTorch -> RTNeural JSON -> RTNeural C++ parity.

Tasks:

- Train a tiny LSTM and GRU on synthetic data.
- Export JSON.
- Load with `RTNeural::json_parser::parseJson<float>()`.
- Compare PyTorch and RTNeural output sample-by-sample.
- Benchmark dynamic JSON model with Eigen/xsimd/STL.

Exit criteria:

- Tolerance report is stable.
- JSON schema is documented.
- CI can run the parity suite.

### Phase 1: CLI Trainer

Goal: build the product core without UI.

Tasks:

- `rttrainer prepare input.wav target.wav`
- `rttrainer train project.json`
- `rttrainer export run-id`
- `rtneural-validator model.json test-input.wav`

Exit criteria:

- A user can train from paired WAVs and get a validated RTNeural JSON.
- Metrics and preview audio are produced.

### Phase 2: Desktop MVP

Goal: make the workflow approachable.

Tasks:

- Tauri project shell.
- Project database and filesystem layout.
- Audio import/validation UI.
- Training job UI and progress stream.
- Evaluation and export screens.
- Packaged Python sidecar.

Exit criteria:

- A non-developer can complete a model from paired WAVs without a terminal.

### Phase 3: Runtime Integrations

Goal: make models immediately useful.

Options:

- Export `.aidax`-compatible metadata envelope.
- Generate a tiny JUCE standalone/player project.
- Generate C++ compile-time model type for known architecture.
- Add cloud training as a paid acceleration path.

## Technical Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PyTorch packaging is large and platform-specific | App install/update complexity | Package trainer as a sidecar, document hardware support, allow advanced external Python env |
| RTNeural supports a limited layer set | Arbitrary model import fails | Curated presets only; export compatibility tests |
| Exported weights are subtly wrong | Bad models or crashes | Golden parity tests against PyTorch for every layer/preset |
| Audio latency alignment is wrong | Model learns phase/latency errors | Impulse-based alignment, cross-correlation fallback, manual nudge UI |
| User data clips or lacks coverage | Poor captures | Preflight analysis and capture checklist |
| Heavy models do not run in real time | Bad downstream experience | Benchmark before export; label target CPU cost |
| Open RTNeural issues affect edge cases | Crashes or mismatches | Pin RTNeural commit, stay conservative, maintain known-good preset matrix |
| Licensing contamination from reference trainers | Product/legal risk | Use RTNeural BSD code as dependency; do not copy GPL/all-rights-reserved trainer code without review |

## Recommendation

Build the app around a narrow, reliable promise:

"Give us a dry file and a matching processed file; we will train a real-time-safe
RTNeural model, prove that it loads, prove that it matches the trained model, and
tell you how expensive it is to run."

Do not begin with arbitrary model design, ONNX import, or live low-latency
monitoring. Those are tempting, but they expand the problem before the core
value is proven.

The first internal milestone should be a command-line pipeline that trains and
validates one LSTM preset. Once parity and export are boring, wrap it in the
desktop UI.

## Source Links

- [RTNeural repository](https://github.com/jatinchowdhury18/RTNeural)
- [RTNeural paper on arXiv](https://arxiv.org/abs/2106.03037)
- [Real-Time Neural Network Inferencing for Audio Processing](https://jatinchowdhury18.medium.com/real-time-neural-network-inferencing-for-audio-processing-857313fd84e1)
- [RTNeural `model_utils.py`](https://github.com/jatinchowdhury18/RTNeural/blob/main/python/model_utils.py)
- [RTNeural `torch_helpers.h`](https://github.com/jatinchowdhury18/RTNeural/blob/main/RTNeural/torch_helpers.h)
- [RTNeural open issue #102](https://github.com/jatinchowdhury18/RTNeural/issues/102)
- [RTNeural open issue #120](https://github.com/jatinchowdhury18/RTNeural/issues/120)
- [RTNeural open issue #124](https://github.com/jatinchowdhury18/RTNeural/issues/124)
- [RTNeural open issue #157](https://github.com/jatinchowdhury18/RTNeural/issues/157)
- [AIDA-X repository](https://github.com/AidaDSP/AIDA-X)
- [AIDA-X / MOD Colab training guide](https://mod.audio/aida-x-colab-training-guide/)
- [AIDA-X model trainer notebook](https://github.com/AidaDSP/Automated-GuitarAmpModelling/blob/aidadsp_devel/AIDA_X_Model_Trainer.ipynb)
- [AIDA-X `modelToRTNeural.py`](https://github.com/AidaDSP/Automated-GuitarAmpModelling/blob/aidadsp_devel/modelToRTNeural.py)
- [Neural Amp Modeler docs](https://neural-amp-modeler.readthedocs.io/en/latest/)
- [NAM local GUI training tutorial](https://neural-amp-modeler.readthedocs.io/en/latest/tutorials/gui.html)
- [NAM calibration tutorial](https://neural-amp-modeler.readthedocs.io/en/latest/tutorials/calibration.html)
- [PyTorch MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch `torch.no_grad` docs](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
- [Tauri v2 sidecar docs](https://v2.tauri.app/develop/sidecar/)
- [Real-Time Guitar Amplifier Emulation with Deep Learning](https://www.mdpi.com/2076-3417/10/3/766)
