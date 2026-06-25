# NAM to RTNeural Conversion Research

Date: 2026-06-25

## Summary

Lossless conversion from arbitrary `.nam` files to our current RTNeural JSON
format is not a general solution. NAM files can describe architectures whose
graph structure is richer than the sequential Keras/RTNeural path we currently
export: residual additions, skip accumulation, gating, input mix-ins, optional
FiLM, heads, packed/slimmable submodels, and A2 quality scaling.

There are still useful paths:

1. **NAM LSTM direct conversion** is plausible as a constrained proof of
   concept, with careful gate-order, transpose, bias, and initial-state parity
   tests.
2. **NAM A1 WaveNet direct conversion** is not representable by the current
   sequential RTNeural JSON. It would require either a custom RTNeural graph
   extension/static block, or plugin-side native NAM support.
3. **NAM A2 direct conversion** should not be the first target. A2 is a new
   composite/slimmable/packed architecture, not just a larger A1 WaveNet.
4. **Model distillation is the best trainer-app path**: load/render a NAM model
   as the teacher, generate target audio from a known DI, then train one of our
   RTNeural-safe presets against that rendered target. This is not lossless, but
   it fits the app we already have.

Recommended near-term project: add a `.nam` inspector and an optional "Distill
NAM to RTNeural" workflow. Keep direct LSTM conversion as a small technical
spike. Defer direct A1/A2 WaveNet conversion until the plugin/runtime strategy
is decided.

## Sources Reviewed

- NAM `.nam` file spec:
  [`neural-amp-modeler` docs](https://neural-amp-modeler.readthedocs.io/en/latest/model-file.html)
- NAM trainer repository:
  [`sdatkinson/neural-amp-modeler`](https://github.com/sdatkinson/neural-amp-modeler)
- NAM A1 WaveNet implementation, pinned from the model-file docs:
  [`nam/models/wavenet.py`](https://github.com/sdatkinson/neural-amp-modeler/blob/cb100787af4b16764ac94a2edf9bcf7dc5ae59a7/nam/models/wavenet.py)
- NAM LSTM implementation:
  [`nam/models/recurrent.py`](https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam/models/recurrent.py)
- Current NAM loader code:
  [`nam/models/_from_nam.py`](https://github.com/sdatkinson/neural-amp-modeler/blob/main/nam/models/_from_nam.py)
- A2 release note:
  [A2 is released](https://www.neuralampmodeler.com/post/a2-is-released)
- A2 integration note:
  [NeuralAmpModelerCore v0.5.0 is released](https://www.neuralampmodeler.com/post/neuralampmodelercore-v0-5-0-is-released)
- A2 technical guide:
  [NAM A2: The Complete Guide](https://www.tone3000.com/guides/nam-a2-the-complete-guide)
- RTNeural supported layer list:
  [`jatinchowdhury18/RTNeural`](https://github.com/jatinchowdhury18/RTNeural)
- NeuralAudio bridge/runtime reference:
  [`mikeoliphant/NeuralAudio`](https://github.com/mikeoliphant/NeuralAudio)
- `.namb` binary loader:
  [`tone-3000/nam-binary-loader`](https://github.com/tone-3000/nam-binary-loader)

## NAM File Format Notes

NAM `.nam` files are JSON dictionaries. The expected top-level keys are:

- `version`
- `architecture`
- `config`
- `weights`

Optional fields include:

- `sample_rate`
- `metadata`
- level calibration fields such as `input_level_dbu` and `output_level_dbu`

The important conversion detail is that `weights` is a flat list, and the way it
maps to tensors is architecture-specific. The NAM docs explicitly point readers
to each architecture's `_export_weights()` implementation for the definitive
weight layout.

Current NAM model factory support includes:

- `Linear`
- `LSTM`
- `WaveNet`

The current trainer code also has registry entries for richer model families
such as `ConvNet`, `Sequential`, and `PackedWaveNet`.

## RTNeural Constraints

RTNeural supports the basic primitives we care about for many amp models:

- Dense
- GRU
- LSTM
- Conv1D / Conv2D
- MaxPooling
- BatchNorm
- tanh, ReLU, Sigmoid, SoftMax, ELU, PReLU

Our app's current RTNeural exporter is narrower than RTNeural's C++ capabilities:
it serializes a Keras `Sequential` layer list. That works for our Dense, GRU,
LSTM, Conv1D stack, WaveNet-style sequential TCN, BatchNorm, and PReLU presets.

It does **not** currently serialize graph edges or arbitrary functional topology:

- residual addition
- skip accumulation
- split/gate/multiply
- parallel branches
- condition/mixin inputs
- quality-scaled composite models

This is the main barrier to direct NAM WaveNet conversion.

## A1 WaveNet Conversion Feasibility

NAM A1 WaveNet is not just a sequential stack of Conv1D layers.

The pinned A1 implementation does roughly this per layer:

```text
zconv = dilated_conv(x)
z = zconv + input_mixer(condition)
if gated:
    post = activation(z[:channels]) * sigmoid(z[channels:])
else:
    post = activation(z)
residual = x[-len(post):] + conv1x1(post)
skip/head += post
```

Then the accumulated head path is rechanneled through 1x1 convolutions and an
optional head module.

That topology is central to the model. Flattening the weights into a sequential
Conv1D stack would change the function. The only lossless-ish options are:

1. **Extend RTNeural JSON/validator/plugin graph support** with Add, Multiply,
   Slice/Split, and named tensors.
2. **Implement a custom RTNeural-backed WaveNet block** in C++ and use NAM's
   config/weight ordering to fill it.
3. **Support NAM natively** in the plugin via NeuralAmpModelerCore or
   NeuralAudio instead of converting.

Option 1 is attractive long-term because it would also unlock more modern model
families in our own trainer. It is too large for a quick trainer-app import.

Option 2 is feasible for official A1 shapes, but it is really a new native
runtime path, not a normal RTNeural JSON export.

Option 3 is the most pragmatic plugin path if we want to load existing NAM
libraries.

## LSTM Conversion Feasibility

NAM LSTM direct conversion is the best candidate for a small proof of concept.

NAM's LSTM export layout is:

For each LSTM layer:

```text
combined_weight: (4 * hidden, input_dim + hidden_dim)
bias:            (4 * hidden)
initial_hidden:  (hidden)
initial_cell:    (hidden)
```

Then the linear head weights follow.

To convert to our Keras/RTNeural JSON layout:

1. Split `combined_weight` into `w_ih` and `w_hh`.
2. Confirm gate order. PyTorch uses input, forget, cell/update, output order.
   Keras/RTNeural JSON should be checked against our golden LSTM fixture before
   assuming parity.
3. Transpose PyTorch-style matrices into Keras-style kernels:
   - `kernel`: `(input_dim, 4 * hidden)`
   - `recurrent_kernel`: `(hidden_dim, 4 * hidden)`
4. Map bias into the expected Keras/RTNeural bias representation.
5. Convert PyTorch `Linear` head `(out, hidden)` to Dense kernel `(hidden, out)`.
6. Decide how to handle NAM's learned initial hidden/cell state.

The initial state is the main behavioral wrinkle. RTNeural's ordinary recurrent
reset path is generally zero-state. If we ignore NAM's initial state, parity may
be poor at the very start of a render but converge after burn-in. For real-time
plugin use, that may be acceptable if we pad or prewarm. For strict conversion,
we would need either:

- RTNeural state initialization support,
- a plugin-side prewarm/burn-in policy,
- or a custom stateful LSTM wrapper.

## A2 Conversion Feasibility

NAM A2 should be treated as a native NAM/Core integration target first, not as a
direct RTNeural JSON target.

Reasons:

- A2 is now NAM's standard architecture for new captures.
- A2 is not backward compatible with old A1 hosts; hosts need updates.
- Official guidance points builders to `neural-amp-modeler` v0.13.0,
  NeuralAmpModelerCore v0.5.2, and NeuralAmpModelerPlugin v0.7.14.
- A2 can run as A2-Full or A2-Lite from a single model.
- A2 uses slimmable/packed training: one model contains multiple runtime sizes.
- The technical guide describes WaveNet-like residual/skip structure, mixed
  kernel sizes, packed submodels, quality scaling, and A2-specific fast paths.

This does not map naturally to our current sequential RTNeural JSON. A2 support
should be evaluated as:

- native NAM Core integration,
- NeuralAudio integration,
- or distillation into one of our RTNeural-safe architectures.

## Sample A2 Model Inspection

Local sample:

```text
/Users/shortwavlabs/Downloads/Diezel Herbert CH2 DI 2.nam
```

Inspection result:

| Field | Value |
| --- | --- |
| File size | `295,022 bytes` |
| NAM version | `0.7.0` |
| Top-level architecture | `SlimmableContainer` |
| Sample rate | `48,000 Hz` |
| Name | `Diezel Herbert CH2 DI 2` |
| Modeled by | `ultimatemetalguitartones` |
| Gear type | `amp` |
| Trainer | `TONE3000` |
| Top-level loudness | `-16.13 dB` |
| Top-level gain | `0.8167` |

The file has no top-level weights. Instead, weights live inside two nested
WaveNet submodels:

| Submodel | Selector | Channels | Weights | Layers | Kernel sizes | Receptive field |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Lite | `max_value = 0.5` | `3` | `1,871` floats | `23` | `6`, `15` | `6,332 samples` / `131.9 ms` |
| Full | `max_value = 1.0` | `8` | `12,146` floats | `23` | `6`, `15` | `6,332 samples` / `131.9 ms` |

Both submodels are `WaveNet` configs with:

- one layer array,
- dilation pattern
  `[1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]`,
- per-layer `LeakyReLU(negative_slope = 0.01)`,
- `gating_mode = none`,
- active `layer1x1`,
- inactive `head1x1`,
- 16-sample head convolution,
- inactive FiLM hooks.

This confirms the A2 guide's description: a single file contains multiple
runtime sizes, and the architecture uses a mixed-kernel, long-receptive-field,
WaveNet-like residual structure. Even though this particular model disables
gating and FiLM, it still needs residual adds, input mix-ins, layer/head 1x1
paths, mixed kernel sizes, and submodel selection. Current RTNeural JSON cannot
represent that graph losslessly.

Best use of this sample in the trainer app:

1. Use it as the first fixture for a `.nam` inspector.
2. Use it as a teacher model for a future NAM-to-RTNeural distillation workflow.
3. Use it to validate that the app labels A2 as "native NAM/distill required",
   not "direct RTNeural conversion available".

## Distillation Path

The best trainer-app feature is not "convert `.nam` weights"; it is "convert
NAM behavior into an RTNeural-safe model."

Workflow:

1. User selects a `.nam` file and a DI/reference WAV.
2. App loads the NAM model with the official Python package, NAM Core, or
   NeuralAudio.
3. App renders the DI through the NAM model to produce a target WAV.
4. App trains an RTNeural-safe preset against the generated pair.
5. App validates:
   - teacher NAM vs student RTNeural audio,
   - native RTNeural parity,
   - aliasing,
   - runtime benchmark,
   - listening preview.

This mirrors the broader ecosystem idea that model-to-model conversion is really
synthetic retraining. The TONE3000 A2 guide says they retrained some A1 models
using synthetic training data generated from the A1 model when original captures
were unavailable.

Distillation advantages:

- Works for A1, A2, and future NAM architectures as long as we can render them.
- Keeps our export path unchanged.
- Lets us choose RTNeural presets by target CPU budget.
- Produces normal package metadata and reports.

Distillation disadvantages:

- Not lossless.
- Requires a varied DI to excite the teacher model.
- Can miss behavior outside the excitation data.
- Needs clear UI language: "distilled from NAM", not "converted from NAM".

## Native Runtime Alternative

NeuralAudio is an important reference because it already supports both worlds:

- NAM WaveNet/LSTM A1 and A2
- RTNeural Keras models for LSTM/GRU

It also has internal static implementations for official NAM A1 WaveNet and
LSTM shapes, dynamic fallback paths, and optional RTNeural load modes for some
models. For A2, NeuralAudio currently uses NAM Core.

This suggests a practical plugin architecture:

```text
Model package
  ├─ RTNeural JSON path for our trained exports
  ├─ NAM/Core path for .nam models
  └─ optional distilled RTNeural student model for constrained devices
```

That is cleaner than forcing every NAM model through RTNeural JSON. It also
keeps A2-Full/A2-Lite quality scaling available.

## `.namb` Note

TONE3000's `.namb` project is not a NAM-to-RTNeural converter. It is a compact
binary representation of NAM models for embedded systems.

Useful lessons for us:

- `.nam` JSON parsing is expensive for embedded devices.
- Text floats are much larger than raw float32.
- Binary model loading can avoid dynamic JSON parsing at runtime.

If we later design `.aidax`, this is relevant: the envelope should be compact,
versioned, and capable of holding model metadata plus raw binary weights. But
`.namb` itself is for NAM Core-style loading, not RTNeural JSON.

## Recommended Implementation Plan

### Phase 1: `.nam` Inspector

Add a read-only inspector command:

```text
rttrainer inspect-nam model.nam
```

Report:

- architecture
- version
- sample rate
- metadata name/gear/tone type
- input/output dBu fields
- weight count and approximate size
- whether it looks like A1 LSTM, A1 WaveNet, A2/PackedWaveNet, or unsupported
- recommended path:
  - direct LSTM spike
  - distill
  - native NAM/plugin support

### Phase 2: NAM Render Sidecar

Add an optional render command:

```text
rttrainer render-nam --model model.nam --input di.wav --output target.wav
```

Implementation candidates:

- Python `neural-amp-modeler` package for fastest prototype.
- NeuralAmpModelerCore CLI for plugin-aligned parity.
- NeuralAudio if we want one C++ abstraction for NAM and RTNeural.

### Phase 3: Distill NAM to RTNeural

Add an app workflow:

```text
Import NAM -> choose DI -> render teacher target -> train RTNeural student
```

Recommended initial student presets:

- `wavenet_tcn_quality`
- `wavenet_tcn_quality_tanh15`
- `wavenet_tcn_fast` for lower runtime checks

Reports should compare:

- NAM teacher vs RTNeural student ESR/RMSE/correlation
- residual spectrum
- aliasing warning
- native RTNeural backend benchmark
- preview audio

### Phase 4: Direct NAM LSTM Converter Spike

Build a small offline script:

```text
scripts/convert_nam_lstm_to_rtneural.py model.nam out.rtneural.json
```

Acceptance tests:

- Load a known NAM LSTM.
- Render impulse, sine sweep, noise, and guitar DI through NAM.
- Render same inputs through RTNeural JSON.
- Check max abs error after burn-in.
- Document whether NAM initial hidden/cell state needs runtime support.

### Phase 5: Decide Plugin Strategy

Before trying direct WaveNet conversion, choose one:

1. Add NAM Core / NeuralAudio as a native plugin runtime beside RTNeural.
2. Extend RTNeural JSON/validator/plugin graph support.
3. Keep NAM support as distillation-only.

My recommendation: **distillation now, native NAM runtime later**, direct LSTM
conversion as a narrow learning spike, and no direct A2 conversion attempt until
we have a plugin runtime strategy.

## Open Questions

- Do we want to support existing `.nam` libraries directly in the eventual
  plugin, or only produce RTNeural exports from our trainer?
- Is the `.aidax` envelope allowed to contain NAM or `.namb` payloads after
  license/format review?
- Should distillation use a generated excitation file, a user's real DI, or both?
- How much teacher/student error is acceptable before the app labels a distilled
  model "not faithful"?
- Should plugin runtime support quality scaling if we adopt A2 natively?
