# RTNeural A2 Runtime Feasibility

Date: 2026-06-26

## Short Answer

The practical path is to add a fused A2-style WaveNet layer to
`shortwavlabs/rtneural-extended`, not to first rebuild RTNeural as a generic
graph runtime.

Current RTNeural can already host A2-inspired sequential Conv1D presets, and
`wavenet_tcn_a2_prelu` proves that this direction is musically useful. But it
does not represent NAM A2 exactly. True A2 needs residual layer outputs,
per-layer skip accumulation, input mix-in paths, a layer-array head convolution,
and optional slimmable submodel selection. Those are graph/topology features
that RTNeural's current dynamic `Model<T>` does not expose.

Best next step:

1. Keep `wavenet_tcn_a2_prelu` as the current trainer/export candidate.
2. Add a new fused dynamic RTNeural layer such as `a2_wavenet` or
   `residual_tcn`.
3. Target the exact A2 subset used by the provided model first.
4. Validate against NAM Core output before exposing it as a trainer export.

## Sources Inspected

- Provided A2 model:
  `/Users/shortwavlabs/Downloads/Diezel Herbert CH2 DI 2.nam`
- User RTNeural fork:
  `/Users/shortwavlabs/Workspace/rt-neural/RTNeural`
  (`shortwavlabs/rtneural-extended`)
- NAM Core local clone:
  `/tmp/NeuralAmpModelerCore`
- NAM Core upstream:
  <https://github.com/sdatkinson/NeuralAmpModelerCore>
- Existing project notes:
  - `docs/Research-NAM-To-RTNeural-Conversion.md`
  - `docs/Research-NAM-Performance-And-WaveNet.md`
  - `docs/Research-WaveNet-Amp-Simulation-Papers-2026-06-24.md`
  - `docs/Research-RTNeural-WaveNet-JUCE-Performance.md`

## What The Provided A2 Model Contains

The provided file is a NAM `0.7.0` model with top-level architecture
`SlimmableContainer`. It contains two full nested WaveNet submodels:

| Submodel | Selector | Channels | Weights | Purpose |
| --- | ---: | ---: | ---: | --- |
| Nano/lite | `max_value = 0.5` | `3` | `1,871` | Lower-cost A2 path |
| Standard/full | `max_value = 1.0` | `8` | `12,146` | Full quality A2 path |

Both submodels share the same A2 shape:

- `input_size = 1`
- `condition_size = 1`
- `channels == bottleneck` (`3` or `8`)
- One layer array
- 23 dilated layers
- Layer kernel sizes:
  `[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]`
- Layer dilations:
  `[1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]`
- Activation: `LeakyReLU(negative_slope = 0.01)` on every layer
- `gating_mode = none`
- `secondary_activation = null`
- `layer1x1.active = true`
- `head1x1.active = false`
- No FiLM paths active
- No grouped convolutions
- Layer-array head convolution:
  - `out_channels = 1`
  - `kernel_size = 16`
  - `bias = true`
- Top-level post-stack `head = null`

The layer-stack receptive field is `6,332` samples including the current sample.
NAM Core's A2 fast path adds the 16-sample head convolution and prewarms
`6,347` samples total, which is about `132.2 ms` at 48 kHz.

## What NAM Core Does For A2

NAM Core does not treat this A2 shape as a random stack of generic operations
when the fast path is enabled. It has a dedicated A2 implementation in
`NAM/wavenet/a2_fast.h` and `NAM/wavenet/a2_fast.cpp`.

Key observations:

- `NAM_ENABLE_A2_FAST` enables a specialized A2 runtime path.
- `is_a2_shape()` strictly checks for:
  - one layer array
  - no condition DSP
  - channels `3` or `8`
  - exact A2 kernel-size and dilation arrays
  - `LeakyReLU(0.01)`
  - no gating
  - no FiLM
  - no head1x1
  - active layer1x1
  - head rechannel kernel size `16`
- `A2FastModel<Channels>` fuses the full block:
  - input rechannel
  - per-layer dilated Conv1D
  - input mix-in
  - LeakyReLU
  - residual 1x1
  - skip/head accumulation
  - head convolution
- It has separate templates for `Channels == 3` and `Channels == 8`.
- It uses preallocated buffers and ring-buffer history.
- For A2 standard (`Channels == 8`), it maps operations to Eigen block kernels.
- For A2 nano (`Channels == 3`), it uses hand-unrolled scalar loops.

That design is the strongest signal for our RTNeural work: A2 is worth treating
as a known real-time DSP primitive.

## What RTNeural Currently Supports

RTNeural's dynamic model is a sequential chain:

- `RTNeural/Model.h` stores `std::vector<Layer<T>*> layers`.
- `Model<T>::forward()` feeds each layer's output to the next layer.
- `Layer<T>::forward(const T* input, T* out)` exposes one input buffer and one
  output buffer.
- `model_loader.h` parses dynamic JSON layers and appends supported layer types:
  `dense`, `conv1d`, `conv2d`, `gru`, `lstm`, `prelu`, `batchnorm`,
  `batchnorm2d`, and `activation`.
- `conv1d` already supports causal state, dilation, groups, weights, and bias.

This is enough for the current exported sequential WaveNet-like presets, but
it cannot directly express NAM A2's residual and skip topology:

- no fan-in/fan-out nodes
- no graph node IDs
- no residual add op
- no per-layer skip accumulation into a head buffer
- no layer-array concept
- no slimmable container or quality selector
- no direct A2/NAM weight stream parser

Static `ModelT` can host custom layers, as shown by the RTNeural custom layer
example, but the dynamic JSON path still needs new library support to load a
complex A2 block from exported packages.

## Current Trainer A2-Inspired Preset

The trainer currently has `wavenet_tcn_a2_prelu`, which is intentionally
RTNeural-safe:

- 12 sequential Conv1D/PReLU blocks
- mixed `6` and `15` sample kernels
- NAM-like non-power-of-two dilations
- default learning rate `3.5e-4`
- receptive field `2,481` samples
- exports as existing RTNeural JSON layers

It performed extremely well on RHYTHM4:

| Preset | Run | ESR | Worst ASR | Avg ASR | Native RTF |
| --- | --- | ---: | ---: | ---: | ---: |
| `wavenet_tcn_quality_tanh15` | `run_cc3dc9235cf7426b8529c546003e0e75` | `0.0646` | `0.0670` | `0.0419` | `11.74x` |
| `wavenet_tcn_a2_prelu` | `run_e61c249debfa4f04a140cf0ff9d7f4ff` | `0.0440` | `0.0354` | `0.0205` | `6.54x` |
| `wavenet_tcn_a2_prelu` continued | `run_0c18cca414014233bf5cd3824768021a` | `0.0381` | `0.0201` | `0.0145` | `6.62x` |

The Logic AU smoke test also looked better than expected: four instances of the
debug RTNeural loader at a 32-sample buffer barely moved CPU on the M5 Max test
machine. That result lowers the urgency of premature plugin optimization, but it
does not remove the need for a smaller/faster runtime path for less powerful
machines.

## Feasibility Options

### Option A: Keep Sequential RTNeural Presets Only

Status: already working.

Pros:

- No RTNeural fork changes.
- Keeps exporter, validator, and plugin simple.
- Good enough for current product testing.
- `wavenet_tcn_a2_prelu` already beats the older quality/tanh15 path on RHYTHM4.

Cons:

- Not true NAM A2.
- Cannot import or faithfully convert A2 `.nam` files.
- Cannot represent the A2 residual/skip/head structure.
- Needs more layers/filters to approximate long-memory behavior.

Recommendation: keep this as the product lane while deeper runtime work happens.

### Option B: Add Generic Graph Execution To RTNeural

This would add graph nodes, named tensors, fan-in/fan-out routing, `add`, `mul`,
`concat`, and topological execution.

Pros:

- Most flexible architecture support.
- Could represent A2, future NAM variants, and other modern model graphs.

Cons:

- Large RTNeural change.
- More allocator and memory-planning work.
- More difficult to keep real-time safe.
- More work for static `ModelT`.
- More export and validation complexity.
- Bigger upstream merge burden.

Recommendation: defer. It is too much infrastructure for the first A2 step.

### Option C: Add A Fused A2/ResidualTCN Layer To RTNeural

This is the best near-term RTNeural extension.

Shape:

- A new dynamic layer type, for example `a2_wavenet` or `residual_tcn`.
- It still conforms to `Layer<T>` with one input buffer and one output buffer.
- Internally, it owns the residual stack, input mix-in, skip accumulator, head
  convolution, and state buffers.
- It can be appended inside the existing sequential `Model<T>`.

Pros:

- Preserves RTNeural's sequential outer model.
- Avoids a general graph runtime.
- Matches NAM Core's own A2 fast-path strategy.
- Can be made real-time safe with preallocated state.
- Can support the provided A2 model shape exactly.
- Can later grow into a more general residual TCN layer.

Cons:

- Requires changes to `shortwavlabs/rtneural-extended`.
- Requires new JSON schema and parser code.
- Requires new exporter and validator support.
- First version should target a constrained A2 subset, not all possible NAM
  WaveNet configs.

Recommendation: do this after the current trainer app stabilizes around the
WaveNet-only preset set.

### Option D: Embed NAM Core Beside RTNeural In The Plugin

This would keep RTNeural exports and add native `.nam` loading separately.

Pros:

- Fastest path to arbitrary NAM file support.
- Uses NAM Core's maintained runtime.
- Avoids reimplementing every NAM edge case.

Cons:

- Does not make RTNeural itself A2-capable.
- Adds another runtime and model format to ship/test.
- Licensing, UI, model metadata, and preset management need review.

Recommendation: keep as a separate future product question. It is useful for
NAM interoperability, but it does not answer the RTNeural A2 export path.

## Recommended RTNeural Extension Plan

### Phase 1: Define A Constrained A2 JSON Layer

Add a new RTNeural JSON layer type for the exact supported subset:

```json
{
  "type": "a2_wavenet",
  "shape": [null, 1],
  "channels": 8,
  "kernel_sizes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6],
  "dilations": [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239],
  "activation": "leaky_relu",
  "negative_slope": 0.01,
  "head_kernel_size": 16,
  "head_scale": 0.00890730263951642,
  "weights": []
}
```

The first version should reject unsupported variations:

- `channels` not in `{3, 8, maybe 16 for our trained exports}`
- active FiLM
- active gating
- active head1x1
- grouped convolutions
- condition DSP
- multiple layer arrays
- slimmable switching inside a single dynamic layer

### Phase 2: Implement The Dynamic Layer In The RTNeural Fork

Candidate files:

- `RTNeural/a2_wavenet/a2_wavenet.h`
- `RTNeural/a2_wavenet/a2_wavenet.tpp`
- parser case in `RTNeural/model_loader.h`

Implementation notes:

- Use a single `Layer<T>` subclass.
- Preallocate all circular history and scratch buffers at construction.
- Keep all work in `forward()` allocation-free.
- Use a ring-buffer strategy for long dilations.
- Start with `float`/`double` template support matching RTNeural conventions.
- Keep weight layout explicit and documented.
- Add `reset()` that clears all state.
- Support one-sample `forward()` first because RTNeural dynamic models are
  currently sample-at-a-time.
- Consider an additional block-processing API later if plugin performance ever
  needs it.

### Phase 3: Add Parity Tests Against NAM Core

Test layers:

1. Impulse input.
2. Step input.
3. Low-level noise.
4. Guitar DI excerpts.
5. Reset/prewarm consistency.
6. Small-buffer behavior (`1`, `16`, `32`, `64`, `128` samples) on the plugin
   side if a block API is added.

Reference path:

- Load `/Users/shortwavlabs/Downloads/Diezel Herbert CH2 DI 2.nam` with NAM Core.
- Render the standard `max_value = 1.0` path.
- Render the nano `max_value = 0.5` path.
- Convert the same weights to RTNeural `a2_wavenet` JSON.
- Compare max absolute error and RMS error.

The acceptance threshold should be strict for synthetic tests and practical for
real guitar files. Start with `max_abs_error < 1e-5` for pure parity where
possible.

### Phase 4: Add Trainer Export Support

Once the RTNeural fork can run the fused layer:

- Add a Keras architecture that matches the fused A2 block.
- Export to `a2_wavenet` JSON instead of sequential Conv1D layers.
- Add golden fixtures.
- Extend `rtneural-validator` for the new layer.
- Add package metadata:
  - `runtime_family: "a2_wavenet"`
  - `a2_compatible: true`
  - `receptive_field_samples`
  - `prewarm_samples`
  - `channels`
  - `quality_selector` if slimmable exports are added later
- Update the export UI to show that the package needs an RTNeural runtime with
  A2 support.

### Phase 5: Add Optional Slimmable Export

The provided NAM file uses two submodels selected by `max_value`.

For our own exported RTNeural packages, this could become:

- one package with two `a2_wavenet` models: `nano` and `standard`
- plugin quality switch that loads/switches between them outside the audio
  thread
- or two separate exports for simpler v1 behavior

Do not start here. Get one fixed full-quality A2 layer working first.

## Why The Fused Layer Is Better Than Sequential Approximation

Sequential Conv1D stacks have to pass all information through one path. A2's
layer array has two paths:

1. A residual path that keeps refining the hidden state.
2. A skip/head path that accumulates each layer's activated contribution.

That skip accumulation matters because early, mid, and late receptive-field
features all reach the output directly. It also explains why A2 can use low
channel counts while still modeling dense high-gain behavior.

Our current `wavenet_tcn_a2_prelu` borrows the dilation/kernels idea, but it
still lacks the skip accumulator and layer-array head. A fused RTNeural layer is
the smallest change that closes that gap.

## Risks

- Weight layout mismatch between NAM Core, Keras, and RTNeural.
- Off-by-one errors in causal convolution history.
- Different reset/prewarm behavior.
- Sample-by-sample RTNeural execution may be slower than NAM Core's block
  processing.
- Supporting arbitrary NAM A2 files may require more than the provided subset.
- Generic graph work can distract from the smaller high-value A2 path.
- Exact `.nam` import may raise product and licensing questions separate from
  training our own RTNeural exports.

## Decision

For this project, the next RTNeural-side research implementation should be a
constrained fused A2/ResidualTCN layer in `shortwavlabs/rtneural-extended`.

Do not attempt full NAM conversion first. The first milestone should be:

- hand-convert the provided A2 standard submodel into a prototype RTNeural
  `a2_wavenet` JSON
- render through RTNeural and NAM Core
- prove numerical parity
- then wire the trainer exporter to emit the same layer from TensorFlow/Keras

If that works, RTNeural becomes capable of the model topology that our listening
tests are already pointing toward, while keeping the trainer app and plugin
runtime simpler than a full graph runtime.
