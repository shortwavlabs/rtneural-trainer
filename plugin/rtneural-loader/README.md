# RTNeural Loader Plugin

Tiny JUCE plugin for auditioning RTNeural JSON exports in a DAW.

This is intentionally plain, but it is now useful enough for real DAW smoke
tests:

- load an exported package folder or a raw `.json` / `.rtneural.json` model;
- load an optional cabinet impulse response (`.wav`, `.aif`, `.aiff`, `.flac`);
- restore the last loaded path when a DAW session reopens;
- show export metadata such as preset, ESR, ASR, validation, native benchmark
  headroom, sample rate, latency, and receptive field;
- expose input gain, output gain, model bypass, cab IR enable, and a simple
  low/mid/high EQ shell;
- show a lightweight output peak/clip indicator.

If no model is loaded, audio is passed through with output trim applied.

## Build

```bash
cmake -S plugin/rtneural-loader -B plugin/rtneural-loader/build -DCMAKE_BUILD_TYPE=Release
cmake --build plugin/rtneural-loader/build --config Release
```

The CMake project prefers the local RTNeural fork at:

```text
/Users/shortwavlabs/Workspace/rt-neural/RTNeural
```

Override it with:

```bash
RTNEURAL_LOCAL_PATH=/path/to/RTNeural cmake -S plugin/rtneural-loader -B plugin/rtneural-loader/build
```

By default, JUCE builds AU, VST3, and Standalone targets and copies plugin
bundles after build. Disable copying with:

```bash
cmake -S plugin/rtneural-loader -B plugin/rtneural-loader/build -DRTNEURAL_LOADER_COPY_PLUGIN_AFTER_BUILD=OFF
```

## Notes

- Model loading happens from the editor button on the message thread.
- DAW state stores the selected model path, package path, parameters, and model
  name, plus the selected impulse response path. On session restore, the plugin
  reloads the model and IR if the paths still exist.
- The audio thread only reads the latest loaded model pointer, cached EQ
  coefficients, the prepared convolution stage, atomics, and sample data.
- Loaded model objects are retained for the lifetime of the processor so a model
  swap cannot delete an object still being read by the audio callback.
- Package-folder loading expects `model.rtneural.json` at the selected folder
  root. Metadata is read opportunistically from `validation-report.json`,
  `benchmark-report.json`, `aliasing-report.json`, and `package.json`.
- The IR stage runs after model inference and EQ, matching an amp-head into
  cabinet flow. IRs are loaded with JUCE `dsp::Convolution`, trimmed and
  normalised on load.
- This is a test harness, not the final AIDA-X-style player UI.

## First Smoke Result

Validated on June 25, 2026:

- AU installed to `~/Library/Audio/Plug-Ins/Components/RTNeural Loader.component`.
- `auval -v aufx RtL1 SwLv` passed.
- Logic Pro loaded the AU and opened the continued RHYTHM4 A2 PReLU export:
  `export_d56825caf0394b4bad518fdba58a9ddc/model.rtneural.json`.
- Single-instance CPU appeared minimal in Logic, and the live model sounded
  good.
- A later Logic smoke ran four plugin instances at a `32` sample buffer with
  minimal apparent CPU increase on the MacBook Pro M5 Max test machine.
- The hardened loader build added path restore, package-folder loading,
  metadata display, input/output gain, bypass, low/mid/high EQ, and an output
  peak indicator; `auval -v aufx RtL1 SwLv` still passes.
- A follow-up build added a cabinet IR loader with a DAW-persisted IR path and
  `Cab IR` enable parameter; `auval -v aufx RtL1 SwLv` still passes.

Next checks: reload a saved Logic project to verify model path restore in the
host, test `64`/`128` buffers, test 96 kHz sessions, and try less powerful
machines.
