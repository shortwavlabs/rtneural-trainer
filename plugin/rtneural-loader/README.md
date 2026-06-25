# RTNeural Loader Plugin

Tiny JUCE plugin for auditioning RTNeural JSON exports in a DAW.

This is intentionally plain: one output volume knob and one file chooser button
that loads a `.json` / `.rtneural.json` model. If no model is loaded, audio is
passed through with the volume control applied.

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
- The audio thread only reads the latest loaded model pointer and applies output
  gain.
- Loaded model objects are retained for the lifetime of the processor so a model
  swap cannot delete an object still being read by the audio callback.
- This is a test harness, not the final AIDA-X-style player UI.

## First Smoke Result

Validated on June 25, 2026:

- AU installed to `~/Library/Audio/Plug-Ins/Components/RTNeural Loader.component`.
- `auval -v aufx RtL1 SwLv` passed.
- Logic Pro loaded the AU and opened the continued RHYTHM4 A2 PReLU export:
  `export_d56825caf0394b4bad518fdba58a9ddc/model.rtneural.json`.
- Single-instance CPU appeared minimal in Logic, and the live model sounded
  good.

Next checks: small Logic buffers (`32`, `64`, `128` samples), duplicated plugin
instances, 96 kHz sessions, and testing on less powerful machines.
