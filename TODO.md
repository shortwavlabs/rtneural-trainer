# TODO

Current status: local desktop V1 is mostly implemented. Remaining work is
productization, release confidence, real-world tuning, and optional downstream
runtime integrations.

## P0: Release-Blocking

- [ ] Run real capture projects through the full app.
  - Cover at least amp, pedal, line/generic, short capture, long capture, quiet
    capture, clipped capture, stereo capture, and high-latency capture cases.
  - Record which warnings fired, which preset was recommended, final ESR/RMSE,
    residual audibility, native validation status, and benchmark status.
  - Tune capture/gain/preset recommendation thresholds from the results.

- [ ] Add a real Tauri UI smoke suite.
  - Exercise first-run empty state and generated sample project creation.
  - Exercise Capture, Align, Train, Evaluate, Export, Runtime, Notes, and error
    states in a real Tauri window.
  - Verify keyboard tab order, visible focus, workflow tabs, disabled states,
    and preview playback controls.
  - Run in CI or document why a platform-specific runner is required.

- [ ] Add a packaged-app tiny train/export smoke.
  - Install or launch the built bundle, not only the debug no-bundle shell.
  - Use the packaged sidecars from the app bundle.
  - Create or load a tiny sample project, prepare audio, train, evaluate, export,
    open/read the export folder, and verify validation/benchmark reports exist.
  - Run on Linux, macOS, and Windows release-packaging jobs where feasible.

- [ ] Decide and implement release signing policy.
  - macOS: Developer ID signing, hardened runtime, entitlements, notarization,
    stapling, and unsigned local-dev fallback.
  - Windows: code-signing certificate, timestamping, NSIS signing, and
    antivirus false-positive mitigation notes.
  - Linux: final bundle targets, app metadata, icons, and artifact naming.
  - Document required secrets and failure recovery in release docs.

- [ ] Finalize release artifact policy.
  - Decide tag format, artifact retention, draft/release publishing flow, and
    whether sidecars are shipped only inside bundles or also separately.
  - Verify `release-artifacts-manifest.json` is sufficient for support/debugging.

## P1: Test And Contract Hardening

- [ ] Add Python checkpoint save/resume tests.
  - Cover completed, cancelled, interrupted, and failed runs.
  - Confirm resume uses the latest safe checkpoint and appends durable events.

- [ ] Add stronger manifest/report validation.
  - Introduce JSON Schema files for prepare, train, evaluate, export, progress
    events, metrics, validation reports, benchmark reports, and packages.
  - Validate manifests at the Python boundary with clear user-facing errors.
  - Add migration notes for any breaking schema change.

- [ ] Add native validator failure fixtures.
  - Invalid JSON.
  - Unsupported layer.
  - NaN/Inf output.
  - Sample-rate or length mismatch.
  - Benchmark below target threshold.

- [ ] Add export gate edge-case tests.
  - Failed Python parity.
  - Failed native validation.
  - Failed benchmark.
  - Missing best checkpoint.
  - Missing preview artifacts.
  - Missing or stale package metadata.

- [ ] Expand real-world audio edge-case tests.
  - Near-silence, clipped input, clipped target, DC offset, mismatched active
    duration, extreme latency, stereo rejection, resample on/off, and very long
    captures.

## P1: Product UX Follow-Up

- [ ] Complete an accessibility audit after the polish pass.
  - Verify screen-reader names for icon buttons, tabs, waveform comparison,
    audio controls, project rows, runtime controls, and notices.
  - Verify keyboard-only operation for the full workflow.
  - Check color contrast for muted text, warnings, errors, badges, and disabled
    states.

- [ ] Improve waveform and spectral inspection beyond mini peak envelopes.
  - Show target/prediction/residual overlays from real preview audio.
  - Add residual spectrum or frequency-band error view.
  - Add zoom or segment selection around the evaluation excerpt.
  - Reuse the same visual language for alignment, preview, and residual analysis.

- [ ] Tune report language with real captures.
  - Calibrate good/usable/needs-work thresholds against listening tests.
  - Make report actions specific to failure mode: alignment, gain, capture
    length, preset capacity, or runtime cost.

- [ ] Harden sidecar/runtime error recovery.
  - Missing sidecar.
  - External Python missing or incompatible.
  - TensorFlow extra missing.
  - Native validator missing.
  - Permission denied when opening export folders.

## P2: Documentation Splits

- [ ] Create `docs/RTNeural-Export-Schema.md`.
  - Document the current RTNeural JSON metadata envelope, package metadata,
    validation report, benchmark report, and compatibility flags.

- [ ] Create `docs/Preset-Compatibility-Matrix.md`.
  - Generate or copy from the code-owned support matrix.
  - Include Keras/PyTorch support, native parity, benchmark tier, and UI exposure
    rules.

- [ ] Create `docs/Audio-Capture-Guidelines.md`.
  - Explain input/target pairing, gain staging, silence, clipping, duration,
    latency, stereo policy, resampling, and recommended capture material.

- [ ] Create `docs/Packaging-And-Sidecars.md`.
  - Explain dev shims, production sidecars, Tauri `externalBin`, PyInstaller,
    CMake validator builds, release artifacts, and signing/notarization.

- [ ] Create `docs/Troubleshooting.md`.
  - Cover failed prepare/train/export, runtime inspection failures, external
    Python setup, sidecar paths, native validator failures, and CI failures.

## P2: Optional Runtime Integrations

- [ ] Review `.aidax` format and license obligations.
  - Decide whether to support `.aidax` directly, use a compatible envelope, or
    defer permanently.
  - Add import/export tests only after the format decision is explicit.

- [ ] Prototype generated JUCE player/plugin export.
  - Use RTNeural-example as a reference for dynamic JSON loading.
  - Keep dynamic RTNeural JSON as the canonical model path.
  - Include smoke tests that load an exported model and process audio.

- [ ] Investigate compile-time RTNeural model generation.
  - Limit to known presets.
  - Compare output and benchmark against dynamic JSON loading.
  - Proceed only if speed/size gains justify the extra exporter complexity.

- [ ] Keep cloud training deferred.
  - Local training remains the default.
  - Revisit only after release packaging and local validation are boring.

## Recurring Gates

Run before release candidates:

```bash
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m unittest discover -s tests -v)
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python ../scripts/generate_golden_rtneural_fixtures.py --check)
cmake -S native/rtneural-validator -B native/rtneural-validator/build
cmake --build native/rtneural-validator/build
python3 scripts/smoke_rtneural_validator.py
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python ../scripts/smoke_rtneural_keras_layers.py)
pnpm --filter rtneural-trainer-app build
(cd app/src-tauri && cargo test)
pnpm --filter rtneural-trainer-app smoke:tauri-workflow
pnpm --filter rtneural-trainer-app smoke:packaged-app
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles app,dmg
```
