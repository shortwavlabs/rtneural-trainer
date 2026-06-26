# TODO

Current status: local desktop V1 is mostly implemented. Remaining work is
productization, release confidence, real-world tuning, and optional downstream
runtime integrations.

Ownership labels:

- `Owner: You` means product decisions, credentials, real audio captures,
  listening judgment, or platform access are needed from you.
- `Owner: Codex` means I can implement, test, document, or automate it in this
  repo.
- `Owner: Shared` means you provide decisions/data/access and I turn that into
  code, docs, tests, or release automation.

Recommended next move:

1. `Shared` Run 2-3 real capture projects through the app and record outcomes,
   starting long amp/pedal captures with `wavenet_tcn_balanced`, using
   `wavenet_tcn_quality` for conservative crunch/rhythm/high-gain refinement,
   and using `wavenet_tcn_a2_prelu` as the current high-gain candidate.
   This is the highest-value next step because it calibrates preset
   recommendations, gain warnings, report language, and export confidence.
   WaveNet defaults, excellent/good/usable/needs-work report language, latency
   candidate review with window agreement, export parity snapshots, and the
   native benchmark matrix are now in place; edge-of-breakup, lead, pedal,
   quiet, clipped, and stereo captures are the next useful evidence.
2. `Codex` Add the packaged-app tiny train/export smoke.
   This protects the production bundle path, not just the dev shell and jsdom
   workflow.
3. `You` Decide the release/signing policy before we spend more time on release
   packaging details.

## P0: Release-Blocking

- [ ] Run real capture projects through the full app.
  - Owner: Shared.
  - You: provide or record representative amp, pedal, line/generic, short
    capture, long capture, quiet capture, clipped capture, stereo capture, and
    high-latency capture cases.
  - You: do the listening judgment for target/prediction/residual and mark
    whether each report verdict feels right.
  - You: for low-confidence or low-agreement latency warnings, try the
    candidate sample offsets shown in Align before committing to long WaveNet
    runs.
  - Codex: create or maintain a repeatable capture-results template if needed.
  - Codex: tune capture/gain/preset recommendation thresholds from the results.
  - Codex: keep the trainer UI and built-in recipes WaveNet-only unless new
    captures show a repeatable non-WaveNet exception.
  - Codex: keep non-WaveNet layer fixtures only as internal RTNeural export
    coverage, not as product training recommendations.
  - Track which warnings fired, which preset was recommended, final ESR/RMSE,
    residual audibility, native validation status, native benchmark worst-case
    real-time factor, ASR aliasing summary, and benchmark matrix weak spots by
    block size/channel count.

- [x] Add a Tauri UI smoke suite.
  - Owner: Codex.
  - Implemented as `pnpm --filter rtneural-trainer-app smoke:tauri-ui`.
  - Covers first-run empty state, generated sample project creation,
    project-specific WAV path switching, rename/delete, Runtime settings,
    Capture, Align, Train, Evaluate, Export, preview report loading, and
    open-export-folder command wiring with mocked Tauri commands.
  - Tauri's desktop WebDriver path is Linux/Windows-only; macOS keeps this
    jsdom suite as the portable CI/local smoke. A true `tauri-driver` window
    smoke can be added later on Linux/Windows runners if needed.

- [ ] Add a packaged-app tiny train/export smoke.
  - Owner: Codex.
  - Install or launch the built bundle, not only the debug no-bundle shell.
  - Use the packaged sidecars from the app bundle.
  - Create or load a tiny sample project, prepare audio, train, evaluate, export,
    open/read the export folder, and verify validation/benchmark reports exist.
  - Run on Linux, macOS, and Windows release-packaging jobs where feasible.

- [ ] Decide and implement release signing policy.
  - Owner: Shared.
  - You: decide whether first public artifacts must be signed/notarized or
    whether unsigned internal builds are acceptable.
  - You: provide Apple Developer ID credentials, Windows code-signing
    certificate details, and release-secret policy when signing is required.
  - Codex: implement macOS hardened runtime, entitlements, notarization,
    stapling, and unsigned local-dev fallback after the policy is set.
  - Codex: implement Windows signing, timestamping, NSIS signing, and
    antivirus false-positive mitigation notes after certificate details exist.
  - Codex: finalize Linux bundle targets, app metadata, icons, artifact naming,
    required secrets, and failure recovery docs.

- [ ] Finalize release artifact policy.
  - Owner: You for decisions, Codex for implementation.
  - You: decide tag format, artifact retention, draft/release publishing flow,
    and whether sidecars are shipped only inside bundles or also separately.
  - Codex: update release workflows and docs after those decisions.
  - Codex: verify `release-artifacts-manifest.json` is sufficient for
    support/debugging.

## P1: Test And Contract Hardening

- [ ] Add Python checkpoint save/resume tests.
  - Owner: Codex.
  - Cover completed, cancelled, interrupted, and failed runs.
  - Confirm resume uses the latest safe checkpoint and appends durable events.

- [ ] Retire non-WaveNet product paths from remaining docs/tests where safe.
  - Owner: Codex.
  - Keep Dense/GRU/LSTM/Conv1D fixtures only where they prove RTNeural exporter
    layer compatibility.
  - Remove non-WaveNet wording from product guidance, UI smoke expectations, and
    run comparison docs.
  - Do not spend more time on recurrent-state training unless a future product
    requirement reopens that path.

- [ ] Add stronger manifest/report validation.
  - Owner: Codex.
  - Introduce JSON Schema files for prepare, train, evaluate, export, progress
    events, metrics, validation reports, benchmark reports, and packages.
  - Validate manifests at the Python boundary with clear user-facing errors.
  - Add migration notes for any breaking schema change.

- [ ] Add native validator failure fixtures.
  - Owner: Codex.
  - Invalid JSON.
  - Unsupported layer.
  - NaN/Inf output.
  - Sample-rate or length mismatch.
  - Benchmark below target threshold.

- [ ] Add export gate edge-case tests.
  - Owner: Codex.
  - Failed Python parity.
  - Failed native validation.
  - Failed benchmark.
  - Missing best checkpoint.
  - Missing preview artifacts.
  - Missing parity snapshot artifacts.
  - Missing or stale package metadata.

- [ ] Expand real-world audio edge-case tests.
  - Owner: Shared.
  - You: provide or approve representative audio cases, especially anything
    that should be considered product-realistic.
  - Codex: add fixtures/tests for near-silence, clipped input, clipped target,
    DC offset, mismatched active duration, extreme latency, stereo rejection,
    resample on/off, and very long captures.

## P1: Product UX Follow-Up

- [ ] Complete an accessibility audit after the polish pass.
  - Owner: Codex, with optional user review.
  - Verify screen-reader names for icon buttons, tabs, waveform comparison,
    audio controls, project rows, runtime controls, and notices.
  - Verify keyboard-only operation for the full workflow.
  - Check color contrast for muted text, warnings, errors, badges, and disabled
    states.
  - You: sanity-check the final workflow if you have a preferred assistive tech
    setup or platform target.

- [ ] Improve waveform and spectral inspection beyond mini peak envelopes.
  - Owner: Codex.
  - Show target/prediction/residual overlays from real preview audio.
  - Add residual spectrum or frequency-band error view.
  - Add zoom or segment selection around the evaluation excerpt.
  - Reuse the same visual language for alignment, preview, and residual
    analysis.

- [ ] Tune report language with real captures.
  - Owner: Shared.
  - Current WaveNet thresholds landed: very strong previews can now report
    `excellent`, strong previews report `good`, non-isolated residual peaks can
    still downgrade to `usable`, severe sustained residual problems remain
    `needs_work`, and native runtime viability starts at `>= 1x` real-time.
  - You: judge whether excellent/good/usable/needs-work matches listening results.
  - Codex: calibrate thresholds against listening notes.
  - Codex: make report actions specific to failure mode: alignment, gain,
    capture length, preset capacity, aliasing, or runtime cost.

- [ ] Harden sidecar/runtime error recovery.
  - Owner: Codex.
  - Missing sidecar.
  - External Python missing or incompatible.
  - TensorFlow extra missing.
  - Native validator missing.
  - Permission denied when opening export folders.

## P2: Documentation Splits

- [ ] Create `docs/RTNeural-Export-Schema.md`.
  - Owner: Codex.
  - Document the current RTNeural JSON metadata envelope, package metadata,
    validation report, benchmark report, and compatibility flags.

- [ ] Create `docs/Preset-Compatibility-Matrix.md`.
  - Owner: Codex.
  - Generate or copy from the code-owned support matrix.
  - Include TensorFlow/Keras support, native parity, benchmark tier, and UI
    exposure rules.

- [x] Create `docs/Audio-Capture-Guidelines.md`.
  - Owner: Shared.
  - You: provide capture preferences, target use cases, and any house style for
    recommended source material.
  - Codex: document input/target pairing, gain staging, silence, clipping,
    duration, latency, stereo policy, resampling, and recommended capture
    material.
  - Initial version created from the 2025 profile-family level analysis; refine
    as more real captures are reviewed.

- [ ] Create `docs/Packaging-And-Sidecars.md`.
  - Owner: Codex after release policy decisions.
  - Explain dev shims, production sidecars, Tauri `externalBin`, PyInstaller,
    CMake validator builds, release artifacts, and signing/notarization.

- [ ] Create `docs/Troubleshooting.md`.
  - Owner: Codex.
  - Cover failed prepare/train/export, runtime inspection failures, external
    Python setup, sidecar paths, native validator failures, and CI failures.

## P2: Optional Runtime Integrations

- [ ] Review `.aidax` format and license obligations.
  - Owner: You for format/license decision, Codex for implementation after
    approval.
  - You: decide whether to support `.aidax` directly, use a compatible envelope,
    or defer permanently.
  - Codex: add import/export tests only after the format decision is explicit.

- [ ] Prototype generated JUCE player/plugin export.
  - Owner: Shared.
  - You: decide whether a generated plugin/player is in scope for V1.x.
  - Codex: use RTNeural-example as a reference for dynamic JSON loading if this
    becomes active.
  - Keep dynamic RTNeural JSON as the canonical model path.
  - Include smoke tests that load an exported model and process audio.

- [ ] Investigate compile-time RTNeural model generation.
  - Owner: Codex only if you approve the added exporter complexity.
  - Limit to known presets.
  - Compare output and benchmark against dynamic JSON loading.
  - Proceed only if speed/size gains justify the extra exporter complexity.

- [ ] Keep cloud training deferred.
  - Owner: You to reopen; Codex to implement only after local release work is
    boring.
  - Local training remains the default.
  - Revisit only after release packaging and local validation are boring.

## Recurring Gates

Run before release candidates:

```bash
pyright
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python -m unittest discover -s tests -v)
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python ../scripts/generate_golden_rtneural_fixtures.py --check)
cmake -S native/rtneural-validator -B native/rtneural-validator/build
cmake --build native/rtneural-validator/build
python3 scripts/smoke_rtneural_validator.py
(cd trainer && UV_CACHE_DIR=../.uv-cache uv run --extra tensorflow python ../scripts/smoke_rtneural_keras_layers.py)
pnpm --filter rtneural-trainer-app build
(cd app/src-tauri && cargo test)
pnpm --filter rtneural-trainer-app smoke:tauri-ui
pnpm --filter rtneural-trainer-app smoke:tauri-workflow
pnpm --filter rtneural-trainer-app smoke:packaged-app
pnpm --filter rtneural-trainer-app smoke:release-package -- --bundles app,dmg
```
