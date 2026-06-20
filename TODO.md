We’re past the “prove the core path” stage. The app now has real prepare/train/export, SQLite job state/recovery, cancel/resume, previews, native validation/benchmarking, sidecar packaging, and smoke/parity tests.

What’s left is mostly productization:

1. **Release automation**
   GitHub Actions now runs Python tests, golden fixture checks, native validator build/smoke, frontend build, Rust tests, Tauri workflow smoke, and packaged-app smoke. Remaining release work is cross-platform bundle smoke, signing/notarization where needed, and artifact publishing.

2. **Expand actual exported presets**
   The support matrix covers Dense, GRU, LSTM, Conv1D, activations, BatchNorm/PReLU, but the real app presets are still only `lstm_light` and `lstm_standard` in [presets.py](/Users/shortwavlabs/Workspace/shortwavlabs/rtneural-trainer/trainer/rttrainer/models/presets.py:14). Next step is product-ready Dense-only, GRU, Conv1D, and maybe hybrid presets exposed in the UI, each with golden JSON and native parity.

3. **Production packaging**
   The debug packaged-app smoke passes, but real release packaging still needs cross-platform release sidecars, PyInstaller validation, Tauri bundle smoke, signing/notarization where needed, and release artifacts.

4. **Training quality controls**
   Add stronger real-world training UX: validation curves, early stopping controls, better preset recommendations, manual alignment override, longer capture handling, normalization/gain guidance, and clearer “this model is good/bad” report language.

5. **Docs catch-up**
   README is current-ish, but the implementation guide is still partly aspirational. It should be updated to reflect what’s done, what’s deferred, and the current smoke/CI gates.

6. **Runtime integrations**
   `.aidax` is still intentionally deferred pending format/license review. Generated JUCE/plugin/standalone auditioning and RTNeural compile-time model generation are also still Phase 3 work.

7. **Polish pass**
   Error copy, edge-case UI states, onboarding/sample project, accessibility pass, and maybe waveform visualization for target/prediction/residual.

My recommended next move: expand the preset catalog now that CI covers the core export and smoke paths.
