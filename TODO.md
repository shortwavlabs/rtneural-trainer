We’re past the “prove the core path” stage. The app now has real prepare/train/export, SQLite job state/recovery, cancel/resume, previews, native validation/benchmarking, sidecar packaging, and smoke/parity tests.

What’s left is mostly productization:

1. **Release automation**
   GitHub Actions now runs Python tests, golden fixture checks, native validator build/smoke, frontend build, Rust tests, Tauri workflow smoke, packaged-app smoke, and a separate cross-platform release packaging workflow that builds real sidecars, smokes Tauri bundles, and uploads artifacts. Remaining release work is signing/notarization credentials, release publishing policy, and any installer-specific polish.

2. **Preset tuning**
   The app now exposes Dense-only, GRU, LSTM, Conv1D, Conv1D BatchNorm/PReLU, and Conv+GRU hybrid presets, each covered by golden JSON plus Python/native RTNeural parity. The train UI also recommends presets from target type, backend, capture duration, alignment confidence, and gain warnings. Remaining work is real-world capture tuning and PyTorch parity only where it is worth supporting.

3. **Production packaging**
   Real release packaging now has PyInstaller sidecar validation, native validator staging, Tauri bundle smoke, and artifact manifests/uploads. Remaining work is signed/notarized production distribution, installer metadata polish, and validating the artifact set against the final release channel.

4. **Training quality controls**
   The app now has validation curves, early stopping controls, preset recommendations, saved manual alignment override, sampled-window handling for longer captures, gain/headroom guidance, and clearer good/usable/needs-work report language. Remaining work is validating those thresholds against real captures and adding more visual waveform inspection.

5. **Docs catch-up**
   README is current-ish, but the implementation guide is still partly aspirational. It should be updated to reflect what’s done, what’s deferred, and the current smoke/CI gates.

6. **Runtime integrations**
   `.aidax` is still intentionally deferred pending format/license review. Generated JUCE/plugin/standalone auditioning and RTNeural compile-time model generation are also still Phase 3 work.

7. **Polish pass**
   Error copy, edge-case UI states, onboarding/sample project, accessibility pass, and maybe waveform visualization for target/prediction/residual.

My recommended next move: run a real capture project through the new training controls, tune the recommendation thresholds, then decide the signing/notarization and release-publishing path.
