# AGENTS.md

Guidance for coding agents working in this repository.

## Project Shape

RTNeural Trainer is a Tauri desktop app with a React UI, Rust orchestration,
Python trainer sidecar, native RTNeural validator, and markdown research/docs.

Common commands:

- `pnpm --filter rtneural-trainer-app tauri dev`
- `pnpm --dir app build`
- `pnpm --dir app test`
- `cargo test --manifest-path app/src-tauri/Cargo.toml`
- `UV_CACHE_DIR=.uv-cache uv run --project trainer pytest -q`
- `pyright`

Use `uv` for Python dependency execution and `pnpm` for app/frontend commands.

## Adding Or Changing Training Presets

Preset changes must be kept in sync across the full app. Do not add a preset in
only Python or only the UI.

When adding a preset, update all applicable places:

1. Python trainer definition:
   - `trainer/rttrainer/models/presets.py`
   - Add the `PRESETS` entry, architecture parameters, default loss, and default
     learning rate if the preset should have one.

2. Rust validation/orchestration allowlist:
   - `app/src-tauri/src/lib.rs`
   - Add the preset to `MODEL_PRESETS`.
   - Add or update `estimated_realtime_factor_for_preset` if the UI/export
     should show a runtime estimate.
   - Add a unit test proving `normalize_model_preset("<preset_id>")` accepts it.

3. React UI:
   - `app/src/App.tsx`
   - Add the visible `PresetOption`.
   - Add a built-in training recipe when users should be able to launch it
     directly.
   - Update resume/continuation defaults if the preset belongs to a continuation
     family.

4. Golden fixtures and parity:
   - `scripts/generate_golden_rtneural_fixtures.py`
   - Add a deterministic seed when useful.
   - Regenerate fixtures with:
     `UV_CACHE_DIR=.uv-cache uv run --project trainer --extra tensorflow python scripts/generate_golden_rtneural_fixtures.py`
   - Commit the new `fixtures/rtneural-json/golden/*.rtneural.json`.

5. Research/search tooling:
   - `scripts/search_rtneural_presets.py`
   - `scripts/compare_training_runs.py`
   - Any benchmark/runtime matrix scripts that enumerate presets.

6. Tests:
   - `trainer/tests/test_training_resume.py`
   - `trainer/tests/test_rtneural_golden_fixtures.py` if fixture behavior changes.
   - `app/src-tauri/src/lib.rs` Rust unit tests for preset allowlist/runtime
     estimates.
   - `app/src/test/tauri-ui-smoke.test.tsx` if the UI flow changes.

7. Docs:
   - `README.md`
   - `docs/Implementation-Guide-RTNeural-Training-Desktop-App.md`
   - Relevant research notes in `docs/`, especially WaveNet/capture docs.

Minimum verification for preset changes:

```bash
cargo test --manifest-path app/src-tauri/Cargo.toml
UV_CACHE_DIR=.uv-cache uv run --project trainer pytest trainer/tests/test_training_resume.py trainer/tests/test_rtneural_golden_fixtures.py -q
pnpm --dir app build
git diff --check
```

If the desktop app reports `Unknown model preset`, check Rust `MODEL_PRESETS`
first. That error is emitted by Rust before Python training can proceed.

## Dev Sidecars

In dev, the app should prefer local trainer source via `uv` when
`trainer/rttrainer` exists. If behavior seems stale, inspect the Progress panel
for the trainer launch mode:

- `using local rttrainer source via uv ...`
- `using external Python ...`
- `using bundled rttrainer sidecar`

The wrong launch mode can make the UI, Rust, and Python appear out of sync.

## Local Reference Repos

The user has cloned RTNeural-related reference repos locally under:

```text
/Users/shortwavlabs/Workspace/rt-neural
```

Use those local clones before web lookups when inspecting RTNeural, the example
plugin, or benchmark behavior.

