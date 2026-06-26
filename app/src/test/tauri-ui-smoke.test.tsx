// @vitest-environment jsdom

import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import type {
  AppStatus,
  AudioReport,
  DeviceInspection,
  ExportPackage,
  ProjectDetail,
  ProjectSummary,
  ProjectWaveform,
  RuntimeSettings,
  RunPreview,
  SidecarProgressEvent,
  TargetKind,
  TrainingRecipe,
  TrainingRun,
  UpdateAudioRequest,
} from "../types";

const tauriMocks = vi.hoisted(() => ({
  convertFileSrc: vi.fn((path: string) => `asset://${path}`),
  invoke: vi.fn(),
  listen: vi.fn(async () => () => undefined),
  open: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: tauriMocks.convertFileSrc,
  invoke: tauriMocks.invoke,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: tauriMocks.listen,
}));

vi.mock("@tauri-apps/plugin-dialog", () => ({
  open: tauriMocks.open,
}));

type SmokeCommand = {
  command: string;
  args?: Record<string, unknown>;
};

type SmokeState = {
  commands: SmokeCommand[];
  details: Map<string, ProjectDetail>;
  openedExportIds: string[];
  projects: ProjectSummary[];
  runtimeSettings: RuntimeSettings;
};

const fixedDate = "2026-06-22T12:00:00.000Z";

beforeEach(() => {
  Object.defineProperty(window, "__TAURI_INTERNALS__", {
    configurable: true,
    value: {},
  });
  Object.defineProperty(window.HTMLMediaElement.prototype, "play", {
    configurable: true,
    value: vi.fn(async () => undefined),
  });
  tauriMocks.convertFileSrc.mockClear();
  tauriMocks.invoke.mockReset();
  tauriMocks.listen.mockClear();
  tauriMocks.open.mockReset();
});

afterEach(() => {
  cleanup();
  Reflect.deleteProperty(window, "__TAURI_INTERNALS__");
});

describe("Tauri UI smoke", () => {
  it("boots the empty state and creates a generated sample project", async () => {
    const user = userEvent.setup();
    const state = installTauriSmokeBackend([]);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Create a capture project" }))
      .toBeInTheDocument();
    expect(screen.getByText("No captures yet.")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /create sample project/i }));

    expect(await screen.findByRole("heading", { name: "Sample Amp" })).toBeInTheDocument();
    await user.click(tab("Capture"));
    expect(await screen.findByText("Capture Source")).toBeInTheDocument();
    expect(dryInput()).toHaveValue("/samples/sample-input.wav");
    expect(targetInput()).toHaveValue("/samples/sample-target.wav");
    expect(state.commands.some((item) => item.command === "create_sample_project")).toBe(true);
  });

  it("keeps capture WAV paths scoped to the selected project", async () => {
    const user = userEvent.setup();
    const lead = projectFixture({
      id: "project_lead",
      name: "Lead Amp",
      kind: "amp",
      inputPath: "/captures/lead/DI.wav",
      targetPath: "/captures/lead/LEAD.wav",
      latencySamples: 8,
      confidence: 0.42,
    });
    const drive = projectFixture({
      id: "project_drive",
      name: "Overdrive Pedal",
      kind: "pedal",
      inputPath: "/captures/drive/DI.wav",
      targetPath: "/captures/drive/DRIVE.wav",
      latencySamples: 3,
      confidence: 0.86,
      resample: true,
      targetSampleRate: 44_100,
      channelPolicy: "first",
    });
    installTauriSmokeBackend([lead, drive]);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Lead Amp" })).toBeInTheDocument();
    expect(dryInput()).toHaveValue("/captures/lead/DI.wav");
    expect(targetInput()).toHaveValue("/captures/lead/LEAD.wav");

    await user.click(screen.getByRole("button", { name: /Overdrive Pedal/ }));

    expect(await screen.findByRole("heading", { name: "Overdrive Pedal" })).toBeInTheDocument();
    await waitFor(() => {
      expect(dryInput()).toHaveValue("/captures/drive/DI.wav");
      expect(targetInput()).toHaveValue("/captures/drive/DRIVE.wav");
    });
    expect(screen.getByRole("checkbox", { name: /Resample prepared audio/i })).toBeChecked();
    expect(screen.getByLabelText("Prepared sample rate")).toHaveValue("44100");
    expect(screen.getByLabelText("Stereo and multichannel")).toHaveValue("first");
  });

  it("covers rename, delete, and runtime settings actions", async () => {
    const user = userEvent.setup();
    const lead = projectFixture({
      id: "project_lead",
      name: "Lead Amp",
      kind: "amp",
      inputPath: "/captures/lead/DI.wav",
      targetPath: "/captures/lead/LEAD.wav",
    });
    const drive = projectFixture({
      id: "project_drive",
      name: "Overdrive Pedal",
      kind: "pedal",
      inputPath: "/captures/drive/DI.wav",
      targetPath: "/captures/drive/DRIVE.wav",
    });
    const state = installTauriSmokeBackend([lead, drive]);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Lead Amp" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Rename project" }));
    await user.clear(screen.getByLabelText("Project name"));
    await user.type(screen.getByLabelText("Project name"), "Lead Amp Revised");
    await user.click(screen.getByRole("button", { name: "Save name" }));

    expect(await screen.findByRole("heading", { name: "Lead Amp Revised" }))
      .toBeInTheDocument();
    expect(state.details.get("project_lead")?.name).toBe("Lead Amp Revised");

    await user.type(screen.getByLabelText("External Python"), "/usr/bin/python3");
    await user.click(screen.getByRole("button", { name: "Save runtime" }));
    await waitFor(() => {
      expect(state.runtimeSettings.external_python_path).toBe("/usr/bin/python3");
    });

    await user.click(screen.getByRole("button", { name: "Delete project" }));
    await user.click(screen.getByRole("button", { name: "Confirm delete" }));

    expect(await screen.findByRole("heading", { name: "Overdrive Pedal" })).toBeInTheDocument();
    expect(state.projects.map((item) => item.id)).toEqual(["project_drive"]);
  });

  it("covers Capture, Align, Train, Evaluate, and Export surfaces", async () => {
    const user = userEvent.setup();
    const project = projectFixture({
      id: "project_rhythm",
      name: "Rhythm Amp",
      kind: "amp",
      inputPath: "/captures/rhythm/DI.wav",
      targetPath: "/captures/rhythm/RHYTHM.wav",
      latencySamples: 10,
      confidence: 0.61,
    });
    const state = installTauriSmokeBackend([project]);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Rhythm Amp" })).toBeInTheDocument();
    expect(screen.getByText("Capture Source")).toBeInTheDocument();

    await user.click(tab("Align"));
    expect(await screen.findByText("Latency Alignment")).toBeInTheDocument();
    expect(await screen.findByText("Window agreement")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /10 samples/ })).toBeInTheDocument();
    expect(screen.getByText("4,096 samples")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Zoom waveform in" }));
    expect(await screen.findByText("2,048 samples")).toBeInTheDocument();
    await waitFor(() => {
      expect(
        state.commands.some(
          (item) =>
            item.command === "get_project_waveform" &&
            (item.args?.payload as { window_samples?: number } | undefined)?.window_samples ===
              2048,
        ),
      ).toBe(true);
    });
    await user.click(screen.getByRole("button", { name: "Zoom waveform out" }));
    expect(await screen.findByText("4,096 samples")).toBeInTheDocument();
    await waitFor(() => {
      expect(document.querySelector(".alignment-wave-track.target .wave-bars")).toBeTruthy();
    });
    const targetBars = document.querySelector(".alignment-wave-track.target .wave-bars");
    expect(targetBars).not.toHaveAttribute("transform");
    fireEvent.change(screen.getByRole("slider"), { target: { value: "42" } });
    expect(targetBars).toHaveAttribute("transform", expect.stringMatching(/^translate\(-/));

    await user.click(tab("Train"));
    expect(await screen.findByText("Training Setup")).toBeInTheDocument();
    await user.click(lastButtonNamed("Train"));
    await waitFor(() => {
      expect(state.details.get("project_rhythm")?.runs).toHaveLength(1);
    });
    expect(screen.getByText("completed")).toBeInTheDocument();

    await user.click(tab("Evaluate"));
    expect(await screen.findByText("Prediction Quality")).toBeInTheDocument();
    expect(await screen.findByText("Training Report")).toBeInTheDocument();
    expect(screen.getAllByText("Target").length).toBeGreaterThan(0);
    expect(state.commands.some((item) => item.command === "get_run_preview")).toBe(true);

    await user.click(tab("Export"));
    expect(await screen.findByText("Export Gate")).toBeInTheDocument();
    expect(screen.getByLabelText("Training run")).toHaveValue("run_smoke");
    await user.click(screen.getByRole("button", { name: "Export selected RTNeural JSON" }));
    expect(await screen.findByText("export_smoke")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Open" }));

    expect(state.openedExportIds).toEqual(["export_smoke"]);
  });

  it("exports the training run selected in the export tab", async () => {
    const user = userEvent.setup();
    const project = projectFixture({
      id: "project_export_choice",
      name: "Export Choice Amp",
      kind: "amp",
      inputPath: "/captures/export-choice/DI.wav",
      targetPath: "/captures/export-choice/TARGET.wav",
    });
    const qualityRun = completedRunFixture("run_quality", "wavenet_tcn_quality", 120);
    const balancedRun = completedRunFixture("run_balanced", "wavenet_tcn_balanced", 80);
    qualityRun.metrics!.esr = 0.101;
    balancedRun.metrics!.esr = 0.208;
    balancedRun.created_at = "2026-06-23T12:00:00.000Z";
    project.runs = [qualityRun, balancedRun];
    const state = installTauriSmokeBackend([project]);

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Export Choice Amp" }))
      .toBeInTheDocument();
    await user.click(tab("Export"));

    const runSelect = await screen.findByLabelText("Training run");
    expect(runSelect).toHaveValue("run_quality");
    expect(screen.getAllByText("Recommended").length).toBeGreaterThan(0);

    await user.selectOptions(runSelect, "run_balanced");
    await user.click(screen.getByRole("button", { name: "Export selected RTNeural JSON" }));

    await waitFor(() => {
      expect(
        state.commands.some(
          (item) =>
            item.command === "export_run" &&
            (item.args?.payload as { run_id?: string } | undefined)?.run_id === "run_balanced",
        ),
      ).toBe(true);
    });
  });
});

function installTauriSmokeBackend(projects: ProjectDetail[]): SmokeState {
  const state: SmokeState = {
    commands: [],
    details: new Map(projects.map((project) => [project.id, clone(project)])),
    openedExportIds: [],
    projects: projects.map(projectSummary),
    runtimeSettings: {
      external_python_path: null,
      selected_backend: "keras",
      selected_device: "tensorflow-cpu",
    },
  };

  tauriMocks.invoke.mockImplementation(async (command: string, args?: Record<string, unknown>) => {
    state.commands.push({ command, args });
    switch (command) {
      case "app_status":
        return clone(appStatusFixture());
      case "get_runtime_settings":
        return clone(state.runtimeSettings);
      case "update_runtime_settings":
        state.runtimeSettings = clone(args?.payload as RuntimeSettings);
        return clone(state.runtimeSettings);
      case "inspect_device":
        return clone(deviceInspectionFixture());
      case "list_projects":
        return clone(state.projects);
      case "list_training_recipes":
        return clone(trainingRecipeFixtures());
      case "list_project_events":
        return [] satisfies SidecarProgressEvent[];
      case "get_project":
        return clone(requireProject(state, String(args?.projectId)));
      case "get_project_waveform": {
        const payload = args?.payload as { project_id: string; window_samples?: number };
        return clone(projectWaveformFixture(payload.project_id, payload.window_samples));
      }
      case "create_sample_project":
        return clone(addProject(state, sampleProjectFixture()));
      case "create_project": {
        const payload = args?.payload as { name: string; target_kind: TargetKind };
        return clone(
          addProject(
            state,
            projectFixture({
              id: `project_${state.projects.length + 1}`,
              inputPath: "",
              kind: payload.target_kind,
              name: payload.name,
              targetPath: "",
              withAudio: false,
            }),
          ),
        );
      }
      case "rename_project": {
        const payload = args?.payload as { project_id: string; name: string };
        const project = requireProject(state, payload.project_id);
        project.name = payload.name;
        project.updated_at = fixedDate;
        state.projects = state.projects.map((summary) =>
          summary.id === project.id ? projectSummary(project) : summary,
        );
        return clone(project);
      }
      case "delete_project": {
        const payload = args?.payload as { project_id: string };
        state.details.delete(payload.project_id);
        state.projects = state.projects.filter((summary) => summary.id !== payload.project_id);
        return clone(state.projects);
      }
      case "update_project_audio": {
        const payload = args?.payload as UpdateAudioRequest;
        const project = requireProject(state, payload.project_id);
        project.audio = audioReportFixture({
          inputPath: payload.input_path,
          targetPath: payload.target_path,
          resample: payload.resample,
          targetSampleRate: payload.target_sample_rate,
          channelPolicy: payload.channel_policy,
        });
        project.status = "ready";
        state.projects = state.projects.map((summary) =>
          summary.id === project.id ? projectSummary(project) : summary,
        );
        return clone(project);
      }
      case "update_project_alignment": {
        const payload = args?.payload as {
          project_id: string;
          manual_latency_adjustment_samples: number;
        };
        const project = requireProject(state, payload.project_id);
        if (project.audio) {
          project.audio.manual_latency_adjustment_samples =
            payload.manual_latency_adjustment_samples;
          project.audio.latency_samples =
            (project.audio.latency_auto_samples ?? project.audio.latency_samples) +
            payload.manual_latency_adjustment_samples;
        }
        return clone(project);
      }
      case "start_training": {
        const payload = args?.payload as { project_id: string; preset: string; epochs: number };
        const project = requireProject(state, payload.project_id);
        project.runs.push(completedRunFixture("run_smoke", payload.preset, payload.epochs));
        state.projects = state.projects.map((summary) =>
          summary.id === project.id ? projectSummary(project) : summary,
        );
        return clone(project);
      }
      case "cancel_training_run":
      case "resume_training_run":
        return clone(requireProject(state, String((args?.payload as { project_id: string }).project_id)));
      case "get_run_preview": {
        const payload = args?.payload as { project_id: string; run_id: string };
        return clone(runPreviewFixture(payload.project_id, payload.run_id));
      }
      case "export_run": {
        const payload = args?.payload as { project_id: string; run_id: string };
        const project = requireProject(state, payload.project_id);
        project.exports.push(exportPackageFixture(payload.run_id));
        state.projects = state.projects.map((summary) =>
          summary.id === project.id ? projectSummary(project) : summary,
        );
        return clone(project);
      }
      case "open_export_folder": {
        const payload = args?.payload as { export_id: string };
        state.openedExportIds.push(payload.export_id);
        return undefined;
      }
      case "update_notes": {
        const payload = args?.payload as { project_id: string; notes: string };
        const project = requireProject(state, payload.project_id);
        project.notes = payload.notes;
        return clone(project);
      }
      case "save_training_recipe":
        return clone((args?.payload as TrainingRecipe) ?? trainingRecipeFixtures()[0]);
      case "delete_training_recipe":
        return [] satisfies TrainingRecipe[];
      default:
        throw new Error(`Unhandled Tauri command in smoke test: ${command}`);
    }
  });

  return state;
}

function addProject(state: SmokeState, project: ProjectDetail): ProjectDetail {
  state.details.set(project.id, clone(project));
  state.projects = [...state.projects, projectSummary(project)];
  return requireProject(state, project.id);
}

function requireProject(state: SmokeState, projectId: string): ProjectDetail {
  const project = state.details.get(projectId);
  if (!project) throw new Error(`Missing smoke project: ${projectId}`);
  return project;
}

function projectFixture({
  id,
  inputPath,
  kind,
  name,
  targetPath,
  channelPolicy = "mixdown",
  confidence = 0.9,
  latencySamples = 0,
  resample = false,
  targetSampleRate = 48_000,
  withAudio = true,
}: {
  id: string;
  inputPath: string;
  kind: TargetKind;
  name: string;
  targetPath: string;
  channelPolicy?: "mixdown" | "first" | "reject";
  confidence?: number;
  latencySamples?: number;
  resample?: boolean;
  targetSampleRate?: number;
  withAudio?: boolean;
}): ProjectDetail {
  return {
    audio: withAudio
      ? audioReportFixture({
          channelPolicy,
          confidence,
          inputPath,
          latencySamples,
          resample,
          targetPath,
          targetSampleRate,
        })
      : null,
    created_at: fixedDate,
    exports: [],
    id,
    name,
    notes: "",
    project_dir: `/projects/${id}`,
    runs: [],
    status: withAudio ? "ready" : "draft",
    target_kind: kind,
    updated_at: fixedDate,
  };
}

function sampleProjectFixture(): ProjectDetail {
  return projectFixture({
    id: "project_sample",
    inputPath: "/samples/sample-input.wav",
    kind: "amp",
    name: "Sample Amp",
    targetPath: "/samples/sample-target.wav",
  });
}

function audioReportFixture({
  inputPath,
  targetPath,
  channelPolicy = "mixdown",
  confidence = 0.9,
  latencySamples = 0,
  resample = false,
  targetSampleRate = 48_000,
}: {
  inputPath: string;
  targetPath: string;
  channelPolicy?: "mixdown" | "first" | "reject";
  confidence?: number;
  latencySamples?: number;
  resample?: boolean;
  targetSampleRate?: number;
}): AudioReport {
  return {
    capture_profile: {
      duration_seconds: 120,
      handling: "sampled_windows",
      recommended_max_windows: 2048,
    },
    gain: {
      guidance: "Levels are usable.",
      headroom_db: 5.2,
      rms_delta_db: 7.4,
      verdict: "usable",
    },
    input: audioFileFixture(inputPath),
    latency: {
      agreement: confidence > 0.7 ? 0.83 : 0.42,
      auto_estimated_samples: latencySamples,
      candidates: [
        {
          agreement: confidence > 0.7 ? 0.83 : 0.42,
          onset_score: 0.72,
          samples: latencySamples,
          score: confidence,
          vote_count: confidence > 0.7 ? 10 : 5,
          window_count: 12,
        },
        {
          agreement: 0.25,
          samples: latencySamples + 8,
          score: Math.max(0, confidence - 0.05),
          vote_count: 3,
          window_count: 12,
        },
      ],
      confidence,
      effective_samples: latencySamples,
      estimated_samples: latencySamples,
      manual_adjustment_samples: 0,
      method: "active_window_correlation",
      score_margin: confidence > 0.7 ? 0.08 : 0.01,
    },
    latency_auto_samples: latencySamples,
    latency_confidence: confidence,
    latency_samples: latencySamples,
    manual_latency_adjustment_samples: 0,
    options: {
      channel_policy: channelPolicy,
      resample,
      target_sample_rate: targetSampleRate,
    },
    prepared: {
      channel_policy: channelPolicy,
      duration_seconds: 120,
      input_path: `${inputPath}.prepared`,
      resampled: resample,
      sample_rate: targetSampleRate,
      samples: targetSampleRate * 120,
      target_path: `${targetPath}.prepared`,
    },
    status: "ready",
    target: audioFileFixture(targetPath),
    warning_details:
      confidence < 0.65
        ? [
            {
              action: "Audition the detected candidates before long training runs.",
              code: "latency_estimate_review",
              detail: `Best candidate is ${latencySamples} samples with ${confidence.toFixed(
                2,
              )} confidence; top candidates include ${latencySamples} samples, ${
                latencySamples + 8
              } samples.`,
              message: "Latency estimate should be reviewed.",
              severity: "info",
            },
          ]
        : [],
    warnings: [],
  };
}

function audioFileFixture(path: string) {
  return {
    channels: 1,
    clipped_samples: 0,
    dc_offset: 0,
    duration_seconds: 120,
    path,
    peak_dbfs: -5.2,
    rms_dbfs: -18.4,
    sample_rate: 48_000,
  };
}

function projectSummary(project: ProjectDetail): ProjectSummary {
  return {
    audio_status: project.audio?.status ?? "missing",
    best_quality:
      project.runs
        .map((run) => run.metrics?.esr ?? null)
        .filter((value): value is number => value !== null)
        .sort((left, right) => left - right)[0] ?? null,
    created_at: project.created_at,
    export_status: project.exports[project.exports.length - 1]?.status ?? null,
    id: project.id,
    name: project.name,
    status: project.status,
    target_kind: project.target_kind,
    updated_at: project.updated_at,
  };
}

function completedRunFixture(id: string, preset = "wavenet_tcn_balanced", epochs = 2): TrainingRun {
  return {
    backend: "keras",
    created_at: fixedDate,
    device: "tensorflow-cpu",
    epochs,
    id,
    log_path: `/runs/${id}/events.jsonl`,
    metrics: {
      esr: 0.123,
      mae: 0.045,
      peak_residual: 0.32,
      realtime_factor: 3.2,
      rmse: 0.067,
      rms_residual: 0.061,
      state_continuous_correlation: 0.98,
    },
    preset,
    status: "completed",
    updated_at: fixedDate,
  };
}

function exportPackageFixture(runId: string): ExportPackage {
  return {
    benchmark_path: "/exports/export_smoke/native-benchmark-report.json",
    benchmark_report: {
      realtime_factor: 3.2,
      status: "pass",
      worst_case_realtime_factor: 2.4,
    },
    created_at: fixedDate,
    export_dir: "/exports/export_smoke",
    id: "export_smoke",
    model_path: "/exports/export_smoke/model.rtneural.json",
    package_metadata: {
      backend: "keras",
      preset: "wavenet_tcn_balanced",
      schema_version: 2,
    },
    package_path: "/exports/export_smoke/package.json",
    run_id: runId,
    status: "ready",
    validation_path: "/exports/export_smoke/native-validation-report.json",
    validation_report: {
      max_abs_error: 0.0001,
      status: "pass",
    },
  };
}

function runPreviewFixture(projectId: string, runId: string): RunPreview {
  return {
    artifacts: ["target", "prediction", "residual"].map((kind) => ({
      duration_seconds: 3,
      exists: true,
      kind,
      label: titleCase(kind),
      path: `/runs/${runId}/previews/${kind}.wav`,
      peak: 0.8,
      peaks: [0.2, 0.4, 0.6, 0.5],
      sample_rate: 48_000,
      size_bytes: 128_000,
      waveform: waveformBins(),
    })),
    project_id: projectId,
    report: {
      backend: "keras",
      created_at: fixedDate,
      early_stopping: {
        best_epoch: 2,
        stopped: false,
      },
      epochs: 2,
      quality_assessment: {
        action: "Export if native benchmark margin is acceptable.",
        summary: "Prediction is close to target.",
        verdict: "good",
      },
    },
    report_path: `/runs/${runId}/training-report.json`,
    run_dir: `/runs/${runId}`,
    run_id: runId,
  };
}

function projectWaveformFixture(projectId: string, windowSamples = 4096): ProjectWaveform {
  const sampleRate = 48_000;
  const durationSeconds = windowSamples / sampleRate;
  return {
    duration_seconds: durationSeconds,
    input: waveformTrackFixture("input", "Dry input", durationSeconds),
    project_id: projectId,
    sample_rate: sampleRate,
    target: waveformTrackFixture("target", "Processed target", durationSeconds),
  };
}

function waveformTrackFixture(kind: "input" | "target", label: string, durationSeconds: number) {
  return {
    duration_seconds: durationSeconds,
    kind,
    label,
    path: `/waveforms/${kind}.wav`,
    peak: 0.8,
    sample_rate: 48_000,
    waveform: waveformBins(),
  };
}

function waveformBins() {
  return Array.from({ length: 12 }, (_, index) => ({
    max: Math.sin(index) * 0.4 + 0.5,
    min: -0.4,
    peak: 0.6,
  }));
}

function appStatusFixture(): AppStatus {
  return {
    data_dir: "/tmp/rtneural-trainer-smoke",
    trainer_sidecar_present: true,
    validator_sidecar_present: true,
    version: "0.1.0",
  };
}

function deviceInspectionFixture(): DeviceInspection {
  return {
    cpu_available: true,
    cuda_available: false,
    keras_version: "3.14.1",
    mps_available: true,
    mps_built: true,
    package_versions: {
      keras: "3.14.1",
      python: "3.12.13",
      rttrainer: "0.1.0",
      tensorflow: "2.21.0",
    },
    platform: "darwin",
    python: "3.12.13",
    schema_version: 1,
    selected_device: "tensorflow-cpu",
    tensorflow_gpus: [],
    tensorflow_status: "available",
    tensorflow_version: "2.21.0",
    trainer_version: "0.1.0",
  };
}

function trainingRecipeFixtures(): TrainingRecipe[] {
  return [];
}

function dryInput() {
  return screen.getByPlaceholderText("/path/to/input.wav");
}

function targetInput() {
  return screen.getByPlaceholderText("/path/to/target.wav");
}

function tab(name: string) {
  return screen.getByRole("tab", { name });
}

function lastButtonNamed(name: string) {
  const buttons = screen.getAllByRole("button", { name });
  const button = buttons[buttons.length - 1];
  if (!button) throw new Error(`No button named ${name}`);
  return button;
}

function titleCase(value: string) {
  return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
