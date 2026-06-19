import { invoke } from "@tauri-apps/api/core";
import type {
  AppStatus,
  CreateProjectRequest,
  ExportRunRequest,
  ProjectDetail,
  ProjectSummary,
  StartTrainingRequest,
  TargetKind,
  UpdateAudioRequest,
  UpdateNotesRequest,
} from "../types";

const STORAGE_KEY = "rtneural-trainer.mock-store.v1";

interface MockStore {
  projects: ProjectDetail[];
}

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (isTauri()) {
    return invoke<T>(command, args);
  }

  return mockInvoke<T>(command, args ?? {});
}

export const api = {
  appStatus: () => call<AppStatus>("app_status"),
  listProjects: () => call<ProjectSummary[]>("list_projects"),
  createProject: (payload: CreateProjectRequest) =>
    call<ProjectDetail>("create_project", { payload }),
  getProject: (projectId: string) =>
    call<ProjectDetail>("get_project", { project_id: projectId }),
  updateAudio: (payload: UpdateAudioRequest) =>
    call<ProjectDetail>("update_project_audio", { payload }),
  startTraining: (payload: StartTrainingRequest) =>
    call<ProjectDetail>("start_training", { payload }),
  exportRun: (payload: ExportRunRequest) =>
    call<ProjectDetail>("export_run", { payload }),
  updateNotes: (payload: UpdateNotesRequest) =>
    call<ProjectDetail>("update_notes", { payload }),
};

function loadStore(): MockStore {
  const fallback: MockStore = { projects: [] };
  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) return fallback;

  try {
    return JSON.parse(raw) as MockStore;
  } catch {
    return fallback;
  }
}

function saveStore(store: MockStore) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
}

function summarize(project: ProjectDetail): ProjectSummary {
  const best = project.runs
    .map((run) => run.metrics?.esr ?? null)
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b)[0];

  return {
    id: project.id,
    name: project.name,
    target_kind: project.target_kind,
    status: project.status,
    created_at: project.created_at,
    updated_at: project.updated_at,
    audio_status: project.audio?.status ?? "missing",
    best_quality: best ?? null,
    export_status: project.exports.length
      ? project.exports[project.exports.length - 1].status
      : null,
  };
}

async function mockInvoke<T>(
  command: string,
  args: Record<string, unknown>,
): Promise<T> {
  const store = loadStore();
  const now = new Date().toISOString();

  switch (command) {
    case "app_status":
      return {
        version: "0.1.0-web-preview",
        data_dir: "browser-localStorage",
        trainer_sidecar_present: false,
        validator_sidecar_present: false,
      } as T;

    case "list_projects":
      return store.projects.map(summarize) as T;

    case "create_project": {
      const payload = args.payload as { name: string; target_kind: TargetKind };
      const id = `project_${Date.now()}`;
      const project: ProjectDetail = {
        id,
        name: payload.name.trim() || "Untitled capture",
        target_kind: payload.target_kind,
        status: "draft",
        created_at: now,
        updated_at: now,
        notes: "",
        project_dir: `mock://${id}`,
        audio: null,
        runs: [],
        exports: [],
      };
      store.projects.unshift(project);
      saveStore(store);
      return project as T;
    }

    case "get_project": {
      const project = requireProject(store, args.project_id as string);
      return project as T;
    }

    case "update_project_audio": {
      const payload = args.payload as {
        project_id: string;
        input_path: string;
        target_path: string;
      };
      const project = requireProject(store, payload.project_id);
      const warnings = [
        payload.input_path ? null : "Input path is empty.",
        payload.target_path ? null : "Target path is empty.",
      ].filter((item): item is string => Boolean(item));

      project.audio = {
        input: mockAudioReport(payload.input_path || "input.wav", -1.1, -18.4),
        target: mockAudioReport(payload.target_path || "target.wav", -0.8, -15.8),
        latency_samples: 123,
        latency_confidence: warnings.length > 0 ? 0.2 : 0.94,
        warnings,
        status: warnings.length > 0 ? "warning" : "ready",
      };
      project.status = warnings.length > 0 ? "draft" : "ready";
      project.updated_at = now;
      saveStore(store);
      return project as T;
    }

    case "start_training": {
      const payload = args.payload as { project_id: string; preset: string };
      const project = requireProject(store, payload.project_id);
      const index = project.runs.length + 1;
      const quality = payload.preset.includes("heavy")
        ? 0.028
        : payload.preset.includes("standard")
          ? 0.044
          : 0.072;

      project.status = "training";
      project.runs.push({
        id: `run_${Date.now()}`,
        preset: payload.preset,
        status: "completed",
        device: "mock-cpu",
        epochs: 60,
        created_at: now,
        updated_at: now,
        metrics: {
          esr: quality,
          mae: quality / 2.8,
          rmse: quality / 1.8,
          peak_residual: quality * 2.6,
          rms_residual: quality / 2.1,
          realtime_factor: payload.preset.includes("heavy") ? 24 : 118,
        },
        log_path: `${project.project_dir}/runs/run_${index}/events.jsonl`,
      });
      project.status = "ready";
      project.updated_at = now;
      saveStore(store);
      return project as T;
    }

    case "export_run": {
      const payload = args.payload as { project_id: string; run_id: string };
      const project = requireProject(store, payload.project_id);
      const run = project.runs.find((item) => item.id === payload.run_id);
      if (!run) throw new Error("Run not found");
      const exportId = `export_${Date.now()}`;
      project.exports.push({
        id: exportId,
        run_id: run.id,
        status: "ready",
        created_at: now,
        model_path: `${project.project_dir}/exports/${exportId}/model.rtneural.json`,
        package_path: `${project.project_dir}/exports/${exportId}/package.json`,
        validation_path: `${project.project_dir}/exports/${exportId}/validation-report.json`,
        benchmark_path: `${project.project_dir}/exports/${exportId}/benchmark-report.json`,
      });
      project.status = "exported";
      project.updated_at = now;
      saveStore(store);
      return project as T;
    }

    case "update_notes": {
      const payload = args.payload as { project_id: string; notes: string };
      const project = requireProject(store, payload.project_id);
      project.notes = payload.notes;
      project.updated_at = now;
      saveStore(store);
      return project as T;
    }

    default:
      throw new Error(`Unknown mock command: ${command}`);
  }
}

function requireProject(store: MockStore, projectId: string) {
  const project = store.projects.find((item) => item.id === projectId);
  if (!project) throw new Error("Project not found");
  return project;
}

function mockAudioReport(path: string, peak: number, rms: number) {
  return {
    sample_rate: 48000,
    channels: 1,
    duration_seconds: 95.4,
    peak_dbfs: peak,
    rms_dbfs: rms,
    clipped_samples: 0,
    dc_offset: 0.0002,
    path,
  };
}
