import { invoke } from "@tauri-apps/api/core";
import type {
  AppStatus,
  CreateProjectRequest,
  ExportRunRequest,
  ProjectDetail,
  ProjectSummary,
  RunControlRequest,
  SidecarProgressEvent,
  StartTrainingRequest,
  UpdateAudioRequest,
  UpdateNotesRequest,
} from "../types";

const isTauri = () =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isTauri()) {
    throw new Error("RTNeural Trainer must run inside the Tauri app runtime.");
  }

  return invoke<T>(command, args);
}

export const api = {
  appStatus: () => call<AppStatus>("app_status"),
  listProjects: () => call<ProjectSummary[]>("list_projects"),
  listProjectEvents: (projectId: string) =>
    call<SidecarProgressEvent[]>("list_project_events", { project_id: projectId }),
  createProject: (payload: CreateProjectRequest) =>
    call<ProjectDetail>("create_project", { payload }),
  getProject: (projectId: string) =>
    call<ProjectDetail>("get_project", { project_id: projectId }),
  updateAudio: (payload: UpdateAudioRequest) =>
    call<ProjectDetail>("update_project_audio", { payload }),
  startTraining: (payload: StartTrainingRequest) =>
    call<ProjectDetail>("start_training", { payload }),
  cancelTrainingRun: (payload: RunControlRequest) =>
    call<ProjectDetail>("cancel_training_run", { payload }),
  resumeTrainingRun: (payload: RunControlRequest) =>
    call<ProjectDetail>("resume_training_run", { payload }),
  exportRun: (payload: ExportRunRequest) =>
    call<ProjectDetail>("export_run", { payload }),
  updateNotes: (payload: UpdateNotesRequest) =>
    call<ProjectDetail>("update_notes", { payload }),
};
