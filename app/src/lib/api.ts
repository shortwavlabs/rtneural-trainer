import { invoke } from "@tauri-apps/api/core";
import type {
  AppStatus,
  CreateProjectRequest,
  DeviceInspection,
  ExportFolderRequest,
  ExportRunRequest,
  ProjectDetail,
  ProjectSummary,
  RunControlRequest,
  RunPreview,
  RunPreviewRequest,
  RuntimeSettings,
  SidecarProgressEvent,
  StartTrainingRequest,
  UpdateAlignmentRequest,
  UpdateAudioRequest,
  UpdateNotesRequest,
  UpdateRuntimeSettingsRequest,
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
  getRuntimeSettings: () => call<RuntimeSettings>("get_runtime_settings"),
  updateRuntimeSettings: (payload: UpdateRuntimeSettingsRequest) =>
    call<RuntimeSettings>("update_runtime_settings", { payload }),
  inspectDevice: () => call<DeviceInspection>("inspect_device"),
  listProjects: () => call<ProjectSummary[]>("list_projects"),
  listProjectEvents: (projectId: string) =>
    call<SidecarProgressEvent[]>("list_project_events", { project_id: projectId }),
  getRunPreview: (payload: RunPreviewRequest) =>
    call<RunPreview>("get_run_preview", { payload }),
  createProject: (payload: CreateProjectRequest) =>
    call<ProjectDetail>("create_project", { payload }),
  getProject: (projectId: string) =>
    call<ProjectDetail>("get_project", { project_id: projectId }),
  updateAudio: (payload: UpdateAudioRequest) =>
    call<ProjectDetail>("update_project_audio", { payload }),
  updateAlignment: (payload: UpdateAlignmentRequest) =>
    call<ProjectDetail>("update_project_alignment", { payload }),
  startTraining: (payload: StartTrainingRequest) =>
    call<ProjectDetail>("start_training", { payload }),
  cancelTrainingRun: (payload: RunControlRequest) =>
    call<ProjectDetail>("cancel_training_run", { payload }),
  resumeTrainingRun: (payload: RunControlRequest) =>
    call<ProjectDetail>("resume_training_run", { payload }),
  exportRun: (payload: ExportRunRequest) =>
    call<ProjectDetail>("export_run", { payload }),
  openExportFolder: (payload: ExportFolderRequest) =>
    call<void>("open_export_folder", { payload }),
  updateNotes: (payload: UpdateNotesRequest) =>
    call<ProjectDetail>("update_notes", { payload }),
};
