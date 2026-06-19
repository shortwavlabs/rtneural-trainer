export type ProjectStatus = "draft" | "ready" | "training" | "exported";
export type TargetKind = "amp" | "pedal" | "line" | "generic";
export type AudioStatus = "missing" | "warning" | "ready";
export type RunStatus = "running" | "completed" | "failed" | "interrupted";
export type ExportStatus = "blocked" | "ready";

export interface AppStatus {
  version: string;
  data_dir: string;
  trainer_sidecar_present: boolean;
  validator_sidecar_present: boolean;
}

export interface ProjectSummary {
  id: string;
  name: string;
  target_kind: TargetKind;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
  audio_status: AudioStatus;
  best_quality: number | null;
  export_status: ExportStatus | null;
}

export interface AudioFileReport {
  sample_rate: number;
  channels: number;
  duration_seconds: number;
  peak_dbfs: number;
  rms_dbfs: number;
  clipped_samples: number;
  dc_offset: number;
  path: string;
}

export interface AudioReport {
  input: AudioFileReport;
  target: AudioFileReport;
  latency_samples: number;
  latency_confidence: number;
  warnings: string[];
  status: AudioStatus;
}

export interface TrainingMetrics {
  esr: number;
  mae: number;
  rmse: number;
  peak_residual: number;
  rms_residual: number;
  realtime_factor: number;
}

export interface TrainingRun {
  id: string;
  preset: string;
  status: RunStatus;
  device: string;
  epochs: number;
  created_at: string;
  updated_at: string;
  metrics: TrainingMetrics | null;
  log_path: string;
}

export interface ExportPackage {
  id: string;
  run_id: string;
  status: ExportStatus;
  created_at: string;
  model_path: string;
  package_path: string;
  validation_path: string;
  benchmark_path: string;
}

export interface ProjectDetail {
  id: string;
  name: string;
  target_kind: TargetKind;
  status: ProjectStatus;
  created_at: string;
  updated_at: string;
  notes: string;
  project_dir: string;
  audio: AudioReport | null;
  runs: TrainingRun[];
  exports: ExportPackage[];
}

export interface CreateProjectRequest {
  name: string;
  target_kind: TargetKind;
}

export interface UpdateAudioRequest {
  project_id: string;
  input_path: string;
  target_path: string;
}

export interface StartTrainingRequest {
  project_id: string;
  preset: string;
}

export interface ExportRunRequest {
  project_id: string;
  run_id: string;
}

export interface UpdateNotesRequest {
  project_id: string;
  notes: string;
}
