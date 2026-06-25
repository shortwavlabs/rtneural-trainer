export type ProjectStatus = "draft" | "ready" | "training" | "exported";
export type TargetKind = "amp" | "pedal" | "line" | "generic";
export type AudioStatus = "missing" | "warning" | "ready";
export type CaptureChannelPolicy = "mixdown" | "first" | "reject";
export type RunStatus =
  | "queued"
  | "preparing"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "interrupted";
export type ExportStatus = "blocked" | "pending" | "validating" | "failed" | "ready";

export interface AppStatus {
  version: string;
  data_dir: string;
  trainer_sidecar_present: boolean;
  validator_sidecar_present: boolean;
}

export type RuntimeBackend = "keras" | "pytorch";

export interface RuntimeSettings {
  selected_backend: RuntimeBackend;
  selected_device: string;
  external_python_path: string | null;
}

export interface UpdateRuntimeSettingsRequest {
  selected_backend: RuntimeBackend;
  selected_device: string;
  external_python_path: string | null;
}

export interface DeviceInspection extends Record<string, unknown> {
  schema_version: number;
  trainer_version: string;
  platform: string;
  python: string;
  cpu_available: boolean;
  selected_device: string;
  tensorflow_status: string;
  torch_status: string;
  cuda_available: boolean;
  mps_available: boolean;
  mps_built: boolean;
  package_versions?: Record<string, string>;
  tensorflow_version?: string;
  keras_version?: string;
  torch_version?: string;
  cuda_device_count?: number;
  cuda_devices?: string[];
  tensorflow_gpus?: string[];
  torch_selected_device?: string;
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
  latency_auto_samples?: number | null;
  manual_latency_adjustment_samples: number;
  latency_confidence: number;
  latency?: LatencyReport | null;
  warnings: string[];
  warning_details: AudioWarning[];
  prepared?: Record<string, unknown> | null;
  capture_profile?: Record<string, unknown> | null;
  gain?: Record<string, unknown> | null;
  options?: Record<string, unknown> | null;
  status: AudioStatus;
}

export interface LatencyReport {
  estimated_samples: number;
  auto_estimated_samples?: number | null;
  manual_adjustment_samples?: number | null;
  effective_samples?: number | null;
  confidence: number;
  method?: string | null;
  agreement?: number | null;
  search_radius_samples?: number | null;
  window_length_samples?: number | null;
  analysis_window_count?: number | null;
  score_margin?: number | null;
  candidates?: LatencyCandidate[];
}

export interface LatencyCandidate {
  samples: number;
  score?: number | null;
  feature_score?: number | null;
  signed_score?: number | null;
  preemphasis_score?: number | null;
  onset_score?: number | null;
  window_count?: number | null;
  vote_count?: number | null;
  agreement?: number | null;
}

export interface AudioWarning {
  code: string;
  severity: "info" | "warning" | "error" | string;
  message: string;
  detail: string;
  action: string;
}

export interface TrainingMetrics {
  esr: number;
  mae: number;
  rmse: number;
  peak_residual: number;
  rms_residual: number;
  realtime_factor: number;
  state_continuous_correlation?: number;
  correlation?: number;
}

export interface TrainingRun {
  id: string;
  preset: string;
  backend: RuntimeBackend | "unknown" | string;
  status: RunStatus;
  device: string;
  epochs: number;
  created_at: string;
  updated_at: string;
  metrics: TrainingMetrics | null;
  log_path: string;
}

export interface TrainingRecipe {
  id: string;
  name: string;
  model_preset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  sequence_length: number;
  max_windows: number;
  resample_training_windows: boolean;
  resample_interval_epochs: number;
  early_stopping_patience: number;
  early_stopping_min_delta: number;
  created_at: string;
  updated_at: string;
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
  export_dir: string;
  package_metadata: Record<string, unknown> | null;
  validation_report: Record<string, unknown> | null;
  benchmark_report: Record<string, unknown> | null;
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

export interface DeleteProjectRequest {
  project_id: string;
}

export interface RenameProjectRequest {
  project_id: string;
  name: string;
}

export interface UpdateAudioRequest {
  project_id: string;
  input_path: string;
  target_path: string;
  target_sample_rate: number;
  resample: boolean;
  channel_policy: CaptureChannelPolicy;
  known_latency_samples?: number | null;
}

export interface StartTrainingRequest {
  project_id: string;
  preset: string;
  resume_from_run_id?: string | null;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  sequence_length: number;
  early_stopping_patience: number;
  early_stopping_min_delta: number;
  max_windows: number;
  resample_training_windows: boolean;
  resample_interval_epochs: number;
}

export interface SaveTrainingRecipeRequest {
  id?: string | null;
  name: string;
  model_preset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  sequence_length: number;
  max_windows: number;
  resample_training_windows: boolean;
  resample_interval_epochs: number;
  early_stopping_patience: number;
  early_stopping_min_delta: number;
}

export interface DeleteTrainingRecipeRequest {
  id: string;
}

export interface UpdateAlignmentRequest {
  project_id: string;
  manual_latency_adjustment_samples: number;
}

export interface ExportRunRequest {
  project_id: string;
  run_id: string;
}

export interface ExportFolderRequest {
  project_id: string;
  export_id: string;
}

export interface RunControlRequest {
  project_id: string;
  run_id: string;
}

export interface RunPreviewRequest {
  project_id: string;
  run_id: string;
}

export interface RunPreviewArtifact {
  kind: "target" | "prediction" | "residual" | string;
  label: string;
  path: string;
  exists: boolean;
  size_bytes: number | null;
  sample_rate: number | null;
  duration_seconds: number | null;
  peak: number | null;
  peaks: number[];
  waveform: WaveformBin[];
}

export interface RunPreview {
  project_id: string;
  run_id: string;
  run_dir: string;
  report_path: string | null;
  report: Record<string, unknown> | null;
  artifacts: RunPreviewArtifact[];
}

export interface WaveformBin {
  min: number;
  max: number;
  peak: number;
}

export interface ProjectWaveformTrack {
  kind: "input" | "target" | string;
  label: string;
  path: string;
  sample_rate: number;
  duration_seconds: number;
  peak: number;
  waveform: WaveformBin[];
}

export interface ProjectWaveform {
  project_id: string;
  sample_rate: number;
  duration_seconds: number;
  input: ProjectWaveformTrack;
  target: ProjectWaveformTrack;
}

export interface ProjectWaveformRequest {
  project_id: string;
  bins?: number;
  window_samples?: number;
}

export interface UpdateNotesRequest {
  project_id: string;
  notes: string;
}

export interface SidecarProgressEvent {
  job_id: string;
  operation: string;
  stream: "stdout" | "stderr" | "system" | string;
  line: string;
  json: Record<string, unknown> | null;
  project_id: string | null;
  run_id: string | null;
  export_id: string | null;
  timestamp: string;
}
