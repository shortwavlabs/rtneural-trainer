use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt, fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command as StdCommand, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};
use tauri::{Emitter, Manager};
use tauri_plugin_shell::{
    process::{Command as ShellCommand, CommandEvent},
    ShellExt,
};
use uuid::Uuid;

const RTTRAINER_SIDECAR: &str = "rttrainer";
const RTNEURAL_VALIDATOR_SIDECAR: &str = "rtneural-validator";
const RUNTIME_SETTINGS_KEY: &str = "runtime";
const MODEL_PRESETS: &[&str] = &[
    "dense_only",
    "gru_light",
    "lstm_light",
    "lstm_standard",
    "conv1d_light",
    "conv1d_bn_prelu",
    "conv1d_stack_prelu",
    "wavenet_tcn",
    "conv_gru_hybrid",
];

#[derive(Clone, Serialize)]
struct AppStatus {
    version: String,
    data_dir: String,
    trainer_sidecar_present: bool,
    validator_sidecar_present: bool,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProjectStatus {
    Draft,
    Ready,
    Training,
    Exported,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum TargetKind {
    Amp,
    Pedal,
    Line,
    Generic,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AudioStatus {
    Missing,
    Warning,
    Ready,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RunStatus {
    Queued,
    Preparing,
    Running,
    Cancelling,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExportStatus {
    Blocked,
    Pending,
    Validating,
    Failed,
    Ready,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobStatus {
    Queued,
    Running,
    Cancelling,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Clone, Serialize, Deserialize)]
struct ProjectSummary {
    id: String,
    name: String,
    target_kind: TargetKind,
    status: ProjectStatus,
    created_at: String,
    updated_at: String,
    audio_status: AudioStatus,
    best_quality: Option<f64>,
    export_status: Option<ExportStatus>,
}

#[derive(Clone, Serialize, Deserialize)]
struct AudioFileReport {
    sample_rate: u32,
    channels: u16,
    duration_seconds: f64,
    peak_dbfs: f64,
    rms_dbfs: f64,
    clipped_samples: u32,
    dc_offset: f64,
    path: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct AudioReport {
    input: AudioFileReport,
    target: AudioFileReport,
    latency_samples: i32,
    #[serde(default)]
    latency_auto_samples: Option<i32>,
    #[serde(default)]
    manual_latency_adjustment_samples: i32,
    latency_confidence: f64,
    warnings: Vec<String>,
    #[serde(default)]
    warning_details: Vec<AudioWarning>,
    #[serde(default)]
    prepared: Option<serde_json::Value>,
    #[serde(default)]
    capture_profile: Option<serde_json::Value>,
    #[serde(default)]
    gain: Option<serde_json::Value>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    status: AudioStatus,
}

#[derive(Clone, Serialize, Deserialize)]
struct AudioWarning {
    code: String,
    severity: String,
    message: String,
    detail: String,
    action: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TrainingMetrics {
    esr: f64,
    mae: f64,
    rmse: f64,
    peak_residual: f64,
    rms_residual: f64,
    realtime_factor: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct TrainingRun {
    id: String,
    preset: String,
    #[serde(default = "default_training_backend")]
    backend: String,
    status: RunStatus,
    device: String,
    epochs: u32,
    created_at: String,
    updated_at: String,
    metrics: Option<TrainingMetrics>,
    log_path: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct TrainingRecipe {
    id: String,
    name: String,
    model_preset: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    sequence_length: u32,
    max_windows: u32,
    early_stopping_patience: u32,
    early_stopping_min_delta: f64,
    created_at: String,
    updated_at: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct ExportPackage {
    id: String,
    run_id: String,
    status: ExportStatus,
    created_at: String,
    model_path: String,
    package_path: String,
    validation_path: String,
    benchmark_path: String,
    #[serde(default)]
    export_dir: String,
    #[serde(default)]
    package_metadata: Option<serde_json::Value>,
    #[serde(default)]
    validation_report: Option<serde_json::Value>,
    #[serde(default)]
    benchmark_report: Option<serde_json::Value>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ProjectDetail {
    id: String,
    name: String,
    target_kind: TargetKind,
    status: ProjectStatus,
    created_at: String,
    updated_at: String,
    notes: String,
    project_dir: String,
    audio: Option<AudioReport>,
    runs: Vec<TrainingRun>,
    exports: Vec<ExportPackage>,
}

#[derive(Deserialize)]
struct CreateProjectRequest {
    name: String,
    target_kind: TargetKind,
}

#[derive(Deserialize)]
struct DeleteProjectRequest {
    project_id: String,
}

#[derive(Deserialize)]
struct RenameProjectRequest {
    project_id: String,
    name: String,
}

#[derive(Deserialize)]
struct UpdateAudioRequest {
    project_id: String,
    input_path: String,
    target_path: String,
    #[serde(default = "default_capture_sample_rate")]
    target_sample_rate: u32,
    #[serde(default)]
    resample: bool,
    #[serde(default = "default_channel_policy")]
    channel_policy: String,
}

#[derive(Deserialize)]
struct UpdateAlignmentRequest {
    project_id: String,
    manual_latency_adjustment_samples: i32,
}

#[derive(Deserialize)]
struct StartTrainingRequest {
    project_id: String,
    preset: String,
    #[serde(default)]
    resume_from_run_id: Option<String>,
    #[serde(default = "default_training_epochs")]
    epochs: u32,
    #[serde(default = "default_early_stopping_patience")]
    early_stopping_patience: u32,
    #[serde(default = "default_early_stopping_min_delta")]
    early_stopping_min_delta: f64,
    #[serde(default = "default_training_max_windows")]
    max_windows: u32,
    #[serde(default = "default_training_batch_size")]
    batch_size: u32,
    #[serde(default = "default_training_learning_rate")]
    learning_rate: f64,
    #[serde(default = "default_training_sequence_length")]
    sequence_length: u32,
}

#[derive(Deserialize)]
struct SaveTrainingRecipeRequest {
    #[serde(default)]
    id: Option<String>,
    name: String,
    model_preset: String,
    epochs: u32,
    batch_size: u32,
    learning_rate: f64,
    sequence_length: u32,
    max_windows: u32,
    early_stopping_patience: u32,
    early_stopping_min_delta: f64,
}

#[derive(Deserialize)]
struct DeleteTrainingRecipeRequest {
    id: String,
}

#[derive(Deserialize)]
struct RunControlRequest {
    project_id: String,
    run_id: String,
}

#[derive(Deserialize)]
struct RunPreviewRequest {
    project_id: String,
    run_id: String,
}

#[derive(Deserialize)]
struct ProjectWaveformRequest {
    project_id: String,
    #[serde(default = "default_waveform_bins")]
    bins: usize,
}

#[derive(Deserialize)]
struct ExportRunRequest {
    project_id: String,
    run_id: String,
}

#[derive(Deserialize)]
struct ExportFolderRequest {
    project_id: String,
    export_id: String,
}

#[derive(Deserialize)]
struct UpdateNotesRequest {
    project_id: String,
    notes: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct RuntimeSettings {
    selected_backend: String,
    #[serde(default = "default_runtime_device")]
    selected_device: String,
    external_python_path: Option<String>,
}

#[derive(Deserialize)]
struct UpdateRuntimeSettingsRequest {
    selected_backend: String,
    #[serde(default = "default_runtime_device")]
    selected_device: String,
    external_python_path: Option<String>,
}

#[derive(Default, Serialize, Deserialize)]
struct Store {
    projects: Vec<ProjectDetail>,
}

struct AppState {
    db_path: PathBuf,
    projects_dir: PathBuf,
    workspace_root: PathBuf,
    db: Mutex<Connection>,
    active_jobs: Mutex<HashMap<String, ActiveProcess>>,
}

struct SidecarOutput {
    stdout: String,
    stderr: String,
}

#[derive(Clone)]
struct ActiveProcess {
    pid: u32,
}

#[derive(Clone)]
struct SidecarContext {
    job_id: String,
    operation: String,
    project_id: Option<String>,
    run_id: Option<String>,
    export_id: Option<String>,
}

#[derive(Clone, Serialize)]
struct SidecarProgressEvent {
    job_id: String,
    operation: String,
    stream: String,
    line: String,
    json: Option<serde_json::Value>,
    project_id: Option<String>,
    run_id: Option<String>,
    export_id: Option<String>,
    timestamp: String,
}

#[derive(Serialize)]
struct RunPreview {
    project_id: String,
    run_id: String,
    run_dir: String,
    report_path: Option<String>,
    report: Option<serde_json::Value>,
    artifacts: Vec<RunPreviewArtifact>,
}

#[derive(Serialize)]
struct RunPreviewArtifact {
    kind: String,
    label: String,
    path: String,
    exists: bool,
    size_bytes: Option<u64>,
    sample_rate: Option<u32>,
    duration_seconds: Option<f64>,
    peak: Option<f64>,
    peaks: Vec<f64>,
    waveform: Vec<WaveformBin>,
}

#[derive(Serialize)]
struct ProjectWaveform {
    project_id: String,
    sample_rate: u32,
    duration_seconds: f64,
    input: WaveformTrack,
    target: WaveformTrack,
}

#[derive(Serialize)]
struct WaveformTrack {
    kind: String,
    label: String,
    path: String,
    sample_rate: u32,
    duration_seconds: f64,
    peak: f64,
    waveform: Vec<WaveformBin>,
}

#[derive(Clone, Copy, Serialize)]
struct WaveformBin {
    min: f64,
    max: f64,
    peak: f64,
}

struct WavPreviewSummary {
    sample_rate: u32,
    duration_seconds: f64,
    peak: f64,
    peaks: Vec<f64>,
    waveform: Vec<WaveformBin>,
}

struct FinalExportPackageInput<'a> {
    project_name: &'a str,
    project_id: &'a str,
    export_id: &'a str,
    run: &'a TrainingRun,
    sample_rate: u32,
    latency_samples: i32,
    export_dir: &'a Path,
    model_path: &'a Path,
    package_path: &'a Path,
    validation_path: &'a Path,
    benchmark_path: &'a Path,
}

#[derive(Deserialize)]
struct PrepareReport {
    input: AudioFileReport,
    target: AudioFileReport,
    #[serde(default)]
    prepared: Option<serde_json::Value>,
    #[serde(default)]
    capture_profile: Option<serde_json::Value>,
    #[serde(default)]
    gain: Option<serde_json::Value>,
    #[serde(default)]
    options: Option<serde_json::Value>,
    latency: LatencyReport,
    warnings: Vec<String>,
    #[serde(default)]
    warning_details: Vec<AudioWarning>,
    status: AudioStatus,
}

#[derive(Deserialize)]
struct LatencyReport {
    estimated_samples: i32,
    #[serde(default)]
    auto_estimated_samples: Option<i32>,
    #[serde(default)]
    manual_adjustment_samples: i32,
    #[serde(default)]
    effective_samples: Option<i32>,
    confidence: f64,
}

#[derive(Deserialize)]
struct TrainingReport {
    run_id: String,
    preset: String,
    #[serde(default = "default_training_backend")]
    backend: String,
    device: String,
    epochs: u32,
    metrics: TrainingMetrics,
    created_at: String,
}

#[tauri::command]
fn app_status(state: tauri::State<AppState>) -> AppStatus {
    AppStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_dir: state
            .db_path
            .parent()
            .unwrap_or(&state.projects_dir)
            .display()
            .to_string(),
        trainer_sidecar_present: bundled_sidecar_exists(RTTRAINER_SIDECAR),
        validator_sidecar_present: bundled_sidecar_exists(RTNEURAL_VALIDATOR_SIDECAR),
    }
}

#[tauri::command]
fn get_runtime_settings(state: tauri::State<AppState>) -> Result<RuntimeSettings, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_runtime_settings(&db)
}

#[tauri::command]
fn update_runtime_settings(
    state: tauri::State<AppState>,
    payload: UpdateRuntimeSettingsRequest,
) -> Result<RuntimeSettings, String> {
    let settings = RuntimeSettings {
        selected_backend: normalize_backend(&payload.selected_backend)?.to_string(),
        selected_device: normalize_runtime_device(&payload.selected_device)?.to_string(),
        external_python_path: payload
            .external_python_path
            .and_then(|path| non_empty_string(&path)),
    };
    let db = state.db.lock().map_err(lock_error)?;
    save_runtime_settings(&db, &settings)?;
    Ok(settings)
}

#[tauri::command]
fn list_training_recipes(state: tauri::State<AppState>) -> Result<Vec<TrainingRecipe>, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_training_recipes(&db)
}

#[tauri::command]
fn save_training_recipe(
    state: tauri::State<AppState>,
    payload: SaveTrainingRecipeRequest,
) -> Result<TrainingRecipe, String> {
    let db = state.db.lock().map_err(lock_error)?;
    let recipe = normalize_training_recipe(payload)?;
    upsert_training_recipe(&db, &recipe)?;
    Ok(recipe)
}

#[tauri::command]
fn delete_training_recipe(
    state: tauri::State<AppState>,
    payload: DeleteTrainingRecipeRequest,
) -> Result<Vec<TrainingRecipe>, String> {
    let recipe_id = non_empty_string(&payload.id)
        .ok_or_else(|| "Training recipe id is required.".to_string())?;
    let db = state.db.lock().map_err(lock_error)?;
    db.execute(
        "DELETE FROM training_recipes WHERE id = ?1",
        params![recipe_id],
    )
    .map_err(to_error)?;
    load_training_recipes(&db)
}

#[tauri::command]
async fn inspect_device(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    run_blocking_command("inspect-device", move || {
        let state = app.state::<AppState>();
        inspect_device_blocking(&app, state.inner())
    })
    .await
}

fn inspect_device_blocking(
    app: &tauri::AppHandle,
    state: &AppState,
) -> Result<serde_json::Value, String> {
    let context = SidecarContext {
        job_id: "runtime_inspection".to_string(),
        operation: "inspect_device".to_string(),
        project_id: None,
        run_id: None,
        export_id: None,
    };
    emit_sidecar_line(app, &context, "system", "rttrainer inspect-device started");
    let args = vec!["inspect-device".to_string(), "--json".to_string()];
    let output = run_rttrainer_args(app, state, "inspect-device", &args, context.clone())?;
    if !output.status.success() {
        emit_sidecar_line(app, &context, "system", "rttrainer inspect-device failed");
        return Err(format!(
            "rttrainer inspect-device failed with status {}.\nstdout:\n{}\nstderr:\n{}",
            output.status, output.stdout, output.stderr
        ));
    }
    emit_sidecar_line(app, &context, "system", "rttrainer inspect-device finished");
    parse_json_from_stdout(&output.stdout)
}

#[tauri::command]
fn list_projects(state: tauri::State<AppState>) -> Result<Vec<ProjectSummary>, String> {
    let db = state.db.lock().map_err(lock_error)?;
    let projects = load_all_projects(&db)?;
    Ok(projects.iter().map(summarize_project).collect())
}

#[tauri::command]
fn get_project(state: tauri::State<AppState>, project_id: String) -> Result<ProjectDetail, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_project_detail(&db, &project_id)
}

#[tauri::command]
fn delete_project(
    state: tauri::State<AppState>,
    payload: DeleteProjectRequest,
) -> Result<Vec<ProjectSummary>, String> {
    let db = state.db.lock().map_err(lock_error)?;
    delete_project_by_id(&db, &payload.project_id, &state.projects_dir)?;
    let projects = load_all_projects(&db)?;
    Ok(projects.iter().map(summarize_project).collect())
}

#[tauri::command]
fn rename_project(
    state: tauri::State<AppState>,
    payload: RenameProjectRequest,
) -> Result<ProjectDetail, String> {
    let db = state.db.lock().map_err(lock_error)?;
    ensure_no_active_project_job(&db, &payload.project_id)?;
    let name = normalize_project_name(&payload.name)?;
    update_project_name(&db, &payload.project_id, &name, &now())?;
    load_project_detail(&db, &payload.project_id)
}

#[tauri::command]
fn list_project_events(
    state: tauri::State<AppState>,
    project_id: String,
) -> Result<Vec<SidecarProgressEvent>, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_project_events(&db, &project_id, 120)
}

#[tauri::command]
fn get_run_preview(
    state: tauri::State<AppState>,
    payload: RunPreviewRequest,
) -> Result<RunPreview, String> {
    let (project_dir, run_dir) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        if !project.runs.iter().any(|run| run.id == payload.run_id) {
            return Err("Run not found".to_string());
        }
        let project_dir = PathBuf::from(&project.project_dir);
        let run_dir = artifact_path(&project_dir, &training_run_dir(&db, &payload.run_id)?);
        (project_dir, run_dir)
    };

    ensure_artifact_inside(&project_dir, &run_dir)?;
    let report_path = run_dir.join("training-report.json");
    let report = if report_path.exists() {
        Some(read_json::<serde_json::Value>(&report_path)?)
    } else {
        None
    };
    let preview_dir = run_dir.join("previews");
    let artifacts = [
        ("target", "Target", "target.wav"),
        ("prediction", "Prediction", "prediction.wav"),
        ("residual", "Residual", "residual.wav"),
        (
            "chunk_prediction",
            "Chunk-reset prediction",
            "chunk-reset-prediction.wav",
        ),
        (
            "chunk_residual",
            "Chunk-reset residual",
            "chunk-reset-residual.wav",
        ),
    ]
    .iter()
    .filter(|(_kind, _label, file_name)| {
        *file_name == "target.wav"
            || *file_name == "prediction.wav"
            || *file_name == "residual.wav"
            || preview_dir.join(*file_name).exists()
    })
    .map(|(kind, label, file_name)| {
        run_preview_artifact(kind, label, &preview_dir.join(*file_name))
    })
    .collect::<Result<Vec<_>, _>>()?;

    Ok(RunPreview {
        project_id: payload.project_id,
        run_id: payload.run_id,
        run_dir: run_dir.display().to_string(),
        report_path: report_path
            .exists()
            .then(|| report_path.display().to_string()),
        report,
        artifacts,
    })
}

#[tauri::command]
fn get_project_waveform(
    state: tauri::State<AppState>,
    payload: ProjectWaveformRequest,
) -> Result<ProjectWaveform, String> {
    let db = state.db.lock().map_err(lock_error)?;
    let project = load_project_detail(&db, &payload.project_id)?;
    let audio = project
        .audio
        .as_ref()
        .ok_or_else(|| "Prepare audio before loading waveforms.".to_string())?;
    let project_dir = PathBuf::from(&project.project_dir);
    let input_path = prepared_audio_path(
        &project_dir,
        audio.prepared.as_ref(),
        "input_path",
        "audio/prepared/input.wav",
    )?;
    let target_path = prepared_audio_path(
        &project_dir,
        audio.prepared.as_ref(),
        "target_path",
        "audio/prepared/target.wav",
    )?;
    ensure_artifact_inside(&project_dir, &input_path)?;
    ensure_artifact_inside(&project_dir, &target_path)?;

    let bins = payload.bins.clamp(64, 600);
    let input = waveform_track("input", "Dry input", &input_path, bins)?;
    let target = waveform_track("target", "Processed target", &target_path, bins)?;
    Ok(ProjectWaveform {
        project_id: payload.project_id,
        sample_rate: input.sample_rate,
        duration_seconds: input.duration_seconds.min(target.duration_seconds),
        input,
        target,
    })
}

#[tauri::command]
fn create_project(
    state: tauri::State<AppState>,
    payload: CreateProjectRequest,
) -> Result<ProjectDetail, String> {
    create_project_record(state.inner(), payload.name.trim(), payload.target_kind, "")
}

#[tauri::command]
fn create_sample_project(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
) -> Result<ProjectDetail, String> {
    let notes = "Generated sample project. The dry input is a short composite tone and the target adds gain, soft saturation, and filtering so the full prepare/train/export workflow can be tested without external audio.";
    let project =
        create_project_record(state.inner(), "Sample amp capture", TargetKind::Amp, notes)?;
    let project_dir = PathBuf::from(&project.project_dir);
    let original_dir = project_dir.join("audio/original");
    let input_path = original_dir.join("sample-input.wav");
    let target_path = original_dir.join("sample-target.wav");
    let (input, target) = generated_sample_capture(48_000, 4.0);
    write_pcm16_wav(&input_path, &input, 48_000)?;
    write_pcm16_wav(&target_path, &target, 48_000)?;

    let payload = UpdateAudioRequest {
        project_id: project.id.clone(),
        input_path: input_path.display().to_string(),
        target_path: target_path.display().to_string(),
        target_sample_rate: 48_000,
        resample: false,
        channel_policy: "mixdown".to_string(),
    };
    prepare_project_audio(&app, state.inner(), &payload)
}

fn create_project_record(
    state: &AppState,
    name: &str,
    target_kind: TargetKind,
    notes: &str,
) -> Result<ProjectDetail, String> {
    let id = format!("project_{}", Uuid::new_v4().simple());
    let run_created_at = now();
    let project_dir = state.projects_dir.join(&id);
    fs::create_dir_all(project_dir.join("audio/original")).map_err(to_error)?;
    fs::create_dir_all(project_dir.join("audio/prepared")).map_err(to_error)?;
    fs::create_dir_all(project_dir.join("runs")).map_err(to_error)?;
    fs::create_dir_all(project_dir.join("exports")).map_err(to_error)?;

    let project = ProjectDetail {
        id,
        name: if name.trim().is_empty() {
            "Untitled capture".to_string()
        } else {
            name.trim().to_string()
        },
        target_kind,
        status: ProjectStatus::Draft,
        created_at: run_created_at.clone(),
        updated_at: run_created_at,
        notes: notes.to_string(),
        project_dir: project_dir.display().to_string(),
        audio: None,
        runs: Vec::new(),
        exports: Vec::new(),
    };

    let db = state.db.lock().map_err(lock_error)?;
    insert_project(&db, &project)?;
    Ok(project)
}

fn delete_project_by_id(
    db: &Connection,
    project_id: &str,
    projects_dir: &Path,
) -> Result<(), String> {
    let project_id = project_id.trim();
    if project_id.is_empty() {
        return Err("Project id is required.".to_string());
    }

    ensure_no_active_project_job(db, project_id)?;
    let project_dir: String = db
        .query_row(
            "SELECT project_dir FROM projects WHERE id = ?1",
            params![project_id],
            |row| row.get(0),
        )
        .optional()
        .map_err(to_error)?
        .ok_or_else(|| "Project not found.".to_string())?;
    let project_path = PathBuf::from(project_dir);
    ensure_managed_project_dir(&project_path, projects_dir)?;

    db.execute("DELETE FROM projects WHERE id = ?1", params![project_id])
        .map_err(to_error)?;

    match fs::remove_dir_all(&project_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "Project was removed from the database, but its folder could not be deleted: {error}"
        )),
    }
}

fn ensure_managed_project_dir(project_dir: &Path, projects_dir: &Path) -> Result<(), String> {
    let root = projects_dir.canonicalize().map_err(to_error)?;
    let project = if project_dir.exists() {
        project_dir.canonicalize().map_err(to_error)?
    } else {
        project_dir.to_path_buf()
    };

    if project == root || !project.starts_with(&root) {
        return Err(
            "Project folder is outside the managed project directory; delete it manually."
                .to_string(),
        );
    }
    Ok(())
}

#[tauri::command]
async fn update_project_audio(
    app: tauri::AppHandle,
    payload: UpdateAudioRequest,
) -> Result<ProjectDetail, String> {
    run_blocking_command("prepare audio", move || {
        let state = app.state::<AppState>();
        prepare_project_audio(&app, state.inner(), &payload)
    })
    .await
}

fn prepare_project_audio(
    app: &tauri::AppHandle,
    state: &AppState,
    payload: &UpdateAudioRequest,
) -> Result<ProjectDetail, String> {
    let input_path = validate_capture_wav_path(&payload.input_path, "Dry input")?;
    let target_path = validate_capture_wav_path(&payload.target_path, "Processed target")?;
    ensure_distinct_capture_paths(&input_path, &target_path)?;
    let target_sample_rate = normalize_capture_sample_rate(payload.target_sample_rate)?;
    let channel_policy = normalize_channel_policy_name(&payload.channel_policy)?;

    let project_dir = {
        let db = state.db.lock().map_err(lock_error)?;
        ensure_no_active_project_job(&db, &payload.project_id)?;
        PathBuf::from(load_project_detail(&db, &payload.project_id)?.project_dir)
    };

    let prepared_dir = project_dir.join("audio/prepared");
    fs::create_dir_all(&prepared_dir).map_err(to_error)?;

    let manifest_path = project_dir.join("audio/prepare-manifest.json");
    write_json(
        &manifest_path,
        &serde_json::json!({
            "input_path": input_path,
            "target_path": target_path,
            "output_dir": prepared_dir,
            "target_sample_rate": target_sample_rate,
            "resample": payload.resample,
            "channel_policy": channel_policy,
            "manual_latency_adjustment_samples": 0
        }),
    )?;
    let job_id = create_job(state, "prepare", Some(&payload.project_id), None, None)?;
    let prepare_result = run_rttrainer(
        app,
        state,
        "prepare",
        &manifest_path,
        SidecarContext {
            job_id: job_id.clone(),
            operation: "prepare".to_string(),
            project_id: Some(payload.project_id.clone()),
            run_id: None,
            export_id: None,
        },
    );
    if let Err(error) = prepare_result {
        mark_job_failed(state, &job_id, &error)?;
        return Err(error);
    }
    mark_job_completed(state, &job_id)?;

    let prepare_report_path = prepared_dir.join("preparation-report.json");
    let prepare_report: PrepareReport = read_json(&prepare_report_path)?;

    let report = AudioReport {
        input: prepare_report.input,
        target: prepare_report.target,
        latency_samples: prepare_report
            .latency
            .effective_samples
            .unwrap_or(prepare_report.latency.estimated_samples),
        latency_auto_samples: prepare_report.latency.auto_estimated_samples,
        manual_latency_adjustment_samples: prepare_report.latency.manual_adjustment_samples,
        latency_confidence: prepare_report.latency.confidence,
        warnings: prepare_report.warnings,
        warning_details: prepare_report.warning_details,
        prepared: prepare_report.prepared,
        capture_profile: prepare_report.capture_profile,
        gain: prepare_report.gain,
        options: prepare_report.options,
        status: prepare_report.status,
    };

    let project_status = if matches!(report.status, AudioStatus::Ready) {
        ProjectStatus::Ready
    } else {
        ProjectStatus::Draft
    };
    let updated_at = now();
    let db = state.db.lock().map_err(lock_error)?;
    upsert_audio_report(&db, &payload.project_id, &report)?;
    update_project_status(&db, &payload.project_id, &project_status, &updated_at)?;
    load_project_detail(&db, &payload.project_id)
}

#[tauri::command]
async fn update_project_alignment(
    app: tauri::AppHandle,
    payload: UpdateAlignmentRequest,
) -> Result<ProjectDetail, String> {
    run_blocking_command("update alignment", move || {
        let state = app.state::<AppState>();
        update_project_alignment_blocking(&app, state.inner(), payload)
    })
    .await
}

fn update_project_alignment_blocking(
    app: &tauri::AppHandle,
    state: &AppState,
    payload: UpdateAlignmentRequest,
) -> Result<ProjectDetail, String> {
    let manual_adjustment = payload
        .manual_latency_adjustment_samples
        .clamp(-48_000, 48_000);
    let (project_dir, input_path, target_path, target_sample_rate, resample, channel_policy) = {
        let db = state.db.lock().map_err(lock_error)?;
        ensure_no_active_project_job(&db, &payload.project_id)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        let audio = project
            .audio
            .ok_or_else(|| "Prepare audio before applying alignment.".to_string())?;
        let options = audio
            .options
            .clone()
            .unwrap_or_else(|| serde_json::json!({}));
        let target_sample_rate = options
            .get("target_sample_rate")
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(audio.input.sample_rate);
        let resample = options
            .get("resample")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let channel_policy = options
            .get("channel_policy")
            .and_then(|value| value.as_str())
            .unwrap_or("mixdown")
            .to_string();
        (
            PathBuf::from(project.project_dir),
            audio.input.path,
            audio.target.path,
            target_sample_rate,
            resample,
            channel_policy,
        )
    };

    let input_path = validate_capture_wav_path(&input_path, "Dry input")?;
    let target_path = validate_capture_wav_path(&target_path, "Processed target")?;
    ensure_distinct_capture_paths(&input_path, &target_path)?;
    let target_sample_rate = normalize_capture_sample_rate(target_sample_rate)?;
    let channel_policy = normalize_channel_policy_name(&channel_policy)?;

    let prepared_dir = project_dir.join("audio/prepared");
    fs::create_dir_all(&prepared_dir).map_err(to_error)?;
    let manifest_path = project_dir.join("audio/prepare-manifest.json");
    write_json(
        &manifest_path,
        &serde_json::json!({
            "input_path": input_path,
            "target_path": target_path,
            "output_dir": prepared_dir,
            "target_sample_rate": target_sample_rate,
            "resample": resample,
            "channel_policy": channel_policy,
            "manual_latency_adjustment_samples": manual_adjustment
        }),
    )?;

    let job_id = create_job(state, "prepare", Some(&payload.project_id), None, None)?;
    let prepare_result = run_rttrainer(
        app,
        state,
        "prepare",
        &manifest_path,
        SidecarContext {
            job_id: job_id.clone(),
            operation: "prepare".to_string(),
            project_id: Some(payload.project_id.clone()),
            run_id: None,
            export_id: None,
        },
    );
    if let Err(error) = prepare_result {
        mark_job_failed(state, &job_id, &error)?;
        return Err(error);
    }
    mark_job_completed(state, &job_id)?;

    let prepare_report_path = prepared_dir.join("preparation-report.json");
    let prepare_report: PrepareReport = read_json(&prepare_report_path)?;
    let report = AudioReport {
        input: prepare_report.input,
        target: prepare_report.target,
        latency_samples: prepare_report
            .latency
            .effective_samples
            .unwrap_or(prepare_report.latency.estimated_samples),
        latency_auto_samples: prepare_report.latency.auto_estimated_samples,
        manual_latency_adjustment_samples: prepare_report.latency.manual_adjustment_samples,
        latency_confidence: prepare_report.latency.confidence,
        warnings: prepare_report.warnings,
        warning_details: prepare_report.warning_details,
        prepared: prepare_report.prepared,
        capture_profile: prepare_report.capture_profile,
        gain: prepare_report.gain,
        options: prepare_report.options,
        status: prepare_report.status,
    };

    let project_status = if matches!(report.status, AudioStatus::Ready) {
        ProjectStatus::Ready
    } else {
        ProjectStatus::Draft
    };
    let updated_at = now();
    let db = state.db.lock().map_err(lock_error)?;
    upsert_audio_report(&db, &payload.project_id, &report)?;
    update_project_status(&db, &payload.project_id, &project_status, &updated_at)?;
    load_project_detail(&db, &payload.project_id)
}

#[tauri::command]
fn start_training(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    payload: StartTrainingRequest,
) -> Result<ProjectDetail, String> {
    let run_id = format!("run_{}", Uuid::new_v4().simple());
    let model_preset = normalize_model_preset(&payload.preset)?.to_string();
    let epochs = normalize_training_epochs(payload.epochs);
    let early_stopping_patience =
        normalize_early_stopping_patience(payload.early_stopping_patience);
    let early_stopping_min_delta =
        normalize_early_stopping_min_delta(payload.early_stopping_min_delta);
    let max_windows = normalize_training_max_windows(payload.max_windows);
    let batch_size = normalize_training_batch_size(payload.batch_size);
    let learning_rate = normalize_training_learning_rate(payload.learning_rate);
    let sequence_length = normalize_training_sequence_length(payload.sequence_length);
    let (
        project_dir,
        selected_backend,
        selected_device,
        resume_checkpoint_path,
        resume_source_run_id,
    ) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        let audio_ready = project
            .audio
            .as_ref()
            .map(|audio| audio.status == AudioStatus::Ready)
            .unwrap_or(false);
        if !audio_ready {
            return Err("Audio must pass preflight before training.".to_string());
        }
        ensure_no_active_project_job(&db, &payload.project_id)?;
        let settings = load_runtime_settings(&db)?;
        let selected_backend = normalize_backend(&settings.selected_backend)?.to_string();
        let selected_device = normalize_runtime_device(&settings.selected_device)?.to_string();
        let project_dir = PathBuf::from(&project.project_dir);
        let resume_source_run_id = payload
            .resume_from_run_id
            .as_deref()
            .and_then(non_empty_string);
        let resume_checkpoint_path = if let Some(source_run_id) = resume_source_run_id.as_deref() {
            let source_run = project
                .runs
                .iter()
                .find(|run| run.id == source_run_id)
                .ok_or_else(|| "Resume source run was not found.".to_string())?;
            if !matches!(source_run.status, RunStatus::Completed) {
                return Err(
                    "Choose a completed run when starting from a previous checkpoint.".to_string(),
                );
            }
            if source_run.preset != model_preset {
                return Err(format!(
                    "Resume source uses preset '{}'. Select the same preset before resuming.",
                    source_run.preset
                ));
            }
            if let Ok(source_backend) = normalize_backend(&source_run.backend) {
                if source_backend != selected_backend {
                    return Err(format!(
                        "Resume source uses {source_backend}; selected runtime backend is {selected_backend}."
                    ));
                }
            }
            let source_run_dir =
                artifact_path(&project_dir, &training_run_dir(&db, source_run_id)?);
            let checkpoint_path = best_checkpoint_path(&source_run_dir)
                .ok_or_else(|| "Selected run does not have a best checkpoint.".to_string())?;
            validate_resume_checkpoint_backend(&checkpoint_path, &selected_backend)?;
            Some(checkpoint_path)
        } else {
            None
        };
        (
            project_dir,
            selected_backend,
            selected_device,
            resume_checkpoint_path,
            resume_source_run_id,
        )
    };

    let run_dir = project_dir.join("runs").join(&run_id);
    fs::create_dir_all(&run_dir).map_err(to_error)?;
    let manifest_path = run_dir.join("train-manifest.json");
    let mut manifest = serde_json::json!({
            "run_id": &run_id,
            "run_dir": &run_dir,
            "prepared_dir": project_dir.join("audio/prepared"),
            "preset": &model_preset,
            "backend": &selected_backend,
            "device": &selected_device,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "max_windows": max_windows,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "seed": 1337
    });
    if let Some(checkpoint_path) = resume_checkpoint_path {
        manifest["resume_from_checkpoint"] = serde_json::Value::Bool(true);
        manifest["resume_epochs_are_additional"] = serde_json::Value::Bool(true);
        manifest["checkpoint_path"] =
            serde_json::Value::String(checkpoint_path.display().to_string());
    }
    if let Some(source_run_id) = resume_source_run_id {
        manifest["resume_source_run_id"] = serde_json::Value::String(source_run_id);
    }
    write_json(&manifest_path, &manifest)?;

    let created_at = now();
    let log_path = run_dir.join("events.jsonl");
    let run = TrainingRun {
        id: run_id.clone(),
        preset: model_preset,
        backend: selected_backend,
        status: RunStatus::Running,
        device: "pending".to_string(),
        epochs: 0,
        created_at: created_at.clone(),
        updated_at: created_at.clone(),
        metrics: None,
        log_path: relative_path_string(&project_dir, &log_path),
    };
    {
        let db = state.db.lock().map_err(lock_error)?;
        insert_training_run(&db, &payload.project_id, &run, &project_dir, &run_dir)?;
        update_project_status(
            &db,
            &payload.project_id,
            &ProjectStatus::Training,
            &created_at,
        )?;
    }

    let job_id = create_job(
        &state,
        "train",
        Some(&payload.project_id),
        Some(&run_id),
        None,
    )?;
    spawn_training_worker(
        app.clone(),
        payload.project_id.clone(),
        run_id.clone(),
        job_id,
        manifest_path,
        project_dir.clone(),
        run_dir,
        log_path,
    );

    let db = state.db.lock().map_err(lock_error)?;
    load_project_detail(&db, &payload.project_id)
}

#[tauri::command]
fn cancel_training_run(
    state: tauri::State<AppState>,
    payload: RunControlRequest,
) -> Result<ProjectDetail, String> {
    let (job_id, project_id) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        let run = project
            .runs
            .iter()
            .find(|run| run.id == payload.run_id)
            .ok_or_else(|| "Run not found".to_string())?;
        if !matches!(
            run.status,
            RunStatus::Queued | RunStatus::Preparing | RunStatus::Running | RunStatus::Cancelling
        ) {
            return Err("Only active training runs can be cancelled.".to_string());
        }
        let job_id = active_training_job_for_run(&db, &payload.run_id)?
            .ok_or_else(|| "No active process is registered for this run.".to_string())?;
        update_training_run_status(
            &db,
            &payload.run_id,
            &RunStatus::Cancelling,
            Some("Cancellation requested."),
        )?;
        update_job_status_in_db(&db, &job_id, &JobStatus::Cancelling, None)?;
        (job_id, project.id)
    };

    let _ = terminate_active_process(&state, &job_id)?;
    let db = state.db.lock().map_err(lock_error)?;
    load_project_detail(&db, &project_id)
}

#[tauri::command]
fn resume_training_run(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    payload: RunControlRequest,
) -> Result<ProjectDetail, String> {
    let (project_dir, run_dir, manifest_path, log_path) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        ensure_no_active_project_job(&db, &payload.project_id)?;
        let run = project
            .runs
            .iter()
            .find(|run| run.id == payload.run_id)
            .ok_or_else(|| "Run not found".to_string())?;
        if !matches!(run.status, RunStatus::Failed | RunStatus::Interrupted) {
            return Err("Only failed or interrupted runs can be resumed.".to_string());
        }
        let project_dir = PathBuf::from(&project.project_dir);
        let run_dir = artifact_path(&project_dir, &training_run_dir(&db, &payload.run_id)?);
        let manifest_path = run_dir.join("train-manifest.json");
        if !manifest_path.exists() {
            return Err("Run manifest is missing; this run cannot be resumed.".to_string());
        }
        let checkpoint_path = best_checkpoint_path(&run_dir)
            .ok_or_else(|| "No checkpoint exists for this run yet.".to_string())?;
        let mut manifest: serde_json::Value = read_json(&manifest_path)?;
        manifest["resume_from_checkpoint"] = serde_json::Value::Bool(true);
        manifest["checkpoint_path"] =
            serde_json::Value::String(checkpoint_path.display().to_string());
        write_json(&manifest_path, &manifest)?;
        let log_path = run_dir.join("events.jsonl");
        update_training_run_status(&db, &payload.run_id, &RunStatus::Running, None)?;
        update_project_status(&db, &payload.project_id, &ProjectStatus::Training, &now())?;
        (project_dir, run_dir, manifest_path, log_path)
    };

    let job_id = create_job(
        &state,
        "train",
        Some(&payload.project_id),
        Some(&payload.run_id),
        None,
    )?;
    spawn_training_worker(
        app.clone(),
        payload.project_id.clone(),
        payload.run_id.clone(),
        job_id,
        manifest_path,
        project_dir,
        run_dir,
        log_path,
    );

    let db = state.db.lock().map_err(lock_error)?;
    load_project_detail(&db, &payload.project_id)
}

#[tauri::command]
async fn export_run(
    app: tauri::AppHandle,
    payload: ExportRunRequest,
) -> Result<ProjectDetail, String> {
    run_blocking_command("export run", move || {
        let state = app.state::<AppState>();
        export_run_blocking(&app, state.inner(), payload)
    })
    .await
}

fn export_run_blocking(
    app: &tauri::AppHandle,
    state: &AppState,
    payload: ExportRunRequest,
) -> Result<ProjectDetail, String> {
    let export_id = format!("export_{}", Uuid::new_v4().simple());
    let (project_name, project_id, project_dir, run, sample_rate, latency_samples) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        ensure_no_active_project_job(&db, &payload.project_id)?;
        let run = project
            .runs
            .iter()
            .find(|run| run.id == payload.run_id)
            .cloned()
            .ok_or_else(|| "Run not found".to_string())?;
        if run.metrics.is_none() {
            return Err("Run has no metrics to export.".to_string());
        }
        let audio = project
            .audio
            .as_ref()
            .ok_or_else(|| "Audio must be prepared before export.".to_string())?;
        (
            project.name.clone(),
            project.id.clone(),
            PathBuf::from(&project.project_dir),
            run,
            audio.input.sample_rate,
            audio.latency_samples,
        )
    };

    let export_dir = project_dir.join("exports").join(&export_id);
    fs::create_dir_all(&export_dir).map_err(to_error)?;

    let model_path = export_dir.join("model.rtneural.json");
    let package_path = export_dir.join("package.json");
    let validation_path = export_dir.join("validation-report.json");
    let benchmark_path = export_dir.join("benchmark-report.json");
    let manifest_path = export_dir.join("export-manifest.json");
    let run_dir = project_dir.join("runs").join(&run.id);
    let prediction_path = run_dir.join("previews/prediction.wav");
    let test_input_path = run_dir.join("test-input.wav");
    let run_backend = read_optional_json_value(&run_dir.join("training-report.json"))
        .and_then(|report| {
            report
                .get("backend")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "keras".to_string());

    write_json(
        &manifest_path,
        &serde_json::json!({
            "name": project_name,
            "run_dir": run_dir,
            "export_dir": export_dir,
            "backend": run_backend,
            "sample_rate": sample_rate,
            "latency_samples": latency_samples,
            "parity_tolerance": 0.0001
        }),
    )?;

    let package = ExportPackage {
        id: export_id.clone(),
        run_id: run.id.clone(),
        status: ExportStatus::Pending,
        created_at: now(),
        model_path: relative_path_string(&project_dir, &model_path),
        package_path: relative_path_string(&project_dir, &package_path),
        validation_path: relative_path_string(&project_dir, &validation_path),
        benchmark_path: relative_path_string(&project_dir, &benchmark_path),
        export_dir: relative_path_string(&project_dir, &export_dir),
        package_metadata: None,
        validation_report: None,
        benchmark_report: None,
    };
    {
        let db = state.db.lock().map_err(lock_error)?;
        insert_export_package(&db, &project_id, &package, &project_dir, &export_dir)?;
    }

    let export_job_id = create_job(
        state,
        "export",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let sidecar_result = run_rttrainer(
        app,
        state,
        "export",
        &manifest_path,
        SidecarContext {
            job_id: export_job_id.clone(),
            operation: "export".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
    );
    let sidecar = match sidecar_result {
        Ok(output) => output,
        Err(error) => {
            mark_job_failed(state, &export_job_id, &error)?;
            let db = state.db.lock().map_err(lock_error)?;
            update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
            return Err(error);
        }
    };
    mark_job_completed(state, &export_job_id)?;
    fs::write(export_dir.join("export-events.jsonl"), sidecar.stdout).map_err(to_error)?;
    if !sidecar.stderr.trim().is_empty() {
        fs::write(export_dir.join("stderr.log"), sidecar.stderr).map_err(to_error)?;
    }

    {
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Validating, None)?;
    }

    let validate_job_id = create_job(
        state,
        "native_validate",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let validate_result = run_validator(
        app,
        state,
        vec![
            "validate".to_string(),
            "--model".to_string(),
            model_path.display().to_string(),
            "--input".to_string(),
            test_input_path.display().to_string(),
            "--reference".to_string(),
            prediction_path.display().to_string(),
            "--report".to_string(),
            validation_path.display().to_string(),
            "--tolerance".to_string(),
            "0.0001".to_string(),
        ],
        SidecarContext {
            job_id: validate_job_id.clone(),
            operation: "native_validate".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
    );
    if let Err(error) = validate_result {
        mark_job_failed(state, &validate_job_id, &error)?;
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
        return Err(error);
    }
    mark_job_completed(state, &validate_job_id)?;

    let benchmark_job_id = create_job(
        state,
        "native_benchmark",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let benchmark_result = run_validator(
        app,
        state,
        vec![
            "benchmark".to_string(),
            "--model".to_string(),
            model_path.display().to_string(),
            "--sample-rate".to_string(),
            sample_rate.to_string(),
            "--seconds".to_string(),
            "10".to_string(),
            "--report".to_string(),
            benchmark_path.display().to_string(),
        ],
        SidecarContext {
            job_id: benchmark_job_id.clone(),
            operation: "native_benchmark".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
    );
    if let Err(error) = benchmark_result {
        mark_job_failed(state, &benchmark_job_id, &error)?;
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
        return Err(error);
    }
    mark_job_completed(state, &benchmark_job_id)?;

    write_final_export_package_metadata(FinalExportPackageInput {
        project_name: &project_name,
        project_id: &project_id,
        export_id: &export_id,
        run: &run,
        sample_rate,
        latency_samples,
        export_dir: &export_dir,
        model_path: &model_path,
        package_path: &package_path,
        validation_path: &validation_path,
        benchmark_path: &benchmark_path,
    })?;

    let db = state.db.lock().map_err(lock_error)?;
    update_export_status(&db, &export_id, &ExportStatus::Ready, None)?;
    update_project_status(&db, &project_id, &ProjectStatus::Exported, &now())?;
    load_project_detail(&db, &project_id)
}

#[tauri::command]
fn open_export_folder(
    state: tauri::State<AppState>,
    payload: ExportFolderRequest,
) -> Result<(), String> {
    let (project_dir, export_dir) = {
        let db = state.db.lock().map_err(lock_error)?;
        let project = load_project_detail(&db, &payload.project_id)?;
        let export_dir = db
            .query_row(
                "SELECT export_dir FROM exports WHERE id = ?1 AND project_id = ?2",
                params![payload.export_id, payload.project_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(to_error)?
            .ok_or_else(|| "Export package not found.".to_string())?;
        let project_dir = PathBuf::from(project.project_dir);
        let export_dir = artifact_path(&project_dir, &export_dir);
        (project_dir, export_dir)
    };
    ensure_artifact_inside(&project_dir, &export_dir)?;
    if !export_dir.is_dir() {
        return Err("Export folder is missing.".to_string());
    }
    open_folder(&export_dir)
}

#[tauri::command]
fn update_notes(
    state: tauri::State<AppState>,
    payload: UpdateNotesRequest,
) -> Result<ProjectDetail, String> {
    let db = state.db.lock().map_err(lock_error)?;
    update_project_notes(&db, &payload.project_id, &payload.notes, &now())?;
    load_project_detail(&db, &payload.project_id)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_dir = app
                .path()
                .app_data_dir()
                .expect("failed to resolve app data directory");
            let projects_dir = app_dir.join("projects");
            fs::create_dir_all(&projects_dir).expect("failed to create project directory");
            let store_path = app_dir.join("store.json");
            let db_path = app_dir.join("rtneural-trainer.sqlite3");
            let mut db = Connection::open(&db_path).expect("failed to open SQLite project store");
            configure_database(&mut db).expect("failed to migrate SQLite project store");
            migrate_legacy_store(&db, &store_path).expect("failed to migrate legacy JSON store");
            recover_interrupted_jobs(&db).expect("failed to recover interrupted jobs");
            audit_missing_artifacts(&db).expect("failed to audit project artifacts");
            app.manage(AppState {
                db_path,
                projects_dir,
                workspace_root: workspace_root(),
                db: Mutex::new(db),
                active_jobs: Mutex::new(HashMap::new()),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_status,
            get_runtime_settings,
            update_runtime_settings,
            list_training_recipes,
            save_training_recipe,
            delete_training_recipe,
            inspect_device,
            list_projects,
            get_project,
            delete_project,
            rename_project,
            list_project_events,
            get_run_preview,
            get_project_waveform,
            create_project,
            create_sample_project,
            update_project_audio,
            update_project_alignment,
            start_training,
            cancel_training_run,
            resume_training_run,
            export_run,
            open_export_folder,
            update_notes
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let raw = fs::read_to_string(path).map_err(to_error)?;
    serde_json::from_str(&raw).map_err(to_error)
}

fn read_optional_json_value(path: &Path) -> Option<serde_json::Value> {
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
}

fn default_runtime_settings() -> RuntimeSettings {
    RuntimeSettings {
        selected_backend: "keras".to_string(),
        selected_device: default_runtime_device(),
        external_python_path: None,
    }
}

fn default_runtime_device() -> String {
    "auto".to_string()
}

fn load_runtime_settings(db: &Connection) -> Result<RuntimeSettings, String> {
    let raw = db
        .query_row(
            "SELECT value_json FROM app_settings WHERE key = ?1",
            params![RUNTIME_SETTINGS_KEY],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(to_error)?;
    let Some(raw) = raw else {
        return Ok(default_runtime_settings());
    };
    let mut settings: RuntimeSettings = serde_json::from_str(&raw).map_err(to_error)?;
    settings.selected_backend = normalize_backend(&settings.selected_backend)?.to_string();
    settings.selected_device = normalize_runtime_device(&settings.selected_device)?.to_string();
    settings.external_python_path = settings
        .external_python_path
        .and_then(|path| non_empty_string(&path));
    Ok(settings)
}

fn save_runtime_settings(db: &Connection, settings: &RuntimeSettings) -> Result<(), String> {
    let normalized = RuntimeSettings {
        selected_backend: normalize_backend(&settings.selected_backend)?.to_string(),
        selected_device: normalize_runtime_device(&settings.selected_device)?.to_string(),
        external_python_path: settings
            .external_python_path
            .as_deref()
            .and_then(non_empty_string),
    };
    let raw = serde_json::to_string(&normalized).map_err(to_error)?;
    db.execute(
        "INSERT INTO app_settings (key, value_json, updated_at)
         VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = excluded.updated_at",
        params![RUNTIME_SETTINGS_KEY, raw, now()],
    )
    .map_err(to_error)?;
    Ok(())
}

fn load_training_recipes(db: &Connection) -> Result<Vec<TrainingRecipe>, String> {
    let mut stmt = db
        .prepare(
            "SELECT id, name, model_preset, epochs, batch_size, learning_rate, sequence_length,
                    max_windows, early_stopping_patience, early_stopping_min_delta,
                    created_at, updated_at
             FROM training_recipes
             ORDER BY updated_at DESC, name ASC",
        )
        .map_err(to_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok(TrainingRecipe {
                id: row.get(0)?,
                name: row.get(1)?,
                model_preset: row.get(2)?,
                epochs: row.get(3)?,
                batch_size: row.get(4)?,
                learning_rate: row.get(5)?,
                sequence_length: row.get(6)?,
                max_windows: row.get(7)?,
                early_stopping_patience: row.get(8)?,
                early_stopping_min_delta: row.get(9)?,
                created_at: row.get(10)?,
                updated_at: row.get(11)?,
            })
        })
        .map_err(to_error)?;

    let mut recipes = Vec::new();
    for row in rows {
        recipes.push(row.map_err(to_error)?);
    }
    Ok(recipes)
}

fn upsert_training_recipe(db: &Connection, recipe: &TrainingRecipe) -> Result<(), String> {
    db.execute(
        "INSERT INTO training_recipes
         (id, name, model_preset, epochs, batch_size, learning_rate, sequence_length,
          max_windows, early_stopping_patience, early_stopping_min_delta, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            model_preset = excluded.model_preset,
            epochs = excluded.epochs,
            batch_size = excluded.batch_size,
            learning_rate = excluded.learning_rate,
            sequence_length = excluded.sequence_length,
            max_windows = excluded.max_windows,
            early_stopping_patience = excluded.early_stopping_patience,
            early_stopping_min_delta = excluded.early_stopping_min_delta,
            updated_at = excluded.updated_at",
        params![
            recipe.id,
            recipe.name,
            recipe.model_preset,
            recipe.epochs,
            recipe.batch_size,
            recipe.learning_rate,
            recipe.sequence_length,
            recipe.max_windows,
            recipe.early_stopping_patience,
            recipe.early_stopping_min_delta,
            recipe.created_at,
            recipe.updated_at
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn normalize_backend(value: &str) -> Result<&'static str, String> {
    match value.trim().to_lowercase().as_str() {
        "" | "keras" | "tensorflow" | "tf" => Ok("keras"),
        "pytorch" | "torch" => Ok("pytorch"),
        _ => Err("Backend must be 'keras' or 'pytorch'.".to_string()),
    }
}

fn normalize_runtime_device(value: &str) -> Result<&'static str, String> {
    match value.trim().to_lowercase().as_str() {
        "" | "auto" => Ok("auto"),
        "cpu" | "tensorflow-cpu" => Ok("cpu"),
        "mps" | "metal" => Ok("mps"),
        "cuda" | "gpu" | "tensorflow-gpu" => Ok("cuda"),
        _ => Err("Device must be 'auto', 'cpu', 'mps', or 'cuda'.".to_string()),
    }
}

fn non_empty_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

fn runtime_settings_from_state(state: &AppState) -> Result<RuntimeSettings, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_runtime_settings(&db)
}

fn external_python_executable(state: &AppState) -> Result<Option<PathBuf>, String> {
    let settings = runtime_settings_from_state(state)?;
    let Some(path) = settings.external_python_path.as_deref() else {
        return Ok(None);
    };
    resolve_python_executable(path).map(Some)
}

fn resolve_python_executable(value: &str) -> Result<PathBuf, String> {
    let path = expand_home_path(value);
    if path.is_file() {
        return Ok(path);
    }
    if path.is_dir() {
        let candidates = if cfg!(windows) {
            vec![path.join("Scripts/python.exe"), path.join("python.exe")]
        } else {
            vec![path.join("bin/python"), path.join("bin/python3")]
        };
        for candidate in candidates {
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "External Python environment not found: {value}. Provide a Python executable or venv folder."
    ))
}

fn expand_home_path(value: &str) -> PathBuf {
    if value == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    }
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(value)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
}

fn parse_json_from_stdout(stdout: &str) -> Result<serde_json::Value, String> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(stdout.trim()) {
        return Ok(value);
    }
    let start = stdout
        .find('{')
        .ok_or_else(|| "rttrainer inspect-device did not return JSON.".to_string())?;
    let end = stdout
        .rfind('}')
        .ok_or_else(|| "rttrainer inspect-device did not return complete JSON.".to_string())?;
    serde_json::from_str(&stdout[start..=end]).map_err(to_error)
}

fn default_capture_sample_rate() -> u32 {
    48_000
}

fn default_channel_policy() -> String {
    "mixdown".to_string()
}

fn default_training_epochs() -> u32 {
    20
}

fn default_early_stopping_patience() -> u32 {
    5
}

fn default_early_stopping_min_delta() -> f64 {
    0.0001
}

fn default_training_max_windows() -> u32 {
    512
}

fn default_training_batch_size() -> u32 {
    16
}

fn default_training_learning_rate() -> f64 {
    0.001
}

fn default_training_sequence_length() -> u32 {
    8192
}

fn default_training_backend() -> String {
    "unknown".to_string()
}

fn default_waveform_bins() -> usize {
    220
}

fn validate_capture_wav_path(value: &str, label: &str) -> Result<PathBuf, String> {
    let Some(trimmed) = non_empty_string(value) else {
        return Err(format!("{label} WAV path is required."));
    };
    let path = expand_home_path(&trimmed);
    if !path.exists() {
        return Err(format!(
            "{label} WAV file does not exist: {}",
            path.display()
        ));
    }
    if !path.is_file() {
        return Err(format!("{label} path is not a file: {}", path.display()));
    }
    let is_wav = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            extension.eq_ignore_ascii_case("wav") || extension.eq_ignore_ascii_case("wave")
        })
        .unwrap_or(false);
    if !is_wav {
        return Err(format!("{label} must be a .wav file: {}", path.display()));
    }
    Ok(path)
}

fn ensure_distinct_capture_paths(input_path: &Path, target_path: &Path) -> Result<(), String> {
    let same_path = match (input_path.canonicalize(), target_path.canonicalize()) {
        (Ok(input), Ok(target)) => input == target,
        _ => input_path == target_path,
    };
    if same_path {
        return Err("Dry input and processed target must be different WAV files.".to_string());
    }
    Ok(())
}

fn generated_sample_capture(sample_rate: u32, seconds: f32) -> (Vec<f32>, Vec<f32>) {
    let sample_count = (sample_rate as f32 * seconds).round().max(1.0) as usize;
    let mut input = Vec::with_capacity(sample_count);
    let mut target = Vec::with_capacity(sample_count);
    let mut filter_state = 0.0_f32;

    for index in 0..sample_count {
        let t = index as f32 / sample_rate as f32;
        let envelope = if t < 0.08 {
            t / 0.08
        } else if t > seconds - 0.12 {
            ((seconds - t) / 0.12).max(0.0)
        } else {
            1.0
        };
        let sweep = 110.0 + 330.0 * (t / seconds).min(1.0);
        let dry = envelope
            * (0.34 * (std::f32::consts::TAU * sweep * t).sin()
                + 0.18 * (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.08 * (std::f32::consts::TAU * 880.0 * t).sin());
        filter_state = 0.86 * filter_state + 0.14 * (dry * 2.4).tanh();
        let wet = (0.82 * filter_state + 0.12 * dry).clamp(-0.92, 0.92);
        input.push(dry.clamp(-0.95, 0.95));
        target.push(wet);
    }

    (input, target)
}

fn write_pcm16_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(to_error)?;
    }
    let data_size = samples
        .len()
        .checked_mul(2)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| "Sample WAV is too large to write.".to_string())?;
    let mut bytes = Vec::with_capacity(44 + data_size as usize);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&1_u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate.to_le_bytes());
    bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    bytes.extend_from_slice(&2_u16.to_le_bytes());
    bytes.extend_from_slice(&16_u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_size.to_le_bytes());
    for sample in samples {
        let quantized = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        bytes.extend_from_slice(&quantized.to_le_bytes());
    }
    fs::write(path, bytes).map_err(to_error)
}

fn normalize_capture_sample_rate(value: u32) -> Result<u32, String> {
    let sample_rate = if value == 0 {
        default_capture_sample_rate()
    } else {
        value
    };
    if !(8_000..=384_000).contains(&sample_rate) {
        return Err("Target sample rate must be between 8 kHz and 384 kHz.".to_string());
    }
    Ok(sample_rate)
}

fn normalize_channel_policy_name(value: &str) -> Result<&'static str, String> {
    match value.trim().to_lowercase().replace('-', "_").as_str() {
        "" | "mixdown" | "mono_mixdown" | "mix_to_mono" => Ok("mixdown"),
        "first" | "first_channel" | "left" => Ok("first"),
        "reject" | "reject_multichannel" | "mono_only" => Ok("reject"),
        _ => Err("Channel policy must be mixdown, first, or reject.".to_string()),
    }
}

fn normalize_training_epochs(value: u32) -> u32 {
    value.clamp(1, 500)
}

fn normalize_early_stopping_patience(value: u32) -> u32 {
    value.min(100)
}

fn normalize_early_stopping_min_delta(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        default_early_stopping_min_delta()
    }
}

fn normalize_training_max_windows(value: u32) -> u32 {
    value.clamp(32, 16_384)
}

fn normalize_training_batch_size(value: u32) -> u32 {
    if value == 0 {
        return default_training_batch_size();
    }
    value.clamp(1, 512)
}

fn normalize_training_learning_rate(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value.clamp(0.000001, 1.0)
    } else {
        default_training_learning_rate()
    }
}

fn normalize_training_sequence_length(value: u32) -> u32 {
    if value == 0 {
        return default_training_sequence_length();
    }
    value.clamp(32, 65_536)
}

fn normalize_project_name(value: &str) -> Result<String, String> {
    let name = value.trim();
    if name.is_empty() {
        return Err("Project name is required.".to_string());
    }
    if name.chars().count() > 120 {
        return Err("Project name must be 120 characters or fewer.".to_string());
    }
    Ok(name.to_string())
}

fn normalize_model_preset(value: &str) -> Result<&'static str, String> {
    let normalized = value.trim();
    MODEL_PRESETS
        .iter()
        .copied()
        .find(|preset| *preset == normalized)
        .ok_or_else(|| {
            format!(
                "Unknown model preset '{normalized}'. Known presets: {}.",
                MODEL_PRESETS.join(", ")
            )
        })
}

fn normalize_training_recipe(payload: SaveTrainingRecipeRequest) -> Result<TrainingRecipe, String> {
    let name = non_empty_string(&payload.name)
        .ok_or_else(|| "Training recipe name is required.".to_string())?;
    let timestamp = now();
    Ok(TrainingRecipe {
        id: payload
            .id
            .and_then(|id| non_empty_string(&id))
            .unwrap_or_else(|| format!("recipe_{}", Uuid::new_v4().simple())),
        name,
        model_preset: normalize_model_preset(&payload.model_preset)?.to_string(),
        epochs: normalize_training_epochs(payload.epochs),
        batch_size: normalize_training_batch_size(payload.batch_size),
        learning_rate: normalize_training_learning_rate(payload.learning_rate),
        sequence_length: normalize_training_sequence_length(payload.sequence_length),
        max_windows: normalize_training_max_windows(payload.max_windows),
        early_stopping_patience: normalize_early_stopping_patience(payload.early_stopping_patience),
        early_stopping_min_delta: normalize_early_stopping_min_delta(
            payload.early_stopping_min_delta,
        ),
        created_at: timestamp.clone(),
        updated_at: timestamp,
    })
}

fn configure_database(db: &mut Connection) -> Result<(), String> {
    db.execute_batch(
        r#"
        PRAGMA foreign_keys = ON;
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        "#,
    )
    .map_err(to_error)?;
    apply_migrations(db)
}

fn apply_migrations(db: &mut Connection) -> Result<(), String> {
    for (id, sql) in MIGRATIONS {
        let already_applied = db
            .query_row(
                "SELECT 1 FROM schema_migrations WHERE id = ?1",
                params![id],
                |_| Ok(()),
            )
            .optional()
            .map_err(to_error)?
            .is_some();
        if already_applied {
            continue;
        }

        let tx = db.transaction().map_err(to_error)?;
        tx.execute_batch(sql).map_err(to_error)?;
        tx.execute(
            "INSERT INTO schema_migrations (id, applied_at) VALUES (?1, ?2)",
            params![id, now()],
        )
        .map_err(to_error)?;
        tx.commit().map_err(to_error)?;
    }
    Ok(())
}

const MIGRATIONS: &[(&str, &str)] = &[
    (
        "001_initial_project_store",
        r#"
    CREATE TABLE projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        target_kind TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        project_dir TEXT NOT NULL
    );

    CREATE TABLE audio_files (
        project_id TEXT PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        report_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE training_runs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        preset TEXT NOT NULL,
        status TEXT NOT NULL,
        device TEXT NOT NULL,
        epochs INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metrics_json TEXT,
        log_path TEXT NOT NULL,
        run_dir TEXT NOT NULL,
        error TEXT
    );

    CREATE INDEX idx_training_runs_project
        ON training_runs(project_id, created_at);

    CREATE TABLE exports (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        model_path TEXT NOT NULL,
        package_path TEXT NOT NULL,
        validation_path TEXT NOT NULL,
        benchmark_path TEXT NOT NULL,
        export_dir TEXT NOT NULL,
        error TEXT
    );

    CREATE INDEX idx_exports_project
        ON exports(project_id, created_at);

    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        project_id TEXT REFERENCES projects(id) ON DELETE CASCADE,
        run_id TEXT,
        export_id TEXT,
        operation TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        error TEXT
    );

    CREATE INDEX idx_jobs_project
        ON jobs(project_id, created_at);

    CREATE TABLE job_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        project_id TEXT,
        run_id TEXT,
        export_id TEXT,
        operation TEXT NOT NULL,
        stream TEXT NOT NULL,
        line TEXT NOT NULL,
        event_json TEXT,
        created_at TEXT NOT NULL
    );

    CREATE INDEX idx_job_events_job
        ON job_events(job_id, id);

    CREATE TABLE app_settings (
        key TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    "#,
    ),
    (
        "002_training_recipes",
        r#"
    CREATE TABLE training_recipes (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        model_preset TEXT NOT NULL,
        epochs INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        learning_rate REAL NOT NULL,
        sequence_length INTEGER NOT NULL,
        max_windows INTEGER NOT NULL,
        early_stopping_patience INTEGER NOT NULL,
        early_stopping_min_delta REAL NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    "#,
    ),
];

fn migrate_legacy_store(db: &Connection, store_path: &Path) -> Result<(), String> {
    let existing_projects: i64 = db
        .query_row("SELECT COUNT(*) FROM projects", [], |row| row.get(0))
        .map_err(to_error)?;
    if existing_projects > 0 || !store_path.exists() {
        return Ok(());
    }

    let raw = fs::read_to_string(store_path).map_err(to_error)?;
    let legacy: Store = serde_json::from_str(&raw).map_err(to_error)?;
    for project in legacy.projects {
        insert_project(db, &project)?;
        if let Some(audio) = &project.audio {
            upsert_audio_report(db, &project.id, audio)?;
        }
        let project_dir = PathBuf::from(&project.project_dir);
        for run in &project.runs {
            let run_dir = project_dir.join("runs").join(&run.id);
            insert_training_run(db, &project.id, run, &project_dir, &run_dir)?;
        }
        for export in &project.exports {
            let export_dir = project_dir.join("exports").join(&export.id);
            insert_export_package(db, &project.id, export, &project_dir, &export_dir)?;
        }
    }
    Ok(())
}

fn recover_interrupted_jobs(db: &Connection) -> Result<(), String> {
    let timestamp = now();
    let message = "App restarted before the job completed.";
    db.execute(
        "UPDATE jobs
         SET status = 'interrupted', updated_at = ?1, finished_at = ?1, error = COALESCE(error, ?2)
         WHERE status IN ('queued', 'running', 'cancelling')",
        params![timestamp, message],
    )
    .map_err(to_error)?;
    db.execute(
        "UPDATE training_runs
         SET status = 'interrupted', updated_at = ?1, error = COALESCE(error, ?2)
         WHERE status IN ('queued', 'preparing', 'running', 'cancelling')",
        params![timestamp, message],
    )
    .map_err(to_error)?;
    db.execute(
        "UPDATE exports
         SET status = 'failed', error = COALESCE(error, ?2)
         WHERE status IN ('pending', 'validating')",
        params![timestamp, message],
    )
    .map_err(to_error)?;
    db.execute(
        "UPDATE projects
         SET status = CASE
             WHEN EXISTS (SELECT 1 FROM exports WHERE exports.project_id = projects.id AND exports.status = 'ready') THEN 'exported'
             WHEN EXISTS (SELECT 1 FROM audio_files WHERE audio_files.project_id = projects.id AND audio_files.status = 'ready') THEN 'ready'
             ELSE 'draft'
         END,
         updated_at = ?1
         WHERE status = 'training'",
        params![timestamp],
    )
    .map_err(to_error)?;
    Ok(())
}

fn audit_missing_artifacts(db: &Connection) -> Result<(), String> {
    let mut stmt = db
        .prepare(
            "SELECT exports.id, projects.project_dir, exports.model_path, exports.package_path,
                    exports.validation_path, exports.benchmark_path
             FROM exports
             JOIN projects ON projects.id = exports.project_id
             WHERE exports.status = 'ready'",
        )
        .map_err(to_error)?;
    let rows = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(to_error)?;

    let mut missing_exports = Vec::new();
    for row in rows {
        let (id, project_dir, model, package, validation, benchmark) = row.map_err(to_error)?;
        let root = PathBuf::from(project_dir);
        let missing = [model, package, validation, benchmark]
            .iter()
            .any(|path| !artifact_path(&root, path).exists());
        if missing {
            missing_exports.push(id);
        }
    }

    for export_id in missing_exports {
        update_export_status(
            db,
            &export_id,
            &ExportStatus::Failed,
            Some("One or more export artifacts are missing."),
        )?;
    }
    Ok(())
}

fn load_all_projects(db: &Connection) -> Result<Vec<ProjectDetail>, String> {
    let mut stmt = db
        .prepare("SELECT id FROM projects ORDER BY updated_at DESC, created_at DESC")
        .map_err(to_error)?;
    let rows = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(to_error)?;
    let mut projects = Vec::new();
    for row in rows {
        projects.push(load_project_detail(db, &row.map_err(to_error)?)?);
    }
    Ok(projects)
}

fn load_project_detail(db: &Connection, project_id: &str) -> Result<ProjectDetail, String> {
    let mut stmt = db
        .prepare(
            "SELECT id, name, target_kind, status, created_at, updated_at, notes, project_dir
             FROM projects
             WHERE id = ?1",
        )
        .map_err(to_error)?;
    let mut rows = stmt.query(params![project_id]).map_err(to_error)?;
    let Some(row) = rows.next().map_err(to_error)? else {
        return Err("Project not found".to_string());
    };

    let project_dir = row.get::<_, String>(7).map_err(to_error)?;
    let mut project = ProjectDetail {
        id: row.get(0).map_err(to_error)?,
        name: row.get(1).map_err(to_error)?,
        target_kind: enum_from_string(&row.get::<_, String>(2).map_err(to_error)?)?,
        status: enum_from_string(&row.get::<_, String>(3).map_err(to_error)?)?,
        created_at: row.get(4).map_err(to_error)?,
        updated_at: row.get(5).map_err(to_error)?,
        notes: row.get(6).map_err(to_error)?,
        project_dir,
        audio: load_audio_report(db, project_id)?,
        runs: Vec::new(),
        exports: Vec::new(),
    };
    let root = PathBuf::from(&project.project_dir);
    project.runs = load_training_runs(db, project_id, &root)?;
    project.exports = load_exports(db, project_id, &root)?;
    Ok(project)
}

fn load_audio_report(db: &Connection, project_id: &str) -> Result<Option<AudioReport>, String> {
    let raw = db
        .query_row(
            "SELECT report_json FROM audio_files WHERE project_id = ?1",
            params![project_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(to_error)?;
    raw.map(|value| serde_json::from_str(&value).map_err(to_error))
        .transpose()
}

fn load_training_runs(
    db: &Connection,
    project_id: &str,
    project_dir: &Path,
) -> Result<Vec<TrainingRun>, String> {
    let mut stmt = db
        .prepare(
            "SELECT id, preset, status, device, epochs, created_at, updated_at, metrics_json, log_path, run_dir
             FROM training_runs
             WHERE project_id = ?1
             ORDER BY created_at ASC",
        )
        .map_err(to_error)?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, u32>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
                row.get::<_, String>(9)?,
            ))
        })
        .map_err(to_error)?;

    let mut runs = Vec::new();
    for row in rows {
        let (
            id,
            preset,
            status,
            device,
            epochs,
            created_at,
            updated_at,
            metrics_json,
            log_path,
            run_dir,
        ) = row.map_err(to_error)?;
        let run_dir = artifact_path(project_dir, &run_dir);
        runs.push(TrainingRun {
            id,
            preset,
            backend: training_run_backend(&run_dir),
            status: enum_from_string(&status)?,
            device,
            epochs,
            created_at,
            updated_at,
            metrics: metrics_json
                .map(|value| serde_json::from_str(&value).map_err(to_error))
                .transpose()?,
            log_path: artifact_path(project_dir, &log_path).display().to_string(),
        });
    }
    Ok(runs)
}

fn load_exports(
    db: &Connection,
    project_id: &str,
    project_dir: &Path,
) -> Result<Vec<ExportPackage>, String> {
    let mut stmt = db
        .prepare(
            "SELECT id, run_id, status, created_at, model_path, package_path, validation_path, benchmark_path, export_dir
             FROM exports
             WHERE project_id = ?1
             ORDER BY created_at ASC",
        )
        .map_err(to_error)?;
    let rows = stmt
        .query_map(params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
                row.get::<_, String>(8)?,
            ))
        })
        .map_err(to_error)?;

    let mut exports = Vec::new();
    for row in rows {
        let (
            id,
            run_id,
            status,
            created_at,
            model_path,
            package_path,
            validation_path,
            benchmark_path,
            export_dir,
        ) = row.map_err(to_error)?;
        let model_path = artifact_path(project_dir, &model_path);
        let package_path = artifact_path(project_dir, &package_path);
        let validation_path = artifact_path(project_dir, &validation_path);
        let benchmark_path = artifact_path(project_dir, &benchmark_path);
        let export_dir = artifact_path(project_dir, &export_dir);
        exports.push(ExportPackage {
            id,
            run_id,
            status: enum_from_string(&status)?,
            created_at,
            model_path: model_path.display().to_string(),
            package_path: package_path.display().to_string(),
            validation_path: validation_path.display().to_string(),
            benchmark_path: benchmark_path.display().to_string(),
            export_dir: export_dir.display().to_string(),
            package_metadata: read_optional_json_value(&package_path),
            validation_report: read_optional_json_value(&validation_path),
            benchmark_report: read_optional_json_value(&benchmark_path),
        });
    }
    Ok(exports)
}

fn load_project_events(
    db: &Connection,
    project_id: &str,
    limit: u32,
) -> Result<Vec<SidecarProgressEvent>, String> {
    let mut stmt = db
        .prepare(
            "SELECT job_id, operation, stream, line, event_json, project_id, run_id, export_id, created_at
             FROM (
                SELECT id, job_id, operation, stream, line, event_json, project_id, run_id, export_id, created_at
                FROM job_events
                WHERE project_id = ?1
                ORDER BY id DESC
                LIMIT ?2
             )
             ORDER BY id ASC",
        )
        .map_err(to_error)?;
    let rows = stmt
        .query_map(params![project_id, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, Option<String>>(4)?,
                row.get::<_, Option<String>>(5)?,
                row.get::<_, Option<String>>(6)?,
                row.get::<_, Option<String>>(7)?,
                row.get::<_, String>(8)?,
            ))
        })
        .map_err(to_error)?;

    let mut events = Vec::new();
    for row in rows {
        let (job_id, operation, stream, line, event_json, project_id, run_id, export_id, timestamp) =
            row.map_err(to_error)?;
        events.push(SidecarProgressEvent {
            job_id,
            operation,
            stream,
            line,
            json: event_json
                .map(|value| serde_json::from_str(&value).map_err(to_error))
                .transpose()?,
            project_id,
            run_id,
            export_id,
            timestamp,
        });
    }
    Ok(events)
}

fn insert_project(db: &Connection, project: &ProjectDetail) -> Result<(), String> {
    db.execute(
        "INSERT OR REPLACE INTO projects
         (id, name, target_kind, status, created_at, updated_at, notes, project_dir)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            project.id,
            project.name,
            enum_to_string(&project.target_kind)?,
            enum_to_string(&project.status)?,
            project.created_at,
            project.updated_at,
            project.notes,
            project.project_dir
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn upsert_audio_report(
    db: &Connection,
    project_id: &str,
    report: &AudioReport,
) -> Result<(), String> {
    db.execute(
        "INSERT INTO audio_files (project_id, status, report_json, updated_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(project_id) DO UPDATE SET
            status = excluded.status,
            report_json = excluded.report_json,
            updated_at = excluded.updated_at",
        params![
            project_id,
            enum_to_string(&report.status)?,
            serde_json::to_string(report).map_err(to_error)?,
            now()
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn insert_training_run(
    db: &Connection,
    project_id: &str,
    run: &TrainingRun,
    project_dir: &Path,
    run_dir: &Path,
) -> Result<(), String> {
    db.execute(
        "INSERT OR REPLACE INTO training_runs
         (id, project_id, preset, status, device, epochs, created_at, updated_at, metrics_json, log_path, run_dir, error)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, NULL)",
        params![
            run.id,
            project_id,
            run.preset,
            enum_to_string(&run.status)?,
            run.device,
            run.epochs,
            run.created_at,
            run.updated_at,
            optional_json(&run.metrics)?,
            relative_path_string(project_dir, Path::new(&run.log_path)),
            relative_path_string(project_dir, run_dir)
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_training_run_success(db: &Connection, run: &TrainingRun) -> Result<(), String> {
    db.execute(
        "UPDATE training_runs
         SET status = ?1, device = ?2, epochs = ?3, updated_at = ?4, metrics_json = ?5, error = NULL
         WHERE id = ?6",
        params![
            enum_to_string(&RunStatus::Completed)?,
            run.device,
            run.epochs,
            run.updated_at,
            optional_json(&run.metrics)?,
            run.id
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_training_run_failure(db: &Connection, run_id: &str, error: &str) -> Result<(), String> {
    db.execute(
        "UPDATE training_runs
         SET status = ?1, updated_at = ?2, error = ?3
         WHERE id = ?4",
        params![enum_to_string(&RunStatus::Failed)?, now(), error, run_id],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_training_run_status(
    db: &Connection,
    run_id: &str,
    status: &RunStatus,
    error: Option<&str>,
) -> Result<(), String> {
    db.execute(
        "UPDATE training_runs
         SET status = ?1, updated_at = ?2, error = ?3
         WHERE id = ?4",
        params![enum_to_string(status)?, now(), error, run_id],
    )
    .map_err(to_error)?;
    Ok(())
}

fn active_training_job_for_run(db: &Connection, run_id: &str) -> Result<Option<String>, String> {
    db.query_row(
        "SELECT id
         FROM jobs
         WHERE run_id = ?1
           AND operation = 'train'
           AND status IN ('queued', 'running', 'cancelling')
         ORDER BY created_at DESC
         LIMIT 1",
        params![run_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(to_error)
}

fn job_status(db: &Connection, job_id: &str) -> Result<Option<String>, String> {
    db.query_row(
        "SELECT status FROM jobs WHERE id = ?1",
        params![job_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(to_error)
}

fn ensure_no_active_project_job(db: &Connection, project_id: &str) -> Result<(), String> {
    let active_job = db
        .query_row(
            "SELECT operation
             FROM jobs
             WHERE project_id = ?1
               AND status IN ('queued', 'running', 'cancelling')
             ORDER BY created_at DESC
             LIMIT 1",
            params![project_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(to_error)?;
    if let Some(operation) = active_job {
        return Err(format!(
            "Project already has an active {operation} job. Wait, cancel, or resume before changing this project."
        ));
    }
    Ok(())
}

fn training_run_dir(db: &Connection, run_id: &str) -> Result<String, String> {
    db.query_row(
        "SELECT run_dir FROM training_runs WHERE id = ?1",
        params![run_id],
        |row| row.get(0),
    )
    .map_err(to_error)
}

fn best_checkpoint_path(run_dir: &Path) -> Option<PathBuf> {
    let keras = run_dir.join("checkpoints/best-model.keras");
    if keras.exists() {
        return Some(keras);
    }
    let torch = run_dir.join("checkpoints/best-checkpoint.pt");
    if torch.exists() {
        return Some(torch);
    }
    None
}

fn training_run_backend(run_dir: &Path) -> String {
    for file_name in ["training-report.json", "train-manifest.json"] {
        if let Some(value) = read_optional_json_value(&run_dir.join(file_name)) {
            if let Some(backend) = value
                .get("backend")
                .and_then(|backend| backend.as_str())
                .and_then(|backend| normalize_backend(backend).ok())
            {
                return backend.to_string();
            }
        }
    }

    if run_dir.join("checkpoints/best-model.keras").exists() {
        return "keras".to_string();
    }
    if run_dir.join("checkpoints/best-checkpoint.pt").exists() {
        return "pytorch".to_string();
    }
    default_training_backend()
}

fn validate_resume_checkpoint_backend(checkpoint_path: &Path, backend: &str) -> Result<(), String> {
    let checkpoint_backend = if checkpoint_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("keras"))
    {
        "keras"
    } else if checkpoint_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pt"))
    {
        "pytorch"
    } else {
        return Err(format!(
            "Resume checkpoint format is not supported: {}",
            checkpoint_path.display()
        ));
    };

    if checkpoint_backend != backend {
        return Err(format!(
            "Selected runtime backend is {backend}, but the chosen checkpoint is {checkpoint_backend}. Switch backend or choose a matching run."
        ));
    }
    Ok(())
}

fn write_final_export_package_metadata(input: FinalExportPackageInput<'_>) -> Result<(), String> {
    let existing_package =
        read_optional_json_value(input.package_path).unwrap_or_else(|| serde_json::json!({}));
    let validation_report = read_optional_json_value(input.validation_path);
    let benchmark_report = read_optional_json_value(input.benchmark_path);
    let model_json = read_optional_json_value(input.model_path);
    let model_metadata = model_json
        .as_ref()
        .and_then(|value| value.get("metadata"))
        .cloned();
    let backend = existing_package
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            model_metadata
                .as_ref()
                .and_then(|value| value.get("backend"))
                .and_then(serde_json::Value::as_str)
        })
        .unwrap_or("unknown");
    let rtneural_commit = existing_package
        .get("compatibility")
        .and_then(|value| value.get("rtneural_commit"))
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            model_metadata
                .as_ref()
                .and_then(|value| value.get("rtneural_commit"))
                .and_then(serde_json::Value::as_str)
        });
    let created_at = existing_package
        .get("created_at")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(now);
    let updated_at = now();
    let package = serde_json::json!({
        "schema_version": 2,
        "package_format": "rtneural-trainer-export",
        "id": input.export_id,
        "name": input.project_name,
        "status": "ready",
        "preset": input.run.preset,
        "backend": backend,
        "sample_rate": input.sample_rate,
        "latency_samples": input.latency_samples,
        "project": {
            "id": input.project_id,
            "name": input.project_name
        },
        "run": {
            "id": input.run.id,
            "preset": input.run.preset,
            "device": input.run.device,
            "epochs": input.run.epochs,
            "created_at": input.run.created_at,
            "updated_at": input.run.updated_at,
            "metrics": input.run.metrics
        },
        "model": {
            "format": "rtneural-json",
            "path": relative_path_string(input.export_dir, input.model_path),
            "sample_rate": input.sample_rate,
            "latency_samples": input.latency_samples,
            "backend": backend,
            "metadata": model_metadata
        },
        "artifacts": [
            export_artifact_metadata(input.export_dir, "model", input.model_path, "application/json"),
            export_artifact_metadata(input.export_dir, "validation_report", input.validation_path, "application/json"),
            export_artifact_metadata(input.export_dir, "benchmark_report", input.benchmark_path, "application/json")
        ],
        "model_path": relative_path_string(input.export_dir, input.model_path),
        "validation_path": relative_path_string(input.export_dir, input.validation_path),
        "benchmark_path": relative_path_string(input.export_dir, input.benchmark_path),
        "package_path": relative_path_string(input.export_dir, input.package_path),
        "quality": input.run.metrics,
        "validation": validation_report,
        "benchmark": benchmark_report,
        "generated_by": {
            "app": "RTNeural Trainer",
            "version": env!("CARGO_PKG_VERSION"),
            "pipeline": "rttrainer export + native rtneural-validator"
        },
        "compatibility": {
            "rtneural_commit": rtneural_commit,
            "rtneural_json": true,
            "dynamic_json": true,
            "schema": "rttrainer-rtneural-json-v0",
            "aidax": {
                "status": "deferred",
                "reason": "Pending format and license review before emitting an AIDA-X envelope."
            }
        },
        "created_at": created_at,
        "updated_at": updated_at
    });
    write_json(input.package_path, &package)
}

fn export_artifact_metadata(
    export_dir: &Path,
    role: &str,
    path: &Path,
    media_type: &str,
) -> serde_json::Value {
    let metadata = fs::metadata(path).ok();
    serde_json::json!({
        "role": role,
        "path": relative_path_string(export_dir, path),
        "media_type": media_type,
        "exists": metadata.is_some(),
        "size_bytes": metadata.map(|item| item.len())
    })
}

fn insert_export_package(
    db: &Connection,
    project_id: &str,
    package: &ExportPackage,
    project_dir: &Path,
    export_dir: &Path,
) -> Result<(), String> {
    db.execute(
        "INSERT OR REPLACE INTO exports
         (id, project_id, run_id, status, created_at, model_path, package_path, validation_path, benchmark_path, export_dir, error)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL)",
        params![
            package.id,
            project_id,
            package.run_id,
            enum_to_string(&package.status)?,
            package.created_at,
            relative_path_string(project_dir, Path::new(&package.model_path)),
            relative_path_string(project_dir, Path::new(&package.package_path)),
            relative_path_string(project_dir, Path::new(&package.validation_path)),
            relative_path_string(project_dir, Path::new(&package.benchmark_path)),
            relative_path_string(project_dir, export_dir)
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_export_status(
    db: &Connection,
    export_id: &str,
    status: &ExportStatus,
    error: Option<&str>,
) -> Result<(), String> {
    db.execute(
        "UPDATE exports SET status = ?1, error = ?2 WHERE id = ?3",
        params![enum_to_string(status)?, error, export_id],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_project_status(
    db: &Connection,
    project_id: &str,
    status: &ProjectStatus,
    updated_at: &str,
) -> Result<(), String> {
    db.execute(
        "UPDATE projects SET status = ?1, updated_at = ?2 WHERE id = ?3",
        params![enum_to_string(status)?, updated_at, project_id],
    )
    .map_err(to_error)?;
    Ok(())
}

fn update_project_name(
    db: &Connection,
    project_id: &str,
    name: &str,
    updated_at: &str,
) -> Result<(), String> {
    let rows_updated = db
        .execute(
            "UPDATE projects SET name = ?1, updated_at = ?2 WHERE id = ?3",
            params![name, updated_at, project_id],
        )
        .map_err(to_error)?;
    if rows_updated == 0 {
        return Err("Project not found.".to_string());
    }
    Ok(())
}

fn update_project_notes(
    db: &Connection,
    project_id: &str,
    notes: &str,
    updated_at: &str,
) -> Result<(), String> {
    db.execute(
        "UPDATE projects SET notes = ?1, updated_at = ?2 WHERE id = ?3",
        params![notes, updated_at, project_id],
    )
    .map_err(to_error)?;
    Ok(())
}

fn create_job(
    state: &AppState,
    operation: &str,
    project_id: Option<&str>,
    run_id: Option<&str>,
    export_id: Option<&str>,
) -> Result<String, String> {
    let job_id = format!("job_{}", Uuid::new_v4().simple());
    let timestamp = now();
    let db = state.db.lock().map_err(lock_error)?;
    db.execute(
        "INSERT INTO jobs
         (id, project_id, run_id, export_id, operation, status, created_at, updated_at, started_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7, ?7)",
        params![
            job_id,
            project_id,
            run_id,
            export_id,
            operation,
            enum_to_string(&JobStatus::Running)?,
            timestamp
        ],
    )
    .map_err(to_error)?;
    Ok(job_id)
}

fn mark_job_completed(state: &AppState, job_id: &str) -> Result<(), String> {
    update_job_status(state, job_id, &JobStatus::Completed, None)
}

fn mark_job_failed(state: &AppState, job_id: &str, error: &str) -> Result<(), String> {
    update_job_status(state, job_id, &JobStatus::Failed, Some(error))
}

fn update_job_status(
    state: &AppState,
    job_id: &str,
    status: &JobStatus,
    error: Option<&str>,
) -> Result<(), String> {
    let db = state.db.lock().map_err(lock_error)?;
    update_job_status_in_db(&db, job_id, status, error)
}

fn update_job_status_in_db(
    db: &Connection,
    job_id: &str,
    status: &JobStatus,
    error: Option<&str>,
) -> Result<(), String> {
    let timestamp = now();
    let finished_at = if matches!(
        status,
        JobStatus::Completed | JobStatus::Failed | JobStatus::Interrupted
    ) {
        Some(timestamp.as_str())
    } else {
        None
    };
    db.execute(
        "UPDATE jobs
         SET status = ?1, updated_at = ?2, finished_at = ?3, error = ?4
         WHERE id = ?5",
        params![
            enum_to_string(status)?,
            timestamp,
            finished_at,
            error,
            job_id
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn persist_job_event(event: &SidecarProgressEvent, state: &AppState) -> Result<(), String> {
    let db = state.db.lock().map_err(lock_error)?;
    db.execute(
        "INSERT INTO job_events
         (job_id, project_id, run_id, export_id, operation, stream, line, event_json, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            event.job_id,
            event.project_id,
            event.run_id,
            event.export_id,
            event.operation,
            event.stream,
            event.line,
            optional_json(&event.json)?,
            event.timestamp
        ],
    )
    .map_err(to_error)?;
    Ok(())
}

fn enum_to_string<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_value(value)
        .map_err(to_error)?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| "Enum did not serialize to a string.".to_string())
}

fn enum_from_string<T: DeserializeOwned>(value: &str) -> Result<T, String> {
    serde_json::from_value(serde_json::Value::String(value.to_string())).map_err(to_error)
}

fn optional_json<T: Serialize>(value: &Option<T>) -> Result<Option<String>, String> {
    value
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(to_error)
}

fn relative_path_string(project_dir: &Path, path: &Path) -> String {
    path.strip_prefix(project_dir)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn artifact_path(project_dir: &Path, stored_path: &str) -> PathBuf {
    let path = PathBuf::from(stored_path);
    if path.is_absolute() {
        path
    } else {
        project_dir.join(path)
    }
}

fn ensure_artifact_inside(project_dir: &Path, path: &Path) -> Result<(), String> {
    let root = project_dir.canonicalize().map_err(to_error)?;
    let target = path.canonicalize().map_err(to_error)?;
    if target.starts_with(&root) {
        Ok(())
    } else {
        Err("Artifact path is outside the project directory.".to_string())
    }
}

fn run_preview_artifact(
    kind: &str,
    label: &str,
    path: &Path,
) -> Result<RunPreviewArtifact, String> {
    let metadata = fs::metadata(path).ok();
    let summary = if metadata.is_some() {
        Some(read_wav_preview_summary(path, 120)?)
    } else {
        None
    };
    Ok(RunPreviewArtifact {
        kind: kind.to_string(),
        label: label.to_string(),
        path: path.display().to_string(),
        exists: metadata.is_some(),
        size_bytes: metadata.map(|item| item.len()),
        sample_rate: summary.as_ref().map(|item| item.sample_rate),
        duration_seconds: summary.as_ref().map(|item| item.duration_seconds),
        peak: summary.as_ref().map(|item| item.peak),
        peaks: summary
            .as_ref()
            .map(|item| item.peaks.clone())
            .unwrap_or_default(),
        waveform: summary.map(|item| item.waveform).unwrap_or_default(),
    })
}

fn prepared_audio_path(
    project_dir: &Path,
    prepared: Option<&serde_json::Value>,
    key: &str,
    fallback: &str,
) -> Result<PathBuf, String> {
    let stored_path = prepared
        .and_then(|value| value.get(key))
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(fallback);
    Ok(artifact_path(project_dir, stored_path))
}

fn waveform_track(
    kind: &str,
    label: &str,
    path: &Path,
    bins: usize,
) -> Result<WaveformTrack, String> {
    let summary = read_wav_preview_summary(path, bins)?;
    Ok(WaveformTrack {
        kind: kind.to_string(),
        label: label.to_string(),
        path: path.display().to_string(),
        sample_rate: summary.sample_rate,
        duration_seconds: summary.duration_seconds,
        peak: summary.peak,
        waveform: summary.waveform,
    })
}

fn read_wav_preview_summary(path: &Path, bins: usize) -> Result<WavPreviewSummary, String> {
    let bytes = fs::read(path).map_err(to_error)?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{} is not a RIFF/WAVE file.", path.display()));
    }

    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;
    let mut data_range = None;
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size =
            u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().map_err(to_error)?)
                as usize;
        let chunk_start = offset + 8;
        let chunk_end = chunk_start.saturating_add(chunk_size);
        if chunk_end > bytes.len() {
            return Err(format!("{} has a truncated WAV chunk.", path.display()));
        }

        if chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err(format!("{} has an invalid fmt chunk.", path.display()));
            }
            let format = u16::from_le_bytes(
                bytes[chunk_start..chunk_start + 2]
                    .try_into()
                    .map_err(to_error)?,
            );
            if format != 1 {
                return Err(format!(
                    "{} uses unsupported WAV format {format}; expected PCM.",
                    path.display()
                ));
            }
            channels = Some(u16::from_le_bytes(
                bytes[chunk_start + 2..chunk_start + 4]
                    .try_into()
                    .map_err(to_error)?,
            ));
            sample_rate = Some(u32::from_le_bytes(
                bytes[chunk_start + 4..chunk_start + 8]
                    .try_into()
                    .map_err(to_error)?,
            ));
            bits_per_sample = Some(u16::from_le_bytes(
                bytes[chunk_start + 14..chunk_start + 16]
                    .try_into()
                    .map_err(to_error)?,
            ));
        } else if chunk_id == b"data" {
            data_range = Some((chunk_start, chunk_end));
        }

        offset = chunk_end + (chunk_size % 2);
    }

    let sample_rate =
        sample_rate.ok_or_else(|| format!("{} is missing sample rate.", path.display()))?;
    let channels =
        channels.ok_or_else(|| format!("{} is missing channel count.", path.display()))?;
    let bits_per_sample =
        bits_per_sample.ok_or_else(|| format!("{} is missing bit depth.", path.display()))?;
    if channels == 0 {
        return Err(format!("{} has zero channels.", path.display()));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "{} uses {bits_per_sample}-bit samples; expected 16-bit PCM.",
            path.display()
        ));
    }

    let (data_start, data_end) =
        data_range.ok_or_else(|| format!("{} is missing audio data.", path.display()))?;
    let bytes_per_frame = usize::from(channels) * 2;
    let frame_count = (data_end - data_start) / bytes_per_frame;
    if frame_count == 0 {
        let bin_count = bins.max(1);
        return Ok(WavPreviewSummary {
            sample_rate,
            duration_seconds: 0.0,
            peak: 0.0,
            peaks: vec![0.0; bin_count],
            waveform: vec![
                WaveformBin {
                    min: 0.0,
                    max: 0.0,
                    peak: 0.0
                };
                bin_count
            ],
        });
    }

    let bin_count = bins.max(1);
    let mut peaks = vec![0.0_f64; bin_count];
    let mut waveform = vec![
        WaveformBin {
            min: 0.0,
            max: 0.0,
            peak: 0.0
        };
        bin_count
    ];
    let mut peak = 0.0_f64;
    for frame_index in 0..frame_count {
        let frame_start = data_start + frame_index * bytes_per_frame;
        let mut frame_value = 0.0_f64;
        for channel in 0..usize::from(channels) {
            let sample_start = frame_start + channel * 2;
            let sample = i16::from_le_bytes(
                bytes[sample_start..sample_start + 2]
                    .try_into()
                    .map_err(to_error)?,
            );
            frame_value += f64::from(sample) / 32768.0;
        }
        frame_value /= f64::from(channels);
        let frame_peak = frame_value.abs();
        peak = peak.max(frame_peak);
        let bin = ((frame_index * peaks.len()) / frame_count).min(peaks.len() - 1);
        peaks[bin] = peaks[bin].max(frame_peak);
        waveform[bin].min = waveform[bin].min.min(frame_value);
        waveform[bin].max = waveform[bin].max.max(frame_value);
        waveform[bin].peak = waveform[bin].peak.max(frame_peak);
    }

    Ok(WavPreviewSummary {
        sample_rate,
        duration_seconds: frame_count as f64 / f64::from(sample_rate),
        peak,
        peaks,
        waveform,
    })
}

fn open_folder(path: &Path) -> Result<(), String> {
    let status = open_folder_command(path).status().map_err(to_error)?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Failed to open export folder: {status}"))
    }
}

#[cfg(target_os = "macos")]
fn open_folder_command(path: &Path) -> StdCommand {
    let mut command = StdCommand::new("open");
    command.arg(path);
    command
}

#[cfg(target_os = "windows")]
fn open_folder_command(path: &Path) -> StdCommand {
    let mut command = StdCommand::new("explorer");
    command.arg(path);
    command
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_folder_command(path: &Path) -> StdCommand {
    let mut command = StdCommand::new("xdg-open");
    command.arg(path);
    command
}

fn summarize_project(project: &ProjectDetail) -> ProjectSummary {
    let best_quality = project
        .runs
        .iter()
        .filter_map(|run| run.metrics.as_ref().map(|metrics| metrics.esr))
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    ProjectSummary {
        id: project.id.clone(),
        name: project.name.clone(),
        target_kind: project.target_kind.clone(),
        status: project.status.clone(),
        created_at: project.created_at.clone(),
        updated_at: project.updated_at.clone(),
        audio_status: project
            .audio
            .as_ref()
            .map(|audio| audio.status.clone())
            .unwrap_or(AudioStatus::Missing),
        best_quality,
        export_status: project.exports.last().map(|export| export.status.clone()),
    }
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(value).map_err(to_error)?;
    fs::write(path, format!("{raw}\n")).map_err(to_error)
}

fn spawn_training_worker(
    app: tauri::AppHandle,
    project_id: String,
    run_id: String,
    job_id: String,
    manifest_path: PathBuf,
    project_dir: PathBuf,
    run_dir: PathBuf,
    log_path: PathBuf,
) {
    thread::spawn(move || {
        let state = app.state::<AppState>();
        let context = SidecarContext {
            job_id: job_id.clone(),
            operation: "train".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run_id.clone()),
            export_id: None,
        };
        let cancelled_before_start = {
            let db = match state.db.lock().map_err(lock_error) {
                Ok(db) => db,
                Err(error) => {
                    finish_training_failure(&app, &state, &project_id, &run_id, &job_id, &error);
                    return;
                }
            };
            matches!(job_status(&db, &job_id), Ok(Some(status)) if status == "cancelling")
        };
        if cancelled_before_start {
            finish_training_failure(
                &app,
                &state,
                &project_id,
                &run_id,
                &job_id,
                "Training was cancelled before the process started.",
            );
            return;
        }
        let result = run_rttrainer(&app, &state, "train", &manifest_path, context);
        match result {
            Ok(sidecar) => {
                if let Err(error) = fs::write(&log_path, sidecar.stdout).map_err(to_error) {
                    finish_training_failure(&app, &state, &project_id, &run_id, &job_id, &error);
                    return;
                }
                if !sidecar.stderr.trim().is_empty() {
                    let _ = fs::write(run_dir.join("stderr.log"), sidecar.stderr);
                }
                match read_json::<TrainingReport>(&run_dir.join("training-report.json")) {
                    Ok(report) => {
                        let completed_run = TrainingRun {
                            id: report.run_id,
                            preset: report.preset,
                            backend: report.backend,
                            status: RunStatus::Completed,
                            device: report.device,
                            epochs: report.epochs,
                            created_at: report.created_at.clone(),
                            updated_at: now(),
                            metrics: Some(report.metrics),
                            log_path: relative_path_string(&project_dir, &log_path),
                        };
                        let update_result = (|| -> Result<(), String> {
                            let db = state.db.lock().map_err(lock_error)?;
                            update_training_run_success(&db, &completed_run)?;
                            update_project_status(&db, &project_id, &ProjectStatus::Ready, &now())?;
                            Ok(())
                        })();
                        if let Err(error) = update_result {
                            finish_training_failure(
                                &app,
                                &state,
                                &project_id,
                                &run_id,
                                &job_id,
                                &error,
                            );
                            return;
                        }
                        if let Err(error) = mark_job_completed(&state, &job_id) {
                            emit_sidecar_line(
                                &app,
                                &job_context(&job_id, &project_id, &run_id),
                                "system",
                                &error,
                            );
                        } else {
                            emit_sidecar_line(
                                &app,
                                &job_context(&job_id, &project_id, &run_id),
                                "system",
                                "training completed",
                            );
                        }
                    }
                    Err(error) => {
                        finish_training_failure(
                            &app,
                            &state,
                            &project_id,
                            &run_id,
                            &job_id,
                            &error,
                        );
                    }
                }
            }
            Err(error) => {
                finish_training_failure(&app, &state, &project_id, &run_id, &job_id, &error);
            }
        }
    });
}

fn finish_training_failure(
    app: &tauri::AppHandle,
    state: &AppState,
    project_id: &str,
    run_id: &str,
    job_id: &str,
    error: &str,
) {
    let context = job_context(job_id, project_id, run_id);
    let mut final_line = "training failed";
    let update_result = (|| -> Result<(), String> {
        let db = state.db.lock().map_err(lock_error)?;
        let cancelling = job_status(&db, job_id)?
            .map(|status| status == "cancelling")
            .unwrap_or(false);
        if cancelling {
            final_line = "training interrupted";
            update_training_run_status(
                &db,
                run_id,
                &RunStatus::Interrupted,
                Some("Training was cancelled."),
            )?;
            update_job_status_in_db(
                &db,
                job_id,
                &JobStatus::Interrupted,
                Some("Training was cancelled."),
            )?;
        } else {
            update_training_run_failure(&db, run_id, error)?;
            update_job_status_in_db(&db, job_id, &JobStatus::Failed, Some(error))?;
        }
        update_project_status(&db, project_id, &ProjectStatus::Ready, &now())?;
        Ok(())
    })();
    if let Err(update_error) = update_result {
        emit_sidecar_line(app, &context, "system", &update_error);
    } else {
        emit_sidecar_line(app, &context, "system", final_line);
    }
}

fn job_context(job_id: &str, project_id: &str, run_id: &str) -> SidecarContext {
    SidecarContext {
        job_id: job_id.to_string(),
        operation: "train".to_string(),
        project_id: Some(project_id.to_string()),
        run_id: Some(run_id.to_string()),
        export_id: None,
    }
}

fn run_rttrainer(
    app: &tauri::AppHandle,
    state: &AppState,
    command: &str,
    manifest_path: &Path,
    context: SidecarContext,
) -> Result<SidecarOutput, String> {
    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rttrainer {command} started"),
    );
    let args = vec![
        command.to_string(),
        "--manifest".to_string(),
        manifest_path.display().to_string(),
    ];
    let output = run_rttrainer_args(app, state, command, &args, context.clone())?;
    if output.status.success() {
        emit_sidecar_line(
            app,
            &context,
            "system",
            &format!("rttrainer {command} finished"),
        );
        return Ok(SidecarOutput {
            stdout: output.stdout,
            stderr: output.stderr,
        });
    }

    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rttrainer {command} failed"),
    );
    Err(format!(
        "rttrainer {command} failed with status {}.\nstdout:\n{}\nstderr:\n{}",
        output.status, output.stdout, output.stderr
    ))
}

fn run_rttrainer_args(
    app: &tauri::AppHandle,
    state: &AppState,
    action: &str,
    args: &[String],
    context: SidecarContext,
) -> Result<StreamingProcessOutput, String> {
    if let Some(python) = external_python_executable(state)? {
        emit_sidecar_line(
            app,
            &context,
            "system",
            &format!("using external Python {}", python.display()),
        );
        let mut process = StdCommand::new(&python);
        process
            .current_dir(&state.workspace_root)
            .arg("-m")
            .arg("rttrainer")
            .args(args);
        return run_streaming_process(app, &context, process);
    }

    if bundled_sidecar_exists(RTTRAINER_SIDECAR) {
        let process = app
            .shell()
            .sidecar(RTTRAINER_SIDECAR)
            .map_err(to_error)?
            .args(args.to_vec());
        return run_streaming_shell_command(app, &context, process);
    }

    let trainer_dir = state.workspace_root.join("trainer");
    let cache_dir = state.workspace_root.join(".uv-cache");
    let mut process = StdCommand::new("uv");
    process
        .current_dir(&trainer_dir)
        .env("UV_CACHE_DIR", &cache_dir)
        .arg("run");
    for extra in uv_extras_for_rttrainer(action, args) {
        process.arg("--extra").arg(extra);
    }
    process.arg("python").arg("-m").arg("rttrainer").args(args);
    run_streaming_process(app, &context, process)
}

fn uv_extras_for_rttrainer(action: &str, args: &[String]) -> Vec<&'static str> {
    if action == "inspect-device" {
        return vec!["tensorflow", "training"];
    }
    if !matches!(action, "train" | "evaluate" | "export") {
        return Vec::new();
    }
    if rttrainer_args_backend(args)
        .as_deref()
        .map(|backend| backend == "pytorch")
        .unwrap_or(false)
    {
        vec!["training"]
    } else {
        vec!["tensorflow"]
    }
}

fn rttrainer_args_backend(args: &[String]) -> Option<String> {
    let manifest_path = args
        .windows(2)
        .find(|items| items[0] == "--manifest")
        .map(|items| PathBuf::from(&items[1]))?;
    let manifest = read_optional_json_value(&manifest_path)?;
    manifest
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .and_then(|value| normalize_backend(value).ok())
        .map(str::to_string)
}

fn run_validator(
    app: &tauri::AppHandle,
    state: &AppState,
    args: Vec<String>,
    context: SidecarContext,
) -> Result<SidecarOutput, String> {
    let action = args.first().cloned().unwrap_or_else(|| "run".to_string());

    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rtneural-validator {action} started"),
    );
    let (output, validator_label) = if bundled_sidecar_exists(RTNEURAL_VALIDATOR_SIDECAR) {
        let process = app
            .shell()
            .sidecar(RTNEURAL_VALIDATOR_SIDECAR)
            .map_err(to_error)?
            .args(args);
        (
            run_streaming_shell_command(app, &context, process)?,
            RTNEURAL_VALIDATOR_SIDECAR.to_string(),
        )
    } else {
        let validator = validator_binary_path(state)?;
        let mut process = StdCommand::new(&validator);
        process.args(args);
        (
            run_streaming_process(app, &context, process)?,
            validator.display().to_string(),
        )
    };
    if output.status.success() {
        emit_sidecar_line(
            app,
            &context,
            "system",
            &format!("rtneural-validator {action} finished"),
        );
        return Ok(SidecarOutput {
            stdout: output.stdout,
            stderr: output.stderr,
        });
    }

    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rtneural-validator {action} failed"),
    );
    Err(format!(
        "{} failed with status {}.\nstdout:\n{}\nstderr:\n{}",
        validator_label, output.status, output.stdout, output.stderr
    ))
}

struct StreamingProcessOutput {
    status: StreamingExitStatus,
    stdout: String,
    stderr: String,
}

struct StreamingExitStatus {
    code: Option<i32>,
    signal: Option<i32>,
}

impl StreamingExitStatus {
    fn success(&self) -> bool {
        self.code == Some(0)
    }
}

impl From<std::process::ExitStatus> for StreamingExitStatus {
    fn from(status: std::process::ExitStatus) -> Self {
        Self {
            code: status.code(),
            signal: None,
        }
    }
}

impl fmt::Display for StreamingExitStatus {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.code, self.signal) {
            (Some(code), _) => write!(formatter, "exit code {code}"),
            (None, Some(signal)) => write!(formatter, "signal {signal}"),
            (None, None) => write!(formatter, "terminated without exit code"),
        }
    }
}

fn run_streaming_process(
    app: &tauri::AppHandle,
    context: &SidecarContext,
    mut process: StdCommand,
) -> Result<StreamingProcessOutput, String> {
    let mut child = process
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(to_error)?;
    register_active_process(app, context, child.id())?;
    if context.operation == "train" && job_is_cancelling(app, &context.job_id)? {
        if let Err(error) = terminate_pid(child.id()) {
            emit_sidecar_line(app, context, "system", &error);
        }
    }

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture sidecar stdout.".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Failed to capture sidecar stderr.".to_string())?;

    let stdout_reader = spawn_stream_reader(app.clone(), context.clone(), "stdout", stdout);
    let stderr_reader = spawn_stream_reader(app.clone(), context.clone(), "stderr", stderr);
    let status = child.wait().map_err(to_error)?;
    unregister_active_process(app, &context.job_id);
    let stdout = join_stream_reader(stdout_reader, "stdout")?;
    let stderr = join_stream_reader(stderr_reader, "stderr")?;

    Ok(StreamingProcessOutput {
        status: status.into(),
        stdout,
        stderr,
    })
}

fn run_streaming_shell_command(
    app: &tauri::AppHandle,
    context: &SidecarContext,
    process: ShellCommand,
) -> Result<StreamingProcessOutput, String> {
    let (mut rx, child) = process.spawn().map_err(to_error)?;
    register_active_process(app, context, child.pid())?;
    if context.operation == "train" && job_is_cancelling(app, &context.job_id)? {
        if let Err(error) = terminate_pid(child.pid()) {
            emit_sidecar_line(app, context, "system", &error);
        }
    }

    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut status = None;
    let mut command_error = None;

    tauri::async_runtime::block_on(async {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    collect_shell_output(app, context, "stdout", &mut stdout, bytes);
                }
                CommandEvent::Stderr(bytes) => {
                    collect_shell_output(app, context, "stderr", &mut stderr, bytes);
                }
                CommandEvent::Error(error) => {
                    command_error = Some(error);
                }
                CommandEvent::Terminated(payload) => {
                    status = Some(StreamingExitStatus {
                        code: payload.code,
                        signal: payload.signal,
                    });
                }
                _ => {}
            }
        }
    });

    unregister_active_process(app, &context.job_id);
    if let Some(error) = command_error {
        return Err(error);
    }

    Ok(StreamingProcessOutput {
        status: status.unwrap_or(StreamingExitStatus {
            code: None,
            signal: None,
        }),
        stdout,
        stderr,
    })
}

fn collect_shell_output(
    app: &tauri::AppHandle,
    context: &SidecarContext,
    stream: &str,
    collected: &mut String,
    bytes: Vec<u8>,
) {
    let text = String::from_utf8_lossy(&bytes);
    collected.push_str(&text);
    if !text.ends_with('\n') {
        collected.push('\n');
    }
    for line in text.lines() {
        if !line.trim().is_empty() {
            emit_sidecar_line(app, context, stream, line);
        }
    }
}

fn spawn_stream_reader<R>(
    app: tauri::AppHandle,
    context: SidecarContext,
    stream: &'static str,
    reader: R,
) -> JoinHandle<Result<String, String>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut collected = String::new();
        for line in BufReader::new(reader).lines() {
            let line = line.map_err(to_error)?;
            collected.push_str(&line);
            collected.push('\n');
            if !line.trim().is_empty() {
                emit_sidecar_line(&app, &context, stream, &line);
            }
        }
        Ok(collected)
    })
}

fn join_stream_reader(
    handle: JoinHandle<Result<String, String>>,
    stream: &str,
) -> Result<String, String> {
    handle
        .join()
        .map_err(|_| format!("Sidecar {stream} reader panicked."))?
}

fn register_active_process(
    app: &tauri::AppHandle,
    context: &SidecarContext,
    pid: u32,
) -> Result<(), String> {
    let state = app.state::<AppState>();
    let mut active_jobs = state.active_jobs.lock().map_err(lock_error)?;
    active_jobs.insert(context.job_id.clone(), ActiveProcess { pid });
    Ok(())
}

fn unregister_active_process(app: &tauri::AppHandle, job_id: &str) {
    let state = app.state::<AppState>();
    if let Ok(mut active_jobs) = state.active_jobs.lock() {
        active_jobs.remove(job_id);
    };
}

fn terminate_active_process(state: &AppState, job_id: &str) -> Result<bool, String> {
    let process = {
        let active_jobs = state.active_jobs.lock().map_err(lock_error)?;
        active_jobs.get(job_id).cloned()
    };
    let Some(process) = process else {
        return Ok(false);
    };
    terminate_pid(process.pid)?;
    Ok(true)
}

fn job_is_cancelling(app: &tauri::AppHandle, job_id: &str) -> Result<bool, String> {
    let state = app.state::<AppState>();
    let db = state.db.lock().map_err(lock_error)?;
    Ok(job_status(&db, job_id)?
        .map(|status| status == "cancelling")
        .unwrap_or(false))
}

#[cfg(unix)]
fn terminate_pid(pid: u32) -> Result<(), String> {
    let status = StdCommand::new("kill")
        .arg("-TERM")
        .arg(pid.to_string())
        .status()
        .map_err(to_error)?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Failed to terminate process {pid}: {status}"))
    }
}

#[cfg(windows)]
fn terminate_pid(pid: u32) -> Result<(), String> {
    let status = StdCommand::new("taskkill")
        .arg("/PID")
        .arg(pid.to_string())
        .arg("/T")
        .arg("/F")
        .status()
        .map_err(to_error)?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("Failed to terminate process {pid}: {status}"))
    }
}

fn emit_sidecar_line(app: &tauri::AppHandle, context: &SidecarContext, stream: &str, line: &str) {
    let payload = SidecarProgressEvent {
        job_id: context.job_id.clone(),
        operation: context.operation.clone(),
        stream: stream.to_string(),
        line: line.to_string(),
        json: serde_json::from_str::<serde_json::Value>(line).ok(),
        project_id: context.project_id.clone(),
        run_id: context.run_id.clone(),
        export_id: context.export_id.clone(),
        timestamp: now(),
    };
    let state = app.state::<AppState>();
    let _ = persist_job_event(&payload, &state);
    let _ = app.emit("sidecar-progress", payload);
}

fn validator_binary_path(state: &AppState) -> Result<PathBuf, String> {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(path) = existing_binary(dir, RTNEURAL_VALIDATOR_SIDECAR) {
                return Ok(path);
            }
        }
    }

    let dev_dir = state.workspace_root.join("native/rtneural-validator/build");
    if let Some(path) = existing_binary(&dev_dir, "rtneural-validator") {
        return Ok(path);
    }

    Err("rtneural-validator binary not found. Build it with: cmake --build native/rtneural-validator/build".to_string())
}

fn bundled_sidecar_exists(stem: &str) -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.parent()
                .map(|dir| existing_binary(dir, stem).is_some())
        })
        .unwrap_or(false)
}

fn existing_binary(dir: &Path, stem: &str) -> Option<PathBuf> {
    let direct = dir.join(stem);
    if direct.exists() {
        return Some(direct);
    }
    let exe = dir.join(format!("{stem}.exe"));
    if exe.exists() {
        return Some(exe);
    }
    None
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or(manifest_dir)
}

fn now() -> String {
    Utc::now().to_rfc3339()
}

fn lock_error<T>(_: std::sync::PoisonError<T>) -> String {
    "Application state lock failed.".to_string()
}

fn to_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

async fn run_blocking_command<T, F>(label: &'static str, task: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, String> + Send + 'static,
{
    tauri::async_runtime::spawn_blocking(task)
        .await
        .map_err(|error| format!("{label} worker failed: {error}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_preview_summary_reads_pcm16_peaks() {
        let root =
            std::env::temp_dir().join(format!("rttrainer-wav-test-{}", Uuid::new_v4().simple()));
        fs::create_dir_all(&root).expect("create temp dir");
        let wav_path = root.join("preview.wav");
        write_test_wav(&wav_path, &[0, 16_384, -32_768, 8_192], 48_000);

        let summary = read_wav_preview_summary(&wav_path, 4).expect("read preview summary");
        assert_eq!(summary.sample_rate, 48_000);
        assert_eq!(summary.peaks.len(), 4);
        assert_eq!(summary.waveform.len(), 4);
        assert!((summary.duration_seconds - (4.0 / 48_000.0)).abs() < f64::EPSILON);
        assert_eq!(summary.peak, 1.0);
        assert_eq!(summary.waveform[0].peak, 0.0);
        assert_eq!(summary.waveform[1].max, 0.5);
        assert_eq!(summary.waveform[2].min, -1.0);
        assert_eq!(summary.waveform[3].max, 0.25);
    }

    #[test]
    fn generated_sample_capture_writes_valid_preview_wavs() {
        let root = std::env::temp_dir().join(format!(
            "rttrainer-sample-wav-test-{}",
            Uuid::new_v4().simple()
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        let input_path = root.join("sample-input.wav");
        let target_path = root.join("sample-target.wav");
        let (input, target) = generated_sample_capture(48_000, 0.25);

        assert_eq!(input.len(), target.len());
        assert!(input.iter().any(|sample| sample.abs() > 0.01));
        assert!(target.iter().any(|sample| sample.abs() > 0.01));

        write_pcm16_wav(&input_path, &input, 48_000).expect("write input wav");
        write_pcm16_wav(&target_path, &target, 48_000).expect("write target wav");

        let input_summary =
            read_wav_preview_summary(&input_path, 16).expect("read input preview summary");
        let target_summary =
            read_wav_preview_summary(&target_path, 16).expect("read target preview summary");

        assert_eq!(input_summary.sample_rate, 48_000);
        assert_eq!(target_summary.sample_rate, 48_000);
        assert_eq!(input_summary.peaks.len(), 16);
        assert_eq!(target_summary.peaks.len(), 16);
        assert_eq!(input_summary.waveform.len(), 16);
        assert_eq!(target_summary.waveform.len(), 16);
        assert!(input_summary.peak > 0.01);
        assert!(target_summary.peak > 0.01);
        assert!(input_summary.waveform.iter().any(|bin| bin.max > 0.0));
        assert!(input_summary.waveform.iter().any(|bin| bin.min < 0.0));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn final_export_package_metadata_includes_native_reports() {
        let root =
            std::env::temp_dir().join(format!("rttrainer-export-test-{}", Uuid::new_v4().simple()));
        let export_dir = root.join("exports/export_test");
        fs::create_dir_all(&export_dir).expect("create export dir");
        let model_path = export_dir.join("model.rtneural.json");
        let package_path = export_dir.join("package.json");
        let validation_path = export_dir.join("validation-report.json");
        let benchmark_path = export_dir.join("benchmark-report.json");
        write_json(
            &model_path,
            &serde_json::json!({
                "metadata": {
                    "backend": "keras",
                    "schema": "rttrainer-rtneural-json-v0"
                }
            }),
        )
        .expect("write model json");
        write_json(
            &validation_path,
            &serde_json::json!({
                "status": "pass",
                "max_abs_error": 0.00001
            }),
        )
        .expect("write validation report");
        write_json(
            &benchmark_path,
            &serde_json::json!({
                "status": "pass",
                "realtime_factor": 42.0
            }),
        )
        .expect("write benchmark report");
        let run = TrainingRun {
            id: "run_test".to_string(),
            preset: "lstm_light".to_string(),
            backend: "keras".to_string(),
            status: RunStatus::Completed,
            device: "tensorflow-cpu".to_string(),
            epochs: 2,
            created_at: now(),
            updated_at: now(),
            metrics: None,
            log_path: "runs/run_test/events.jsonl".to_string(),
        };

        write_final_export_package_metadata(FinalExportPackageInput {
            project_name: "Package Test",
            project_id: "project_test",
            export_id: "export_test",
            run: &run,
            sample_rate: 48_000,
            latency_samples: 12,
            export_dir: &export_dir,
            model_path: &model_path,
            package_path: &package_path,
            validation_path: &validation_path,
            benchmark_path: &benchmark_path,
        })
        .expect("write final package metadata");

        let package: serde_json::Value = read_json(&package_path).expect("read package");
        assert_eq!(package["schema_version"], 2);
        assert_eq!(package["backend"], "keras");
        assert_eq!(package["validation"]["status"], "pass");
        assert_eq!(package["benchmark"]["realtime_factor"], 42.0);
        assert_eq!(package["compatibility"]["aidax"]["status"], "deferred");
    }

    #[test]
    fn sqlite_store_persists_events_and_recovers_running_jobs() {
        let mut db = Connection::open_in_memory().expect("open in-memory sqlite");
        configure_database(&mut db).expect("migrate sqlite");

        let root = std::env::temp_dir().join(format!("rttrainer-test-{}", Uuid::new_v4().simple()));
        fs::create_dir_all(&root).expect("create temp project root");
        let timestamp = now();
        let project = ProjectDetail {
            id: "project_test".to_string(),
            name: "SQLite test".to_string(),
            target_kind: TargetKind::Generic,
            status: ProjectStatus::Draft,
            created_at: timestamp.clone(),
            updated_at: timestamp,
            notes: String::new(),
            project_dir: root.display().to_string(),
            audio: None,
            runs: Vec::new(),
            exports: Vec::new(),
        };
        insert_project(&db, &project).expect("insert project");
        let audio = AudioReport {
            input: AudioFileReport {
                sample_rate: 48_000,
                channels: 1,
                duration_seconds: 1.0,
                peak_dbfs: -3.0,
                rms_dbfs: -18.0,
                clipped_samples: 0,
                dc_offset: 0.0,
                path: "input.wav".to_string(),
            },
            target: AudioFileReport {
                sample_rate: 48_000,
                channels: 1,
                duration_seconds: 1.0,
                peak_dbfs: -3.0,
                rms_dbfs: -18.0,
                clipped_samples: 0,
                dc_offset: 0.0,
                path: "target.wav".to_string(),
            },
            latency_samples: 0,
            latency_auto_samples: Some(0),
            manual_latency_adjustment_samples: 0,
            latency_confidence: 1.0,
            warnings: Vec::new(),
            warning_details: Vec::new(),
            prepared: None,
            capture_profile: None,
            gain: None,
            options: None,
            status: AudioStatus::Ready,
        };
        upsert_audio_report(&db, &project.id, &audio).expect("insert audio report");
        update_project_status(&db, &project.id, &ProjectStatus::Training, &now())
            .expect("mark project training");
        let run = TrainingRun {
            id: "run_test".to_string(),
            preset: "lstm_standard".to_string(),
            backend: "keras".to_string(),
            status: RunStatus::Running,
            device: "pending".to_string(),
            epochs: 0,
            created_at: now(),
            updated_at: now(),
            metrics: None,
            log_path: "runs/run_test/events.jsonl".to_string(),
        };
        let run_dir = root.join("runs/run_test");
        insert_training_run(
            &db,
            &project.id,
            &run,
            Path::new(&project.project_dir),
            &run_dir,
        )
        .expect("insert running training run");

        let state = AppState {
            db_path: PathBuf::from(":memory:"),
            projects_dir: root.clone(),
            workspace_root: root,
            db: Mutex::new(db),
            active_jobs: Mutex::new(HashMap::new()),
        };
        let job_id = create_job(
            &state,
            "train",
            Some("project_test"),
            Some("run_test"),
            None,
        )
        .expect("create job");
        let event = SidecarProgressEvent {
            job_id: job_id.clone(),
            operation: "train".to_string(),
            stream: "stdout".to_string(),
            line: r#"{"type":"epoch","epoch":1}"#.to_string(),
            json: Some(serde_json::json!({"type": "epoch", "epoch": 1})),
            project_id: Some("project_test".to_string()),
            run_id: Some("run_test".to_string()),
            export_id: None,
            timestamp: now(),
        };
        persist_job_event(&event, &state).expect("persist job event");

        let db = state.db.lock().expect("lock sqlite");
        let events = load_project_events(&db, "project_test", 120).expect("load events");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].job_id, job_id);
        recover_interrupted_jobs(&db).expect("recover interrupted jobs");
        let status: String = db
            .query_row(
                "SELECT status FROM jobs WHERE id = ?1",
                params![events[0].job_id],
                |row| row.get(0),
            )
            .expect("query recovered job status");
        assert_eq!(status, "interrupted");
        let run_status: String = db
            .query_row(
                "SELECT status FROM training_runs WHERE id = 'run_test'",
                [],
                |row| row.get(0),
            )
            .expect("query recovered run status");
        assert_eq!(run_status, "interrupted");
        let project_status: String = db
            .query_row(
                "SELECT status FROM projects WHERE id = 'project_test'",
                [],
                |row| row.get(0),
            )
            .expect("query recovered project status");
        assert_eq!(project_status, "ready");
    }

    #[test]
    fn delete_project_removes_sqlite_rows_and_managed_folder() {
        let mut db = Connection::open_in_memory().expect("open in-memory sqlite");
        configure_database(&mut db).expect("migrate sqlite");

        let root =
            std::env::temp_dir().join(format!("rttrainer-delete-test-{}", Uuid::new_v4().simple()));
        let project_dir = root.join("project_delete_test");
        fs::create_dir_all(project_dir.join("runs")).expect("create temp project dir");
        fs::write(project_dir.join("runs/progress.jsonl"), "{}\n").expect("write artifact");
        let timestamp = now();
        let project = ProjectDetail {
            id: "project_delete_test".to_string(),
            name: "Delete me".to_string(),
            target_kind: TargetKind::Generic,
            status: ProjectStatus::Draft,
            created_at: timestamp.clone(),
            updated_at: timestamp,
            notes: String::new(),
            project_dir: project_dir.display().to_string(),
            audio: None,
            runs: Vec::new(),
            exports: Vec::new(),
        };
        insert_project(&db, &project).expect("insert project");

        let run = TrainingRun {
            id: "run_delete_test".to_string(),
            preset: "lstm_standard".to_string(),
            backend: "keras".to_string(),
            status: RunStatus::Completed,
            device: "cpu".to_string(),
            epochs: 1,
            created_at: now(),
            updated_at: now(),
            metrics: None,
            log_path: "runs/run_delete_test/events.jsonl".to_string(),
        };
        insert_training_run(
            &db,
            &project.id,
            &run,
            Path::new(&project.project_dir),
            &project_dir.join("runs/run_delete_test"),
        )
        .expect("insert completed run");

        delete_project_by_id(&db, &project.id, &root).expect("delete project");

        let project_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM projects WHERE id = ?1",
                params![project.id],
                |row| row.get(0),
            )
            .expect("count projects");
        let run_count: i64 = db
            .query_row(
                "SELECT COUNT(*) FROM training_runs WHERE id = 'run_delete_test'",
                [],
                |row| row.get(0),
            )
            .expect("count runs");
        assert_eq!(project_count, 0);
        assert_eq!(run_count, 0);
        assert!(!project_dir.exists());

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rename_project_trims_and_persists_name() {
        let mut db = Connection::open_in_memory().expect("open in-memory sqlite");
        configure_database(&mut db).expect("migrate sqlite");

        let timestamp = now();
        let project = ProjectDetail {
            id: "project_rename_test".to_string(),
            name: "Old name".to_string(),
            target_kind: TargetKind::Amp,
            status: ProjectStatus::Draft,
            created_at: timestamp.clone(),
            updated_at: timestamp,
            notes: String::new(),
            project_dir: "/tmp/project_rename_test".to_string(),
            audio: None,
            runs: Vec::new(),
            exports: Vec::new(),
        };
        insert_project(&db, &project).expect("insert project");

        assert!(normalize_project_name("   ").is_err());
        assert!(normalize_project_name(&"x".repeat(121)).is_err());
        let name = normalize_project_name("  Deluxe pedal capture  ").expect("normalize name");
        let updated_at = now();
        update_project_name(&db, &project.id, &name, &updated_at).expect("rename project");

        let renamed = load_project_detail(&db, &project.id).expect("load renamed project");
        assert_eq!(renamed.name, "Deluxe pedal capture");
        assert_eq!(renamed.updated_at, updated_at);
    }

    #[test]
    fn capture_path_validation_rejects_non_wav_and_normalizes_policy() {
        let root = std::env::temp_dir().join(format!(
            "rttrainer-capture-test-{}",
            Uuid::new_v4().simple()
        ));
        fs::create_dir_all(&root).expect("create temp dir");
        let text_path = root.join("capture.txt");
        fs::write(&text_path, "not audio").expect("write non-wav file");

        let error = validate_capture_wav_path(&text_path.display().to_string(), "Dry input")
            .expect_err("reject non-wav");
        assert!(error.contains(".wav"));
        assert_eq!(normalize_channel_policy_name("mono_mixdown"), Ok("mixdown"));
        assert_eq!(normalize_channel_policy_name("left"), Ok("first"));
        assert!(normalize_capture_sample_rate(48_000).is_ok());
        assert!(normalize_capture_sample_rate(1).is_err());
    }

    #[test]
    fn runtime_settings_round_trip_through_sqlite() {
        let mut db = Connection::open_in_memory().expect("open in-memory sqlite");
        configure_database(&mut db).expect("migrate sqlite");

        let defaults = load_runtime_settings(&db).expect("load default settings");
        assert_eq!(defaults.selected_backend, "keras");
        assert_eq!(defaults.selected_device, "auto");
        assert_eq!(defaults.external_python_path, None);

        save_runtime_settings(
            &db,
            &RuntimeSettings {
                selected_backend: "torch".to_string(),
                selected_device: "metal".to_string(),
                external_python_path: Some("  /tmp/rttrainer-python  ".to_string()),
            },
        )
        .expect("save runtime settings");

        let settings = load_runtime_settings(&db).expect("load saved settings");
        assert_eq!(settings.selected_backend, "pytorch");
        assert_eq!(settings.selected_device, "mps");
        assert_eq!(
            settings.external_python_path,
            Some("/tmp/rttrainer-python".to_string())
        );
        assert!(normalize_runtime_device("quantum").is_err());
    }

    #[test]
    fn training_recipes_round_trip_through_sqlite() {
        let mut db = Connection::open_in_memory().expect("open in-memory sqlite");
        configure_database(&mut db).expect("migrate sqlite");

        let recipe = normalize_training_recipe(SaveTrainingRecipeRequest {
            id: None,
            name: "  Rhythm long run  ".to_string(),
            model_preset: "conv_gru_hybrid".to_string(),
            epochs: 80,
            batch_size: 24,
            learning_rate: 0.0005,
            sequence_length: 2048,
            max_windows: 4096,
            early_stopping_patience: 12,
            early_stopping_min_delta: 0.00005,
        })
        .expect("normalize recipe");
        upsert_training_recipe(&db, &recipe).expect("save recipe");

        let saved = load_training_recipes(&db).expect("load recipes");
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].name, "Rhythm long run");
        assert_eq!(saved[0].model_preset, "conv_gru_hybrid");
        assert_eq!(saved[0].epochs, 80);
        assert_eq!(saved[0].sequence_length, 2048);

        let updated = normalize_training_recipe(SaveTrainingRecipeRequest {
            id: Some(recipe.id.clone()),
            name: "Rhythm production".to_string(),
            model_preset: "lstm_standard".to_string(),
            epochs: 999,
            batch_size: 0,
            learning_rate: f64::NAN,
            sequence_length: 0,
            max_windows: 99_999,
            early_stopping_patience: 999,
            early_stopping_min_delta: 2.0,
        })
        .expect("normalize updated recipe");
        upsert_training_recipe(&db, &updated).expect("update recipe");

        let saved = load_training_recipes(&db).expect("reload recipes");
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].name, "Rhythm production");
        assert_eq!(saved[0].model_preset, "lstm_standard");
        assert_eq!(saved[0].epochs, 500);
        assert_eq!(saved[0].batch_size, default_training_batch_size());
        assert_eq!(saved[0].learning_rate, default_training_learning_rate());
        assert_eq!(saved[0].sequence_length, default_training_sequence_length());
        assert_eq!(saved[0].max_windows, 16_384);
        assert_eq!(saved[0].early_stopping_patience, 100);
        assert_eq!(saved[0].early_stopping_min_delta, 1.0);

        db.execute(
            "DELETE FROM training_recipes WHERE id = ?1",
            params![recipe.id],
        )
        .expect("delete recipe");
        let saved = load_training_recipes(&db).expect("reload after delete");
        assert!(saved.is_empty());
    }

    fn write_test_wav(path: &Path, samples: &[i16], sample_rate: u32) {
        let data_size = samples.len() as u32 * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_size).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&1_u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2_u16.to_le_bytes());
        bytes.extend_from_slice(&16_u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_size.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        fs::write(path, bytes).expect("write wav");
    }
}
