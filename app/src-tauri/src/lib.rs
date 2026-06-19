use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tauri::Manager;
use uuid::Uuid;

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
    Running,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExportStatus {
    Blocked,
    Ready,
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
    latency_confidence: f64,
    warnings: Vec<String>,
    status: AudioStatus,
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
    status: RunStatus,
    device: String,
    epochs: u32,
    created_at: String,
    updated_at: String,
    metrics: Option<TrainingMetrics>,
    log_path: String,
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
struct UpdateAudioRequest {
    project_id: String,
    input_path: String,
    target_path: String,
}

#[derive(Deserialize)]
struct StartTrainingRequest {
    project_id: String,
    preset: String,
}

#[derive(Deserialize)]
struct ExportRunRequest {
    project_id: String,
    run_id: String,
}

#[derive(Deserialize)]
struct UpdateNotesRequest {
    project_id: String,
    notes: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Store {
    projects: Vec<ProjectDetail>,
}

struct AppState {
    store_path: PathBuf,
    projects_dir: PathBuf,
    store: Mutex<Store>,
}

#[tauri::command]
fn app_status(state: tauri::State<AppState>) -> AppStatus {
    let binaries_dir = std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| state.projects_dir.clone());

    AppStatus {
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_dir: state
            .projects_dir
            .parent()
            .unwrap_or(&state.projects_dir)
            .display()
            .to_string(),
        trainer_sidecar_present: binary_exists(&binaries_dir, "rttrainer"),
        validator_sidecar_present: binary_exists(&binaries_dir, "rtneural-validator"),
    }
}

#[tauri::command]
fn list_projects(state: tauri::State<AppState>) -> Result<Vec<ProjectSummary>, String> {
    let store = state.store.lock().map_err(lock_error)?;
    Ok(store.projects.iter().map(summarize_project).collect())
}

#[tauri::command]
fn get_project(state: tauri::State<AppState>, project_id: String) -> Result<ProjectDetail, String> {
    let store = state.store.lock().map_err(lock_error)?;
    store
        .projects
        .iter()
        .find(|project| project.id == project_id)
        .cloned()
        .ok_or_else(|| "Project not found".to_string())
}

#[tauri::command]
fn create_project(
    state: tauri::State<AppState>,
    payload: CreateProjectRequest,
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
        name: if payload.name.trim().is_empty() {
            "Untitled capture".to_string()
        } else {
            payload.name.trim().to_string()
        },
        target_kind: payload.target_kind,
        status: ProjectStatus::Draft,
        created_at: run_created_at.clone(),
        updated_at: run_created_at,
        notes: String::new(),
        project_dir: project_dir.display().to_string(),
        audio: None,
        runs: Vec::new(),
        exports: Vec::new(),
    };

    let mut store = state.store.lock().map_err(lock_error)?;
    store.projects.insert(0, project.clone());
    persist(&state, &store)?;
    write_json(project_dir.join("project.json"), &project)?;
    Ok(project)
}

#[tauri::command]
fn update_project_audio(
    state: tauri::State<AppState>,
    payload: UpdateAudioRequest,
) -> Result<ProjectDetail, String> {
    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;
    let project_dir = PathBuf::from(&project.project_dir);
    let prepared_dir = project_dir.join("audio/prepared");
    fs::create_dir_all(&prepared_dir).map_err(to_error)?;

    let mut warnings = Vec::new();
    collect_audio_warning(&payload.input_path, "Input", &mut warnings);
    collect_audio_warning(&payload.target_path, "Target", &mut warnings);
    if payload.input_path == payload.target_path {
        warnings.push("Input and target paths should be different files.".to_string());
    }

    let clipped = payload.target_path.to_lowercase().contains("clip");
    if clipped {
        warnings.push("Target path suggests a clipped capture; inspect before training.".to_string());
    }

    let status = if warnings.is_empty() {
        AudioStatus::Ready
    } else {
        AudioStatus::Warning
    };

    let report = AudioReport {
        input: audio_report(&payload.input_path, -1.2, -18.4, false),
        target: audio_report(&payload.target_path, -0.8, -15.6, clipped),
        latency_samples: 123,
        latency_confidence: if warnings.is_empty() { 0.94 } else { 0.42 },
        warnings,
        status: status.clone(),
    };

    project.audio = Some(report.clone());
    project.status = if matches!(status, AudioStatus::Ready) {
        ProjectStatus::Ready
    } else {
        ProjectStatus::Draft
    };
    project.updated_at = now();

    write_json(prepared_dir.join("preparation-report.json"), &report)?;
    write_json(project_dir.join("project.json"), &*project)?;
    let cloned = project.clone();
    persist(&state, &store)?;
    Ok(cloned)
}

#[tauri::command]
fn start_training(
    state: tauri::State<AppState>,
    payload: StartTrainingRequest,
) -> Result<ProjectDetail, String> {
    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;

    let audio_ready = project
        .audio
        .as_ref()
        .map(|audio| audio.status == AudioStatus::Ready)
        .unwrap_or(false);
    if !audio_ready {
        return Err("Audio must pass preflight before training.".to_string());
    }

    let run_id = format!("run_{}", Uuid::new_v4().simple());
    let run_created_at = now();
    let project_dir = PathBuf::from(&project.project_dir);
    let run_dir = project_dir.join("runs").join(&run_id);
    let checkpoint_dir = run_dir.join("checkpoints");
    let preview_dir = run_dir.join("previews");
    fs::create_dir_all(&checkpoint_dir).map_err(to_error)?;
    fs::create_dir_all(&preview_dir).map_err(to_error)?;

    let metrics = metrics_for_preset(&payload.preset);
    let log_path = run_dir.join("events.jsonl");
    let event_log = [
        json_line("run_started", &run_id, 0, None),
        json_line("epoch", &run_id, 20, Some(metrics.esr * 1.8)),
        json_line("epoch", &run_id, 40, Some(metrics.esr * 1.2)),
        json_line("epoch", &run_id, 60, Some(metrics.esr)),
        json_line("run_finished", &run_id, 60, Some(metrics.esr)),
    ]
    .join("\n");
    fs::write(&log_path, format!("{event_log}\n")).map_err(to_error)?;
    write_json(
        checkpoint_dir.join("best-checkpoint.json"),
        &serde_json::json!({
            "schema_version": 1,
            "preset": &payload.preset,
            "format": "simulated-state-dict",
            "ready_for_export": true
        }),
    )?;
    write_json(run_dir.join("metrics.json"), &metrics)?;
    write_preview_file(preview_dir.join("target-preview.wav"), "target preview")?;
    write_preview_file(preview_dir.join("prediction-preview.wav"), "prediction preview")?;
    write_preview_file(preview_dir.join("residual-preview.wav"), "residual preview")?;

    let run = TrainingRun {
        id: run_id,
        preset: payload.preset,
        status: RunStatus::Completed,
        device: detected_device(),
        epochs: 60,
        created_at: run_created_at.clone(),
        updated_at: run_created_at,
        metrics: Some(metrics),
        log_path: log_path.display().to_string(),
    };

    project.status = ProjectStatus::Ready;
    project.updated_at = now();
    project.runs.push(run);
    write_json(project_dir.join("project.json"), &*project)?;
    let cloned = project.clone();
    persist(&state, &store)?;
    Ok(cloned)
}

#[tauri::command]
fn export_run(
    state: tauri::State<AppState>,
    payload: ExportRunRequest,
) -> Result<ProjectDetail, String> {
    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;
    let run = project
        .runs
        .iter()
        .find(|run| run.id == payload.run_id)
        .cloned()
        .ok_or_else(|| "Run not found".to_string())?;
    let metrics = run
        .metrics
        .clone()
        .ok_or_else(|| "Run has no metrics to export.".to_string())?;

    if metrics.realtime_factor < 20.0 {
        return Err("Benchmark gate failed for realtime export.".to_string());
    }

    let export_id = format!("export_{}", Uuid::new_v4().simple());
    let project_dir = PathBuf::from(&project.project_dir);
    let export_dir = project_dir.join("exports").join(&export_id);
    fs::create_dir_all(&export_dir).map_err(to_error)?;

    let model_path = export_dir.join("model.rtneural.json");
    let package_path = export_dir.join("package.json");
    let validation_path = export_dir.join("validation-report.json");
    let benchmark_path = export_dir.join("benchmark-report.json");

    write_json(
        &model_path,
        &serde_json::json!({
            "in_shape": [null, null, 1],
            "layers": [
                {
                    "type": "lstm",
                    "activation": "",
                    "shape": [null, null, 16],
                    "weights": []
                },
                {
                    "type": "dense",
                    "activation": "",
                    "shape": [null, null, 1],
                    "weights": []
                }
            ],
            "metadata": {
                "schema_version": 1,
                "sample_rate": 48000,
                "latency_samples": project.audio.as_ref().map(|audio| audio.latency_samples).unwrap_or(0),
                "architecture": &run.preset,
                "loss": &metrics,
                "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d"
            }
        }),
    )?;
    write_json(
        &validation_path,
        &serde_json::json!({
            "schema_version": 1,
            "status": "pass",
            "max_abs_error": 0.000001,
            "rmse": 0.0000003,
            "validator": "built-in-simulated"
        }),
    )?;
    write_json(
        &benchmark_path,
        &serde_json::json!({
            "schema_version": 1,
            "status": "pass",
            "backend": "simulated-eigen",
            "sample_rate": 48000,
            "realtime_factor": metrics.realtime_factor
        }),
    )?;
    write_json(
        &package_path,
        &serde_json::json!({
            "schema_version": 1,
            "name": &project.name,
            "project_id": &project.id,
            "run_id": &run.id,
            "preset": &run.preset,
            "sample_rate": 48000,
            "quality": {
                "esr": metrics.esr,
                "rmse": metrics.rmse
            },
            "runtime": {
                "realtime_factor": metrics.realtime_factor,
                "backend": "simulated-eigen"
            },
            "compatibility": {
                "rtneural_commit": "1fb1f075a5d66e85bfc8f488c3f3626840cb3a1d",
                "dynamic_json": true
            }
        }),
    )?;

    let package = ExportPackage {
        id: export_id,
        run_id: run.id,
        status: ExportStatus::Ready,
        created_at: now(),
        model_path: model_path.display().to_string(),
        package_path: package_path.display().to_string(),
        validation_path: validation_path.display().to_string(),
        benchmark_path: benchmark_path.display().to_string(),
    };

    project.status = ProjectStatus::Exported;
    project.updated_at = now();
    project.exports.push(package);
    write_json(project_dir.join("project.json"), &*project)?;
    let cloned = project.clone();
    persist(&state, &store)?;
    Ok(cloned)
}

#[tauri::command]
fn update_notes(
    state: tauri::State<AppState>,
    payload: UpdateNotesRequest,
) -> Result<ProjectDetail, String> {
    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;
    project.notes = payload.notes;
    project.updated_at = now();
    write_json(PathBuf::from(&project.project_dir).join("project.json"), &*project)?;
    let cloned = project.clone();
    persist(&state, &store)?;
    Ok(cloned)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let app_dir = app
                .path()
                .app_data_dir()
                .expect("failed to resolve app data directory");
            let projects_dir = app_dir.join("projects");
            fs::create_dir_all(&projects_dir).expect("failed to create project directory");
            let store_path = app_dir.join("store.json");
            let store = load_store(&store_path);
            app.manage(AppState {
                store_path,
                projects_dir,
                store: Mutex::new(store),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            app_status,
            list_projects,
            get_project,
            create_project,
            update_project_audio,
            start_training,
            export_run,
            update_notes
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn load_store(path: &Path) -> Store {
    let Ok(raw) = fs::read_to_string(path) else {
        return Store::default();
    };
    serde_json::from_str(&raw).unwrap_or_default()
}

fn persist(state: &AppState, store: &Store) -> Result<(), String> {
    write_json(&state.store_path, store)
}

fn find_project_mut<'a>(
    store: &'a mut Store,
    project_id: &str,
) -> Result<&'a mut ProjectDetail, String> {
    store
        .projects
        .iter_mut()
        .find(|project| project.id == project_id)
        .ok_or_else(|| "Project not found".to_string())
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

fn audio_report(path: &str, peak_dbfs: f64, rms_dbfs: f64, clipped: bool) -> AudioFileReport {
    AudioFileReport {
        sample_rate: 48_000,
        channels: 1,
        duration_seconds: 95.4,
        peak_dbfs,
        rms_dbfs,
        clipped_samples: if clipped { 42 } else { 0 },
        dc_offset: 0.0002,
        path: path.to_string(),
    }
}

fn collect_audio_warning(path: &str, label: &str, warnings: &mut Vec<String>) {
    if path.trim().is_empty() {
        warnings.push(format!("{label} path is empty."));
    } else if !path.to_lowercase().ends_with(".wav") {
        warnings.push(format!("{label} should be a WAV file for v1."));
    }
}

fn metrics_for_preset(preset: &str) -> TrainingMetrics {
    let esr = match preset {
        "heavy_recurrent" => 0.028,
        "lstm_standard" => 0.044,
        "dense_memoryless" => 0.061,
        _ => 0.072,
    };

    TrainingMetrics {
        esr,
        mae: esr / 2.8,
        rmse: esr / 1.8,
        peak_residual: esr * 2.6,
        rms_residual: esr / 2.1,
        realtime_factor: if preset == "heavy_recurrent" { 24.0 } else { 118.0 },
    }
}

fn detected_device() -> String {
    #[cfg(target_os = "macos")]
    {
        "mps-ready".to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "cpu".to_string()
    }
}

fn binary_exists(dir: &Path, stem: &str) -> bool {
    let direct = dir.join(stem);
    let exe = dir.join(format!("{stem}.exe"));
    direct.exists() || exe.exists()
}

fn json_line(kind: &str, run_id: &str, epoch: u32, esr: Option<f64>) -> String {
    serde_json::json!({
        "type": kind,
        "run_id": run_id,
        "epoch": epoch,
        "val_esr": esr,
        "timestamp": now()
    })
    .to_string()
}

fn write_preview_file(path: PathBuf, label: &str) -> Result<(), String> {
    fs::write(path, format!("Simulated {label}; replace with rendered WAV bytes.\n")).map_err(to_error)
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(value).map_err(to_error)?;
    fs::write(path, format!("{raw}\n")).map_err(to_error)
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
