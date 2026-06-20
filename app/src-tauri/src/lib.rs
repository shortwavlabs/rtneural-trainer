use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::Mutex,
    thread::{self, JoinHandle},
};
use tauri::{Emitter, Manager};
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
struct RunControlRequest {
    project_id: String,
    run_id: String,
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

#[derive(Deserialize)]
struct PrepareReport {
    input: AudioFileReport,
    target: AudioFileReport,
    latency: LatencyReport,
    warnings: Vec<String>,
    status: AudioStatus,
}

#[derive(Deserialize)]
struct LatencyReport {
    estimated_samples: i32,
    confidence: f64,
}

#[derive(Deserialize)]
struct TrainingReport {
    run_id: String,
    preset: String,
    device: String,
    epochs: u32,
    metrics: TrainingMetrics,
    created_at: String,
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
            .db_path
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
fn list_project_events(
    state: tauri::State<AppState>,
    project_id: String,
) -> Result<Vec<SidecarProgressEvent>, String> {
    let db = state.db.lock().map_err(lock_error)?;
    load_project_events(&db, &project_id, 120)
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

    let db = state.db.lock().map_err(lock_error)?;
    insert_project(&db, &project)?;
    Ok(project)
}

#[tauri::command]
fn update_project_audio(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    payload: UpdateAudioRequest,
) -> Result<ProjectDetail, String> {
    if payload.input_path.trim().is_empty() {
        return Err("Input path is required.".to_string());
    }
    if payload.target_path.trim().is_empty() {
        return Err("Target path is required.".to_string());
    }
    if payload.input_path == payload.target_path {
        return Err("Input and target paths should be different files.".to_string());
    }

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
            "input_path": payload.input_path,
            "target_path": payload.target_path,
            "output_dir": prepared_dir
        }),
    )?;
    let job_id = create_job(&state, "prepare", Some(&payload.project_id), None, None)?;
    let prepare_result = run_rttrainer(
        &app,
        &state,
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
        mark_job_failed(&state, &job_id, &error)?;
        return Err(error);
    }
    mark_job_completed(&state, &job_id)?;

    let prepare_report_path = prepared_dir.join("preparation-report.json");
    let prepare_report: PrepareReport = read_json(&prepare_report_path)?;

    let report = AudioReport {
        input: prepare_report.input,
        target: prepare_report.target,
        latency_samples: prepare_report.latency.estimated_samples,
        latency_confidence: prepare_report.latency.confidence,
        warnings: prepare_report.warnings,
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
    let project_dir = {
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
        PathBuf::from(&project.project_dir)
    };

    let run_dir = project_dir.join("runs").join(&run_id);
    fs::create_dir_all(&run_dir).map_err(to_error)?;
    let manifest_path = run_dir.join("train-manifest.json");
    write_json(
        &manifest_path,
        &serde_json::json!({
            "run_id": &run_id,
            "run_dir": &run_dir,
            "prepared_dir": project_dir.join("audio/prepared"),
            "preset": &payload.preset,
            "backend": "keras",
            "epochs": 20,
            "batch_size": 16,
            "learning_rate": 0.001,
            "sequence_length": 1024,
            "max_windows": 512,
            "seed": 1337
        }),
    )?;

    let created_at = now();
    let log_path = run_dir.join("events.jsonl");
    let run = TrainingRun {
        id: run_id.clone(),
        preset: payload.preset.clone(),
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
fn export_run(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
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

    write_json(
        &manifest_path,
        &serde_json::json!({
            "name": project_name,
            "run_dir": run_dir,
            "export_dir": export_dir,
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
    };
    {
        let db = state.db.lock().map_err(lock_error)?;
        insert_export_package(&db, &project_id, &package, &project_dir, &export_dir)?;
    }

    let export_job_id = create_job(
        &state,
        "export",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let sidecar_result = run_rttrainer(
        &app,
        &state,
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
            mark_job_failed(&state, &export_job_id, &error)?;
            let db = state.db.lock().map_err(lock_error)?;
            update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
            return Err(error);
        }
    };
    mark_job_completed(&state, &export_job_id)?;
    fs::write(export_dir.join("export-events.jsonl"), sidecar.stdout).map_err(to_error)?;
    if !sidecar.stderr.trim().is_empty() {
        fs::write(export_dir.join("stderr.log"), sidecar.stderr).map_err(to_error)?;
    }

    {
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Validating, None)?;
    }

    let validate_job_id = create_job(
        &state,
        "native_validate",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let validate_result = run_validator(
        &app,
        &state,
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
        mark_job_failed(&state, &validate_job_id, &error)?;
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
        return Err(error);
    }
    mark_job_completed(&state, &validate_job_id)?;

    let benchmark_job_id = create_job(
        &state,
        "native_benchmark",
        Some(&project_id),
        Some(&run.id),
        Some(&export_id),
    )?;
    let benchmark_result = run_validator(
        &app,
        &state,
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
        mark_job_failed(&state, &benchmark_job_id, &error)?;
        let db = state.db.lock().map_err(lock_error)?;
        update_export_status(&db, &export_id, &ExportStatus::Failed, Some(&error))?;
        return Err(error);
    }
    mark_job_completed(&state, &benchmark_job_id)?;

    let db = state.db.lock().map_err(lock_error)?;
    update_export_status(&db, &export_id, &ExportStatus::Ready, None)?;
    update_project_status(&db, &project_id, &ProjectStatus::Exported, &now())?;
    load_project_detail(&db, &project_id)
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
            list_projects,
            get_project,
            list_project_events,
            create_project,
            update_project_audio,
            start_training,
            cancel_training_run,
            resume_training_run,
            export_run,
            update_notes
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let raw = fs::read_to_string(path).map_err(to_error)?;
    serde_json::from_str(&raw).map_err(to_error)
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

const MIGRATIONS: &[(&str, &str)] = &[(
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
)];

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
            "SELECT id, preset, status, device, epochs, created_at, updated_at, metrics_json, log_path
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
            ))
        })
        .map_err(to_error)?;

    let mut runs = Vec::new();
    for row in rows {
        let (id, preset, status, device, epochs, created_at, updated_at, metrics_json, log_path) =
            row.map_err(to_error)?;
        runs.push(TrainingRun {
            id,
            preset,
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
            "SELECT id, run_id, status, created_at, model_path, package_path, validation_path, benchmark_path
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
        ) = row.map_err(to_error)?;
        exports.push(ExportPackage {
            id,
            run_id,
            status: enum_from_string(&status)?,
            created_at,
            model_path: artifact_path(project_dir, &model_path)
                .display()
                .to_string(),
            package_path: artifact_path(project_dir, &package_path)
                .display()
                .to_string(),
            validation_path: artifact_path(project_dir, &validation_path)
                .display()
                .to_string(),
            benchmark_path: artifact_path(project_dir, &benchmark_path)
                .display()
                .to_string(),
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
            "Project already has an active {operation} job. Wait, cancel, or resume before starting another job."
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

fn binary_exists(dir: &Path, stem: &str) -> bool {
    let direct = dir.join(stem);
    let exe = dir.join(format!("{stem}.exe"));
    direct.exists() || exe.exists()
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
    let trainer_dir = state.workspace_root.join("trainer");
    let cache_dir = state.workspace_root.join(".uv-cache");
    let mut process = Command::new("uv");
    process
        .current_dir(&trainer_dir)
        .env("UV_CACHE_DIR", &cache_dir)
        .arg("run");
    if matches!(command, "train" | "evaluate" | "export") {
        process.arg("--extra").arg("tensorflow");
    }
    process
        .arg("python")
        .arg("-m")
        .arg("rttrainer")
        .arg(command)
        .arg("--manifest")
        .arg(manifest_path);

    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rttrainer {command} started"),
    );
    let output = run_streaming_process(app, &context, process)?;
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

fn run_validator(
    app: &tauri::AppHandle,
    state: &AppState,
    args: Vec<String>,
    context: SidecarContext,
) -> Result<SidecarOutput, String> {
    let validator = validator_binary_path(state)?;
    let action = args.first().cloned().unwrap_or_else(|| "run".to_string());
    let mut process = Command::new(&validator);
    process.args(args);

    emit_sidecar_line(
        app,
        &context,
        "system",
        &format!("rtneural-validator {action} started"),
    );
    let output = run_streaming_process(app, &context, process)?;
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
        validator.display(),
        output.status,
        output.stdout,
        output.stderr
    ))
}

struct StreamingProcessOutput {
    status: std::process::ExitStatus,
    stdout: String,
    stderr: String,
}

fn run_streaming_process(
    app: &tauri::AppHandle,
    context: &SidecarContext,
    mut process: Command,
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
        status,
        stdout,
        stderr,
    })
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
    let status = Command::new("kill")
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
    let status = Command::new("taskkill")
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
            if let Some(path) = existing_binary(dir, "rtneural-validator") {
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

#[cfg(test)]
mod tests {
    use super::*;

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
            latency_confidence: 1.0,
            warnings: Vec::new(),
            status: AudioStatus::Ready,
        };
        upsert_audio_report(&db, &project.id, &audio).expect("insert audio report");
        update_project_status(&db, &project.id, &ProjectStatus::Training, &now())
            .expect("mark project training");
        let run = TrainingRun {
            id: "run_test".to_string(),
            preset: "lstm_standard".to_string(),
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
}
