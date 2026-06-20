use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::{
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
    workspace_root: PathBuf,
    store: Mutex<Store>,
}

struct SidecarOutput {
    stdout: String,
    stderr: String,
}

#[derive(Clone)]
struct SidecarContext {
    operation: String,
    project_id: Option<String>,
    run_id: Option<String>,
    export_id: Option<String>,
}

#[derive(Clone, Serialize)]
struct SidecarProgressEvent {
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
        let store = state.store.lock().map_err(lock_error)?;
        store
            .projects
            .iter()
            .find(|project| project.id == payload.project_id)
            .map(|project| PathBuf::from(&project.project_dir))
            .ok_or_else(|| "Project not found".to_string())?
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
    run_rttrainer(
        &app,
        &state,
        "prepare",
        &manifest_path,
        SidecarContext {
            operation: "prepare".to_string(),
            project_id: Some(payload.project_id.clone()),
            run_id: None,
            export_id: None,
        },
    )?;

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

    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;
    project.audio = Some(report.clone());
    project.status = if matches!(report.status, AudioStatus::Ready) {
        ProjectStatus::Ready
    } else {
        ProjectStatus::Draft
    };
    project.updated_at = now();

    write_json(project_dir.join("project.json"), &*project)?;
    let cloned = project.clone();
    persist(&state, &store)?;
    Ok(cloned)
}

#[tauri::command]
fn start_training(
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    payload: StartTrainingRequest,
) -> Result<ProjectDetail, String> {
    let run_id = format!("run_{}", Uuid::new_v4().simple());
    let project_dir = {
        let store = state.store.lock().map_err(lock_error)?;
        let project = store
            .projects
            .iter()
            .find(|project| project.id == payload.project_id)
            .ok_or_else(|| "Project not found".to_string())?;
        let audio_ready = project
            .audio
            .as_ref()
            .map(|audio| audio.status == AudioStatus::Ready)
            .unwrap_or(false);
        if !audio_ready {
            return Err("Audio must pass preflight before training.".to_string());
        }
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

    let sidecar = run_rttrainer(
        &app,
        &state,
        "train",
        &manifest_path,
        SidecarContext {
            operation: "train".to_string(),
            project_id: Some(payload.project_id.clone()),
            run_id: Some(run_id.clone()),
            export_id: None,
        },
    )?;
    let log_path = run_dir.join("events.jsonl");
    fs::write(&log_path, sidecar.stdout).map_err(to_error)?;
    if !sidecar.stderr.trim().is_empty() {
        fs::write(run_dir.join("stderr.log"), sidecar.stderr).map_err(to_error)?;
    }

    let report: TrainingReport = read_json(&run_dir.join("training-report.json"))?;

    let run = TrainingRun {
        id: report.run_id,
        preset: report.preset,
        status: RunStatus::Completed,
        device: report.device,
        epochs: report.epochs,
        created_at: report.created_at.clone(),
        updated_at: now(),
        metrics: Some(report.metrics),
        log_path: log_path.display().to_string(),
    };

    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &payload.project_id)?;
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
    app: tauri::AppHandle,
    state: tauri::State<AppState>,
    payload: ExportRunRequest,
) -> Result<ProjectDetail, String> {
    let export_id = format!("export_{}", Uuid::new_v4().simple());
    let (project_name, project_id, project_dir, run, sample_rate, latency_samples) = {
        let store = state.store.lock().map_err(lock_error)?;
        let project = store
            .projects
            .iter()
            .find(|project| project.id == payload.project_id)
            .ok_or_else(|| "Project not found".to_string())?;
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

    let sidecar = run_rttrainer(
        &app,
        &state,
        "export",
        &manifest_path,
        SidecarContext {
            operation: "export".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
    )?;
    fs::write(export_dir.join("export-events.jsonl"), sidecar.stdout).map_err(to_error)?;
    if !sidecar.stderr.trim().is_empty() {
        fs::write(export_dir.join("stderr.log"), sidecar.stderr).map_err(to_error)?;
    }

    run_validator(
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
            operation: "native_validate".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
    )?;
    run_validator(
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
            operation: "native_benchmark".to_string(),
            project_id: Some(project_id.clone()),
            run_id: Some(run.id.clone()),
            export_id: Some(export_id.clone()),
        },
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

    let mut store = state.store.lock().map_err(lock_error)?;
    let project = find_project_mut(&mut store, &project_id)?;
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
    write_json(
        PathBuf::from(&project.project_dir).join("project.json"),
        &*project,
    )?;
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
                workspace_root: workspace_root(),
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

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let raw = fs::read_to_string(path).map_err(to_error)?;
    serde_json::from_str(&raw).map_err(to_error)
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

fn binary_exists(dir: &Path, stem: &str) -> bool {
    let direct = dir.join(stem);
    let exe = dir.join(format!("{stem}.exe"));
    direct.exists() || exe.exists()
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<(), String> {
    let raw = serde_json::to_string_pretty(value).map_err(to_error)?;
    fs::write(path, format!("{raw}\n")).map_err(to_error)
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

fn emit_sidecar_line(app: &tauri::AppHandle, context: &SidecarContext, stream: &str, line: &str) {
    let payload = SidecarProgressEvent {
        operation: context.operation.clone(),
        stream: stream.to_string(),
        line: line.to_string(),
        json: serde_json::from_str::<serde_json::Value>(line).ok(),
        project_id: context.project_id.clone(),
        run_id: context.run_id.clone(),
        export_id: context.export_id.clone(),
        timestamp: now(),
    };
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
