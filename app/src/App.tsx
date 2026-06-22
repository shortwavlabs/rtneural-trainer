import {
  Activity,
  AlertTriangle,
  AudioLines,
  CheckCircle2,
  Cpu,
  Crosshair,
  Download,
  FileAudio,
  FolderOpen,
  FolderPlus,
  Gauge,
  LoaderCircle,
  PackageCheck,
  Pencil,
  Play,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Square,
  Trash2,
  X,
  type LucideIcon,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./lib/api";
import type {
  AppStatus,
  AudioReport,
  AudioStatus,
  AudioWarning,
  CaptureChannelPolicy,
  DeviceInspection,
  ExportPackage,
  ProjectDetail,
  ProjectSummary,
  ProjectWaveform,
  ProjectWaveformTrack,
  RuntimeBackend,
  RuntimeSettings,
  RunPreview,
  RunPreviewArtifact,
  SidecarProgressEvent,
  TargetKind,
  TrainingMetrics,
  TrainingRecipe,
  TrainingRun,
  WaveformBin,
} from "./types";

type TabId = "capture" | "align" | "train" | "evaluate" | "export";

type CaptureAnalyzePayload = {
  inputPath: string;
  targetPath: string;
  targetSampleRate: number;
  resample: boolean;
  channelPolicy: CaptureChannelPolicy;
};

type TrainingOptions = {
  preset: string;
  resumeFromRunId?: string | null;
  epochs: number;
  batchSize: number;
  learningRate: number;
  sequenceLength: number;
  earlyStoppingPatience: number;
  earlyStoppingMinDelta: number;
  maxWindows: number;
};

type TrainingRecipeOption = {
  id: string;
  source: "built_in" | "custom";
  name: string;
  description: string;
  modelPreset: string;
  epochs: number;
  batchSize: number;
  learningRate: number;
  sequenceLength: number;
  maxWindows: number;
  earlyStoppingPatience: number;
  earlyStoppingMinDelta: number;
};

type TrainingHistoryPoint = {
  epoch: number;
  trainLoss: number | null;
  valEsr: number | null;
  valRmse: number | null;
  validationScore: number | null;
  predictionRmsRatio: number | null;
  learningRate: number | null;
  nextLearningRate: number | null;
  learningRateReduced: boolean;
  isBest: boolean;
};

type PresetRecommendation = {
  presetId: string;
  label: string;
  confidence: "high" | "medium" | "low";
  reasons: string[];
};

type QualityDecision = {
  verdict: "good" | "usable" | "needs_work" | "unknown";
  summary: string;
  action: string;
};

const tabs: Array<{ id: TabId; label: string; icon: LucideIcon }> = [
  { id: "capture", label: "Capture", icon: AudioLines },
  { id: "align", label: "Align", icon: Crosshair },
  { id: "train", label: "Train", icon: Cpu },
  { id: "evaluate", label: "Evaluate", icon: Activity },
  { id: "export", label: "Export", icon: PackageCheck },
];

const targetLabels: Record<TargetKind, string> = {
  amp: "Amp",
  pedal: "Pedal",
  line: "Line",
  generic: "Generic",
};

const presets = [
  {
    id: "dense_only",
    label: "Dense",
    detail: "2x Dense, tanh",
    cpu: "Tiny CPU",
    backends: ["keras"],
  },
  {
    id: "gru_light",
    label: "GRU",
    detail: "1x GRU, hidden 10",
    cpu: "Low CPU",
    backends: ["keras"],
  },
  {
    id: "lstm_light",
    label: "Light",
    detail: "1x LSTM, hidden 12",
    cpu: "Low CPU",
    backends: ["keras", "pytorch"],
  },
  {
    id: "lstm_standard",
    label: "Standard",
    detail: "1x LSTM, hidden 16",
    cpu: "Default",
    backends: ["keras", "pytorch"],
  },
  {
    id: "conv1d_light",
    label: "Conv1D",
    detail: "Causal Conv1D, 8 filters",
    cpu: "Fast",
    backends: ["keras"],
  },
  {
    id: "conv1d_bn_prelu",
    label: "Conv + PReLU",
    detail: "Conv1D, BatchNorm, PReLU",
    cpu: "Moderate",
    backends: ["keras"],
  },
  {
    id: "conv1d_stack_prelu",
    label: "Stacked Conv",
    detail: "4x causal Conv1D, PReLU",
    cpu: "Moderate",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn",
    label: "WaveNet TCN",
    detail: "8x dilated causal Conv1D",
    cpu: "Heavy",
    backends: ["keras"],
  },
  {
    id: "conv_gru_hybrid",
    label: "Hybrid",
    detail: "Conv1D front-end + GRU",
    cpu: "Moderate",
    backends: ["keras"],
  },
] satisfies Array<{
  id: string;
  label: string;
  detail: string;
  cpu: string;
  backends: RuntimeBackend[];
}>;

const builtInTrainingRecipes = [
  {
    id: "builtin_smoke",
    source: "built_in",
    name: "Quick smoke",
    description: "Fast pipeline check before spending time on a run.",
    modelPreset: "conv_gru_hybrid",
    epochs: 4,
    batchSize: 16,
    learningRate: 0.001,
    sequenceLength: 4096,
    maxWindows: 512,
    earlyStoppingPatience: 0,
    earlyStoppingMinDelta: 0.0001,
  },
  {
    id: "builtin_balanced",
    source: "built_in",
    name: "Balanced",
    description: "Finite-memory baseline before trying recurrent models.",
    modelPreset: "conv1d_bn_prelu",
    epochs: 40,
    batchSize: 16,
    learningRate: 0.001,
    sequenceLength: 8192,
    maxWindows: 2048,
    earlyStoppingPatience: 6,
    earlyStoppingMinDelta: 0.0001,
  },
  {
    id: "builtin_production",
    source: "built_in",
    name: "Production",
    description: "WaveNet-style TCN for quality experiments; benchmark before export.",
    modelPreset: "wavenet_tcn",
    epochs: 120,
    batchSize: 16,
    learningRate: 0.0007,
    sequenceLength: 8192,
    maxWindows: 4096,
    earlyStoppingPatience: 12,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_long_capture",
    source: "built_in",
    name: "Long capture",
    description: "Broader window coverage with the stacked finite-memory model.",
    modelPreset: "conv1d_stack_prelu",
    epochs: 80,
    batchSize: 16,
    learningRate: 0.001,
    sequenceLength: 8192,
    maxWindows: 4096,
    earlyStoppingPatience: 10,
    earlyStoppingMinDelta: 0.0001,
  },
] satisfies TrainingRecipeOption[];

export default function App() {
  const [status, setStatus] = useState<AppStatus | null>(null);
  const [runtimeSettings, setRuntimeSettings] = useState<RuntimeSettings | null>(null);
  const [deviceInspection, setDeviceInspection] = useState<DeviceInspection | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState<string | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [trainingRecipes, setTrainingRecipes] = useState<TrainingRecipe[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("capture");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [progressEvents, setProgressEvents] = useState<SidecarProgressEvent[]>([]);
  const sidecarBusy =
    busy === "audio" || busy === "sample" || busy === "train" || busy === "export";

  useEffect(() => {
    void boot();
  }, []);

  useEffect(() => {
    if (!isTauriRuntime()) return;

    let mounted = true;
    const unlisten = listen<SidecarProgressEvent>("sidecar-progress", (event) => {
      if (!mounted) return;
      setProgressEvents((current) => [...current, event.payload].slice(-120));
      if (selectedId && shouldRefreshProject(event.payload)) {
        void loadProject(selectedId);
      }
    });

    unlisten.catch((caught) => {
      if (mounted) setError(toFriendlyMessage(caught));
    });

    return () => {
      mounted = false;
      void unlisten.then((dispose) => dispose());
    };
  }, [selectedId]);

  useEffect(() => {
    setProgressEvents([]);
    if (!selectedId) {
      setProject(null);
      return;
    }
    void loadProject(selectedId);
  }, [selectedId]);

  async function boot() {
    try {
      setError(null);
      const [nextStatus, nextSettings, nextProjects, nextRecipes] = await Promise.all([
        api.appStatus(),
        api.getRuntimeSettings(),
        api.listProjects(),
        api.listTrainingRecipes(),
      ]);
      setStatus(nextStatus);
      setRuntimeSettings(nextSettings);
      setProjects(nextProjects);
      setTrainingRecipes(nextRecipes);
      setSelectedId((current) => current ?? nextProjects[0]?.id ?? null);
      void refreshDeviceInspection();
    } catch (caught) {
      setError(toFriendlyMessage(caught));
    }
  }

  async function refreshDeviceInspection() {
    setRuntimeBusy("inspect");
    setRuntimeError(null);
    try {
      const nextInspection = await api.inspectDevice();
      setDeviceInspection(nextInspection);
    } catch (caught) {
      setRuntimeError(toFriendlyMessage(caught));
      setDeviceInspection(null);
    } finally {
      setRuntimeBusy(null);
    }
  }

  async function saveRuntimeSettings(nextSettings: RuntimeSettings) {
    setRuntimeBusy("settings");
    setRuntimeError(null);
    try {
      const saved = await api.updateRuntimeSettings(nextSettings);
      setRuntimeSettings(saved);
      await refreshDeviceInspection();
    } catch (caught) {
      setRuntimeError(toFriendlyMessage(caught));
    } finally {
      setRuntimeBusy(null);
    }
  }

  async function loadProject(projectId: string) {
    try {
      setError(null);
      const [nextProject, nextEvents] = await Promise.all([
        api.getProject(projectId),
        api.listProjectEvents(projectId),
      ]);
      setProject(nextProject);
      setProgressEvents(nextEvents);
    } catch (caught) {
      setError(toFriendlyMessage(caught));
    }
  }

  async function refreshProjects() {
    const nextProjects = await api.listProjects();
    setProjects(nextProjects);
  }

  async function saveTrainingRecipe(options: TrainingOptions, name: string, id?: string) {
    const saved = await api.saveTrainingRecipe({
      id: id ?? null,
      name,
      model_preset: options.preset,
      epochs: options.epochs,
      batch_size: options.batchSize,
      learning_rate: options.learningRate,
      sequence_length: options.sequenceLength,
      max_windows: options.maxWindows,
      early_stopping_patience: options.earlyStoppingPatience,
      early_stopping_min_delta: options.earlyStoppingMinDelta,
    });
    const nextRecipes = await api.listTrainingRecipes();
    setTrainingRecipes(nextRecipes);
    return saved;
  }

  async function deleteTrainingRecipe(recipeId: string) {
    const nextRecipes = await api.deleteTrainingRecipe({ id: recipeId });
    setTrainingRecipes(nextRecipes);
  }

  async function commitProject(nextProject: ProjectDetail, nextTab?: TabId) {
    setProject(nextProject);
    setSelectedId(nextProject.id);
    await refreshProjects();
    if (nextTab) setActiveTab(nextTab);
  }

  async function deleteSelectedProject() {
    if (!project) return;

    setBusy("delete-project");
    setError(null);
    try {
      const deletedIndex = projects.findIndex((item) => item.id === project.id);
      const nextProjects = await api.deleteProject({ project_id: project.id });
      const nextSelectedProject =
        nextProjects[Math.min(Math.max(deletedIndex, 0), nextProjects.length - 1)] ??
        null;

      setProjects(nextProjects);
      setProject(null);
      setProgressEvents([]);
      setSelectedId(nextSelectedProject?.id ?? null);
      setActiveTab("capture");
    } catch (caught) {
      setError(toFriendlyMessage(caught));
    } finally {
      setBusy(null);
    }
  }

  async function renameSelectedProject(name: string) {
    if (!project) return;

    setBusy("rename-project");
    setError(null);
    try {
      const nextProject = await api.renameProject({
        project_id: project.id,
        name,
      });
      await commitProject(nextProject);
    } catch (caught) {
      setError(toFriendlyMessage(caught));
      throw caught;
    } finally {
      setBusy(null);
    }
  }

  const hasActiveRun = Boolean(
    project?.runs.some((run) =>
      ["queued", "preparing", "running", "cancelling"].includes(run.status),
    ),
  );
  const hasActiveExport = Boolean(
    project?.exports.some((item) => ["pending", "validating"].includes(item.status)),
  );
  const progressActive = sidecarBusy || hasActiveRun || hasActiveExport;
  const deleteDisabledReason = progressActive
    ? "Finish or cancel the active training/export job before deleting."
    : busy && busy !== "delete-project"
      ? "Finish the current action before deleting this project."
      : null;
  const renameDisabledReason = progressActive
    ? "Finish or cancel the active training/export job before renaming."
    : busy && busy !== "rename-project"
      ? "Finish the current action before renaming this project."
      : null;

  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <Activity size={22} strokeWidth={2.2} />
          </div>
          <div>
            <p className="eyebrow">Shortwav Labs</p>
            <h1>RTNeural Trainer</h1>
          </div>
        </div>

        <NewProjectForm
          busy={busy === "create"}
          onCreate={async (name, target_kind) => {
            setBusy("create");
            try {
              const nextProject = await api.createProject({ name, target_kind });
              await commitProject(nextProject, "capture");
            } catch (caught) {
              setError(toFriendlyMessage(caught));
            } finally {
              setBusy(null);
            }
          }}
        />

        <ProjectList
          projects={projects}
          selectedId={selectedId}
          onSelect={(id) => setSelectedId(id)}
        />

        <RuntimeStatus
          status={status}
          settings={runtimeSettings}
          inspection={deviceInspection}
          busy={runtimeBusy}
          error={runtimeError}
          onRefresh={() => void refreshDeviceInspection()}
          onSave={(nextSettings) => void saveRuntimeSettings(nextSettings)}
        />
      </aside>

      <main className="workspace" id="workspace" tabIndex={-1}>
        {error ? (
          <ErrorNotice message={error} onDismiss={() => setError(null)} />
        ) : null}

        {project ? (
          <>
            <ProjectHeader
              project={project}
              deleteBusy={busy === "delete-project"}
              deleteDisabledReason={deleteDisabledReason}
              renameBusy={busy === "rename-project"}
              renameDisabledReason={renameDisabledReason}
              onDelete={() => void deleteSelectedProject()}
              onRename={renameSelectedProject}
            />
            <StepTabs activeTab={activeTab} onChange={setActiveTab} />
            <section className="work-surface">
              {activeTab === "capture" ? (
                <CaptureView
                  project={project}
                  busy={busy === "audio"}
                  onAnalyze={async (capture) => {
                    setProgressEvents([]);
                    setBusy("audio");
                    try {
                      const nextProject = await api.updateAudio({
                        project_id: project.id,
                        input_path: capture.inputPath,
                        target_path: capture.targetPath,
                        target_sample_rate: capture.targetSampleRate,
                        resample: capture.resample,
                        channel_policy: capture.channelPolicy,
                      });
                      await commitProject(nextProject, "align");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                />
              ) : null}

              {activeTab === "align" ? (
                <AlignView
                  project={project}
                  busy={busy === "alignment"}
                  onApply={async (manualLatencyAdjustmentSamples) => {
                    setProgressEvents([]);
                    setBusy("alignment");
                    try {
                      const nextProject = await api.updateAlignment({
                        project_id: project.id,
                        manual_latency_adjustment_samples: manualLatencyAdjustmentSamples,
                      });
                      await commitProject(nextProject, "align");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                />
              ) : null}

              {activeTab === "train" ? (
                <TrainView
                  project={project}
                  backend={runtimeSettings?.selected_backend ?? "keras"}
                  busy={busy === "train"}
                  recipeBusy={busy === "recipe"}
                  events={progressEvents}
                  customRecipes={trainingRecipes}
                  onTrain={async (options) => {
                    setProgressEvents([]);
                    setBusy("train");
                    try {
                      const nextProject = await api.startTraining({
                        project_id: project.id,
                        preset: options.preset,
                        resume_from_run_id: options.resumeFromRunId ?? null,
                        epochs: options.epochs,
                        batch_size: options.batchSize,
                        learning_rate: options.learningRate,
                        sequence_length: options.sequenceLength,
                        early_stopping_patience: options.earlyStoppingPatience,
                        early_stopping_min_delta: options.earlyStoppingMinDelta,
                        max_windows: options.maxWindows,
                      });
                      await commitProject(nextProject, "train");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                  onSaveRecipe={async (options, name, recipeId) => {
                    setBusy("recipe");
                    try {
                      return await saveTrainingRecipe(options, name, recipeId);
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                      throw caught;
                    } finally {
                      setBusy(null);
                    }
                  }}
                  onDeleteRecipe={async (recipeId) => {
                    setBusy("recipe");
                    try {
                      await deleteTrainingRecipe(recipeId);
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                      throw caught;
                    } finally {
                      setBusy(null);
                    }
                  }}
                  onCancel={async (runId) => {
                    setBusy("cancel");
                    try {
                      const nextProject = await api.cancelTrainingRun({
                        project_id: project.id,
                        run_id: runId,
                      });
                      await commitProject(nextProject, "train");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                  onResume={async (runId) => {
                    setBusy("resume");
                    try {
                      const nextProject = await api.resumeTrainingRun({
                        project_id: project.id,
                        run_id: runId,
                      });
                      await commitProject(nextProject, "train");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                />
              ) : null}

              {activeTab === "evaluate" ? <EvaluateView project={project} /> : null}

              {activeTab === "export" ? (
                <ExportView
                  project={project}
                  busy={busy === "export"}
                  onExport={async (runId) => {
                    setProgressEvents([]);
                    setBusy("export");
                    try {
                      const nextProject = await api.exportRun({
                        project_id: project.id,
                        run_id: runId,
                      });
                      await commitProject(nextProject, "export");
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                  onOpenExport={async (exportId) => {
                    setBusy("open-export");
                    try {
                      await api.openExportFolder({
                        project_id: project.id,
                        export_id: exportId,
                      });
                    } catch (caught) {
                      setError(toFriendlyMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                />
              ) : null}
            </section>

            <ProgressLog events={progressEvents} active={progressActive} />

            <NotesPanel
              project={project}
              onSave={async (notes) => {
                setBusy("notes");
                try {
                  const nextProject = await api.updateNotes({
                    project_id: project.id,
                    notes,
                  });
                  await commitProject(nextProject);
                } catch (caught) {
                  setError(toFriendlyMessage(caught));
                } finally {
                  setBusy(null);
                }
              }}
            />
          </>
        ) : (
          <EmptyState
            busy={busy === "sample"}
            onCreateSample={async () => {
              setBusy("sample");
              setError(null);
              setProgressEvents([]);
              try {
                const nextProject = await api.createSampleProject();
                await commitProject(nextProject, "train");
              } catch (caught) {
                setError(toFriendlyMessage(caught));
              } finally {
                setBusy(null);
              }
            }}
          />
        )}
      </main>
    </div>
  );
}

function NewProjectForm({
  busy,
  onCreate,
}: {
  busy: boolean;
  onCreate: (name: string, targetKind: TargetKind) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [targetKind, setTargetKind] = useState<TargetKind>("amp");

  return (
    <form
      className="new-project"
      onSubmit={(event) => {
        event.preventDefault();
        void onCreate(name, targetKind).then(() => setName(""));
      }}
    >
      <label>
        Project
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="Deluxe pedal capture"
        />
      </label>
      <div className="segmented compact" role="group" aria-label="Project target">
        {(Object.keys(targetLabels) as TargetKind[]).map((target) => (
          <button
            aria-pressed={targetKind === target}
            className={targetKind === target ? "active" : ""}
            key={target}
            type="button"
            onClick={() => setTargetKind(target)}
          >
            {targetLabels[target]}
          </button>
        ))}
      </div>
      <button className="primary-button" type="submit" disabled={busy}>
        {busy ? <LoaderCircle className="spin" size={16} /> : <FolderPlus size={16} />}
        Create
      </button>
    </form>
  );
}

function ProjectList({
  projects,
  selectedId,
  onSelect,
}: {
  projects: ProjectSummary[];
  selectedId: string | null;
  onSelect: (projectId: string) => void;
}) {
  return (
    <div className="project-list">
      <div className="section-label">Projects</div>
      {projects.length === 0 ? (
        <p className="muted">No captures yet.</p>
      ) : (
        projects.map((project) => (
          <button
            aria-current={project.id === selectedId ? "true" : undefined}
            className={`project-row ${project.id === selectedId ? "active" : ""}`}
            key={project.id}
            type="button"
            onClick={() => onSelect(project.id)}
          >
            <span className={`status-dot ${project.audio_status}`} aria-hidden="true" />
            <span>
              <strong>{project.name}</strong>
              <small>
                {targetLabels[project.target_kind]} · {audioStatusLabel(project.audio_status)} ·{" "}
                {project.best_quality === null
                  ? "No run"
                  : `ESR ${project.best_quality.toFixed(3)}`}
              </small>
            </span>
          </button>
        ))
      )}
    </div>
  );
}

function RuntimeStatus({
  status,
  settings,
  inspection,
  busy,
  error,
  onRefresh,
  onSave,
}: {
  status: AppStatus | null;
  settings: RuntimeSettings | null;
  inspection: DeviceInspection | null;
  busy: string | null;
  error: string | null;
  onRefresh: () => void;
  onSave: (settings: RuntimeSettings) => void;
}) {
  const [backend, setBackend] = useState<RuntimeBackend>(
    settings?.selected_backend ?? "keras",
  );
  const [selectedDevice, setSelectedDevice] = useState(settings?.selected_device ?? "auto");
  const [externalPythonPath, setExternalPythonPath] = useState(
    settings?.external_python_path ?? "",
  );

  useEffect(() => {
    setBackend(settings?.selected_backend ?? "keras");
    setSelectedDevice(settings?.selected_device ?? "auto");
    setExternalPythonPath(settings?.external_python_path ?? "");
  }, [settings?.external_python_path, settings?.selected_backend, settings?.selected_device]);

  const packageVersions = inspection?.package_versions ?? {};
  const deviceOptions = useMemo(
    () => runtimeDeviceOptions(backend, inspection),
    [backend, inspection],
  );
  const selectedDeviceWarning = runtimeDeviceWarning(backend, selectedDevice, inspection);
  const runtimeSource = settings?.external_python_path
    ? "External"
    : status?.trainer_sidecar_present
      ? "Sidecar"
      : "uv dev";
  const hasChanges =
    backend !== (settings?.selected_backend ?? "keras") ||
    selectedDevice !== (settings?.selected_device ?? "auto") ||
    externalPythonPath.trim() !== (settings?.external_python_path ?? "");

  useEffect(() => {
    const selectedOption = deviceOptions.find((option) => option.value === selectedDevice);
    if (!selectedOption || !selectedOption.available) {
      setSelectedDevice("auto");
    }
  }, [deviceOptions, selectedDevice]);

  return (
    <div className="runtime">
      <div className="runtime-heading">
        <div className="section-label">Runtime</div>
        <button
          className="icon-button"
          type="button"
          disabled={busy === "inspect"}
          onClick={onRefresh}
          title="Refresh runtime"
        >
          {busy === "inspect" ? (
            <LoaderCircle className="spin" size={14} />
          ) : (
            <RotateCcw size={14} />
          )}
        </button>
      </div>
      <dl>
        <div>
          <dt>App</dt>
          <dd>{status?.version ?? "Loading"}</dd>
        </div>
        <div>
          <dt>Trainer</dt>
          <dd>{runtimeSource}</dd>
        </div>
        <div>
          <dt>Validator</dt>
          <dd>{status?.validator_sidecar_present ? "Sidecar" : "CMake dev"}</dd>
        </div>
        <div>
          <dt>Backend</dt>
          <dd>{backendLabel(settings?.selected_backend ?? "keras")}</dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>{runtimeDeviceLabel(selectedDevice, backend, inspection)}</dd>
        </div>
      </dl>

      <div className="runtime-chips">
        <RuntimeChip label="CPU" active={Boolean(inspection?.cpu_available ?? true)} />
        <RuntimeChip label="MPS" active={Boolean(inspection?.mps_available && inspection?.mps_built)} />
        <RuntimeChip label="CUDA" active={Boolean(inspection?.cuda_available)} />
      </div>

      <div className="package-list">
        <PackageVersion label="Python" value={packageVersions.python ?? inspection?.python} />
        <PackageVersion label="rttrainer" value={packageVersions.rttrainer ?? inspection?.trainer_version} />
        <PackageVersion label="TensorFlow" value={packageVersions.tensorflow ?? inspection?.tensorflow_version} />
        <PackageVersion label="Keras" value={packageVersions.keras ?? inspection?.keras_version} />
        <PackageVersion label="TF Metal" value={packageVersions["tensorflow-metal"]} />
        <PackageVersion label="PyTorch" value={packageVersions.torch ?? inspection?.torch_version} />
      </div>

      <form
        className="runtime-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSave({
            selected_backend: backend,
            selected_device: selectedDevice,
            external_python_path: externalPythonPath.trim() || null,
          });
        }}
      >
        <label>
          Backend
          <select
            value={backend}
            onChange={(event) => setBackend(event.target.value as RuntimeBackend)}
          >
            <option value="keras">TensorFlow/Keras</option>
            <option value="pytorch">PyTorch</option>
          </select>
        </label>
        <label>
          Training device
          <select
            value={selectedDevice}
            onChange={(event) => setSelectedDevice(event.target.value)}
          >
            {deviceOptions.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={!option.available}
              >
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {selectedDeviceWarning ? (
          <small className="runtime-hint">{selectedDeviceWarning}</small>
        ) : null}
        <label>
          External Python
          <input
            value={externalPythonPath}
            onChange={(event) => setExternalPythonPath(event.target.value)}
            placeholder="/path/to/python"
          />
        </label>
        <button
          className="secondary-button wide"
          type="submit"
          disabled={busy === "settings" || !hasChanges}
        >
          {busy === "settings" ? (
            <LoaderCircle className="spin" size={16} />
          ) : (
            <Save size={16} />
          )}
          Save runtime
        </button>
      </form>

      {error ? (
        <div className="runtime-error">
          <AlertTriangle size={15} />
          <span>
            <strong>{friendlyError(error).title}</strong>
            <small>{friendlyError(error).action}</small>
          </span>
        </div>
      ) : null}
    </div>
  );
}

function ErrorNotice({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  const friendly = friendlyError(message);
  return (
    <div className="notice notice-error app-error" role="alert">
      <AlertTriangle size={18} />
      <span>
        <strong>{friendly.title}</strong>
        <small>{friendly.detail}</small>
        <small>{friendly.action}</small>
      </span>
      <button className="secondary-button" type="button" onClick={onDismiss}>
        Dismiss
      </button>
    </div>
  );
}

function RuntimeChip({ label, active }: { label: string; active: boolean }) {
  return (
    <span
      className={active ? "runtime-chip active" : "runtime-chip"}
      aria-label={`${label} ${active ? "available" : "unavailable"}`}
    >
      {label}
    </span>
  );
}

function PackageVersion({
  label,
  value,
}: {
  label: string;
  value: string | undefined;
}) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value && value !== "not installed" ? value : "none"}</strong>
    </div>
  );
}

function ProjectHeader({
  project,
  deleteBusy,
  deleteDisabledReason,
  renameBusy,
  renameDisabledReason,
  onDelete,
  onRename,
}: {
  project: ProjectDetail;
  deleteBusy: boolean;
  deleteDisabledReason: string | null;
  renameBusy: boolean;
  renameDisabledReason: string | null;
  onDelete: () => void;
  onRename: (name: string) => Promise<void>;
}) {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState(project.name);
  const [renameTouched, setRenameTouched] = useState(false);
  const latestRun = project.runs[project.runs.length - 1];
  const latestExport = project.exports[project.exports.length - 1];
  const trimmedDraftName = draftName.trim();
  const canSaveName =
    trimmedDraftName.length > 0 &&
    trimmedDraftName.length <= 120 &&
    trimmedDraftName !== project.name;
  const deleteLabel = deleteBusy
    ? "Deleting"
    : confirmDelete
      ? "Confirm delete"
      : "Delete project";
  const renameError =
    renameTouched && trimmedDraftName.length === 0
      ? "Project name is required."
      : renameTouched && trimmedDraftName.length > 120
        ? "Project name must be 120 characters or fewer."
        : null;

  useEffect(() => {
    setConfirmDelete(false);
    setEditingName(false);
    setDraftName(project.name);
    setRenameTouched(false);
  }, [project.id, project.name]);

  async function submitRename() {
    setRenameTouched(true);
    if (!canSaveName) {
      if (trimmedDraftName === project.name) {
        setEditingName(false);
      }
      return;
    }

    try {
      await onRename(trimmedDraftName);
      setEditingName(false);
      setRenameTouched(false);
    } catch {
      // The app-level error notice carries the backend message.
    }
  }

  return (
    <header className="project-header">
      <div>
        <p className="eyebrow">{targetLabels[project.target_kind]} capture</p>
        {editingName ? (
          <form
            className="rename-form"
            onSubmit={(event) => {
              event.preventDefault();
              void submitRename();
            }}
          >
            <input
              aria-label="Project name"
              autoFocus
              maxLength={140}
              value={draftName}
              onBlur={() => setRenameTouched(true)}
              onChange={(event) => setDraftName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  setDraftName(project.name);
                  setEditingName(false);
                  setRenameTouched(false);
                }
              }}
            />
            <div className="rename-actions">
              <button
                className="secondary-button"
                type="submit"
                disabled={renameBusy || !canSaveName}
              >
                {renameBusy ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}
                Save name
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={renameBusy}
                onClick={() => {
                  setDraftName(project.name);
                  setEditingName(false);
                  setRenameTouched(false);
                }}
              >
                <X size={16} />
                Cancel
              </button>
            </div>
            {renameError ? <p className="rename-error">{renameError}</p> : null}
          </form>
        ) : (
          <h2>{project.name}</h2>
        )}
        <p className="path-line">{project.project_dir}</p>
        {confirmDelete ? (
          <p className="delete-confirmation" role="status">
            This removes the project, runs, exports, reports, and managed files.
          </p>
        ) : null}
      </div>
      <div className="project-header-side">
        <div className="summary-strip">
          <Metric label="Audio" value={project.audio?.status ?? "missing"} />
          <Metric label="Runs" value={String(project.runs.length)} />
          <Metric
            label="Best ESR"
            value={latestRun?.metrics ? latestRun.metrics.esr.toFixed(3) : "none"}
          />
          <Metric label="Export" value={latestExport?.status ?? "blocked"} />
        </div>
        <div className="project-actions" aria-live="polite">
          {!editingName ? (
            <button
              className="secondary-button"
              type="button"
              disabled={deleteBusy || renameBusy || Boolean(renameDisabledReason)}
              title={renameDisabledReason ?? "Rename project"}
              onClick={() => {
                setConfirmDelete(false);
                setDraftName(project.name);
                setEditingName(true);
                setRenameTouched(false);
              }}
            >
              {renameBusy ? <LoaderCircle className="spin" size={16} /> : <Pencil size={16} />}
              Rename project
            </button>
          ) : null}
          {confirmDelete ? (
            <button
              className="secondary-button"
              type="button"
              disabled={deleteBusy}
              onClick={() => setConfirmDelete(false)}
            >
              Cancel
            </button>
          ) : null}
          <button
            className="danger-button"
            type="button"
            disabled={deleteBusy || Boolean(deleteDisabledReason)}
            title={deleteDisabledReason ?? "Delete project"}
            onClick={() => {
              setEditingName(false);
              if (!confirmDelete) {
                setConfirmDelete(true);
                return;
              }
              onDelete();
            }}
          >
            {deleteBusy ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
            {deleteLabel}
          </button>
        </div>
        {deleteDisabledReason ? (
          <p className="project-action-hint">{deleteDisabledReason}</p>
        ) : null}
        {renameDisabledReason && !deleteDisabledReason ? (
          <p className="project-action-hint">{renameDisabledReason}</p>
        ) : null}
      </div>
    </header>
  );
}

function StepTabs({
  activeTab,
  onChange,
}: {
  activeTab: TabId;
  onChange: (tabId: TabId) => void;
}) {
  return (
    <nav className="step-tabs" aria-label="Workflow" role="tablist">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        return (
          <button
            aria-selected={activeTab === tab.id}
            className={activeTab === tab.id ? "active" : ""}
            key={tab.id}
            role="tab"
            type="button"
            onClick={() => onChange(tab.id)}
          >
            <Icon size={16} />
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}

function CaptureView({
  project,
  busy,
  onAnalyze,
}: {
  project: ProjectDetail;
  busy: boolean;
  onAnalyze: (payload: CaptureAnalyzePayload) => Promise<void>;
}) {
  const [inputPath, setInputPath] = useState(project.audio?.input.path ?? "");
  const [targetPath, setTargetPath] = useState(project.audio?.target.path ?? "");
  const [resample, setResample] = useState(false);
  const [targetSampleRate, setTargetSampleRate] = useState(48_000);
  const [channelPolicy, setChannelPolicy] = useState<CaptureChannelPolicy>("mixdown");
  const captureValidation = validateCaptureForm(inputPath, targetPath);
  const canPickFiles = isTauriRuntime();

  async function pickCaptureFile(kind: "input" | "target") {
    const selected = await openDialog({
      multiple: false,
      directory: false,
      title: kind === "input" ? "Choose dry input WAV" : "Choose processed target WAV",
      filters: [{ name: "WAV audio", extensions: ["wav", "wave"] }],
    });
    if (typeof selected !== "string") return;
    if (kind === "input") {
      setInputPath(selected);
    } else {
      setTargetPath(selected);
    }
  }

  return (
    <div className="screen-grid capture-grid">
      <div className="panel span-7">
        <ScreenTitle
          icon={FileAudio}
          title="Capture Source"
          detail="Import the dry reference and matching processed target."
        />
        <form
          className="path-form"
          onSubmit={(event) => {
            event.preventDefault();
            void onAnalyze({
              inputPath,
              targetPath,
              targetSampleRate,
              resample,
              channelPolicy,
            });
          }}
        >
          <PathPickerField
            label="Dry input WAV"
            value={inputPath}
            placeholder="/path/to/input.wav"
            disabled={busy}
            canPick={canPickFiles}
            onChange={setInputPath}
            onPick={() => void pickCaptureFile("input")}
          />
          <PathPickerField
            label="Processed target WAV"
            value={targetPath}
            placeholder="/path/to/target.wav"
            disabled={busy}
            canPick={canPickFiles}
            onChange={setTargetPath}
            onPick={() => void pickCaptureFile("target")}
          />
          <div className="capture-options">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={resample}
                onChange={(event) => setResample(event.target.checked)}
              />
              <span>
                Resample prepared audio
                <small>Use this when captures are not already at the target rate.</small>
              </span>
            </label>
            <label>
              Prepared sample rate
              <select
                value={targetSampleRate}
                onChange={(event) => setTargetSampleRate(Number(event.target.value))}
                disabled={!resample}
              >
                <option value={48000}>48 kHz</option>
                <option value={44100}>44.1 kHz</option>
                <option value={96000}>96 kHz</option>
              </select>
            </label>
            <label>
              Stereo and multichannel
              <select
                value={channelPolicy}
                onChange={(event) =>
                  setChannelPolicy(event.target.value as CaptureChannelPolicy)
                }
              >
                <option value="mixdown">Mix to mono</option>
                <option value="first">Use first channel</option>
                <option value="reject">Require mono</option>
              </select>
            </label>
          </div>
          {captureValidation.length ? (
            <div className="validation-list">
              {captureValidation.map((message) => (
                <div className="validation-row" key={message}>
                  <AlertTriangle size={15} />
                  <span>{message}</span>
                </div>
              ))}
            </div>
          ) : null}
          <button
            className="primary-button"
            type="submit"
            disabled={busy || captureValidation.length > 0}
          >
            {busy ? <LoaderCircle className="spin" size={16} /> : <Gauge size={16} />}
            Analyze
          </button>
        </form>
      </div>

      <div className="panel span-5">
        <ScreenTitle
          icon={SlidersHorizontal}
          title="Preflight"
          detail="The first pass checks format, gain, duration, and alignment."
        />
        {project.audio ? (
          <AudioReportView project={project} />
        ) : (
          <p className="muted">
            Add paired files to generate a preparation report.
          </p>
        )}
      </div>
    </div>
  );
}

function PathPickerField({
  label,
  value,
  placeholder,
  disabled,
  canPick,
  onChange,
  onPick,
}: {
  label: string;
  value: string;
  placeholder: string;
  disabled: boolean;
  canPick: boolean;
  onChange: (value: string) => void;
  onPick: () => void;
}) {
  return (
    <label>
      {label}
      <div className="path-picker">
        <input
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          disabled={disabled}
        />
        <button
          className="secondary-button"
          type="button"
          disabled={disabled || !canPick}
          onClick={onPick}
          title={canPick ? `Choose ${label}` : "Native file picker is available in Tauri"}
        >
          <FolderOpen size={16} />
          Choose
        </button>
      </div>
    </label>
  );
}

function AlignView({
  project,
  busy,
  onApply,
}: {
  project: ProjectDetail;
  busy: boolean;
  onApply: (manualLatencyAdjustmentSamples: number) => Promise<void>;
}) {
  const [nudge, setNudge] = useState(project.audio?.manual_latency_adjustment_samples ?? 0);
  const [waveform, setWaveform] = useState<ProjectWaveform | null>(null);
  const [waveformBusy, setWaveformBusy] = useState(false);
  const [waveformError, setWaveformError] = useState<string | null>(null);
  const autoLatency =
    project.audio?.latency_auto_samples ?? project.audio?.latency_samples ?? 0;
  const effectiveLatency = autoLatency + nudge;
  const confidence = project.audio?.latency_confidence ?? 0;
  const sampleRate =
    getNumber(project.audio?.prepared ?? null, "sample_rate") ??
    project.audio?.input.sample_rate ??
    48_000;
  const savedNudge = project.audio?.manual_latency_adjustment_samples ?? 0;
  const hasUnsavedNudge = nudge !== savedNudge;

  useEffect(() => {
    setNudge(project.audio?.manual_latency_adjustment_samples ?? 0);
  }, [project.audio?.manual_latency_adjustment_samples, project.id]);

  useEffect(() => {
    let mounted = true;
    setWaveform(null);
    setWaveformError(null);
    if (!project.audio) {
      setWaveformBusy(false);
      return;
    }

    setWaveformBusy(true);
    api
      .getProjectWaveform({ project_id: project.id, bins: 240 })
      .then((nextWaveform) => {
        if (mounted) setWaveform(nextWaveform);
      })
      .catch((caught) => {
        if (mounted) setWaveformError(toFriendlyMessage(caught));
      })
      .finally(() => {
        if (mounted) setWaveformBusy(false);
      });

    return () => {
      mounted = false;
    };
  }, [project.id, project.updated_at]);

  if (!project.audio) {
    return (
      <SetupRequired
        icon={Crosshair}
        title="Prepare audio before alignment"
        detail="Alignment uses the preparation report. Add paired WAV files in Capture, then run preflight."
      />
    );
  }

  return (
    <div className="screen-grid">
      <div className="panel span-8">
        <ScreenTitle
          icon={Crosshair}
          title="Latency Alignment"
          detail="Inspect the detected offset before committing training time."
        />
        <WaveformOverlay
          latency={effectiveLatency}
          loading={waveformBusy}
          error={waveformError}
          waveform={waveform}
        />
      </div>
      <div className="panel span-4">
        <div className="metric-stack">
          <Metric label="Auto estimate" value={`${autoLatency} samples`} />
          <Metric label="Manual adjustment" value={`${nudge} samples`} />
          <Metric label="Training latency" value={`${effectiveLatency} samples`} />
          <Metric label="Confidence" value={`${Math.round(confidence * 100)}%`} />
          <Metric
            label="Milliseconds"
            value={`${((effectiveLatency / sampleRate) * 1000).toFixed(2)} ms`}
          />
        </div>
        <label className="range-control">
          Manual nudge
          <input
            type="range"
            min="-256"
            max="256"
            value={nudge}
            onChange={(event) => setNudge(Number(event.target.value))}
          />
          <span>{nudge} samples</span>
        </label>
        <button
          className="primary-button wide"
          type="button"
          disabled={busy || !hasUnsavedNudge}
          onClick={() => void onApply(nudge)}
        >
          {busy ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}
          Apply alignment
        </button>
        {project.audio?.manual_latency_adjustment_samples ? (
          <div className="notice notice-info">
            <CheckCircle2 size={18} />
            <span>
              Manual alignment is saved and will be used for training and export.
            </span>
          </div>
        ) : null}
        {project.audio?.warning_details.length ? (
          <WarningList warnings={project.audio.warning_details} />
        ) : project.audio?.warnings.length ? (
          <WarningList warnings={legacyWarnings(project.audio.warnings)} />
        ) : (
          <div className="notice notice-ok">
            <CheckCircle2 size={18} />
            <span>Prepared audio is ready for training.</span>
          </div>
        )}
      </div>
    </div>
  );
}

function TrainView({
  project,
  backend,
  busy,
  recipeBusy,
  events,
  customRecipes,
  onTrain,
  onSaveRecipe,
  onDeleteRecipe,
  onCancel,
  onResume,
}: {
  project: ProjectDetail;
  backend: RuntimeBackend;
  busy: boolean;
  recipeBusy: boolean;
  events: SidecarProgressEvent[];
  customRecipes: TrainingRecipe[];
  onTrain: (options: TrainingOptions) => Promise<void>;
  onSaveRecipe: (
    options: TrainingOptions,
    name: string,
    recipeId?: string,
  ) => Promise<TrainingRecipe>;
  onDeleteRecipe: (recipeId: string) => Promise<void>;
  onCancel: (runId: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
}) {
  const recommendation = useMemo(
    () => recommendPreset(project, backend),
    [backend, project.audio, project.target_kind],
  );
  const customRecipeOptions = useMemo(
    () => customRecipes.map(trainingRecipeFromCustom),
    [customRecipes],
  );
  const recipeOptions = useMemo(
    () => [...builtInTrainingRecipes, ...customRecipeOptions],
    [customRecipeOptions],
  );
  const [selectedRecipeId, setSelectedRecipeId] = useState("builtin_balanced");
  const [preset, setPreset] = useState(
    recipeOptions.find((recipe) => recipe.id === "builtin_balanced")?.modelPreset ??
      recommendation.presetId,
  );
  const [epochs, setEpochs] = useState(40);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [sequenceLength, setSequenceLength] = useState(8192);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(6);
  const [earlyStoppingMinDelta, setEarlyStoppingMinDelta] = useState(0.0001);
  const [maxWindows, setMaxWindows] = useState(2048);
  const [resumeFromRunId, setResumeFromRunId] = useState<string | null>(null);
  const [recipeName, setRecipeName] = useState("");
  const [recipeNotice, setRecipeNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<RunPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const selectedRecipe = recipeOptions.find((recipe) => recipe.id === selectedRecipeId) ?? null;
  const selectedPreset = presets.find((item) => item.id === preset) ?? presets[0];
  const selectedPresetSupported = selectedPreset.backends.includes(backend);
  const resumeCandidates = useMemo(
    () => compatibleResumeRuns(project.runs, preset, backend),
    [backend, preset, project.runs],
  );
  const selectedResumeRun =
    resumeCandidates.find((run) => run.id === resumeFromRunId) ?? null;
  const completedRuns = project.runs.filter((run) => run.status === "completed");
  const canTrain = project.audio?.status === "ready";
  const currentOptions: TrainingOptions = {
    preset,
    resumeFromRunId,
    epochs,
    batchSize,
    learningRate,
    sequenceLength,
    earlyStoppingPatience,
    earlyStoppingMinDelta,
    maxWindows,
  };
  const selectedCustomRecipe = customRecipes.find((recipe) => recipe.id === selectedRecipeId);
  const hasRecipeName = recipeName.trim().length > 0;
  const activeRun = [...project.runs]
    .reverse()
    .find((run) => ["queued", "preparing", "running", "cancelling"].includes(run.status));
  const resumableRun = [...project.runs]
    .reverse()
    .find((run) => run.status === "failed" || run.status === "interrupted");
  const evidenceRun =
    activeRun ??
    [...project.runs]
      .reverse()
      .find((run) => run.metrics || ["failed", "interrupted"].includes(run.status)) ??
    null;
  const liveHistory = trainingHistoryFromEvents(events, evidenceRun?.id ?? null);
  const reportHistory = historyFromReport(preview?.report ?? null);
  const curveHistory = liveHistory.length ? liveHistory : reportHistory;
  const reportAssessment = getNestedObject(preview?.report ?? null, ["quality_assessment"]);
  const quality = qualityVerdict(evidenceRun?.metrics ?? null, reportAssessment);

  useEffect(() => {
    if (selectedPresetSupported) return;
    setPreset(recommendation.presetId);
  }, [backend, preset, recommendation.presetId, selectedPresetSupported]);

  useEffect(() => {
    if (!resumeFromRunId) return;
    if (resumeCandidates.some((run) => run.id === resumeFromRunId)) return;
    setResumeFromRunId(null);
  }, [resumeCandidates, resumeFromRunId]);

  useEffect(() => {
    if (!selectedRecipe || recipeModelSupported(selectedRecipe.modelPreset, backend)) return;
    setSelectedRecipeId("manual");
    setRecipeNotice("Recipe model is not available on this backend.");
  }, [backend, selectedRecipe]);

  useEffect(() => {
    if (selectedRecipeId !== "builtin_balanced") return;
    setMaxWindows(Math.max(recommendedWindowBudget(project), 2048));
  }, [project.audio?.capture_profile, project.id, selectedRecipeId]);

  useEffect(() => {
    const recipe = recipeOptions.find((item) => item.id === selectedRecipeId);
    if (!recipe) return;
    setRecipeName(recipe.source === "custom" ? recipe.name : "");
  }, [recipeOptions, selectedRecipeId]);

  useEffect(() => {
    let mounted = true;
    setPreview(null);
    if (!evidenceRun || !["completed", "failed", "interrupted"].includes(evidenceRun.status)) {
      setPreviewBusy(false);
      return;
    }
    setPreviewBusy(true);
    api
      .getRunPreview({ project_id: project.id, run_id: evidenceRun.id })
      .then((nextPreview) => {
        if (mounted) setPreview(nextPreview);
      })
      .catch(() => {
        if (mounted) setPreview(null);
      })
      .finally(() => {
        if (mounted) setPreviewBusy(false);
      });
    return () => {
      mounted = false;
    };
  }, [evidenceRun?.id, evidenceRun?.status, project.id]);

  function applyRecipe(recipe: TrainingRecipeOption) {
    setSelectedRecipeId(recipe.id);
    setPreset(
      recipeModelSupported(recipe.modelPreset, backend) ? recipe.modelPreset : recommendation.presetId,
    );
    setEpochs(recipe.epochs);
    setBatchSize(recipe.batchSize);
    setLearningRate(recipe.learningRate);
    setSequenceLength(recipe.sequenceLength);
    setMaxWindows(recipe.maxWindows);
    setEarlyStoppingPatience(recipe.earlyStoppingPatience);
    setEarlyStoppingMinDelta(recipe.earlyStoppingMinDelta);
    setRecipeName(recipe.source === "custom" ? recipe.name : "");
    setRecipeNotice(null);
  }

  function markManualRecipeEdit() {
    setSelectedRecipeId("manual");
    setRecipeNotice(null);
  }

  async function saveCurrentRecipe() {
    if (!hasRecipeName) return;
    try {
      const saved = await onSaveRecipe(
        currentOptions,
        recipeName.trim(),
        selectedCustomRecipe?.id,
      );
      setSelectedRecipeId(saved.id);
      setRecipeName(saved.name);
      setRecipeNotice(`Saved ${saved.name}.`);
    } catch {
      setRecipeNotice(null);
    }
  }

  async function deleteCurrentRecipe() {
    if (!selectedCustomRecipe) return;
    const deletedName = selectedCustomRecipe.name;
    try {
      await onDeleteRecipe(selectedCustomRecipe.id);
      const fallback = builtInTrainingRecipes.find((recipe) => recipe.id === "builtin_balanced");
      if (fallback) applyRecipe(fallback);
      setRecipeNotice(`Deleted ${deletedName}.`);
    } catch {
      setRecipeNotice(null);
    }
  }

  return (
    <div className="screen-grid">
      <div className="panel span-5">
        <ScreenTitle
          icon={Cpu}
          title="Training Setup"
          detail="Start from a recipe, then tune the run before launching."
        />
        <div className="recipe-panel">
          <label>
            Training recipe
            <select
              value={selectedRecipeId}
              onChange={(event) => {
                const nextId = event.target.value;
                if (nextId === "manual") {
                  setSelectedRecipeId("manual");
                  return;
                }
                const recipe = recipeOptions.find((item) => item.id === nextId);
                if (recipe) applyRecipe(recipe);
              }}
            >
              {selectedRecipeId !== "manual" && !selectedRecipe ? (
                <option value={selectedRecipeId}>Saved recipe</option>
              ) : null}
              <option value="manual">Custom settings</option>
              <optgroup label="Built in">
                {builtInTrainingRecipes.map((recipe) => (
                  <option
                    key={recipe.id}
                    value={recipe.id}
                    disabled={!recipeModelSupported(recipe.modelPreset, backend)}
                  >
                    {recipe.name}
                  </option>
                ))}
              </optgroup>
              {customRecipeOptions.length ? (
                <optgroup label="Saved">
                  {customRecipeOptions.map((recipe) => (
                    <option
                      key={recipe.id}
                      value={recipe.id}
                      disabled={!recipeModelSupported(recipe.modelPreset, backend)}
                    >
                      {recipe.name}
                    </option>
                  ))}
                </optgroup>
              ) : null}
            </select>
          </label>
          <div className="recipe-summary">
            <span>{selectedRecipe?.source === "custom" ? "Saved recipe" : "Recipe"}</span>
            <strong>{selectedRecipe?.name ?? "Custom settings"}</strong>
            <small>
              {selectedRecipe?.description ??
                "Manual values will be used for this run. Save them to reuse later."}
            </small>
          </div>
          <div className="recipe-actions">
            <label>
              Recipe name
              <input
                value={recipeName}
                onChange={(event) => {
                  setRecipeName(event.target.value);
                  setRecipeNotice(null);
                }}
                placeholder="e.g. Rhythm production"
              />
            </label>
            <button
              className="secondary-button"
              type="button"
              disabled={recipeBusy || !hasRecipeName}
              onClick={() => void saveCurrentRecipe()}
            >
              {recipeBusy ? <LoaderCircle className="spin" size={16} /> : <Save size={16} />}
              {selectedCustomRecipe ? "Update recipe" : "Save recipe"}
            </button>
            {selectedCustomRecipe ? (
              <button
                className="danger-button"
                type="button"
                disabled={recipeBusy}
                onClick={() => void deleteCurrentRecipe()}
              >
                <Trash2 size={16} />
                Delete
              </button>
            ) : null}
          </div>
          {recipeNotice ? <small className="recipe-notice">{recipeNotice}</small> : null}
        </div>
        <PresetRecommendation recommendation={recommendation} selectedPreset={selectedPreset} />
        <div className="preset-list">
          {presets.map((item) => {
            const supported = item.backends.includes(backend);
            return (
              <button
                className={preset === item.id ? "preset active" : "preset"}
                disabled={!supported}
                key={item.id}
                title={supported ? item.detail : "Select TensorFlow/Keras to train this preset."}
                type="button"
                onClick={() => {
                  markManualRecipeEdit();
                  setPreset(item.id);
                }}
              >
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.detail}</small>
                </span>
                <em>{supported ? item.cpu : "Keras only"}</em>
              </button>
            );
          })}
        </div>
        <div className="resume-panel">
          <label>
            Start from checkpoint
            <select
              value={resumeFromRunId ?? ""}
              onChange={(event) => {
                const nextRunId = event.target.value || null;
                setResumeFromRunId(nextRunId);
                setRecipeNotice(null);
              }}
            >
              <option value="">Start from scratch</option>
              {resumeCandidates.map((run) => (
                <option key={run.id} value={run.id}>
                  {resumeRunLabel(run)}
                </option>
              ))}
            </select>
          </label>
          <small>
            {selectedResumeRun
              ? "Uses that run's best checkpoint. Epochs below are added after the selected checkpoint."
              : completedRuns.length
                ? "Choose a completed run with the same preset and backend to continue training."
                : "Complete a run first to make checkpoint continuation available."}
          </small>
        </div>
        <div className="training-controls">
          <label>
            Epochs
            <input
              type="number"
              min={1}
              max={500}
              value={epochs}
              onChange={(event) => {
                markManualRecipeEdit();
                setEpochs(clampNumber(event.target.valueAsNumber, 1, 500));
              }}
            />
          </label>
          <label>
            Batch size
            <input
              type="number"
              min={1}
              max={512}
              value={batchSize}
              onChange={(event) => {
                markManualRecipeEdit();
                setBatchSize(clampNumber(event.target.valueAsNumber, 1, 512));
              }}
            />
          </label>
          <label>
            Learning rate
            <input
              type="number"
              min={0.000001}
              max={1}
              step={0.0001}
              value={learningRate}
              onChange={(event) => {
                markManualRecipeEdit();
                setLearningRate(clampFloat(event.target.valueAsNumber, 0.000001, 1, 0.001));
              }}
            />
          </label>
          <label>
            Sequence length
            <input
              type="number"
              min={32}
              max={65536}
              step={32}
              value={sequenceLength}
              onChange={(event) => {
                markManualRecipeEdit();
                setSequenceLength(clampNumber(event.target.valueAsNumber, 32, 65536));
              }}
            />
          </label>
          <label>
            Early-stop patience
            <input
              type="number"
              min={0}
              max={100}
              value={earlyStoppingPatience}
              onChange={(event) => {
                markManualRecipeEdit();
                setEarlyStoppingPatience(clampNumber(event.target.valueAsNumber, 0, 100));
              }}
            />
          </label>
          <label>
            Min ESR improvement
            <input
              type="number"
              min={0}
              max={1}
              step={0.0001}
              value={earlyStoppingMinDelta}
              onChange={(event) => {
                markManualRecipeEdit();
                setEarlyStoppingMinDelta(
                  clampFloat(event.target.valueAsNumber, 0, 1, 0.0001),
                );
              }}
            />
          </label>
          <label>
            Window budget
            <input
              type="number"
              min={32}
              max={16384}
              step={32}
              value={maxWindows}
              onChange={(event) => {
                markManualRecipeEdit();
                setMaxWindows(clampNumber(event.target.valueAsNumber, 32, 16384));
              }}
            />
          </label>
        </div>
        <CaptureTrainingGuidance project={project} />
        <button
          className="primary-button wide"
          type="button"
          disabled={!canTrain || !selectedPresetSupported || busy || Boolean(activeRun)}
          onClick={() =>
            void onTrain({
              preset,
              epochs,
              batchSize,
              learningRate,
              sequenceLength,
              earlyStoppingPatience,
              earlyStoppingMinDelta,
              maxWindows,
              resumeFromRunId,
            })
          }
        >
          {busy ? (
            <LoaderCircle className="spin" size={16} />
          ) : selectedResumeRun ? (
            <RotateCcw size={16} />
          ) : (
            <Play size={16} />
          )}
          {selectedResumeRun ? "Continue training" : "Train"}
        </button>
        {activeRun ? (
          <button
            className="danger-button wide"
            type="button"
            disabled={activeRun.status === "cancelling"}
            onClick={() => void onCancel(activeRun.id)}
          >
            <Square size={16} />
            {activeRun.status === "cancelling" ? "Cancelling" : "Cancel run"}
          </button>
        ) : null}
        {!activeRun && resumableRun ? (
          <button
            className="secondary-button wide"
            type="button"
            onClick={() => void onResume(resumableRun.id)}
          >
            <RotateCcw size={16} />
            Resume checkpoint
          </button>
        ) : null}
        {!canTrain ? (
          <div className="notice notice-warning">
            <AlertTriangle size={18} />
            <span>
              <strong>Training is blocked.</strong>
              <small>Run Capture preflight and resolve blocking audio warnings first.</small>
            </span>
          </div>
        ) : null}
      </div>
      <div className="panel span-7">
        <ScreenTitle
          icon={Activity}
          title="Runs"
          detail="Each run keeps checkpoints, metrics, and preview artifacts."
        />
        <TrainingEvidence
          history={curveHistory}
          quality={quality}
          loading={previewBusy}
          report={preview?.report ?? null}
        />
        <RunTable runs={project.runs} />
      </div>
    </div>
  );
}

function PresetRecommendation({
  recommendation,
  selectedPreset,
}: {
  recommendation: PresetRecommendation;
  selectedPreset: { id: string; label: string; detail: string };
}) {
  const isFollowing = recommendation.presetId === selectedPreset.id;
  return (
    <div className="recommendation-box">
      <div className="recommendation-head">
        <span className={`confidence ${recommendation.confidence}`}>
          {recommendation.confidence}
        </span>
        <strong>{isFollowing ? "Recommended preset selected" : "Recommendation differs"}</strong>
      </div>
      <p>
        {recommendation.label}: {recommendation.reasons.join(" ")}
      </p>
    </div>
  );
}

function CaptureTrainingGuidance({ project }: { project: ProjectDetail }) {
  const profile = project.audio?.capture_profile ?? null;
  const gain = project.audio?.gain ?? null;
  const duration = getNumber(profile, "duration_seconds");
  const windowBudget = getNumber(profile, "recommended_max_windows");
  const gainGuidance = getString(gain, "guidance");
  const gainVerdict = getString(gain, "verdict");
  const headroom = getNumber(gain, "headroom_db");
  const rmsDelta = getNumber(gain, "rms_delta_db");

  if (!project.audio) return null;

  return (
    <div className="training-guidance">
      <div>
        <span>Capture length</span>
        <strong>{duration !== null ? formatSeconds(duration) : "unknown"}</strong>
        <small>
          {windowBudget !== null
            ? `${windowBudget} windows recommended`
            : "Use more windows for longer captures."}
        </small>
      </div>
      <div>
        <span>Gain staging</span>
        <strong>{gainVerdict ? gainVerdict.replace(/_/g, " ") : "unknown"}</strong>
        <small>{gainGuidance ?? "Check peak and RMS before long runs."}</small>
      </div>
      <div>
        <span>Level delta</span>
        <strong>{rmsDelta !== null ? `${rmsDelta.toFixed(1)} dB RMS` : "unknown"}</strong>
        <small>{headroom !== null ? `${headroom.toFixed(1)} dB peak headroom` : ""}</small>
      </div>
    </div>
  );
}

function TrainingEvidence({
  history,
  quality,
  loading,
  report,
}: {
  history: TrainingHistoryPoint[];
  quality: QualityDecision;
  loading: boolean;
  report: Record<string, unknown> | null;
}) {
  const earlyStopping = getNestedObject(report, ["early_stopping"]);
  const stopped = getBoolean(earlyStopping, "stopped");
  const bestEpoch = getNumber(earlyStopping, "best_epoch");
  const dataset = getNestedObject(report, ["dataset"]);
  const selectedWindows = getNumber(dataset, "selected_windows");
  const availableWindows = getNumber(dataset, "available_windows");
  const trend = validationTrend(history);

  return (
    <div className="training-evidence">
      <QualityCallout decision={quality} />
      <ValidationCurve history={history} loading={loading} />
      <div className="evidence-grid">
        <Metric
          label="Curve trend"
          value={trend}
        />
        <Metric
          label="Best epoch"
          value={
            stopped
              ? `Stopped at ${getNumber(earlyStopping, "epoch") ?? "?"}`
              : bestEpoch !== null
                ? String(bestEpoch)
                : "waiting"
          }
        />
        <Metric
          label="Window coverage"
          value={
            selectedWindows !== null && availableWindows !== null
              ? `${selectedWindows}/${availableWindows}`
              : "waiting"
          }
        />
      </div>
    </div>
  );
}

function QualityCallout({ decision }: { decision: QualityDecision }) {
  return (
    <div className={`quality-callout ${decision.verdict}`}>
      <strong>{decision.summary}</strong>
      <span>{decision.action}</span>
    </div>
  );
}

function ValidationCurve({
  history,
  loading,
}: {
  history: TrainingHistoryPoint[];
  loading: boolean;
}) {
  if (loading && history.length === 0) {
    return (
      <div className="curve-empty">
        <LoaderCircle className="spin" size={18} />
        <span>Loading validation curve.</span>
      </div>
    );
  }

  if (history.length === 0) {
    return <p className="muted">Validation curve appears after the first epoch.</p>;
  }

  const values = history
    .map((point) => point.valEsr)
    .filter((value): value is number => value !== null);
  const chart = {
    width: 1000,
    height: 360,
    left: 66,
    right: 24,
    top: 24,
    bottom: 46,
  };
  const plotWidth = chart.width - chart.left - chart.right;
  const plotHeight = chart.height - chart.top - chart.bottom;
  const rawMax = Math.max(...values, 0.001);
  const rawMin = Math.min(...values, rawMax);
  const rawRange = Math.max(0.0001, rawMax - rawMin);
  const domainMax = rawMax + rawRange * 0.08;
  const domainMin = Math.max(0, rawMin - rawRange * 0.08);
  const domainRange = Math.max(0.0001, domainMax - domainMin);
  const xForIndex = (index: number) =>
    chart.left + (history.length === 1 ? plotWidth / 2 : (index / (history.length - 1)) * plotWidth);
  const yForValue = (value: number | null) => {
    const safeValue = value ?? domainMax;
    return chart.top + ((domainMax - safeValue) / domainRange) * plotHeight;
  };
  const path = history
    .map((point, index) => {
      const x = xForIndex(index);
      const y = yForValue(point.valEsr);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  const latest = history[history.length - 1];
  const latestScore = latest.validationScore;
  const best = history.reduce(
    (current, point) =>
      point.valEsr !== null && (current.valEsr === null || point.valEsr < current.valEsr)
        ? point
        : current,
    history[0],
  );
  const selectedPoints = history.filter((point) => point.isBest);
  const bestSelected = selectedPoints[selectedPoints.length - 1] ?? best;
  const first = history.find((point) => point.valEsr !== null);
  const improvement =
    first?.valEsr !== null && first?.valEsr !== undefined && latest.valEsr !== null
      ? Math.max(0, first.valEsr - latest.valEsr)
      : null;
  const stillImproving =
    history.length >= 8 &&
    latest.valEsr !== null &&
    best.valEsr !== null &&
    latest.epoch === best.epoch;
  const gridValues = [domainMax, domainMin + domainRange / 2, domainMin];
  const lrReductions = history.filter((point) => point.learningRateReduced);
  const currentLearningRate = latest.nextLearningRate ?? latest.learningRate;

  return (
    <div className="curve-card">
      <div className="curve-header">
        <div>
          <span>Validation ESR</span>
          <strong>{latest.valEsr !== null ? latest.valEsr.toFixed(4) : "waiting"}</strong>
        </div>
        <small>
          Selected epoch {bestSelected.epoch}
          {improvement !== null ? ` · -${improvement.toFixed(4)} ESR` : ""}
          {latestScore !== null ? ` · score ${latestScore.toFixed(4)}` : ""}
          {latest.predictionRmsRatio !== null
            ? ` · level ${(latest.predictionRmsRatio * 100).toFixed(0)}%`
            : ""}
          {currentLearningRate !== null ? ` · lr ${formatLearningRate(currentLearningRate)}` : ""}
        </small>
      </div>
      <svg
        viewBox={`0 0 ${chart.width} ${chart.height}`}
        role="img"
        aria-label="Validation ESR curve"
      >
        {gridValues.map((value, index) => {
          const y = yForValue(value);
          return (
            <g key={`${value}-${index}`}>
              <path
                d={`M ${chart.left} ${y.toFixed(2)} L ${chart.width - chart.right} ${y.toFixed(2)}`}
                className={index === gridValues.length - 1 ? "curve-baseline" : "curve-gridline"}
              />
              <text x="8" y={y + 4} className="curve-axis-label">
                {value.toFixed(3)}
              </text>
            </g>
          );
        })}
        <path
          d={`M ${chart.left} ${chart.top} L ${chart.left} ${chart.height - chart.bottom} L ${chart.width - chart.right} ${chart.height - chart.bottom}`}
          className="curve-axis"
        />
        <path
          d={`M ${chart.left} ${chart.height - chart.bottom} L ${chart.width - chart.right} ${chart.height - chart.bottom}`}
          className="curve-baseline"
        />
        {lrReductions.map((point) => {
          const index = history.findIndex((item) => item.epoch === point.epoch);
          if (index < 0) return null;
          const x = xForIndex(index);
          return (
            <g key={`lr-${point.epoch}`}>
              <path
                d={`M ${x.toFixed(2)} ${chart.top} L ${x.toFixed(2)} ${chart.height - chart.bottom}`}
                className="curve-lr-marker"
              />
              <text x={x + 6} y={chart.top + 14} className="curve-lr-label">
                lr
              </text>
            </g>
          );
        })}
        <path d={path} className="curve-line" />
        {history.map((point, index) => {
          if (!point.isBest || point.valEsr === null) return null;
          return (
            <circle
              cx={xForIndex(index)}
              cy={yForValue(point.valEsr)}
              r="5"
              key={`${point.epoch}-${index}`}
            />
          );
        })}
      </svg>
      {stillImproving ? (
        <p className="curve-note">
          Best value landed at the final epoch. This run was still improving, so a
          longer run or more patience may be worth testing.
        </p>
      ) : null}
      {lrReductions.length ? (
        <p className="curve-note">
          Learning rate stepped down after epoch{" "}
          {lrReductions.map((point) => point.epoch).join(", ")}.
        </p>
      ) : null}
      <p className="curve-note">
        Checkpoints use validation score: stream ESR plus a short-window diagnostic and
        an underpowered-output penalty.
      </p>
    </div>
  );
}

function EvaluateView({ project }: { project: ProjectDetail }) {
  const completedRuns = useMemo(() => {
    return [...project.runs]
      .filter((run) => run.metrics)
      .sort((a, b) => (a.metrics?.esr ?? 1) - (b.metrics?.esr ?? 1));
  }, [project.runs]);
  const bestRun = completedRuns[0] ?? null;
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [preview, setPreview] = useState<RunPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    if (completedRuns.length === 0) {
      setSelectedRunId(null);
      return;
    }
    if (!selectedRunId || !completedRuns.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(bestRun?.id ?? completedRuns[0].id);
    }
  }, [bestRun?.id, completedRuns, selectedRunId]);

  const selectedRun =
    completedRuns.find((run) => run.id === selectedRunId) ?? bestRun;

  useEffect(() => {
    let mounted = true;
    setPreview(null);
    setPreviewError(null);
    if (!selectedRun) return;

    setPreviewBusy(true);
    api
      .getRunPreview({ project_id: project.id, run_id: selectedRun.id })
      .then((nextPreview) => {
        if (mounted) setPreview(nextPreview);
      })
      .catch((caught) => {
        if (mounted) setPreviewError(toFriendlyMessage(caught));
      })
      .finally(() => {
        if (mounted) setPreviewBusy(false);
      });

    return () => {
      mounted = false;
    };
  }, [project.id, selectedRun?.id]);

  return (
    <div className="screen-grid">
      <div className="panel span-7">
        <ScreenTitle
          icon={Activity}
          title="Prediction Quality"
          detail="Compare target, prediction, and residual before export."
        />
        {completedRuns.length > 1 ? (
          <label className="compact-field">
            Run
            <select
              value={selectedRun?.id ?? ""}
              onChange={(event) => setSelectedRunId(event.target.value)}
            >
              {completedRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.preset} · ESR {run.metrics?.esr.toFixed(3) ?? "none"} ·{" "}
                  {shortId(run.id)}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {selectedRun ? (
          <QualityView run={selectedRun} />
        ) : (
          <SetupRequired
            icon={Activity}
            title="No completed run yet"
            detail="Train a preset first. Evaluation will show quality metrics, the training report, and preview audio."
          />
        )}
        {previewError ? (
          <div className="notice notice-error">
            <AlertTriangle size={18} />
            <span>{previewError}</span>
          </div>
        ) : null}
        <RunReport preview={preview} loading={previewBusy} />
      </div>
      <div className="panel span-5">
        <ScreenTitle
          icon={AudioLines}
          title="Preview"
          detail="Offline renders are generated by the trainer."
        />
        <PreviewPlayer preview={preview} loading={previewBusy} />
      </div>
    </div>
  );
}

function ExportView({
  project,
  busy,
  onExport,
  onOpenExport,
}: {
  project: ProjectDetail;
  busy: boolean;
  onExport: (runId: string) => Promise<void>;
  onOpenExport: (exportId: string) => Promise<void>;
}) {
  const completedRuns = project.runs.filter((run) => run.status === "completed");
  const selectedRun = completedRuns[completedRuns.length - 1] ?? null;

  return (
    <div className="screen-grid">
      <div className="panel span-5">
        <ScreenTitle
          icon={PackageCheck}
          title="Export Gate"
          detail="RTNeural JSON is only ready after validation and benchmark reports."
        />
        <GateList project={project} selectedRun={selectedRun} />
        {!selectedRun ? (
          <div className="notice notice-warning">
            <AlertTriangle size={18} />
            <span>
              <strong>Export is blocked.</strong>
              <small>Complete a training run before writing an RTNeural package.</small>
            </span>
          </div>
        ) : null}
        <button
          className="primary-button wide"
          type="button"
          disabled={!selectedRun || busy}
          onClick={() => selectedRun && void onExport(selectedRun.id)}
        >
          {busy ? <LoaderCircle className="spin" size={16} /> : <Download size={16} />}
          Export RTNeural JSON
        </button>
      </div>
      <div className="panel span-7">
        <ScreenTitle
          icon={PackageCheck}
          title="Packages"
          detail="Each package contains model JSON, metadata, reports, and previews."
        />
        <ExportList exports={project.exports} onOpenExport={onOpenExport} />
      </div>
    </div>
  );
}

function NotesPanel({
  project,
  onSave,
}: {
  project: ProjectDetail;
  onSave: (notes: string) => Promise<void>;
}) {
  const [notes, setNotes] = useState(project.notes);

  useEffect(() => {
    setNotes(project.notes);
  }, [project.id, project.notes]);

  return (
    <section className="notes-panel">
      <label>
        Notes
        <textarea
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          placeholder="Capture chain, gain settings, mic notes, or export target."
        />
      </label>
      <button type="button" onClick={() => void onSave(notes)}>
        <Save size={16} />
        Save notes
      </button>
    </section>
  );
}

function AudioReportView({ project }: { project: ProjectDetail }) {
  const audio = project.audio;
  if (!audio) return null;

  return (
    <div className="report">
      <div className="metric-stack">
        <Metric
          label="Input"
          value={`${audio.input.sample_rate / 1000} kHz · ${audio.input.channels} ch`}
        />
        <Metric
          label="Target"
          value={`${audio.target.sample_rate / 1000} kHz · ${audio.target.channels} ch`}
        />
        <Metric label="Target peak" value={`${audio.target.peak_dbfs.toFixed(1)} dBFS`} />
        <Metric label="Duration" value={`${audio.input.duration_seconds.toFixed(1)} s`} />
        <Metric label="Latency" value={`${audio.latency_samples} samples`} />
      </div>
      <GainGuidance audio={audio} />
      {audio.warning_details.length ? (
        <WarningList warnings={audio.warning_details} />
      ) : audio.warnings.length ? (
        <WarningList warnings={legacyWarnings(audio.warnings)} />
      ) : (
        <div className="notice notice-ok">
          <CheckCircle2 size={18} />
          <span>No blocking preflight warnings.</span>
        </div>
      )}
    </div>
  );
}

function GainGuidance({ audio }: { audio: AudioReport }) {
  const gain = audio.gain ?? null;
  const profile = audio.capture_profile ?? null;
  const verdict = getString(gain, "verdict");
  const guidance = getString(gain, "guidance");
  const rmsDelta = getNumber(gain, "rms_delta_db");
  const headroom = getNumber(gain, "headroom_db");
  const duration = getNumber(profile, "duration_seconds");
  const windows = getNumber(profile, "recommended_max_windows");

  if (!gain && !profile) return null;

  return (
    <div className="guidance-grid">
      <div>
        <span>Gain read</span>
        <strong>{verdict ? verdict.replace(/_/g, " ") : "unknown"}</strong>
        <small>{guidance ?? "Check peak and RMS before long runs."}</small>
      </div>
      <div>
        <span>Headroom</span>
        <strong>{headroom !== null ? `${headroom.toFixed(1)} dB` : "unknown"}</strong>
        <small>{rmsDelta !== null ? `${rmsDelta.toFixed(1)} dB RMS target offset` : ""}</small>
      </div>
      <div>
        <span>Capture handling</span>
        <strong>{duration !== null ? formatSeconds(duration) : "unknown"}</strong>
        <small>{windows !== null ? `${windows} windows recommended` : ""}</small>
      </div>
    </div>
  );
}

function RunTable({ runs }: { runs: TrainingRun[] }) {
  if (runs.length === 0) {
    return (
      <SetupRequired
        icon={Cpu}
        title="No training runs yet"
        detail="Choose a preset and start training. Checkpoints, ESR, runtime cost, and recovery state will appear here."
      />
    );
  }

  return (
    <div className="table">
      <div className="table-head">
        <span>Preset</span>
        <span>Status</span>
        <span>Device</span>
        <span>ESR</span>
        <span>RTF</span>
      </div>
      {runs.map((run) => (
        <div className="table-row" key={run.id}>
          <span>{run.preset}</span>
          <span>{run.status}</span>
          <span>{run.device}</span>
          <span>{run.metrics ? run.metrics.esr.toFixed(3) : run.status}</span>
          <span>{run.metrics ? `${run.metrics.realtime_factor.toFixed(0)}x` : "none"}</span>
        </div>
      ))}
    </div>
  );
}

function QualityView({ run }: { run: TrainingRun }) {
  if (!run.metrics) return null;

  return (
    <div className="quality-grid">
      <Metric label="ESR" value={run.metrics.esr.toFixed(3)} />
      <Metric label="MAE" value={run.metrics.mae.toFixed(3)} />
      <Metric label="RMSE" value={run.metrics.rmse.toFixed(3)} />
      <Metric label="Peak residual" value={run.metrics.peak_residual.toFixed(3)} />
      <Metric label="RMS residual" value={run.metrics.rms_residual.toFixed(3)} />
      <Metric label="Realtime factor" value={`${run.metrics.realtime_factor.toFixed(0)}x`} />
    </div>
  );
}

function RunReport({
  preview,
  loading,
}: {
  preview: RunPreview | null;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="notice notice-ok">
        <LoaderCircle className="spin" size={18} />
        <span>Loading report.</span>
      </div>
    );
  }
  if (!preview) return null;

  const report = preview.report;
  if (!report) {
    return <p className="muted">No training report found.</p>;
  }
  const decision = qualityVerdict(null, getNestedObject(report, ["quality_assessment"]));
  const earlyStopping = getNestedObject(report, ["early_stopping"]);
  const stateDiagnostic = getNestedObject(report, ["state_diagnostic"]);
  const stopped = getBoolean(earlyStopping, "stopped");

  return (
    <div className="report-block">
      <p className="section-label">Training Report</p>
      <QualityCallout decision={decision} />
      <StateDiagnosticPanel diagnostic={stateDiagnostic} />
      <div className="report-grid">
        <Metric label="Backend" value={getString(report, "backend") ?? "unknown"} />
        <Metric label="Epochs" value={String(getNumber(report, "epochs") ?? "unknown")} />
        <Metric
          label="Checkpoint"
          value={String(getNumber(report, "checkpoint_epoch") ?? "unknown")}
        />
        <Metric label="Created" value={formatReportDate(getString(report, "created_at"))} />
        <Metric
          label="Early stopping"
          value={
            stopped
              ? `stopped at ${getNumber(earlyStopping, "epoch") ?? "?"}`
              : `best ${getNumber(earlyStopping, "best_epoch") ?? "unknown"}`
          }
        />
      </div>
      {preview.report_path ? <p className="artifact-path">{preview.report_path}</p> : null}
    </div>
  );
}

function StateDiagnosticPanel({
  diagnostic,
}: {
  diagnostic: Record<string, unknown> | null;
}) {
  if (!diagnostic) return null;
  const verdict = getString(diagnostic, "verdict");
  if (!verdict || verdict === "finite_memory") return null;

  const isDrift = verdict === "state_drift_suspected";
  const continuousEsr = getNumber(diagnostic, "continuous_esr");
  const chunkEsr = getNumber(diagnostic, "chunk_reset_esr");
  const continuousCorrelation = getNumber(diagnostic, "continuous_correlation");
  const chunkCorrelation = getNumber(diagnostic, "chunk_reset_correlation");
  const chunkSize = getNumber(diagnostic, "chunk_size");
  const chunkSeconds = getNumber(diagnostic, "chunk_seconds");

  return (
    <div className={isDrift ? "state-diagnostic warning" : "state-diagnostic"}>
      <div>
        <strong>{getString(diagnostic, "summary") ?? "State diagnostic complete."}</strong>
        <span>{getString(diagnostic, "action") ?? "Inspect the continuous preview before export."}</span>
      </div>
      <div className="state-diagnostic-grid">
        <Metric
          label="Continuous ESR"
          value={continuousEsr !== null ? continuousEsr.toFixed(3) : "unknown"}
        />
        <Metric
          label="Reset ESR"
          value={chunkEsr !== null ? chunkEsr.toFixed(3) : "unknown"}
        />
        <Metric
          label="Correlation"
          value={
            continuousCorrelation !== null && chunkCorrelation !== null
              ? `${continuousCorrelation.toFixed(2)} / ${chunkCorrelation.toFixed(2)}`
              : "unknown"
          }
        />
        <Metric
          label="Reset window"
          value={
            chunkSize !== null && chunkSeconds !== null
              ? `${chunkSize} samples · ${chunkSeconds.toFixed(2)} s`
              : "unknown"
          }
        />
      </div>
    </div>
  );
}

function PreviewPlayer({
  preview,
  loading,
}: {
  preview: RunPreview | null;
  loading: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [activeKind, setActiveKind] = useState("target");
  const [playRequest, setPlayRequest] = useState(0);
  const artifacts = preview?.artifacts ?? [];
  const playable = artifacts.filter((artifact) => artifact.exists);
  const activeArtifact =
    playable.find((artifact) => artifact.kind === activeKind) ?? playable[0] ?? null;
  const activeSrc = activeArtifact ? convertFileSrc(activeArtifact.path) : "";

  useEffect(() => {
    setActiveKind("target");
    setPlayRequest(0);
  }, [preview?.run_id]);

  useEffect(() => {
    if (!playRequest || !audioRef.current) return;
    void audioRef.current.play().catch(() => undefined);
  }, [activeSrc, playRequest]);

  if (loading) {
    return (
      <div className="notice notice-ok">
        <LoaderCircle className="spin" size={18} />
        <span>Loading previews.</span>
      </div>
    );
  }

  if (!preview) {
    return (
      <SetupRequired
        icon={AudioLines}
        title="No preview audio yet"
        detail="Complete a training run to render target, prediction, and residual WAV previews."
      />
    );
  }

  return (
    <div className="preview-player">
      <div className="preview-transport">
        <div>
          <p className="section-label">Now Playing</p>
          <h3>{activeArtifact?.label ?? "No audio"}</h3>
        </div>
        <audio
          ref={audioRef}
          controls
          preload="metadata"
          src={activeSrc}
          aria-label={activeArtifact ? `${activeArtifact.label} preview` : "Preview audio"}
        />
      </div>
      <WaveformComparison artifacts={artifacts} />
      <div className="preview-list">
        {artifacts.map((artifact) => (
          <div
            className={[
              "preview-row",
              artifact.exists ? "" : "disabled",
              activeArtifact?.kind === artifact.kind ? "active" : "",
            ]
              .filter(Boolean)
              .join(" ")}
            key={artifact.kind}
          >
            <button
              type="button"
              aria-pressed={activeArtifact?.kind === artifact.kind}
              disabled={!artifact.exists}
              onClick={() => {
                setActiveKind(artifact.kind);
                setPlayRequest((current) => current + 1);
              }}
              aria-label={`Play ${artifact.label}`}
            >
              <Play size={15} />
            </button>
            <span>
              <strong>{artifact.label}</strong>
              <small>{previewArtifactDetail(artifact)}</small>
            </span>
            <PeakWave peaks={artifact.peaks} waveform={artifact.waveform} />
          </div>
        ))}
      </div>
      <p className="artifact-path">{preview.run_dir}</p>
    </div>
  );
}

function PeakWave({
  peaks,
  waveform,
}: {
  peaks: number[];
  waveform?: WaveformBin[];
}) {
  return (
    <SoundCloudWaveform
      compact
      peaks={peaks}
      waveform={waveform ?? []}
    />
  );
}

function SoundCloudWaveform({
  waveform,
  peaks = [],
  normalizePeak,
  compact = false,
}: {
  waveform: WaveformBin[];
  peaks?: number[];
  normalizePeak?: number;
  compact?: boolean;
}) {
  const bins = waveform.length ? waveform : waveformFromPeaks(peaks);
  const visibleBins = bins.length ? bins : waveformFromPeaks(Array.from({ length: 96 }, () => 0));
  const barWidth = compact ? 2 : 3;
  const gap = compact ? 1 : 2;
  const height = compact ? 32 : 118;
  const center = height / 2;
  const usableHeight = center - (compact ? 3 : 7);
  const peak =
    normalizePeak ??
    Math.max(...visibleBins.map((bin) => bin.peak), 0.000001);
  const width = Math.max(1, visibleBins.length * (barWidth + gap));

  return (
    <svg
      className={compact ? "soundcloud-wave compact" : "soundcloud-wave"}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      <line
        className="wave-centerline"
        x1="0"
        x2={width}
        y1={center}
        y2={center}
      />
      {visibleBins.map((bin, index) => {
        const positive = clamp01(Math.max(0, bin.max) / peak);
        const negative = clamp01(Math.abs(Math.min(0, bin.min)) / peak);
        const fallback = clamp01(bin.peak / peak) * 0.5;
        const topAmount = Math.max(positive, fallback);
        const bottomAmount = Math.max(negative, fallback);
        const x = index * (barWidth + gap) + barWidth / 2;
        const y1 = center - Math.max(compact ? 1.5 : 2.5, topAmount * usableHeight);
        const y2 = center + Math.max(compact ? 1.5 : 2.5, bottomAmount * usableHeight);
        return (
          <line
            className="wave-bar"
            key={`${index}-${bin.min}-${bin.max}`}
            x1={x}
            x2={x}
            y1={y1}
            y2={y2}
            strokeWidth={barWidth}
          />
        );
      })}
    </svg>
  );
}

function waveformFromPeaks(peaks: number[]) {
  return peaks.map((peak) => ({
    min: -Math.max(0, peak),
    max: Math.max(0, peak),
    peak: Math.max(0, peak),
  }));
}

function clamp01(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function WaveformComparison({ artifacts }: { artifacts: RunPreviewArtifact[] }) {
  const ordered = ["target", "prediction", "residual"]
    .map((kind) => artifacts.find((artifact) => artifact.kind === kind))
    .filter((artifact): artifact is RunPreviewArtifact => Boolean(artifact));

  if (ordered.length === 0) return null;

  return (
    <div className="waveform-comparison" role="img" aria-label="Target, prediction, and residual waveform comparison">
      <div className="waveform-comparison-head">
        <span>Waveform comparison</span>
        <small>Actual min/max waveform, normalized per file</small>
      </div>
      {ordered.map((artifact) => (
        <div className={`waveform-track ${artifact.kind}`} key={artifact.kind}>
          <div>
            <strong>{artifact.label}</strong>
            <small>{previewArtifactDetail(artifact)}</small>
          </div>
          <PeakWave peaks={artifact.peaks} waveform={artifact.waveform} />
        </div>
      ))}
    </div>
  );
}

function GateList({
  project,
  selectedRun,
}: {
  project: ProjectDetail;
  selectedRun: TrainingRun | null;
}) {
  const checks = [
    { label: "Audio prepared", ok: project.audio?.status === "ready" },
    { label: "Training completed", ok: Boolean(selectedRun) },
    { label: "Metrics saved", ok: Boolean(selectedRun?.metrics) },
    { label: "Native validation", ok: Boolean(selectedRun?.metrics) },
    {
      label: "Benchmark safe",
      ok: (selectedRun?.metrics?.realtime_factor ?? 0) >= 20,
    },
  ];

  return (
    <div className="gate-list">
      {checks.map((check) => (
        <div className={check.ok ? "gate pass" : "gate"} key={check.label}>
          {check.ok ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          <span>{check.label}</span>
        </div>
      ))}
    </div>
  );
}

function ExportList({
  exports,
  onOpenExport,
}: {
  exports: ExportPackage[];
  onOpenExport: (exportId: string) => Promise<void>;
}) {
  if (exports.length === 0) {
    return (
      <SetupRequired
        icon={PackageCheck}
        title="No export package yet"
        detail="Once a completed run passes the gate, export creates model JSON, metadata, validation, and benchmark reports."
      />
    );
  }

  return (
    <div className="export-list">
      {exports.map((item) => (
        <div className="export-item" key={item.id}>
          <div className="export-main">
            <div>
              <strong>{item.id}</strong>
              <small>{exportPackageSummary(item)}</small>
            </div>
            <div className="export-actions">
              <span className="badge">{item.status}</span>
              <button
                className="secondary-button"
                type="button"
                onClick={() => void onOpenExport(item.id)}
              >
                <FolderOpen size={16} />
                Open
              </button>
            </div>
          </div>
          <div className="report-strip">
            <ReportPill
              label="Validation"
              status={getString(item.validation_report, "status")}
              detail={validationSummary(item.validation_report)}
            />
            <ReportPill
              label="Benchmark"
              status={getString(item.benchmark_report, "status")}
              detail={benchmarkSummary(item.benchmark_report)}
            />
          </div>
          <div className="artifact-grid">
            <span>{compactPath(item.model_path)}</span>
            <span>{compactPath(item.package_path)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ReportPill({
  label,
  status,
  detail,
}: {
  label: string;
  status: string | null;
  detail: string;
}) {
  const normalized = status ?? "pending";
  return (
    <div className={`report-pill ${normalized}`}>
      <span>{label}</span>
      <strong>{normalized}</strong>
      <small>{detail}</small>
    </div>
  );
}

function WaveformOverlay({
  latency,
  loading,
  error,
  waveform,
}: {
  latency: number;
  loading: boolean;
  error: string | null;
  waveform: ProjectWaveform | null;
}) {
  const normalizePeak = waveform
    ? Math.max(waveform.input.peak, waveform.target.peak, 0.000001)
    : 1;

  return (
    <div
      className="waveform"
      role="img"
      aria-label="Prepared dry input and processed target waveforms"
    >
      <div className="waveform-meta">
        <span>Prepared waveform</span>
        <small>
          {waveform
            ? `${formatSeconds(waveform.duration_seconds)} · ${waveform.sample_rate.toLocaleString()} Hz`
            : "Loading actual prepared audio"}
        </small>
      </div>
      <div className="latency-marker">
        <span>{latency} samples</span>
      </div>
      {loading ? (
        <div className="waveform-placeholder">Loading actual waveform.</div>
      ) : error ? (
        <div className="waveform-placeholder warning">Waveform unavailable: {error}</div>
      ) : waveform ? (
        <>
          <AlignmentWaveformTrack
            track={waveform.input}
            normalizePeak={normalizePeak}
          />
          <AlignmentWaveformTrack
            track={waveform.target}
            normalizePeak={normalizePeak}
          />
        </>
      ) : (
        <div className="waveform-placeholder">Run capture preflight to render a waveform.</div>
      )}
    </div>
  );
}

function AlignmentWaveformTrack({
  track,
  normalizePeak,
}: {
  track: ProjectWaveformTrack;
  normalizePeak: number;
}) {
  return (
    <div className={`alignment-wave-track ${track.kind}`}>
      <div className="alignment-wave-label">
        <strong>{track.label}</strong>
        <small>
          {formatDbfs(track.peak)} · {formatSeconds(track.duration_seconds)}
        </small>
      </div>
      <SoundCloudWaveform
        waveform={track.waveform}
        normalizePeak={normalizePeak}
      />
    </div>
  );
}

function WarningList({ warnings }: { warnings: AudioWarning[] }) {
  return (
    <div className="warning-list">
      {warnings.map((warning) => (
        <div className={warning.severity === "info" ? "notice notice-info" : "notice notice-warning"} key={`${warning.code}-${warning.message}`}>
          {warning.severity === "info" ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
          <span>
            <strong>{warning.message}</strong>
            {warning.detail ? <small>{warning.detail}</small> : null}
            {warning.action ? <small>{warning.action}</small> : null}
          </span>
        </div>
      ))}
    </div>
  );
}

function ScreenTitle({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
}) {
  return (
    <div className="screen-title">
      <Icon size={18} />
      <div>
        <h3>{title}</h3>
        <p>{detail}</p>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ProgressLog({
  events,
  active,
}: {
  events: SidecarProgressEvent[];
  active: boolean;
}) {
  if (!active && events.length === 0) return null;

  const latest = events[events.length - 1] ?? null;
  const recent = events.slice(-10).reverse();

  return (
    <section className={active ? "progress-panel active" : "progress-panel"}>
      <div className="progress-header">
        <div>
          <p className="section-label">Progress</p>
          <h3>{latest ? progressTitle(latest) : "Launching sidecar"}</h3>
        </div>
        <span className={active ? "live-pill active" : "live-pill"}>
          {active ? "Live" : "Done"}
        </span>
      </div>

      {recent.length === 0 ? (
        <p className="muted">Waiting for the first event.</p>
      ) : (
        <ol className="progress-list">
          {recent.map((event, index) => (
            <li
              className={event.stream === "stderr" ? "warning" : ""}
              key={`${event.timestamp}-${index}-${event.operation}-${event.stream}`}
            >
              <span className="progress-time">{formatEventTime(event.timestamp)}</span>
              <span>
                <strong>{progressTitle(event)}</strong>
                <small>{progressDetail(event)}</small>
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function SetupRequired({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
}) {
  return (
    <div className="setup-required">
      <Icon size={20} />
      <span>
        <strong>{title}</strong>
        <small>{detail}</small>
      </span>
    </div>
  );
}

function EmptyState({
  busy,
  onCreateSample,
}: {
  busy: boolean;
  onCreateSample: () => Promise<void>;
}) {
  return (
    <div className="empty-state">
      <Activity size={42} />
      <h2>Create a capture project</h2>
      <p>
        Start with paired dry and processed WAV files, or generate a local sample
        capture to exercise the full workflow.
      </p>
      <div className="empty-actions">
        <button
          className="primary-button"
          type="button"
          disabled={busy}
          onClick={() => void onCreateSample()}
        >
          {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
          Create sample project
        </button>
      </div>
      <ol className="onboarding-steps" aria-label="Workflow overview">
        <li>Capture: import or generate paired WAV files.</li>
        <li>Train: choose a curated RTNeural-safe preset.</li>
        <li>Evaluate: compare target, prediction, and residual audio.</li>
        <li>Export: write JSON after parity, validation, and benchmark checks.</li>
      </ol>
    </div>
  );
}

function progressTitle(event: SidecarProgressEvent) {
  const type = eventType(event);
  if (type === "run_started") {
    return `Training ${getString(event.json, "preset") ?? event.run_id ?? ""}`.trim();
  }
  if (type === "epoch") {
    const epoch = getNumber(event.json, "epoch");
    const total = getNumber(event.json, "total_epochs");
    return epoch && total ? `Epoch ${epoch}/${total}` : "Epoch";
  }
  if (type === "checkpoint") return "Checkpoint saved";
  if (type === "learning_rate_reduced") return "Learning rate reduced";
  if (type === "run_finished" || type === "train_command_finished") {
    return "Training completed";
  }
  if (type === "prepare_finished") return "Audio prepared";
  if (type === "export_finished") return "RTNeural export written";
  if (type === "error") return "Sidecar error";
  if (event.stream === "system") return event.line;
  if (event.stream === "stderr") return "Sidecar stderr";
  return operationLabel(event.operation);
}

function progressDetail(event: SidecarProgressEvent) {
  const type = eventType(event);
  if (type === "run_started") {
    const epochs = getNumber(event.json, "epochs");
    const device = getString(event.json, "device");
    return [epochs ? `${epochs} epochs` : null, device].filter(Boolean).join(" · ");
  }
  if (type === "epoch") {
    const trainLoss = getNumber(event.json, "train_loss");
    const valEsr = getNumber(event.json, "val_esr");
    const validationScore = getNumber(event.json, "validation_score");
    const learningRate = getNumber(event.json, "learning_rate");
    const nextLearningRate = getNumber(event.json, "next_learning_rate");
    const lrReduced = getBoolean(event.json, "learning_rate_reduced");
    const isBest = getBoolean(event.json, "is_best");
    return [
      trainLoss !== null ? `loss ${formatMetric(trainLoss)}` : null,
      valEsr !== null ? `val ESR ${formatMetric(valEsr)}` : null,
      validationScore !== null ? `score ${formatMetric(validationScore)}` : null,
      learningRate !== null ? `lr ${formatLearningRate(learningRate)}` : null,
      lrReduced && nextLearningRate !== null
        ? `next ${formatLearningRate(nextLearningRate)}`
        : null,
      isBest ? "best" : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }
  if (type === "learning_rate_reduced") {
    const from = getNumber(event.json, "from");
    const to = getNumber(event.json, "to");
    const epoch = getNumber(event.json, "epoch");
    return [
      epoch !== null ? `after epoch ${epoch}` : null,
      from !== null && to !== null
        ? `${formatLearningRate(from)} -> ${formatLearningRate(to)}`
        : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }
  if (type === "prepare_finished") {
    const status = getString(event.json, "status");
    const warnings = getArray(event.json, "warnings").length;
    return [status, warnings ? `${warnings} warnings` : "no warnings"].filter(Boolean).join(" · ");
  }
  if (type === "export_finished") {
    return getString(event.json, "model_path") ?? getString(event.json, "export_dir") ?? event.line;
  }
  if (type === "error") {
    return getString(event.json, "message") ?? event.line;
  }
  if (event.stream === "system") return operationLabel(event.operation);
  return event.line;
}

function eventType(event: SidecarProgressEvent) {
  return getString(event.json, "type");
}

function recommendPreset(
  project: ProjectDetail,
  backend: RuntimeBackend,
): PresetRecommendation {
  const duration = getNumber(project.audio?.capture_profile ?? null, "duration_seconds");
  const confidence = project.audio?.latency_confidence ?? 0;
  const warningCodes = new Set(project.audio?.warning_details.map((warning) => warning.code) ?? []);
  const reasons: string[] = [];

  if (backend === "pytorch") {
    reasons.push("PyTorch parity is currently limited to LSTM presets.");
    if (duration !== null && duration < 20) {
      return {
        presetId: "lstm_light",
        label: "Light LSTM",
        confidence: "medium",
        reasons: [...reasons, "Short captures benefit from the lower-risk light model."],
      };
    }
    return {
      presetId: "lstm_standard",
      label: "Standard LSTM",
      confidence: "medium",
      reasons: [...reasons, "It is the safest PyTorch-compatible default."],
    };
  }

  if (warningCodes.has("capture_level_low") || warningCodes.has("rms_mismatch")) {
    reasons.push("Gain warnings are present, so start with a stable recurrent baseline.");
    return {
      presetId: "gru_light",
      label: "GRU",
      confidence: "medium",
      reasons,
    };
  }

  if (project.target_kind === "line" || project.target_kind === "generic") {
    reasons.push("Line/generic captures often need a quick memoryless baseline first.");
    return {
      presetId: "dense_only",
      label: "Dense",
      confidence: "medium",
      reasons,
    };
  }

  if (duration !== null && duration >= 60 && confidence < 0.75) {
    reasons.push(
      "Long captures with review-level alignment should start with a finite-memory baseline.",
    );
    return {
      presetId: "conv1d_bn_prelu",
      label: "Conv + PReLU",
      confidence: confidence >= 0.55 ? "medium" : "low",
      reasons: [
        ...reasons,
        "This avoids recurrent hidden-state drift while checking whether the capture can be learned.",
      ],
    };
  }

  if (duration !== null && duration >= 90 && confidence >= 0.75) {
    reasons.push("The capture is long enough for the WaveNet-style finite-memory model.");
    return {
      presetId: "wavenet_tcn",
      label: "WaveNet TCN",
      confidence: "high",
      reasons: [
        ...reasons,
        "This keeps inference finite-memory while adding more dilated-convolution receptive field; check benchmark results before export.",
      ],
    };
  }

  if (duration !== null && duration < 15) {
    reasons.push("Short captures should start with a compact recurrent model.");
    return {
      presetId: "gru_light",
      label: "GRU",
      confidence: "medium",
      reasons,
    };
  }

  reasons.push("Amp and pedal captures usually need short-term memory.");
  return {
    presetId: "lstm_standard",
    label: "Standard LSTM",
    confidence: confidence >= 0.65 ? "high" : "low",
    reasons:
      confidence >= 0.65
        ? reasons
        : [...reasons, "Alignment confidence is low, so verify the nudge before a long run."],
  };
}

function recommendedWindowBudget(project: ProjectDetail) {
  const recommended = getNumber(project.audio?.capture_profile ?? null, "recommended_max_windows");
  if (recommended !== null) return Math.round(recommended);
  const duration = project.audio?.input.duration_seconds ?? 0;
  if (duration >= 120) return 4096;
  if (duration >= 45) return 2048;
  return 512;
}

function trainingRecipeFromCustom(recipe: TrainingRecipe): TrainingRecipeOption {
  return {
    id: recipe.id,
    source: "custom",
    name: recipe.name,
    description: `${recipe.epochs} epochs, ${recipe.max_windows} windows`,
    modelPreset: recipe.model_preset,
    epochs: recipe.epochs,
    batchSize: recipe.batch_size,
    learningRate: recipe.learning_rate,
    sequenceLength: recipe.sequence_length,
    maxWindows: recipe.max_windows,
    earlyStoppingPatience: recipe.early_stopping_patience,
    earlyStoppingMinDelta: recipe.early_stopping_min_delta,
  };
}

function recipeModelSupported(modelPreset: string, backend: RuntimeBackend) {
  return Boolean(presets.find((preset) => preset.id === modelPreset)?.backends.includes(backend));
}

function compatibleResumeRuns(
  runs: TrainingRun[],
  preset: string,
  backend: RuntimeBackend,
) {
  return [...runs]
    .reverse()
    .filter((run) => {
      if (run.status !== "completed" || !run.metrics) return false;
      if (run.preset !== preset) return false;
      return normalizeRunBackend(run.backend) === backend;
    });
}

function normalizeRunBackend(backend: string) {
  const normalized = backend.trim().toLowerCase();
  if (normalized === "tensorflow" || normalized === "tf") return "keras";
  if (normalized === "torch") return "pytorch";
  return normalized;
}

function resumeRunLabel(run: TrainingRun) {
  return [
    presetDisplayLabel(run.preset),
    run.metrics ? `ESR ${run.metrics.esr.toFixed(3)}` : null,
    `${run.epochs} epochs`,
    formatReportDate(run.created_at),
  ]
    .filter(Boolean)
    .join(" · ");
}

function presetDisplayLabel(presetId: string) {
  return presets.find((preset) => preset.id === presetId)?.label ?? presetId;
}

function trainingHistoryFromEvents(
  events: SidecarProgressEvent[],
  runId: string | null,
): TrainingHistoryPoint[] {
  return events
    .filter((event) => eventType(event) === "epoch")
    .filter((event) => !runId || event.run_id === runId || getString(event.json, "run_id") === runId)
    .map((event) => ({
      epoch: getNumber(event.json, "epoch") ?? 0,
      trainLoss: getNumber(event.json, "train_loss"),
      valEsr: getNumber(event.json, "val_esr"),
      valRmse: getNumber(event.json, "val_rmse"),
      validationScore: getNumber(event.json, "validation_score"),
      predictionRmsRatio: getNumber(event.json, "prediction_rms_ratio"),
      learningRate: getNumber(event.json, "learning_rate"),
      nextLearningRate: getNumber(event.json, "next_learning_rate"),
      learningRateReduced: getBoolean(event.json, "learning_rate_reduced"),
      isBest: getBoolean(event.json, "is_best"),
    }))
    .filter((point) => point.epoch > 0);
}

function historyFromReport(report: Record<string, unknown> | null): TrainingHistoryPoint[] {
  const history = getArray(report, "history");
  return history
    .map((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return null;
      const value = item as Record<string, unknown>;
      return {
        epoch: getNumber(value, "epoch") ?? 0,
        trainLoss: getNumber(value, "train_loss"),
        valEsr: getNumber(value, "val_esr"),
        valRmse: getNumber(value, "val_rmse"),
        validationScore: getNumber(value, "validation_score"),
        predictionRmsRatio: getNumber(value, "prediction_rms_ratio"),
        learningRate: getNumber(value, "learning_rate"),
        nextLearningRate: getNumber(value, "next_learning_rate"),
        learningRateReduced: getBoolean(value, "learning_rate_reduced"),
        isBest: getBoolean(value, "is_best"),
      };
    })
    .filter((point): point is TrainingHistoryPoint => Boolean(point && point.epoch > 0));
}

function validationTrend(history: TrainingHistoryPoint[]) {
  const values = history
    .filter((point) => point.valEsr !== null)
    .map((point) => ({ epoch: point.epoch, value: point.valEsr as number }));
  if (values.length < 4) return "waiting";

  const latest = values[values.length - 1];
  const best = values.reduce((current, point) =>
    point.value < current.value ? point : current,
  );
  const first = values[0];
  const improvement = first.value - latest.value;
  const tail = values.slice(-Math.min(8, values.length));
  const tailImprovement = tail[0].value - tail[tail.length - 1].value;

  if (latest.epoch === best.epoch && tailImprovement > 0.0005) {
    return "still improving";
  }
  if (improvement > 0.0005) return "improved";
  return "flat";
}

function qualityVerdict(
  metrics: TrainingMetrics | null,
  reportAssessment: Record<string, unknown> | null,
): QualityDecision {
  const reportVerdict = getString(reportAssessment, "verdict");
  if (reportVerdict) {
    return {
      verdict: normalizeQualityVerdict(reportVerdict),
      summary: getString(reportAssessment, "summary") ?? "Training report is available.",
      action: getString(reportAssessment, "action") ?? "Inspect the previews before export.",
    };
  }
  if (!metrics) {
    return {
      verdict: "unknown",
      summary: "No quality decision yet.",
      action: "Run training to generate validation metrics and preview audio.",
    };
  }
  if (metrics.esr <= 0.03 && metrics.rmse <= 0.03 && metrics.realtime_factor >= 40) {
    return {
      verdict: "good",
      summary: "Good candidate for export.",
      action: "Listen to the residual and export if the preview matches the target.",
    };
  }
  if (metrics.esr <= 0.1 && metrics.rmse <= 0.08 && metrics.realtime_factor >= 20) {
    return {
      verdict: "usable",
      summary: "Usable, but inspect before shipping.",
      action: "Compare target and prediction. Try a richer preset if the residual is audible.",
    };
  }
  return {
    verdict: "needs_work",
    summary: "Needs more work before export.",
    action: "Check alignment and gain, then train longer or choose a stronger preset.",
  };
}

function normalizeQualityVerdict(value: string): QualityDecision["verdict"] {
  if (value === "good" || value === "usable" || value === "needs_work") return value;
  return "unknown";
}

function getString(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return typeof item === "string" ? item : null;
}

function getNumber(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return typeof item === "number" ? item : null;
}

function getBoolean(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return typeof item === "boolean" ? item : false;
}

function getArray(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return Array.isArray(item) ? item : [];
}

function getNestedString(value: Record<string, unknown> | null, keys: string[]) {
  const item = getNestedValue(value, keys);
  return typeof item === "string" ? item : null;
}

function getNestedNumber(value: Record<string, unknown> | null, keys: string[]) {
  const item = getNestedValue(value, keys);
  return typeof item === "number" ? item : null;
}

function getNestedObject(value: Record<string, unknown> | null, keys: string[]) {
  const item = getNestedValue(value, keys);
  return item && typeof item === "object" && !Array.isArray(item)
    ? (item as Record<string, unknown>)
    : null;
}

function getNestedValue(value: Record<string, unknown> | null, keys: string[]) {
  let current: unknown = value;
  for (const key of keys) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return null;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

function clampNumber(value: number, minimum: number, maximum: number) {
  if (!Number.isFinite(value)) return minimum;
  return Math.round(Math.min(maximum, Math.max(minimum, value)));
}

function clampFloat(value: number, minimum: number, maximum: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(maximum, Math.max(minimum, value));
}

function operationLabel(operation: string) {
  return operation
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function validateCaptureForm(inputPath: string, targetPath: string) {
  const messages: string[] = [];
  const input = inputPath.trim();
  const target = targetPath.trim();
  if (!input) messages.push("Choose or enter a dry input WAV.");
  if (!target) messages.push("Choose or enter a processed target WAV.");
  if (input && !isWavPath(input)) messages.push("Dry input must be a .wav file.");
  if (target && !isWavPath(target)) messages.push("Processed target must be a .wav file.");
  if (input && target && input === target) {
    messages.push("Dry input and processed target must be different files.");
  }
  return messages;
}

function isWavPath(path: string) {
  return /\.(wav|wave)$/i.test(path.trim());
}

function legacyWarnings(warnings: string[]): AudioWarning[] {
  return warnings.map((message) => ({
    code: message,
    severity: "warning",
    message,
    detail: "",
    action: "",
  }));
}

function backendLabel(backend: RuntimeBackend) {
  return backend === "pytorch" ? "PyTorch" : "TensorFlow/Keras";
}

function runtimeDeviceOptions(backend: RuntimeBackend, inspection: DeviceInspection | null) {
  const mpsAvailable = Boolean(inspection?.mps_available && inspection?.mps_built);
  const cudaAvailable = Boolean(inspection?.cuda_available);
  const tensorflowGpuAvailable = Boolean(inspection?.tensorflow_gpus?.length);
  if (backend === "keras") {
    return [
      { value: "auto", label: "Auto", available: true },
      { value: "cpu", label: "CPU", available: true },
      {
        value: "mps",
        label:
          mpsAvailable && tensorflowGpuAvailable
            ? "MPS/Metal GPU"
            : mpsAvailable
              ? "MPS needs TF Metal"
              : "MPS unavailable",
        available: mpsAvailable && tensorflowGpuAvailable,
      },
      {
        value: "cuda",
        label:
          cudaAvailable && tensorflowGpuAvailable
            ? "CUDA GPU"
            : cudaAvailable
              ? "CUDA needs TF GPU"
              : "CUDA unavailable",
        available: cudaAvailable && tensorflowGpuAvailable,
      },
    ];
  }

  return [
    { value: "auto", label: "Auto", available: true },
    { value: "cpu", label: "CPU", available: true },
    { value: "mps", label: mpsAvailable ? "MPS" : "MPS unavailable", available: mpsAvailable },
    {
      value: "cuda",
      label: cudaAvailable ? "CUDA" : "CUDA unavailable",
      available: cudaAvailable,
    },
  ];
}

function runtimeDeviceLabel(
  device: string,
  backend: RuntimeBackend,
  inspection: DeviceInspection | null,
) {
  if (device === "auto") {
    const inspectedDevice =
      backend === "pytorch" ? inspection?.torch_selected_device : inspection?.selected_device;
    return inspectedDevice ? `Auto (${inspectedDevice})` : "Auto";
  }
  if (device === "mps") return backend === "keras" ? "MPS/Metal GPU" : "MPS";
  if (device === "cuda") return "CUDA";
  if (device === "cpu") return "CPU";
  return device;
}

function runtimeDeviceWarning(
  backend: RuntimeBackend,
  device: string,
  inspection: DeviceInspection | null,
) {
  const tensorflowGpuAvailable = Boolean(inspection?.tensorflow_gpus?.length);
  if (backend === "keras") {
    if (device !== "auto" && device !== "cpu" && !tensorflowGpuAvailable) {
      return "TensorFlow/Keras does not currently report a GPU for this runtime.";
    }
    if (
      device === "auto" &&
      Boolean(inspection?.mps_available && inspection?.mps_built) &&
      !tensorflowGpuAvailable
    ) {
      return "PyTorch reports MPS, but TensorFlow/Keras does not. Switch to PyTorch for MPS, or install/configure tensorflow-metal for Keras GPU training.";
    }
    return null;
  }
  if (device === "auto" || device === "cpu") return null;
  if (device === "mps" && !Boolean(inspection?.mps_available && inspection?.mps_built)) {
    return "PyTorch does not currently report MPS availability.";
  }
  if (device === "cuda" && !Boolean(inspection?.cuda_available)) {
    return "PyTorch does not currently report CUDA availability.";
  }
  return null;
}

function audioStatusLabel(status: AudioStatus) {
  if (status === "ready") return "Audio ready";
  if (status === "warning") return "Audio warnings";
  return "Audio missing";
}

function formatMetric(value: number) {
  return value < 0.01 ? value.toExponential(2) : value.toFixed(4);
}

function formatLearningRate(value: number) {
  return value < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function previewArtifactDetail(artifact: RunPreviewArtifact) {
  if (!artifact.exists) return "missing";
  return [
    artifact.duration_seconds !== null ? formatSeconds(artifact.duration_seconds) : null,
    artifact.sample_rate !== null ? `${artifact.sample_rate / 1000} kHz` : null,
    artifact.peak !== null ? `peak ${formatDbfs(artifact.peak)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function exportPackageSummary(item: ExportPackage) {
  const metadata = item.package_metadata;
  const backend =
    getNestedString(metadata, ["model", "backend"]) ??
    getString(metadata, "backend") ??
    "unknown";
  const sampleRate =
    getNestedNumber(metadata, ["model", "sample_rate"]) ?? getNumber(metadata, "sample_rate");
  const latency =
    getNestedNumber(metadata, ["model", "latency_samples"]) ??
    getNumber(metadata, "latency_samples");
  return [
    backend,
    sampleRate !== null ? `${sampleRate / 1000} kHz` : null,
    latency !== null ? `${latency} samples` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function validationSummary(report: Record<string, unknown> | null) {
  if (!report) return "waiting for report";
  const maxAbs = getNumber(report, "max_abs_error");
  const rmse = getNumber(report, "rmse");
  const tolerance = getNumber(report, "tolerance");
  return [
    maxAbs !== null ? `max ${formatMetric(maxAbs)}` : null,
    rmse !== null ? `rmse ${formatMetric(rmse)}` : null,
    tolerance !== null ? `tol ${formatMetric(tolerance)}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function benchmarkSummary(report: Record<string, unknown> | null) {
  if (!report) return "waiting for report";
  const realtimeFactor = getNumber(report, "realtime_factor");
  const elapsedMs = getNumber(report, "elapsed_ms");
  const frames = getNumber(report, "frames_processed");
  return [
    realtimeFactor !== null ? `${realtimeFactor.toFixed(0)}x realtime` : null,
    elapsedMs !== null ? `${elapsedMs.toFixed(1)} ms` : null,
    frames !== null ? `${frames.toLocaleString()} frames` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function compactPath(path: string) {
  return path.split(/[\\/]/).slice(-2).join("/");
}

function formatSeconds(value: number) {
  if (value < 1) return `${Math.round(value * 1000)} ms`;
  return `${value.toFixed(2)} s`;
}

function formatDbfs(value: number) {
  if (value <= 0) return "-inf dBFS";
  return `${(20 * Math.log10(value)).toFixed(1)} dBFS`;
}

function formatReportDate(value: string | null) {
  if (!value) return "unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortId(value: string) {
  return value.slice(-8);
}

function formatEventTime(timestamp: string) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function shouldRefreshProject(event: SidecarProgressEvent) {
  const type = eventType(event);
  if (
    [
      "run_finished",
      "train_command_finished",
      "export_finished",
      "prepare_finished",
      "error",
    ].includes(type ?? "")
  ) {
    return true;
  }
  return (
    event.stream === "system" &&
    (event.line.includes("completed") ||
      event.line.includes("failed") ||
      event.line.includes("finished") ||
      event.line.includes("interrupted"))
  );
}

function isTauriRuntime() {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function toFriendlyMessage(caught: unknown) {
  return caught instanceof Error ? caught.message : String(caught);
}

function friendlyError(message: string) {
  const normalized = message.toLowerCase();
  if (normalized.includes("must run inside the tauri app runtime")) {
    return {
      title: "Desktop runtime required.",
      detail: "This action uses local files and sidecars, so it only works in the Tauri app.",
      action: "Open the desktop shell with `pnpm --filter rtneural-trainer-app tauri dev`.",
    };
  }
  if (normalized.includes("external python environment not found")) {
    return {
      title: "External Python was not found.",
      detail: message,
      action: "Choose a Python executable or venv folder in Runtime, then save and refresh.",
    };
  }
  if (normalized.includes("tensorflow") && normalized.includes("required")) {
    return {
      title: "TensorFlow is missing for this path.",
      detail: message,
      action: "Install the TensorFlow extra with `uv sync --extra tensorflow` or use a configured external environment.",
    };
  }
  if (normalized.includes("does not exist") || normalized.includes("not a file")) {
    return {
      title: "A source file could not be opened.",
      detail: message,
      action: "Use Choose to pick the WAV again, then rerun preflight.",
    };
  }
  if (normalized.includes("sidecar") || normalized.includes("rttrainer")) {
    return {
      title: "A sidecar command failed.",
      detail: message,
      action: "Check the progress log and Runtime panel. Regenerate dev sidecars if the binary is missing.",
    };
  }
  return {
    title: "The operation could not finish.",
    detail: message,
    action: "Review the visible state, adjust the input, and try again.",
  };
}
