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
  ZoomIn,
  ZoomOut,
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
  LatencyCandidate,
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

const ALIGNMENT_WAVEFORM_WINDOWS = [2048, 4096, 8192, 16384, 32768] as const;
const DEFAULT_ALIGNMENT_WAVEFORM_WINDOW_INDEX = 1;

type CaptureAnalyzePayload = {
  inputPath: string;
  targetPath: string;
  targetSampleRate: number;
  resample: boolean;
  channelPolicy: CaptureChannelPolicy;
  knownLatencyEnabled: boolean;
  knownLatencySamples: number;
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
  resampleTrainingWindows: boolean;
  resampleIntervalEpochs: number;
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
  resampleTrainingWindows: boolean;
  resampleIntervalEpochs: number;
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
  verdict: "excellent" | "good" | "usable" | "needs_work" | "unknown";
  summary: string;
  action: string;
};

type PresetOption = {
  id: string;
  label: string;
  detail: string;
  cpu: string;
  backends: RuntimeBackend[];
  hidden?: boolean;
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

const presets: PresetOption[] = [
  {
    id: "wavenet_tcn_fast",
    label: "WaveNet Fast",
    detail: "6x dilated causal Conv1D",
    cpu: "Heavy",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_clean",
    label: "WaveNet Clean Linear",
    detail: "Linear long-field dilated Conv1D",
    cpu: "Moderate",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_edge",
    label: "WaveNet Edge",
    detail: "Clean field, gentle tanh breakup",
    cpu: "Moderate",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_separable_fast",
    label: "WaveNet Separable",
    detail: "Depthwise dilated Conv1D + 1x1 mix",
    cpu: "Experimental",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn_balanced",
    label: "WaveNet Balanced",
    detail: "8x dilated causal Conv1D",
    cpu: "Heavy",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_balanced_tanh15",
    label: "WaveNet Tanh 1.5",
    detail: "Balanced WaveNet, smoothed tanh",
    cpu: "Research",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn_balanced_tanh18",
    label: "WaveNet Tanh 1.8",
    detail: "Balanced WaveNet, smoother tanh",
    cpu: "Research",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn",
    label: "WaveNet Balanced",
    detail: "8x dilated causal Conv1D",
    cpu: "Heavy",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn_quality",
    label: "WaveNet Quality",
    detail: "10x dilated causal Conv1D",
    cpu: "Max CPU",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_quality_tanh15",
    label: "WaveNet Quality Tanh 1.5",
    detail: "Quality WaveNet, smoothed tanh",
    cpu: "Research",
    backends: ["keras"],
  },
  {
    id: "wavenet_tcn_high_gain",
    label: "WaveNet High Gain",
    detail: "11x dilated causal Conv1D",
    cpu: "Research",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn_quality_tanh18",
    label: "WaveNet Quality Tanh 1.8",
    detail: "Quality WaveNet, smoother tanh",
    cpu: "Research",
    backends: ["keras"],
    hidden: true,
  },
  {
    id: "wavenet_tcn_a2_prelu",
    label: "WaveNet A2 PReLU",
    detail: "A2 dilations, mixed kernels, PReLU",
    cpu: "Research",
    backends: ["keras"],
  },
];

const visiblePresets = presets.filter((preset) => !preset.hidden && isWaveNetPreset(preset.id));

const builtInTrainingRecipes = [
  {
    id: "builtin_smoke",
    source: "built_in",
    name: "WaveNet quick check",
    description: "Short WaveNet pass to verify alignment, gain, and sidecar flow.",
    modelPreset: "wavenet_tcn_fast",
    epochs: 12,
    batchSize: 16,
    learningRate: 0.0008,
    sequenceLength: 8192,
    maxWindows: 512,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 4,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_balanced",
    source: "built_in",
    name: "WaveNet balanced",
    description: "Default first quality run for amp captures; benchmark before export.",
    modelPreset: "wavenet_tcn_balanced",
    epochs: 120,
    batchSize: 16,
    learningRate: 0.0007,
    sequenceLength: 8192,
    maxWindows: 4096,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 12,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_wavenet_clean",
    source: "built_in",
    name: "WaveNet clean linear",
    description: "Long-field, mostly linear WaveNet path for clean amp captures.",
    modelPreset: "wavenet_tcn_clean",
    epochs: 160,
    batchSize: 16,
    learningRate: 0.0002,
    sequenceLength: 8192,
    maxWindows: 8192,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 20,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_wavenet_edge",
    source: "built_in",
    name: "WaveNet edge breakup",
    description: "Clean-inspired long field with gentle nonlinear breakup.",
    modelPreset: "wavenet_tcn_edge",
    epochs: 180,
    batchSize: 16,
    learningRate: 0.00015,
    sequenceLength: 8192,
    maxWindows: 8192,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 24,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_wavenet_quality",
    source: "built_in",
    name: "WaveNet quality",
    description: "Slower high-gain refinement when balanced is still audible.",
    modelPreset: "wavenet_tcn_quality",
    epochs: 180,
    batchSize: 16,
    learningRate: 0.0005,
    sequenceLength: 8192,
    maxWindows: 8192,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 20,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_wavenet_quality_tanh15",
    source: "built_in",
    name: "WaveNet quality tanh 1.5",
    description: "Quality WaveNet with gentler tanh for high-band residual research.",
    modelPreset: "wavenet_tcn_quality_tanh15",
    epochs: 180,
    batchSize: 16,
    learningRate: 0.0005,
    sequenceLength: 8192,
    maxWindows: 8192,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 20,
    earlyStoppingMinDelta: 0.00005,
  },
  {
    id: "builtin_wavenet_a2_prelu",
    source: "built_in",
    name: "WaveNet A2 PReLU",
    description: "A2-inspired non-power dilations, mixed kernels, and PReLU.",
    modelPreset: "wavenet_tcn_a2_prelu",
    epochs: 180,
    batchSize: 16,
    learningRate: 0.00035,
    sequenceLength: 8192,
    maxWindows: 8192,
    resampleTrainingWindows: true,
    resampleIntervalEpochs: 1,
    earlyStoppingPatience: 20,
    earlyStoppingMinDelta: 0.00005,
  },
] satisfies TrainingRecipeOption[];

const defaultTrainingRecipe =
  builtInTrainingRecipes.find((recipe) => recipe.id === "builtin_balanced") ??
  builtInTrainingRecipes[0];

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
  const projectLoadRequestRef = useRef(0);
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
    projectLoadRequestRef.current += 1;
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
    const requestId = projectLoadRequestRef.current + 1;
    projectLoadRequestRef.current = requestId;
    try {
      setError(null);
      const [nextProject, nextEvents] = await Promise.all([
        api.getProject(projectId),
        api.listProjectEvents(projectId),
      ]);
      if (requestId !== projectLoadRequestRef.current) return;
      setProject(nextProject);
      setProgressEvents(nextEvents);
    } catch (caught) {
      if (requestId !== projectLoadRequestRef.current) return;
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
      resample_training_windows: options.resampleTrainingWindows,
      resample_interval_epochs: options.resampleIntervalEpochs,
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
    projectLoadRequestRef.current += 1;
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
                  key={project.id}
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
                        known_latency_samples: capture.knownLatencyEnabled
                          ? capture.knownLatencySamples
                          : null,
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
                        resample_training_windows: options.resampleTrainingWindows,
                        resample_interval_epochs: options.resampleIntervalEpochs,
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
  const backend: RuntimeBackend = "keras";
  const [selectedDevice, setSelectedDevice] = useState(settings?.selected_device ?? "auto");
  const [externalPythonPath, setExternalPythonPath] = useState(
    settings?.external_python_path ?? "",
  );

  useEffect(() => {
    setSelectedDevice(settings?.selected_device ?? "auto");
    setExternalPythonPath(settings?.external_python_path ?? "");
  }, [settings?.external_python_path, settings?.selected_device]);

  const packageVersions = inspection?.package_versions ?? {};
  const deviceOptions = useMemo(
    () => runtimeDeviceOptions(inspection),
    [inspection],
  );
  const selectedDeviceWarning = runtimeDeviceWarning(selectedDevice, inspection);
  const runtimeSource = settings?.external_python_path
    ? "External"
    : status?.trainer_sidecar_present
      ? "Sidecar"
      : "uv dev";
  const hasChanges =
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
          <dd>TensorFlow/Keras</dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>{runtimeDeviceLabel(selectedDevice, inspection)}</dd>
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
      </div>

      <form
        className="runtime-form"
        onSubmit={(event) => {
          event.preventDefault();
          onSave({
            selected_backend: "keras",
            selected_device: selectedDevice,
            external_python_path: externalPythonPath.trim() || null,
          });
        }}
      >
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
  const initialCapture = captureDefaultsForProject(project);
  const [inputPath, setInputPath] = useState(initialCapture.inputPath);
  const [targetPath, setTargetPath] = useState(initialCapture.targetPath);
  const [resample, setResample] = useState(initialCapture.resample);
  const [targetSampleRate, setTargetSampleRate] = useState(
    initialCapture.targetSampleRate,
  );
  const [channelPolicy, setChannelPolicy] = useState<CaptureChannelPolicy>(
    initialCapture.channelPolicy,
  );
  const [knownLatencyEnabled, setKnownLatencyEnabled] = useState(
    initialCapture.knownLatencyEnabled,
  );
  const [knownLatencySamples, setKnownLatencySamples] = useState(
    initialCapture.knownLatencySamples,
  );
  const captureValidation = validateCaptureForm(
    inputPath,
    targetPath,
    knownLatencyEnabled,
    knownLatencySamples,
  );
  const canPickFiles = isTauriRuntime();

  useEffect(() => {
    const nextCapture = captureDefaultsForProject(project);
    setInputPath(nextCapture.inputPath);
    setTargetPath(nextCapture.targetPath);
    setResample(nextCapture.resample);
    setTargetSampleRate(nextCapture.targetSampleRate);
    setChannelPolicy(nextCapture.channelPolicy);
    setKnownLatencyEnabled(nextCapture.knownLatencyEnabled);
    setKnownLatencySamples(nextCapture.knownLatencySamples);
  }, [project.id]);

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
              knownLatencyEnabled,
              knownLatencySamples: Math.round(knownLatencySamples),
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
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={knownLatencyEnabled}
                onChange={(event) => setKnownLatencyEnabled(event.target.checked)}
              />
              <span>
                Use known latency
                <small>Skips auto-detect for matched DAW renders.</small>
              </span>
            </label>
            <label>
              Latency samples
              <input
                type="number"
                min={-48000}
                max={48000}
                step={1}
                value={knownLatencySamples}
                disabled={!knownLatencyEnabled}
                onChange={(event) => setKnownLatencySamples(Number(event.target.value))}
              />
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
  const [waveformWindowIndex, setWaveformWindowIndex] = useState(
    DEFAULT_ALIGNMENT_WAVEFORM_WINDOW_INDEX,
  );
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
  const latencyCandidates = latencyCandidateDetails(project.audio);
  const latencyAgreement = finiteNumber(project.audio?.latency?.agreement);
  const latencyMargin = finiteNumber(project.audio?.latency?.score_margin);
  const latencyPolarity = project.audio?.latency?.polarity === "inverted" ? "inverted" : "normal";
  const polarityConfidence = finiteNumber(project.audio?.latency?.polarity_confidence);
  const waveformWindowSamples =
    ALIGNMENT_WAVEFORM_WINDOWS[waveformWindowIndex] ??
    ALIGNMENT_WAVEFORM_WINDOWS[DEFAULT_ALIGNMENT_WAVEFORM_WINDOW_INDEX];
  const canZoomIn = waveformWindowIndex > 0;
  const canZoomOut = waveformWindowIndex < ALIGNMENT_WAVEFORM_WINDOWS.length - 1;

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
      .getProjectWaveform({
        project_id: project.id,
        bins: 420,
        window_samples: waveformWindowSamples,
      })
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
  }, [project.id, project.updated_at, waveformWindowSamples]);

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
          alignmentShiftSamples={nudge - savedNudge}
          zoomLabel={`${waveformWindowSamples.toLocaleString()} samples`}
          canZoomIn={canZoomIn}
          canZoomOut={canZoomOut}
          onZoomIn={() => {
            setWaveformWindowIndex((current) => Math.max(0, current - 1));
          }}
          onZoomOut={() => {
            setWaveformWindowIndex((current) =>
              Math.min(ALIGNMENT_WAVEFORM_WINDOWS.length - 1, current + 1),
            );
          }}
        />
      </div>
      <div className="panel span-4">
        <div className="metric-stack">
          <Metric label="Auto estimate" value={`${autoLatency} samples`} />
          <Metric label="Manual adjustment" value={`${nudge} samples`} />
          <Metric label="Training latency" value={`${effectiveLatency} samples`} />
          <Metric label="Confidence" value={`${Math.round(confidence * 100)}%`} />
          {latencyAgreement !== null ? (
            <Metric label="Window agreement" value={`${Math.round(latencyAgreement * 100)}%`} />
          ) : null}
          {latencyMargin !== null ? (
            <Metric label="Score margin" value={latencyMargin.toFixed(3)} />
          ) : null}
          {latencyPolarity === "inverted" ? (
            <Metric
              label="Polarity"
              value={`inverted${
                polarityConfidence !== null ? ` · ${Math.round(polarityConfidence * 100)}%` : ""
              }`}
            />
          ) : null}
          <Metric
            label="Milliseconds"
            value={`${((effectiveLatency / sampleRate) * 1000).toFixed(2)} ms`}
          />
        </div>
        {latencyPolarity === "inverted" ? (
          <div className="notice notice-warning">
            <AlertTriangle size={18} />
            <span>
              Target polarity appears inverted; training will preserve the captured amp output.
            </span>
          </div>
        ) : null}
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
        {latencyCandidates.length ? (
          <div className="candidate-offsets">
            <span>Try detected candidates</span>
            <div>
              {latencyCandidates.map((candidate) => {
                const candidateSamples = candidate.samples;
                const candidateNudge = candidateSamples - autoLatency;
                const clampedNudge = clampNumber(candidateNudge, -256, 256);
                const isActive = effectiveLatency === candidateSamples;
                const isInverted = candidate.polarity === "inverted";
                return (
                  <button
                    className={[
                      "chip",
                      isActive ? "active" : "",
                      isInverted ? "inverted" : "",
                    ]
                      .filter(Boolean)
                      .join(" ")}
                    key={candidateSamples}
                    type="button"
                    title={latencyCandidateTitle(candidate)}
                    onClick={() => setNudge(clampedNudge)}
                  >
                    {latencyCandidateLabel(candidate)}
                  </button>
                );
              })}
            </div>
            <small>
              Low-confidence estimates are worth auditioning before a long WaveNet run.
            </small>
          </div>
        ) : null}
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
  const [selectedRecipeId, setSelectedRecipeId] = useState(defaultTrainingRecipe.id);
  const [preset, setPreset] = useState(
    recipeOptions.find((recipe) => recipe.id === defaultTrainingRecipe.id)?.modelPreset ??
      recommendation.presetId,
  );
  const [epochs, setEpochs] = useState(defaultTrainingRecipe.epochs);
  const [batchSize, setBatchSize] = useState(defaultTrainingRecipe.batchSize);
  const [learningRate, setLearningRate] = useState(defaultTrainingRecipe.learningRate);
  const [sequenceLength, setSequenceLength] = useState(defaultTrainingRecipe.sequenceLength);
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(
    defaultTrainingRecipe.earlyStoppingPatience,
  );
  const [earlyStoppingMinDelta, setEarlyStoppingMinDelta] = useState(
    defaultTrainingRecipe.earlyStoppingMinDelta,
  );
  const [maxWindows, setMaxWindows] = useState(defaultTrainingRecipe.maxWindows);
  const [resampleTrainingWindows, setResampleTrainingWindows] = useState<boolean>(
    defaultTrainingRecipe.resampleTrainingWindows,
  );
  const [resampleIntervalEpochs, setResampleIntervalEpochs] = useState(
    defaultTrainingRecipe.resampleIntervalEpochs,
  );
  const [resumeFromRunId, setResumeFromRunId] = useState<string | null>(null);
  const [recipeName, setRecipeName] = useState("");
  const [recipeNotice, setRecipeNotice] = useState<string | null>(null);
  const [preview, setPreview] = useState<RunPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const selectedRecipe = recipeOptions.find((recipe) => recipe.id === selectedRecipeId) ?? null;
  const selectedPreset = presets.find((item) => item.id === preset) ?? visiblePresets[0];
  const selectedPresetSupported =
    isWaveNetPreset(selectedPreset.id) && selectedPreset.backends.includes(backend);
  const resumeCandidates = useMemo(
    () => compatibleResumeRuns(project.runs, preset, backend),
    [backend, preset, project.runs],
  );
  const qualityResumeRun = useMemo(
    () => bestWaveNetResumeRun(project.runs, backend),
    [backend, project.runs],
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
    resampleTrainingWindows,
    resampleIntervalEpochs,
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
    if (selectedRecipeId !== defaultTrainingRecipe.id) return;
    setMaxWindows(Math.max(recommendedWindowBudget(project), defaultTrainingRecipe.maxWindows));
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
    setResampleTrainingWindows(recipe.resampleTrainingWindows);
    setResampleIntervalEpochs(recipe.resampleIntervalEpochs);
    setEarlyStoppingPatience(recipe.earlyStoppingPatience);
    setEarlyStoppingMinDelta(recipe.earlyStoppingMinDelta);
    setRecipeName(recipe.source === "custom" ? recipe.name : "");
    setRecipeNotice(null);
  }

  function markManualRecipeEdit() {
    setSelectedRecipeId("manual");
    setRecipeNotice(null);
  }

  function applyQualityContinuation(run: TrainingRun) {
    const nextPreset = qualityContinuationPreset(run.preset);
    setSelectedRecipeId("manual");
    setPreset(nextPreset);
    setResumeFromRunId(run.id);
    setEpochs(120);
    setBatchSize(16);
    setLearningRate(qualityContinuationLearningRate(nextPreset));
    setSequenceLength(8192);
    setMaxWindows(
      Math.max(
        recommendedWindowBudget(project),
        isLongWaveNetPreset(nextPreset)
          ? 8192
          : 4096,
      ),
    );
    setResampleTrainingWindows(true);
    setResampleIntervalEpochs(1);
    setEarlyStoppingPatience(20);
    setEarlyStoppingMinDelta(0.00005);
    setRecipeName("");
    setRecipeNotice(
      `Continuing ${presetDisplayLabel(run.preset)} from its best checkpoint with a lower learning rate.`,
    );
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
      applyRecipe(defaultTrainingRecipe);
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
          {visiblePresets.map((item) => {
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
          {qualityResumeRun ? (
            <button
              className="secondary-button"
              type="button"
              onClick={() => applyQualityContinuation(qualityResumeRun)}
            >
              <RotateCcw size={16} />
              Continue best WaveNet
            </button>
          ) : null}
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
          <label className="toggle-row training-toggle">
            <input
              type="checkbox"
              checked={resampleTrainingWindows}
              onChange={(event) => {
                markManualRecipeEdit();
                setResampleTrainingWindows(event.target.checked);
              }}
            />
            <span>
              Rotate training windows
              <small>Keep validation/test fixed while sampling new training chunks.</small>
            </span>
          </label>
          <label>
            Rotation interval
            <input
              type="number"
              min={1}
              max={50}
              value={resampleIntervalEpochs}
              disabled={!resampleTrainingWindows}
              onChange={(event) => {
                markManualRecipeEdit();
                setResampleIntervalEpochs(clampNumber(event.target.valueAsNumber, 1, 50));
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
              resampleTrainingWindows,
              resampleIntervalEpochs,
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
        <RunComparison runs={project.runs} exports={project.exports} />
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
  const latencyConfidence = project.audio?.latency_confidence ?? null;
  const candidates = latencyCandidateSamples(project.audio);

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
      <div>
        <span>Latency review</span>
        <strong>
          {latencyConfidence !== null ? `${Math.round(latencyConfidence * 100)}%` : "unknown"}
        </strong>
        <small>
          {candidates.length
            ? `Try ${candidates.join(", ")} samples before long runs.`
            : "Use the Align view before long WaveNet training."}
        </small>
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
  const completedRuns = useMemo(
    () =>
      [...project.runs]
        .filter((run) => run.status === "completed")
        .sort((left, right) => timestampMs(right.created_at) - timestampMs(left.created_at)),
    [project.runs],
  );
  const recommendedRun = useMemo(
    () => bestExportCandidateRun(completedRuns, project.exports),
    [completedRuns, project.exports],
  );
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    if (completedRuns.length === 0) {
      setSelectedRunId(null);
      return;
    }
    if (!selectedRunId || !completedRuns.some((run) => run.id === selectedRunId)) {
      setSelectedRunId(recommendedRun?.id ?? completedRuns[0].id);
    }
  }, [completedRuns, recommendedRun?.id, selectedRunId]);

  const selectedRun =
    completedRuns.find((run) => run.id === selectedRunId) ?? recommendedRun;

  return (
    <div className="screen-grid">
      <div className="panel span-5">
        <ScreenTitle
          icon={PackageCheck}
          title="Export Gate"
          detail="RTNeural JSON is only ready after validation and benchmark reports."
        />
        {completedRuns.length > 0 ? (
          <label className="compact-field export-run-field">
            Training run
            <select
              aria-label="Training run"
              value={selectedRun?.id ?? ""}
              disabled={busy}
              onChange={(event) => setSelectedRunId(event.target.value)}
            >
              {completedRuns.map((run) => (
                <option key={run.id} value={run.id}>
                  {exportRunOptionLabel(run, recommendedRun?.id ?? null)}
                </option>
              ))}
            </select>
            <small>Select the completed training run that will be packaged.</small>
          </label>
        ) : null}
        {selectedRun ? (
          <SelectedExportRun
            run={selectedRun}
            recommended={selectedRun.id === recommendedRun?.id}
          />
        ) : null}
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
          Export selected RTNeural JSON
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

function SelectedExportRun({
  run,
  recommended,
}: {
  run: TrainingRun;
  recommended: boolean;
}) {
  return (
    <div className="selected-run-card">
      <div>
        <span>Selected training</span>
        <strong>{presetDisplayLabel(run.preset)}</strong>
        <small>
          {shortId(run.id)} · {run.epochs} epochs · {formatReportDate(run.created_at)}
        </small>
      </div>
      <div className="selected-run-stats">
        {recommended ? <span className="badge">Recommended</span> : null}
        <Metric label="ESR" value={run.metrics ? run.metrics.esr.toFixed(3) : "none"} />
        <Metric
          label="RTF"
          value={run.metrics ? `${run.metrics.realtime_factor.toFixed(1)}x` : "none"}
        />
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

function RunComparison({
  runs,
  exports,
}: {
  runs: TrainingRun[];
  exports: ExportPackage[];
}) {
  if (runs.length === 0) {
    return (
      <SetupRequired
        icon={Cpu}
        title="No training runs yet"
        detail="Choose a preset and start training. Checkpoints, ESR, runtime cost, and recovery state will appear here."
      />
    );
  }

  const completedRuns = runs.filter((run) => run.status === "completed" && run.metrics);
  const bestQuality = bestCompletedRun(completedRuns);
  const bestExport = bestExportCandidateRun(completedRuns, exports);
  const bestRuntime = bestRuntimeRun(completedRuns, exports);
  const sortedRuns = [...runs].sort(
    (left, right) => timestampMs(right.created_at) - timestampMs(left.created_at),
  );

  return (
    <div className="run-comparison">
      <div className="run-comparison-summary">
        <RunDecisionCard
          label="Export pick"
          run={bestExport}
          detail="Best score across ESR, ASR, validation, and runtime reports."
        />
        <RunDecisionCard
          label="Lowest ESR"
          run={bestQuality}
          detail="Best training metric before export-side checks."
        />
        <RunDecisionCard
          label="Fastest native"
          run={bestRuntime}
          detail="Highest native RTNeural realtime factor when known."
        />
      </div>
      <div className="table run-table">
        <div className="table-head">
          <span>Decision</span>
          <span>Preset</span>
          <span>Status</span>
          <span>ESR</span>
          <span>ASR</span>
          <span>Native RTF</span>
          <span>Updated</span>
        </div>
        {sortedRuns.map((run) => {
          const item = exportForRun(exports, run.id);
          const worstAsr = exportWorstAsr(item);
          const nativeFactor = exportNativeRealtimeFactor(item) ?? run.metrics?.realtime_factor ?? null;
          const badges = runDecisionBadges(run, item, {
            bestExportRunId: bestExport?.id ?? null,
            bestQualityRunId: bestQuality?.id ?? null,
            bestRuntimeRunId: bestRuntime?.id ?? null,
          });
          return (
            <div className="table-row" key={run.id}>
              <span className="run-decision-cell">
                {badges.map((badge) => (
                  <em className={`decision-badge ${badge.tone}`} key={badge.label}>
                    {badge.label}
                  </em>
                ))}
              </span>
              <span>
                <strong>{presetDisplayLabel(run.preset)}</strong>
                <small>{shortId(run.id)}</small>
              </span>
              <span>{run.status}</span>
              <span>{run.metrics ? run.metrics.esr.toFixed(3) : "none"}</span>
              <span>{worstAsr !== null ? formatPercent(worstAsr) : "not exported"}</span>
              <span>{nativeFactor !== null ? `${nativeFactor.toFixed(2)}x` : "unknown"}</span>
              <span>{formatReportDate(run.updated_at || run.created_at)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RunDecisionCard({
  label,
  run,
  detail,
}: {
  label: string;
  run: TrainingRun | null;
  detail: string;
}) {
  return (
    <div className="run-decision-card">
      <span>{label}</span>
      <strong>{run ? presetDisplayLabel(run.preset) : "pending"}</strong>
      <small>
        {run?.metrics
          ? `${run.metrics.esr.toFixed(3)} ESR · ${shortId(run.id)}`
          : detail}
      </small>
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
  shiftSamples = 0,
  sampleRate,
  durationSeconds,
}: {
  waveform: WaveformBin[];
  peaks?: number[];
  normalizePeak?: number;
  compact?: boolean;
  shiftSamples?: number;
  sampleRate?: number;
  durationSeconds?: number;
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
  const sampleCount =
    sampleRate && durationSeconds ? Math.max(1, sampleRate * durationSeconds) : 0;
  const shiftX = sampleCount ? (-shiftSamples / sampleCount) * width : 0;
  const barsTransform = Math.abs(shiftX) > 0.0001 ? `translate(${shiftX} 0)` : undefined;

  return (
    <svg
      className={[
        "soundcloud-wave",
        compact ? "compact" : null,
        barsTransform ? "shifted" : null,
      ]
        .filter(Boolean)
        .join(" ")}
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
      <g className="wave-bars" transform={barsTransform}>
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
      </g>
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
      label: "Runtime estimate",
      ok: (selectedRun?.metrics?.realtime_factor ?? 0) >= 1,
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
            <ReportPill
              label="Backends"
              status={getNestedString(item.package_metadata, ["benchmark_matrix", "status"])}
              detail={benchmarkMatrixSummary(item.package_metadata)}
            />
            <ReportPill
              label="Aliasing"
              status={getNestedString(item.package_metadata, ["aliasing", "status"])}
              detail={aliasingSummary(item.package_metadata)}
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
  alignmentShiftSamples,
  zoomLabel,
  canZoomIn,
  canZoomOut,
  onZoomIn,
  onZoomOut,
}: {
  latency: number;
  loading: boolean;
  error: string | null;
  waveform: ProjectWaveform | null;
  alignmentShiftSamples: number;
  zoomLabel: string;
  canZoomIn: boolean;
  canZoomOut: boolean;
  onZoomIn: () => void;
  onZoomOut: () => void;
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
        <div>
          <span>Prepared waveform</span>
          <small>
            {waveform
              ? `${formatSeconds(waveform.duration_seconds)} · ${waveform.sample_rate.toLocaleString()} Hz`
              : "Loading actual prepared audio"}
          </small>
        </div>
        <div className="waveform-zoom" aria-label="Waveform zoom">
          <button
            className="icon-button"
            type="button"
            disabled={!canZoomOut}
            onClick={onZoomOut}
            title="Zoom waveform out"
            aria-label="Zoom waveform out"
          >
            <ZoomOut size={14} />
          </button>
          <span>{zoomLabel}</span>
          <button
            className="icon-button"
            type="button"
            disabled={!canZoomIn}
            onClick={onZoomIn}
            title="Zoom waveform in"
            aria-label="Zoom waveform in"
          >
            <ZoomIn size={14} />
          </button>
        </div>
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
            shiftSamples={alignmentShiftSamples}
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
  shiftSamples = 0,
}: {
  track: ProjectWaveformTrack;
  normalizePeak: number;
  shiftSamples?: number;
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
        shiftSamples={shiftSamples}
        sampleRate={track.sample_rate}
        durationSeconds={track.duration_seconds}
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
  _backend: RuntimeBackend,
): PresetRecommendation {
  const duration = getNumber(project.audio?.capture_profile ?? null, "duration_seconds");
  const confidence = project.audio?.latency_confidence ?? 0;
  const gain = project.audio?.gain ?? null;
  const gainVerdict = getString(gain, "verdict");
  const targetRms = getNumber(gain, "target_rms_dbfs");
  const headroom = getNumber(gain, "headroom_db");
  const rmsDelta = getNumber(gain, "rms_delta_db");
  const warningCodes = new Set(project.audio?.warning_details.map((warning) => warning.code) ?? []);
  const reasons: string[] = [];
  const ampLike = project.target_kind === "amp" || project.target_kind === "pedal";
  const denseOrDriven =
    (targetRms !== null && targetRms > -19) ||
    (rmsDelta !== null && rmsDelta >= 5) ||
    (headroom !== null && headroom < 6);
  const cleanOrLowGain =
    ampLike &&
    targetRms !== null &&
    targetRms < -24 &&
    (rmsDelta === null || rmsDelta < 2.5) &&
    (headroom === null || headroom >= 5);

  if (warningCodes.has("capture_level_low")) {
    reasons.push("The capture looks too quiet for a high-confidence quality run.");
    return {
      presetId: "wavenet_tcn_fast",
      label: "WaveNet Fast",
      confidence: "low",
      reasons: [
        ...reasons,
        "Use this only as a quick learnability check, then fix capture gain before a final run.",
      ],
    };
  }

  if (project.target_kind === "line" || project.target_kind === "generic") {
    reasons.push("Line and generic captures still use the same WaveNet export path.");
    return {
      presetId: "wavenet_tcn_fast",
      label: "WaveNet Fast",
      confidence: "medium",
      reasons: [...reasons, "Start compact and compare the preview before a deeper run."],
    };
  }

  if (warningCodes.has("rms_mismatch")) {
    reasons.push("Input and target levels differ enough to review gain before a long run.");
    return {
      presetId: "wavenet_tcn_fast",
      label: "WaveNet Fast",
      confidence: "medium",
      reasons: [
        ...reasons,
        "If the residual sounds mostly level-related, fix the capture gain before a quality run.",
      ],
    };
  }

  if (ampLike && duration !== null && duration >= 120) {
    if (confidence < 0.65) {
      reasons.push("The capture is long enough for WaveNet, but alignment needs review.");
      return {
        presetId: denseOrDriven ? "wavenet_tcn_a2_prelu" : "wavenet_tcn_balanced",
        label: denseOrDriven ? "WaveNet A2 PReLU" : "WaveNet Balanced",
        confidence: "low",
        reasons: [
          ...reasons,
          "Try the top latency candidates in Align before committing to a long run.",
        ],
      };
    }

    if (denseOrDriven) {
      reasons.push("Dense or driven amp captures have consistently favored deeper WaveNet models.");
      if (gainVerdict === "healthy") {
        reasons.push("Gain staging looks healthy enough for a slower quality run.");
      }
      return {
        presetId: "wavenet_tcn_a2_prelu",
        label: "WaveNet A2 PReLU",
        confidence: confidence >= 0.75 ? "high" : "medium",
        reasons: [
          ...reasons,
          "Use WaveNet Quality as the conservative comparison if the A2 run plateaus.",
          "Export and check ASR/native runtime before choosing the final package.",
        ],
      };
    }

    if (cleanOrLowGain) {
      reasons.push("This looks like a quieter, lower-gain amp capture with stable alignment.");
      return {
        presetId: "wavenet_tcn_clean",
        label: "WaveNet Clean Linear",
        confidence: confidence >= 0.75 ? "high" : "medium",
        reasons: [
          ...reasons,
          "Use this long-field linear path before a nonlinear quality run.",
          "If it plateaus early, compare WaveNet Edge before the heavier quality recipes.",
        ],
      };
    }

    reasons.push("Long amp captures are now routed through WaveNet as the quality lane.");
    return {
      presetId: "wavenet_tcn_balanced",
      label: "WaveNet Balanced",
      confidence: "high",
      reasons: [
        ...reasons,
        "Balanced is the first pass for clean and lower-gain captures; quality is the refinement step.",
        "Use WaveNet Fast as the lower-runtime A/B check.",
      ],
    };
  }

  if (ampLike && duration !== null && duration >= 60) {
    if (cleanOrLowGain && confidence >= 0.65) {
      reasons.push("The capture is quiet enough to start with the clean linear WaveNet path.");
      return {
        presetId: "wavenet_tcn_clean",
        label: "WaveNet Clean Linear",
        confidence: confidence >= 0.75 ? "high" : "medium",
        reasons: [...reasons, "Compare WaveNet Edge if the preview misses light breakup."],
      };
    }

    reasons.push(
      confidence < 0.75
        ? "Alignment needs review, but WaveNet is still the better quality starting point."
        : "The capture is long enough for the balanced WaveNet finite-memory model.",
    );
    return {
      presetId: "wavenet_tcn_balanced",
      label: "WaveNet Balanced",
      confidence: confidence >= 0.75 ? "high" : "medium",
      reasons: [
        ...reasons,
        confidence < 0.75
          ? "Use Align to try candidate offsets before extending the run."
          : "Check native benchmark results before export; try WaveNet Fast if headroom is tight.",
      ],
    };
  }

  if (duration !== null && duration >= 45 && confidence >= 0.65) {
    reasons.push("The capture has enough material for a fast WaveNet sanity run.");
    if (headroom !== null && headroom < 3) {
      reasons.push("Peak headroom is tight, so inspect residual peaks before a WaveNet quality run.");
    }
    return {
      presetId: "wavenet_tcn_fast",
      label: "WaveNet Fast",
      confidence: "medium",
      reasons: [
        ...reasons,
        "Use balanced, quality, or A2 next if the preview suggests the capture is learnable.",
      ],
    };
  }

  if (duration !== null && duration < 15) {
    reasons.push("The capture is short for amp modeling, so keep this as a quick check.");
    return {
      presetId: "wavenet_tcn_fast",
      label: "WaveNet Fast",
      confidence: "medium",
      reasons: [...reasons, "A longer capture is still recommended for a final model."],
    };
  }

  reasons.push("Amp and pedal captures are now routed through WaveNet by default.");
  return {
    presetId: "wavenet_tcn_balanced",
    label: "WaveNet Balanced",
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
    resampleTrainingWindows: recipe.resample_training_windows,
    resampleIntervalEpochs: recipe.resample_interval_epochs,
    earlyStoppingPatience: recipe.early_stopping_patience,
    earlyStoppingMinDelta: recipe.early_stopping_min_delta,
  };
}

function recipeModelSupported(modelPreset: string, backend: RuntimeBackend) {
  const preset = presets.find((item) => item.id === modelPreset);
  return Boolean(preset && isWaveNetPreset(modelPreset) && preset.backends.includes(backend));
}

function resumePresetsCompatible(sourcePreset: string, targetPreset: string) {
  if (sourcePreset === targetPreset) return true;
  return (
    (sourcePreset === "wavenet_tcn" && targetPreset === "wavenet_tcn_balanced") ||
    (sourcePreset === "wavenet_tcn_balanced" && targetPreset === "wavenet_tcn")
  );
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
      if (!resumePresetsCompatible(run.preset, preset)) return false;
      return normalizeRunBackend(run.backend) === backend;
    });
}

function isWaveNetPreset(preset: string) {
  return preset === "wavenet_tcn" || preset.startsWith("wavenet_tcn_");
}

function isQualityWaveNetPreset(preset: string) {
  return preset === "wavenet_tcn_quality" || preset.startsWith("wavenet_tcn_quality_");
}

function isLongWaveNetPreset(preset: string) {
  return (
    isQualityWaveNetPreset(preset) ||
    preset === "wavenet_tcn_high_gain" ||
    preset === "wavenet_tcn_a2_prelu" ||
    preset === "wavenet_tcn_clean" ||
    preset === "wavenet_tcn_edge"
  );
}

function bestWaveNetResumeRun(runs: TrainingRun[], backend: RuntimeBackend) {
  const candidates = runs.filter((run) => {
    if (run.status !== "completed" || !run.metrics) return false;
    if (!isWaveNetPreset(run.preset)) return false;
    return normalizeRunBackend(run.backend) === backend;
  });
  return candidates.reduce<TrainingRun | null>((best, run) => {
    if (!best) return run;
    const bestEsr = best.metrics?.esr ?? Number.POSITIVE_INFINITY;
    const runEsr = run.metrics?.esr ?? Number.POSITIVE_INFINITY;
    return runEsr < bestEsr ? run : best;
  }, null);
}

function qualityContinuationPreset(preset: string) {
  return preset === "wavenet_tcn" ? "wavenet_tcn_balanced" : preset;
}

function qualityContinuationLearningRate(preset: string) {
  if (isQualityWaveNetPreset(preset) || preset === "wavenet_tcn_high_gain") {
    return 0.00015;
  }
  if (preset === "wavenet_tcn_a2_prelu") return 0.00012;
  if (preset === "wavenet_tcn_clean") return 0.0001;
  if (preset === "wavenet_tcn_edge") return 0.00008;
  if (preset === "wavenet_tcn_fast") return 0.0003;
  return 0.0002;
}

function normalizeRunBackend(backend: string) {
  const normalized = backend.trim().toLowerCase();
  if (normalized === "tensorflow" || normalized === "tf") return "keras";
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

function bestCompletedRun(runs: TrainingRun[]) {
  return runs.reduce<TrainingRun | null>((best, run) => {
    if (!best) return run;
    const bestEsr = best.metrics?.esr ?? Number.POSITIVE_INFINITY;
    const runEsr = run.metrics?.esr ?? Number.POSITIVE_INFINITY;
    if (runEsr !== bestEsr) return runEsr < bestEsr ? run : best;
    return timestampMs(run.created_at) > timestampMs(best.created_at) ? run : best;
  }, null);
}

function bestExportCandidateRun(runs: TrainingRun[], exports: ExportPackage[]) {
  return runs.reduce<TrainingRun | null>((best, run) => {
    if (!run.metrics) return best;
    if (!best) return run;
    const runScore = runExportScore(run, exportForRun(exports, run.id));
    const bestScore = runExportScore(best, exportForRun(exports, best.id));
    if (runScore !== bestScore) return runScore < bestScore ? run : best;
    return timestampMs(run.created_at) > timestampMs(best.created_at) ? run : best;
  }, null);
}

function bestRuntimeRun(runs: TrainingRun[], exports: ExportPackage[]) {
  return runs.reduce<TrainingRun | null>((best, run) => {
    if (!run.metrics) return best;
    if (!best) return run;
    const runFactor =
      exportNativeRealtimeFactor(exportForRun(exports, run.id)) ??
      run.metrics.realtime_factor ??
      0;
    const bestFactor =
      exportNativeRealtimeFactor(exportForRun(exports, best.id)) ??
      best.metrics?.realtime_factor ??
      0;
    if (runFactor !== bestFactor) return runFactor > bestFactor ? run : best;
    return timestampMs(run.created_at) > timestampMs(best.created_at) ? run : best;
  }, null);
}

function runExportScore(run: TrainingRun, item: ExportPackage | null) {
  const esr = run.metrics?.esr ?? Number.POSITIVE_INFINITY;
  const worstAsr = exportWorstAsr(item);
  const validationPenalty =
    item?.status === "failed" ? 1 : item?.status === "ready" ? 0 : 0.025;
  const aliasingPenalty = worstAsr !== null ? worstAsr * 0.75 : 0.015;
  const runtime = exportNativeRealtimeFactor(item) ?? run.metrics?.realtime_factor ?? null;
  const runtimePenalty = runtime !== null && runtime < 1 ? 0.15 : 0;
  return esr + aliasingPenalty + validationPenalty + runtimePenalty;
}

function exportForRun(exports: ExportPackage[], runId: string) {
  return (
    exports
      .filter((item) => item.run_id === runId)
      .sort((left, right) => timestampMs(right.created_at) - timestampMs(left.created_at))[0] ??
    null
  );
}

function exportWorstAsr(item: ExportPackage | null) {
  return getNestedNumber(item?.package_metadata ?? null, ["aliasing", "worst_asr"]);
}

function exportNativeRealtimeFactor(item: ExportPackage | null) {
  return (
    getNestedNumber(item?.package_metadata ?? null, [
      "benchmark_matrix",
      "fastest_passing_backend",
      "realtime_factor",
    ]) ?? getNumber(item?.benchmark_report ?? null, "realtime_factor")
  );
}

function runDecisionBadges(
  run: TrainingRun,
  item: ExportPackage | null,
  leaders: {
    bestExportRunId: string | null;
    bestQualityRunId: string | null;
    bestRuntimeRunId: string | null;
  },
) {
  const badges: Array<{ label: string; tone: "good" | "warning" | "danger" | "neutral" }> = [];
  if (run.id === leaders.bestExportRunId) badges.push({ label: "Export pick", tone: "good" });
  if (run.id === leaders.bestQualityRunId) badges.push({ label: "Lowest ESR", tone: "neutral" });
  if (run.id === leaders.bestRuntimeRunId) badges.push({ label: "Fastest", tone: "neutral" });
  if (run.status === "running" || run.status === "preparing" || run.status === "queued") {
    badges.push({ label: "Running", tone: "warning" });
  }
  if (run.status === "failed" || run.status === "interrupted") {
    badges.push({ label: "Recover", tone: "danger" });
  }
  if (item?.status === "ready") badges.push({ label: "Exported", tone: "good" });
  if (item?.status === "failed") badges.push({ label: "Validation failed", tone: "danger" });
  const worstAsr = exportWorstAsr(item);
  if (worstAsr !== null && worstAsr >= 0.08) {
    badges.push({ label: "ASR warning", tone: worstAsr >= 0.16 ? "danger" : "warning" });
  }
  if (badges.length === 0 && run.status === "completed") {
    badges.push({ label: item ? "Review" : "Export to validate", tone: "neutral" });
  }
  if (badges.length === 0) badges.push({ label: "Waiting", tone: "neutral" });
  return badges.slice(0, 3);
}

function exportRunOptionLabel(run: TrainingRun, recommendedRunId: string | null) {
  return [
    run.id === recommendedRunId ? "Recommended" : null,
    presetDisplayLabel(run.preset),
    run.metrics ? `ESR ${run.metrics.esr.toFixed(3)}` : "No metrics",
    `${run.epochs} epochs`,
    formatReportDate(run.created_at),
    shortId(run.id),
  ]
    .filter(Boolean)
    .join(" · ");
}

function timestampMs(value: string) {
  const timestamp = new Date(value).getTime();
  return Number.isNaN(timestamp) ? 0 : timestamp;
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
  const correlation = residualCorrelation(metrics);
  const correlationOk = correlation === null || correlation >= 0.93;
  const correlationStrong = correlation === null || correlation >= 0.99;
  const isolatedPeak =
    metrics.peak_residual > 0.55 &&
    metrics.esr <= 0.03 &&
    metrics.rmse <= 0.03 &&
    (correlation === null || correlation >= 0.98);
  if (
    metrics.esr <= 0.015 &&
    metrics.rmse <= 0.02 &&
    correlationStrong &&
    metrics.realtime_factor >= 1
  ) {
    return {
      verdict: "excellent",
      summary: isolatedPeak
        ? "Excellent candidate with isolated peaks."
        : "Excellent candidate for export.",
      action: isolatedPeak
        ? "Residual energy and correlation are strong. Listen for the peak events, then export if the native benchmark passes."
        : "This is a preferred model. Export if the native benchmark has enough realtime margin.",
    };
  }
  if (
    metrics.esr <= 0.12 &&
    metrics.rmse <= 0.06 &&
    correlationOk &&
    metrics.realtime_factor >= 1
  ) {
    if (metrics.peak_residual > 0.55 && !isolatedPeak) {
      return {
        verdict: "usable",
        summary: "Usable with residual peaks to inspect.",
        action:
          "The overall match is strong, but residual peaks are still noticeable. Listen before treating the export as final.",
      };
    }
    return {
      verdict: "good",
      summary: "Good candidate for export.",
      action:
        "Listen for the remaining residual and export if the native benchmark has enough realtime margin.",
    };
  }
  if (
    metrics.esr <= 0.18 &&
    metrics.rmse <= 0.08 &&
    metrics.peak_residual <= 0.8 &&
    (correlation === null || correlation >= 0.88) &&
    metrics.realtime_factor >= 1
  ) {
    return {
      verdict: "usable",
      summary: "Usable, but inspect before shipping.",
      action:
        "Compare target, prediction, and residual. Try a quality preset if the remaining high-band detail is audible.",
    };
  }
  return {
    verdict: "needs_work",
    summary: "Needs more work before export.",
    action: "Check alignment and gain, then train longer or choose a stronger preset.",
  };
}

function normalizeQualityVerdict(value: string): QualityDecision["verdict"] {
  if (
    value === "excellent" ||
    value === "good" ||
    value === "usable" ||
    value === "needs_work"
  ) {
    return value;
  }
  return "unknown";
}

function residualCorrelation(metrics: TrainingMetrics) {
  const fromState = metrics.state_continuous_correlation;
  if (typeof fromState === "number" && Number.isFinite(fromState)) return fromState;
  const fallback = metrics.correlation;
  if (typeof fallback === "number" && Number.isFinite(fallback)) return fallback;
  return null;
}

function latencyCandidateSamples(audio: AudioReport | null | undefined) {
  return latencyCandidateDetails(audio).map((candidate) => candidate.samples);
}

function latencyCandidateDetails(audio: AudioReport | null | undefined): LatencyCandidate[] {
  const candidates = new Map<number, LatencyCandidate>();
  const autoLatency = audio?.latency_auto_samples ?? audio?.latency_samples ?? 0;

  for (const candidate of audio?.latency?.candidates ?? []) {
    const samples = finiteNumber(candidate.samples);
    if (samples === null) continue;
    candidates.set(samples, { ...candidate, samples });
  }

  for (const warning of audio?.warning_details ?? []) {
    if (!warning.code.includes("latency")) continue;
    const source = `${warning.message} ${warning.detail} ${warning.action}`;
    const candidateText = source.match(/top candidates include ([^.]+)/i)?.[1] ?? source;
    for (const match of candidateText.matchAll(/-?\d+(?=\s*samples?)/gi)) {
      const value = Number(match[0]);
      if (Number.isFinite(value) && !candidates.has(value)) {
        candidates.set(value, { samples: value });
      }
    }
  }
  if (!candidates.size) return [];

  if (!candidates.has(autoLatency)) {
    candidates.set(autoLatency, {
      samples: autoLatency,
      score: audio?.latency_confidence ?? null,
      agreement: audio?.latency?.agreement ?? null,
    });
  }

  return [...candidates.values()]
    .filter((candidate) => Math.abs(candidate.samples - autoLatency) <= 256)
    .sort((left, right) => {
      const leftDistance = Math.abs(left.samples - autoLatency);
      const rightDistance = Math.abs(right.samples - autoLatency);
      if (leftDistance !== rightDistance) return leftDistance - rightDistance;
      return (right.agreement ?? 0) - (left.agreement ?? 0);
    })
    .slice(0, 4);
}

function latencyCandidateLabel(candidate: LatencyCandidate) {
  const agreement = finiteNumber(candidate.agreement);
  const parts = [`${candidate.samples} samples`];
  if (agreement !== null) parts.push(`${Math.round(agreement * 100)}%`);
  if (candidate.polarity === "inverted") parts.push("inv");
  return parts.join(" · ");
}

function latencyCandidateTitle(candidate: LatencyCandidate) {
  const score = finiteNumber(candidate.score);
  const onsetScore = finiteNumber(candidate.onset_score);
  const signedScore = finiteNumber(candidate.signed_score);
  const preemphasisScore = finiteNumber(candidate.preemphasis_score);
  const invertedScore = finiteNumber(candidate.inverted_score);
  const voteCount = finiteNumber(candidate.vote_count);
  const windowCount = finiteNumber(candidate.window_count);
  const details = [
    `Set training latency to ${candidate.samples} samples`,
    candidate.polarity === "inverted" ? "target polarity appears inverted" : null,
    voteCount !== null && windowCount !== null
      ? `${voteCount}/${windowCount} windows voted for this offset`
      : null,
    score !== null ? `score ${score.toFixed(3)}` : null,
    invertedScore !== null ? `inverted ${invertedScore.toFixed(3)}` : null,
    signedScore !== null ? `signed ${signedScore.toFixed(3)}` : null,
    preemphasisScore !== null ? `pre ${preemphasisScore.toFixed(3)}` : null,
    onsetScore !== null ? `onset ${onsetScore.toFixed(3)}` : null,
  ];
  return details.filter(Boolean).join(" · ");
}

function getString(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return typeof item === "string" ? item : null;
}

function getNumber(value: Record<string, unknown> | null, key: string) {
  const item = value?.[key];
  return typeof item === "number" ? item : null;
}

function finiteNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
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

function getNestedArray(value: Record<string, unknown> | null, keys: string[]) {
  const item = getNestedValue(value, keys);
  return Array.isArray(item) ? item : [];
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

function captureDefaultsForProject(project: ProjectDetail): CaptureAnalyzePayload {
  const options = project.audio?.options ?? null;
  const savedChannelPolicy = getString(options, "channel_policy");
  const knownLatencySamples = getNumber(options, "known_latency_samples");
  return {
    inputPath: project.audio?.input.path ?? "",
    targetPath: project.audio?.target.path ?? "",
    targetSampleRate: getNumber(options, "target_sample_rate") ?? 48_000,
    resample: getBoolean(options, "resample"),
    channelPolicy: isCaptureChannelPolicy(savedChannelPolicy)
      ? savedChannelPolicy
      : "mixdown",
    knownLatencyEnabled: knownLatencySamples !== null,
    knownLatencySamples:
      knownLatencySamples ?? project.audio?.latency_auto_samples ?? project.audio?.latency_samples ?? 0,
  };
}

function isCaptureChannelPolicy(value: string | null): value is CaptureChannelPolicy {
  return value === "mixdown" || value === "first" || value === "reject";
}

function validateCaptureForm(
  inputPath: string,
  targetPath: string,
  knownLatencyEnabled = false,
  knownLatencySamples = 0,
) {
  const messages: string[] = [];
  const input = inputPath.trim();
  const target = targetPath.trim();
  if (!input) messages.push("Choose or enter a dry input WAV.");
  if (!target) messages.push("Choose or enter a processed target WAV.");
  if (input && !isWavPath(input)) messages.push("Dry input must be a .wav file.");
  if (target && !isWavPath(target)) messages.push("Processed target must be a .wav file.");
  if (
    knownLatencyEnabled &&
    (!Number.isFinite(knownLatencySamples) ||
      Math.round(knownLatencySamples) !== knownLatencySamples ||
      Math.abs(knownLatencySamples) > 48_000)
  ) {
    messages.push("Known latency must be a whole number between -48000 and 48000 samples.");
  }
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

function runtimeDeviceOptions(inspection: DeviceInspection | null) {
  const mpsAvailable = Boolean(inspection?.mps_available && inspection?.mps_built);
  const cudaAvailable = Boolean(inspection?.cuda_available);
  const tensorflowGpuAvailable = Boolean(inspection?.tensorflow_gpus?.length);
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

function runtimeDeviceLabel(
  device: string,
  inspection: DeviceInspection | null,
) {
  if (device === "auto") {
    const inspectedDevice = inspection?.selected_device;
    return inspectedDevice ? `Auto (${inspectedDevice})` : "Auto";
  }
  if (device === "mps") return "MPS/Metal GPU";
  if (device === "cuda") return "CUDA";
  if (device === "cpu") return "CPU";
  return device;
}

function runtimeDeviceWarning(
  device: string,
  inspection: DeviceInspection | null,
) {
  const tensorflowGpuAvailable = Boolean(inspection?.tensorflow_gpus?.length);
  if (device !== "auto" && device !== "cpu" && !tensorflowGpuAvailable) {
    return "TensorFlow/Keras does not currently report a GPU for this runtime.";
  }
  if (
    device === "auto" &&
    Boolean(inspection?.mps_available && inspection?.mps_built) &&
    !tensorflowGpuAvailable
  ) {
    return "TensorFlow/Keras does not currently report a GPU. MPS/CUDA selection may require tensorflow-metal or a TensorFlow GPU build.";
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

function formatPercent(value: number) {
  const percentage = value * 100;
  if (percentage >= 100) return `${percentage.toFixed(0)}%`;
  if (percentage >= 10) return `${percentage.toFixed(1)}%`;
  return `${percentage.toFixed(2)}%`;
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
  const summary = getNestedObject(report, ["summary"]);
  const worstCase = getNestedObject(summary, ["worst_case"]);
  const blockSize = getNumber(worstCase, "block_size");
  const channels = getNumber(worstCase, "channels");
  const modelInfo = getNestedObject(report, ["model_info"]);
  const receptiveField = getNumber(modelInfo, "receptive_field_samples");
  const modelBytes = getNumber(modelInfo, "size_bytes");
  return [
    realtimeFactor !== null ? `worst ${realtimeFactor.toFixed(2)}x realtime` : null,
    blockSize !== null && channels !== null
      ? `${blockSize} samples, ${channels} ch`
      : null,
    receptiveField !== null ? `${Math.round(receptiveField)} sample receptive field` : null,
    modelBytes !== null ? `${formatBytes(modelBytes)} model` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function benchmarkMatrixSummary(metadata: Record<string, unknown> | null) {
  const matrix = getNestedObject(metadata, ["benchmark_matrix"]);
  if (!matrix) return "waiting for matrix";

  const fastest = getNestedObject(matrix, ["fastest_passing_backend"]);
  const fastestId = getString(fastest, "id");
  const fastestFactor = getNumber(fastest, "realtime_factor");
  const headroom = getString(fastest, "headroom");
  const backends = getNestedArray(matrix, ["backends"]);
  const availableCount = backends.filter((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return false;
    const status = getString(entry as Record<string, unknown>, "status");
    return status !== "unavailable";
  }).length;

  return [
    fastestId
      ? `fastest ${fastestId}${fastestFactor !== null ? ` ${fastestFactor.toFixed(2)}x` : ""}`
      : "no passing backend",
    headroom ? headroom.split("_").join(" ") : null,
    `${availableCount}/${backends.length} available`,
  ]
    .filter(Boolean)
    .join(" · ");
}

function aliasingSummary(metadata: Record<string, unknown> | null) {
  const aliasing = getNestedObject(metadata, ["aliasing"]);
  if (!aliasing) return "waiting for ASR";

  const verdict = getString(aliasing, "verdict");
  const worstAsr = getNumber(aliasing, "worst_asr");
  const averageAsr = getNumber(aliasing, "average_asr");
  const tests = getArray(aliasing, "tests");

  return [
    verdict ? verdict.split("_").join(" ") : null,
    aliasingInterpretation(verdict),
    worstAsr !== null ? `worst ASR ${formatPercent(worstAsr)}` : null,
    averageAsr !== null ? `avg ${formatPercent(averageAsr)}` : null,
    tests.length ? `${tests.length} probes` : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

function aliasingInterpretation(verdict: string | null) {
  if (verdict === "low_aliasing") return "confirm by ear";
  if (verdict === "review_aliasing") return "listen for foldback";
  if (verdict === "high_aliasing") return "warning, compare presets";
  return null;
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

function formatBytes(value: number) {
  if (value < 1024) return `${value.toFixed(0)} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
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
