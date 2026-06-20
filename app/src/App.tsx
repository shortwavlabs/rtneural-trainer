import {
  Activity,
  AlertTriangle,
  AudioLines,
  CheckCircle2,
  Cpu,
  Crosshair,
  Download,
  FileAudio,
  FolderPlus,
  Gauge,
  LoaderCircle,
  PackageCheck,
  Play,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Square,
  type LucideIcon,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./lib/api";
import type {
  AppStatus,
  ExportPackage,
  ProjectDetail,
  ProjectSummary,
  RunPreview,
  RunPreviewArtifact,
  SidecarProgressEvent,
  TargetKind,
  TrainingRun,
} from "./types";

type TabId = "capture" | "align" | "train" | "evaluate" | "export";

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
    id: "lstm_light",
    label: "Light",
    detail: "1x LSTM, hidden 12",
    cpu: "Low CPU",
  },
  {
    id: "lstm_standard",
    label: "Standard",
    detail: "1x LSTM, hidden 16",
    cpu: "Default",
  },
];

export default function App() {
  const [status, setStatus] = useState<AppStatus | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("capture");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [progressEvents, setProgressEvents] = useState<SidecarProgressEvent[]>([]);
  const sidecarBusy = busy === "audio" || busy === "train" || busy === "export";

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
      if (mounted) setError(toMessage(caught));
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
      const [nextStatus, nextProjects] = await Promise.all([
        api.appStatus(),
        api.listProjects(),
      ]);
      setStatus(nextStatus);
      setProjects(nextProjects);
      setSelectedId((current) => current ?? nextProjects[0]?.id ?? null);
    } catch (caught) {
      setError(toMessage(caught));
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
      setError(toMessage(caught));
    }
  }

  async function refreshProjects() {
    const nextProjects = await api.listProjects();
    setProjects(nextProjects);
  }

  async function commitProject(nextProject: ProjectDetail, nextTab?: TabId) {
    setProject(nextProject);
    setSelectedId(nextProject.id);
    await refreshProjects();
    if (nextTab) setActiveTab(nextTab);
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

  return (
    <div className="app-shell">
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
              setError(toMessage(caught));
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

        <RuntimeStatus status={status} />
      </aside>

      <main className="workspace">
        {error ? (
          <div className="notice notice-error">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        ) : null}

        {project ? (
          <>
            <ProjectHeader project={project} />
            <StepTabs activeTab={activeTab} onChange={setActiveTab} />
            <section className="work-surface">
              {activeTab === "capture" ? (
                <CaptureView
                  project={project}
                  busy={busy === "audio"}
                  onAnalyze={async (input_path, target_path) => {
                    setProgressEvents([]);
                    setBusy("audio");
                    try {
                      const nextProject = await api.updateAudio({
                        project_id: project.id,
                        input_path,
                        target_path,
                      });
                      await commitProject(nextProject, "align");
                    } catch (caught) {
                      setError(toMessage(caught));
                    } finally {
                      setBusy(null);
                    }
                  }}
                />
              ) : null}

              {activeTab === "align" ? <AlignView project={project} /> : null}

              {activeTab === "train" ? (
                <TrainView
                  project={project}
                  busy={busy === "train"}
                  onTrain={async (preset) => {
                    setProgressEvents([]);
                    setBusy("train");
                    try {
                      const nextProject = await api.startTraining({
                        project_id: project.id,
                        preset,
                      });
                      await commitProject(nextProject, "train");
                    } catch (caught) {
                      setError(toMessage(caught));
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
                      setError(toMessage(caught));
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
                      setError(toMessage(caught));
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
                      setError(toMessage(caught));
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
                  setError(toMessage(caught));
                } finally {
                  setBusy(null);
                }
              }}
            />
          </>
        ) : (
          <EmptyState />
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
      <div className="segmented compact">
        {(Object.keys(targetLabels) as TargetKind[]).map((target) => (
          <button
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
            className={`project-row ${project.id === selectedId ? "active" : ""}`}
            key={project.id}
            type="button"
            onClick={() => onSelect(project.id)}
          >
            <span className={`status-dot ${project.audio_status}`} />
            <span>
              <strong>{project.name}</strong>
              <small>
                {targetLabels[project.target_kind]} ·{" "}
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

function RuntimeStatus({ status }: { status: AppStatus | null }) {
  return (
    <div className="runtime">
      <div className="section-label">Runtime</div>
      <dl>
        <div>
          <dt>App</dt>
          <dd>{status?.version ?? "Loading"}</dd>
        </div>
        <div>
          <dt>Trainer</dt>
          <dd>{status?.trainer_sidecar_present ? "Bundled" : "Built-in"}</dd>
        </div>
        <div>
          <dt>Validator</dt>
          <dd>{status?.validator_sidecar_present ? "Bundled" : "Built-in"}</dd>
        </div>
      </dl>
    </div>
  );
}

function ProjectHeader({ project }: { project: ProjectDetail }) {
  const latestRun = project.runs[project.runs.length - 1];
  const latestExport = project.exports[project.exports.length - 1];

  return (
    <header className="project-header">
      <div>
        <p className="eyebrow">{targetLabels[project.target_kind]} capture</p>
        <h2>{project.name}</h2>
        <p className="path-line">{project.project_dir}</p>
      </div>
      <div className="summary-strip">
        <Metric label="Audio" value={project.audio?.status ?? "missing"} />
        <Metric label="Runs" value={String(project.runs.length)} />
        <Metric
          label="Best ESR"
          value={latestRun?.metrics ? latestRun.metrics.esr.toFixed(3) : "none"}
        />
        <Metric label="Export" value={latestExport?.status ?? "blocked"} />
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
    <nav className="step-tabs" aria-label="Workflow">
      {tabs.map((tab) => {
        const Icon = tab.icon;
        return (
          <button
            className={activeTab === tab.id ? "active" : ""}
            key={tab.id}
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
  onAnalyze: (inputPath: string, targetPath: string) => Promise<void>;
}) {
  const [inputPath, setInputPath] = useState(project.audio?.input.path ?? "");
  const [targetPath, setTargetPath] = useState(project.audio?.target.path ?? "");

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
            void onAnalyze(inputPath, targetPath);
          }}
        >
          <label>
            Dry input WAV
            <input
              value={inputPath}
              onChange={(event) => setInputPath(event.target.value)}
              placeholder="/path/to/input.wav"
            />
          </label>
          <label>
            Processed target WAV
            <input
              value={targetPath}
              onChange={(event) => setTargetPath(event.target.value)}
              placeholder="/path/to/target.wav"
            />
          </label>
          <button className="primary-button" type="submit" disabled={busy}>
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

function AlignView({ project }: { project: ProjectDetail }) {
  const [nudge, setNudge] = useState(0);
  const latency = project.audio?.latency_samples ?? 0;
  const confidence = project.audio?.latency_confidence ?? 0;

  return (
    <div className="screen-grid">
      <div className="panel span-8">
        <ScreenTitle
          icon={Crosshair}
          title="Latency Alignment"
          detail="Inspect the detected offset before committing training time."
        />
        <WaveformOverlay latency={latency + nudge} />
      </div>
      <div className="panel span-4">
        <div className="metric-stack">
          <Metric label="Estimated latency" value={`${latency + nudge} samples`} />
          <Metric label="Confidence" value={`${Math.round(confidence * 100)}%`} />
          <Metric
            label="Milliseconds"
            value={`${(((latency + nudge) / 48000) * 1000).toFixed(2)} ms`}
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
        {project.audio?.warnings.length ? (
          <WarningList warnings={project.audio.warnings} />
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
  busy,
  onTrain,
  onCancel,
  onResume,
}: {
  project: ProjectDetail;
  busy: boolean;
  onTrain: (preset: string) => Promise<void>;
  onCancel: (runId: string) => Promise<void>;
  onResume: (runId: string) => Promise<void>;
}) {
  const [preset, setPreset] = useState("lstm_standard");
  const canTrain = project.audio?.status === "ready";
  const activeRun = [...project.runs]
    .reverse()
    .find((run) => ["queued", "preparing", "running", "cancelling"].includes(run.status));
  const resumableRun = [...project.runs]
    .reverse()
    .find((run) => run.status === "failed" || run.status === "interrupted");

  return (
    <div className="screen-grid">
      <div className="panel span-5">
        <ScreenTitle
          icon={Cpu}
          title="Model Preset"
          detail="Curated architectures keep RTNeural export predictable."
        />
        <div className="preset-list">
          {presets.map((item) => (
            <button
              className={preset === item.id ? "preset active" : "preset"}
              key={item.id}
              type="button"
              onClick={() => setPreset(item.id)}
            >
              <span>
                <strong>{item.label}</strong>
                <small>{item.detail}</small>
              </span>
              <em>{item.cpu}</em>
            </button>
          ))}
        </div>
        <button
          className="primary-button wide"
          type="button"
          disabled={!canTrain || busy || Boolean(activeRun)}
          onClick={() => void onTrain(preset)}
        >
          {busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
          Train
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
            <span>Audio must pass preflight before training starts.</span>
          </div>
        ) : null}
      </div>
      <div className="panel span-7">
        <ScreenTitle
          icon={Activity}
          title="Runs"
          detail="Each run keeps checkpoints, metrics, and preview artifacts."
        />
        <RunTable runs={project.runs} />
      </div>
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
        if (mounted) setPreviewError(toMessage(caught));
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
        {selectedRun ? <QualityView run={selectedRun} /> : <p className="muted">No run yet.</p>}
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
}: {
  project: ProjectDetail;
  busy: boolean;
  onExport: (runId: string) => Promise<void>;
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
        <ExportList exports={project.exports} />
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
        <Metric label="Input" value={`${audio.input.sample_rate / 1000} kHz mono`} />
        <Metric label="Target peak" value={`${audio.target.peak_dbfs.toFixed(1)} dBFS`} />
        <Metric label="Duration" value={`${audio.input.duration_seconds.toFixed(1)} s`} />
        <Metric label="Latency" value={`${audio.latency_samples} samples`} />
      </div>
      {audio.warnings.length ? (
        <WarningList warnings={audio.warnings} />
      ) : (
        <div className="notice notice-ok">
          <CheckCircle2 size={18} />
          <span>No blocking preflight warnings.</span>
        </div>
      )}
    </div>
  );
}

function RunTable({ runs }: { runs: TrainingRun[] }) {
  if (runs.length === 0) {
    return <p className="muted">No training runs yet.</p>;
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

  return (
    <div className="report-block">
      <p className="section-label">Training Report</p>
      <div className="report-grid">
        <Metric label="Backend" value={getString(report, "backend") ?? "unknown"} />
        <Metric label="Epochs" value={String(getNumber(report, "epochs") ?? "unknown")} />
        <Metric
          label="Checkpoint"
          value={String(getNumber(report, "checkpoint_epoch") ?? "unknown")}
        />
        <Metric label="Created" value={formatReportDate(getString(report, "created_at"))} />
      </div>
      {preview.report_path ? <p className="artifact-path">{preview.report_path}</p> : null}
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
    return <p className="muted">No preview artifacts yet.</p>;
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
            <PeakWave peaks={artifact.peaks} />
          </div>
        ))}
      </div>
      <p className="artifact-path">{preview.run_dir}</p>
    </div>
  );
}

function PeakWave({ peaks }: { peaks: number[] }) {
  const visiblePeaks = peaks.length ? peaks : Array.from({ length: 28 }, () => 0);
  return (
    <div className="mini-wave" aria-hidden="true">
      {visiblePeaks.map((peak, index) => (
        <i
          key={`${index}-${peak}`}
          style={{ height: `${Math.max(3, Math.round(Math.min(1, peak) * 28))}px` }}
        />
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

function ExportList({ exports }: { exports: ExportPackage[] }) {
  if (exports.length === 0) {
    return <p className="muted">No export package yet.</p>;
  }

  return (
    <div className="export-list">
      {exports.map((item) => (
        <div className="export-item" key={item.id}>
          <div>
            <strong>{item.id}</strong>
            <small>{item.model_path}</small>
          </div>
          <span className="badge">{item.status}</span>
        </div>
      ))}
    </div>
  );
}

function WaveformOverlay({ latency }: { latency: number }) {
  return (
    <div className="waveform">
      <div className="latency-marker" style={{ left: `${48 + Math.max(-20, Math.min(20, latency / 12))}%` }} />
      <div className="wave-row input">
        {Array.from({ length: 80 }, (_, index) => (
          <i key={index} />
        ))}
      </div>
      <div className="wave-row target">
        {Array.from({ length: 80 }, (_, index) => (
          <i key={index} />
        ))}
      </div>
    </div>
  );
}

function WarningList({ warnings }: { warnings: string[] }) {
  return (
    <div className="warning-list">
      {warnings.map((warning) => (
        <div className="notice notice-warning" key={warning}>
          <AlertTriangle size={18} />
          <span>{warning}</span>
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

function EmptyState() {
  return (
    <div className="empty-state">
      <Activity size={42} />
      <h2>Create a capture project</h2>
      <p>Start with paired dry and processed WAV files, then train and validate a model package.</p>
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
    const isBest = getBoolean(event.json, "is_best");
    return [
      trainLoss !== null ? `loss ${formatMetric(trainLoss)}` : null,
      valEsr !== null ? `val ESR ${formatMetric(valEsr)}` : null,
      isBest ? "best" : null,
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

function operationLabel(operation: string) {
  return operation
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMetric(value: number) {
  return value < 0.01 ? value.toExponential(2) : value.toFixed(4);
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

function toMessage(caught: unknown) {
  return caught instanceof Error ? caught.message : String(caught);
}
