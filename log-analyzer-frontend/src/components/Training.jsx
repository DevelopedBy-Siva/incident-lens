import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowDown,
  CheckCircle,
  Database,
  Loader2,
  Play,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import Navbar from "./Navbar";
import { modelLifecycleAPI } from "../services/api";
import {
  errorMessage,
  invalidateModelLifecycleCache,
  useModelLifecycle,
} from "../hooks/useModelLifecycle";
import {
  displayModelName,
  EmptyState,
  formatDate,
  LifecycleError,
  StatusBadge,
} from "./modelLifecycle/ModelUi";

function WorkflowStep({ icon: Icon, title, description, active }) {
  return (
    <div
      className={`flex-1 rounded-xl border p-4 ${
        active ? "border-sky-500/40 bg-sky-500/5" : "border-gray-800"
      }`}
    >
      <Icon size={16} className={active ? "text-sky-400" : "text-gray-600"} />
      <div className="text-sm text-gray-200 mt-3">{title}</div>
      <div className="text-xs text-gray-600 mt-1">{description}</div>
    </div>
  );
}

function Training() {
  const {
    runtime,
    datasets,
    jobs,
    artifacts,
    loading,
    refreshing,
    error,
    hasActiveJobs,
    refresh,
  } = useModelLifecycle({ pollWhileActive: true });
  const [selectedDataset, setSelectedDataset] = useState("");
  const [building, setBuilding] = useState(false);
  const [runningJob, setRunningJob] = useState(null);
  const [actionError, setActionError] = useState("");
  const [message, setMessage] = useState("");

  const readyDatasets = useMemo(
    () => datasets.filter((dataset) => dataset.status === "READY"),
    [datasets],
  );

  useEffect(() => {
    if (
      readyDatasets.length > 0 &&
      !readyDatasets.some((dataset) => dataset.id === selectedDataset)
    ) {
      setSelectedDataset(readyDatasets[0].id);
    }
  }, [readyDatasets, selectedDataset]);

  const artifactById = useMemo(
    () =>
      Object.fromEntries(artifacts.map((artifact) => [artifact.id, artifact])),
    [artifacts],
  );
  const datasetById = useMemo(
    () => Object.fromEntries(datasets.map((dataset) => [dataset.id, dataset])),
    [datasets],
  );

  const buildDataset = async () => {
    setBuilding(true);
    setActionError("");
    setMessage("");
    try {
      const response = await modelLifecycleAPI.buildDataset();
      invalidateModelLifecycleCache();
      await refresh(true);
      setSelectedDataset(response.data.id);
      setMessage(`${response.data.dataset_version} is READY for training.`);
    } catch (requestError) {
      setActionError(errorMessage(requestError, "Could not build dataset."));
    } finally {
      setBuilding(false);
    }
  };

  const runExistingJob = async (job) => {
    setRunningJob(job.id);
    setActionError("");
    setMessage("");
    try {
      await modelLifecycleAPI.runTrainingJob(job.id);
      invalidateModelLifecycleCache();
      const updated = await refresh(true);
      const finished = updated.jobs.find((item) => item.id === job.id);
      if (finished?.status === "PASSED") {
        setMessage("Training passed and the new adapter is now active.");
      } else if (finished?.status === "FAILED") {
        setActionError("Training failed. Review the job status below.");
      }
    } catch (requestError) {
      setActionError(errorMessage(requestError, "Could not run training job."));
      invalidateModelLifecycleCache();
      await refresh(true).catch(() => {});
    } finally {
      setRunningJob(null);
    }
  };

  const trainAdapter = async () => {
    if (!selectedDataset) return;
    setActionError("");
    setMessage("");
    try {
      const response =
        await modelLifecycleAPI.createTrainingJob(selectedDataset);
      invalidateModelLifecycleCache();
      await refresh(true);
      await runExistingJob(response.data);
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not create training job."),
      );
    }
  };

  const hasExecutingJobs = jobs.some((job) =>
    ["RUNNING", "EVALUATING"].includes(job.status),
  );
  const createBusy = building || runningJob !== null || hasActiveJobs;

  return (
    <div className="min-h-screen bg-black">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-16">
        <div className="flex items-start justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-medium text-white">Training</h1>
            <p className="text-sm text-gray-500 mt-1">
              Turn confirmed incidents into a project-specific LoRA adapter
            </p>
          </div>
          <button
            type="button"
            onClick={() => refresh(true).catch(() => {})}
            disabled={refreshing}
            className="px-3 py-2 border border-gray-700 rounded-lg text-xs text-gray-400 hover:text-white hover:border-gray-600 flex items-center gap-2 disabled:opacity-50"
          >
            <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2 mb-8">
          <WorkflowStep
            icon={Database}
            title="1. Dataset"
            description="Immutable confirmed incident examples"
            active={datasets.length > 0}
          />
          <ArrowDown
            size={15}
            className="text-gray-700 self-center md:-rotate-90"
          />
          <WorkflowStep
            icon={Play}
            title="2. Training Job"
            description="Fine-tune a project LoRA adapter"
            active={jobs.length > 0}
          />
          <ArrowDown
            size={15}
            className="text-gray-700 self-center md:-rotate-90"
          />
          <WorkflowStep
            icon={Sparkles}
            title="3. Model Artifact"
            description="Evaluate, register, and activate"
            active={artifacts.length > 0}
          />
        </div>

        <LifecycleError message={error || actionError} />
        {message && (
          <div className="mb-5 p-3 rounded-lg border border-green-500/30 bg-green-500/10 text-green-300 text-sm">
            {message}
          </div>
        )}

        <section className="rounded-xl border border-gray-800 p-5 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
            <div>
              <h2 className="text-lg font-medium text-white">Datasets</h2>
              <p className="text-xs text-gray-500 mt-1">
                Each build creates a new immutable version.
              </p>
            </div>
            <button
              type="button"
              onClick={buildDataset}
              disabled={building}
              className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-600 text-white text-xs font-medium flex items-center justify-center gap-2 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
            >
              {building ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Database size={13} />
              )}
              {building ? "Building..." : "Build New Dataset"}
            </button>
          </div>

          {loading ? (
            <div className="py-10 text-center">
              <span className="loader" />
            </div>
          ) : datasets.length === 0 ? (
            <EmptyState>
              No datasets yet. Build one from eligible confirmed incidents.
            </EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-gray-600 border-b border-gray-800">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Version</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">Records</th>
                    <th className="py-2 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset) => (
                    <tr key={dataset.id} className="border-b border-gray-900">
                      <td className="py-3 pr-4 text-gray-200">
                        {dataset.dataset_version}
                      </td>
                      <td className="py-3 pr-4">
                        <StatusBadge status={dataset.status} />
                      </td>
                      <td className="py-3 pr-4 text-gray-400">
                        {dataset.record_count}
                      </td>
                      <td className="py-3 text-gray-500">
                        {formatDate(dataset.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="rounded-xl border border-gray-800 p-5 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 mb-5">
            <div>
              <h2 className="text-lg font-medium text-white">Training Jobs</h2>
              <p className="text-xs text-gray-500 mt-1">
                Active jobs refresh every four seconds; completed jobs stop
                polling.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2">
              <select
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                disabled={createBusy || readyDatasets.length === 0}
                className="min-w-52 px-3 py-2 bg-black border border-gray-700 rounded-lg text-xs text-gray-300 disabled:text-gray-600"
              >
                {readyDatasets.length === 0 && (
                  <option value="">No READY datasets</option>
                )}
                {readyDatasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.dataset_version} · {dataset.record_count} records
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={trainAdapter}
                disabled={!selectedDataset || createBusy}
                className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium flex items-center justify-center gap-2 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed"
              >
                {runningJob ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Sparkles size={13} />
                )}
                {runningJob ? "Training..." : "Train Adapter"}
              </button>
            </div>
          </div>

          {jobs.length === 0 ? (
            <EmptyState>No training jobs yet.</EmptyState>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => {
                const dataset = datasetById[job.dataset_id];
                const artifact = artifactById[job.artifact_id];
                return (
                  <div
                    key={job.id}
                    className="rounded-lg border border-gray-800 p-4"
                  >
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <StatusBadge status={job.status} />
                          <span className="text-xs text-gray-600 font-mono truncate">
                            {job.id}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
                          <div>
                            <span className="text-gray-600">Dataset</span>
                            <div className="text-gray-300 mt-1">
                              {dataset?.dataset_version || job.dataset_id}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Started</span>
                            <div className="text-gray-300 mt-1">
                              {formatDate(job.started_at)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Finished</span>
                            <div className="text-gray-300 mt-1">
                              {formatDate(job.finished_at)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Artifact</span>
                            <div className="text-gray-300 mt-1">
                              {artifact?.artifact_version || "—"}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Evaluation</span>
                            <div className="text-gray-300 mt-1">
                              {artifact?.evaluation_score?.toFixed(3) ?? "—"}
                            </div>
                          </div>
                        </div>
                      </div>
                      {job.status === "QUEUED" && (
                        <button
                          type="button"
                          onClick={() => runExistingJob(job)}
                          disabled={
                            building || runningJob !== null || hasExecutingJobs
                          }
                          className="px-3 py-2 border border-sky-500/30 text-sky-400 rounded-lg text-xs hover:bg-sky-500/10 disabled:opacity-50"
                        >
                          Run
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="rounded-xl border border-gray-800 p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-medium text-white">
                Training Results
              </h2>
              <p className="text-xs text-gray-500 mt-1">
                Adapters produced by successful jobs
              </p>
            </div>
            {runtime?.active_artifact_version && (
              <div className="flex items-center gap-1.5 text-xs text-green-400">
                <CheckCircle size={13} /> {runtime.active_artifact_version}{" "}
                active
              </div>
            )}
          </div>
          {artifacts.length === 0 ? (
            <EmptyState>No training results yet.</EmptyState>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {artifacts.slice(0, 6).map((artifact) => (
                <div
                  key={artifact.id}
                  className="rounded-lg border border-gray-800 p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-white">
                      {artifact.artifact_version}
                    </span>
                    <StatusBadge status={artifact.status} />
                  </div>
                  <div className="text-xs text-gray-500">
                    {displayModelName(artifact.base_model)} · score{" "}
                    {artifact.evaluation_score?.toFixed(3) ?? "—"}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default Training;
