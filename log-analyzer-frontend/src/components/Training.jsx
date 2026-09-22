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
      className={`flex flex-1 items-center gap-3 rounded-xl border p-4 ${
        active ? "border-google-blue/40 bg-google-blue/5" : "border-google-border"
      }`}
    >
      <Icon
        size={24}
        className={`shrink-0 ${active ? "text-google-blue" : "text-google-muted"}`}
      />
      <div>
        <div className="text-sm text-google-text">{title}</div>
        <div className="text-xs text-google-muted mt-1">{description}</div>
      </div>
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
      setMessage(`${response.data.dataset_version} is ready to use for training.`);
    } catch (requestError) {
      setActionError(errorMessage(requestError, "Could not create the dataset."));
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
        setMessage(
          "Training finished successfully. The new model will analyze future incidents.",
        );
      } else if (finished?.status === "FAILED") {
        setActionError("Training did not finish successfully. Review the details below and try again.");
      }
    } catch (requestError) {
      setActionError(errorMessage(requestError, "Could not start this training run."));
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
        errorMessage(requestError, "Could not start training."),
      );
    }
  };

  const hasExecutingJobs = jobs.some((job) =>
    ["RUNNING", "EVALUATING"].includes(job.status),
  );
  const createBusy = building || runningJob !== null || hasActiveJobs;

  return (
    <div className="app-shell">
      <div className="app-canvas">
        <Navbar />
      <main className="max-w-[96rem] mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-16">
        <div className="flex items-start justify-between gap-4 mb-12">
          <div>
            <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.04em] text-google-text">Training</h1>
            <p className="text-sm text-google-muted mt-2">
              Train a model using incidents your team has reviewed and confirmed.
            </p>
          </div>
          <button
            type="button"
            onClick={() => refresh(true).catch(() => {})}
            disabled={refreshing}
            className="px-3 py-2 border border-google-border rounded-lg text-xs text-google-muted hover:text-google-text hover:border-google-blue flex items-center gap-2 disabled:opacity-50"
          >
            <RefreshCw size={13} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>

        <div className="flex flex-col md:flex-row items-stretch md:items-center gap-2 mb-8">
          <WorkflowStep
            icon={Database}
            title="Dataset"
            description="Collect confirmed incidents as training examples"
            active={datasets.length > 0}
          />
          <ArrowDown
            size={15}
            className="text-google-muted self-center md:-rotate-90"
          />
          <WorkflowStep
            icon={Play}
            title="Training Job"
            description="Teach the model from those examples"
            active={jobs.length > 0}
          />
          <ArrowDown
            size={15}
            className="text-google-muted self-center md:-rotate-90"
          />
          <WorkflowStep
            icon={Sparkles}
            title="Trained Model"
            description="Review the result and put it to work"
            active={artifacts.length > 0}
          />
        </div>

        <LifecycleError message={error || actionError} />
        {message && (
          <div className="mb-5 p-3 rounded-lg border border-google-green/30 bg-green-50 text-google-green text-sm">
            {message}
          </div>
        )}

        <section className="soft-card p-5 mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-google-text">Datasets</h2>
              <p className="text-xs text-google-muted mt-1">
                Create a reusable set of confirmed incidents for model
                training.
              </p>
            </div>
            <button
              type="button"
              onClick={buildDataset}
              disabled={building}
              className="px-4 py-2 rounded-lg bg-google-blue hover:bg-google-blue-dark text-white text-xs font-medium flex items-center justify-center gap-2 disabled:bg-google-chip disabled:text-google-muted disabled:cursor-not-allowed"
            >
              {building ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <Database size={13} />
              )}
              {building ? "Creating..." : "Create Dataset"}
            </button>
          </div>

          {loading ? (
            <div className="py-10 text-center">
              <span className="loader" />
            </div>
          ) : datasets.length === 0 ? (
            <EmptyState>
              No datasets yet. Create one from incidents your team has already
              reviewed and confirmed.
            </EmptyState>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-google-muted border-b border-google-border">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Version</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">Records</th>
                    <th className="py-2 font-medium">Created</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset) => (
                    <tr key={dataset.id} className="border-b border-google-border">
                      <td className="py-3 pr-4 text-google-text">
                        {dataset.dataset_version}
                      </td>
                      <td className="py-3 pr-4">
                        <StatusBadge status={dataset.status} />
                      </td>
                      <td className="py-3 pr-4 text-google-muted">
                        {dataset.record_count}
                      </td>
                      <td className="py-3 text-google-muted">
                        {formatDate(dataset.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="soft-card p-5 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 mb-5">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-google-text">Training Jobs</h2>
              <p className="text-xs text-google-muted mt-1">
                Choose a dataset and start a training run. Progress updates
                automatically.
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-2">
              <select
                value={selectedDataset}
                onChange={(event) => setSelectedDataset(event.target.value)}
                disabled={createBusy || readyDatasets.length === 0}
                className="min-w-52 px-3 py-2 bg-google-bg border border-google-border rounded-lg text-xs text-google-text disabled:text-google-muted"
              >
                {readyDatasets.length === 0 && (
                  <option value="">No datasets ready for training</option>
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
                className="px-4 py-2 rounded-lg bg-google-blue hover:bg-google-blue-dark text-white text-xs font-medium flex items-center justify-center gap-2 disabled:bg-google-chip disabled:text-google-muted disabled:cursor-not-allowed"
              >
                {runningJob ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Sparkles size={13} />
                )}
                {runningJob ? "Training..." : "Train Model"}
              </button>
            </div>
          </div>

          {jobs.length === 0 ? (
            <EmptyState>
              No training runs yet. Create a dataset above, then select it to
              train a model.
            </EmptyState>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => {
                const dataset = datasetById[job.dataset_id];
                const artifact = artifactById[job.artifact_id];
                return (
                  <div
                    key={job.id}
                    className="rounded-2xl border border-google-border bg-white/80 p-4"
                  >
                    <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <StatusBadge status={job.status} />
                          <span className="text-xs text-google-muted font-mono truncate">
                            {job.id}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
                          <div>
                            <span className="text-google-muted">Dataset</span>
                            <div className="text-google-text mt-1">
                              {dataset?.dataset_version || job.dataset_id}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Started</span>
                            <div className="text-google-text mt-1">
                              {formatDate(job.started_at)}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Finished</span>
                            <div className="text-google-text mt-1">
                              {formatDate(job.finished_at)}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Trained model</span>
                            <div className="text-google-text mt-1">
                              {artifact?.artifact_version || "—"}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Quality score</span>
                            <div className="text-google-text mt-1">
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
                          className="px-3 py-2 border border-google-blue/30 text-google-blue rounded-lg text-xs hover:bg-google-blue/10 disabled:opacity-50"
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

        <section className="soft-card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-google-text">
                Training Results
              </h2>
              <p className="text-xs text-google-muted mt-1">
                Models that completed training and are ready for review.
              </p>
            </div>
            {runtime?.active_artifact_version && (
              <div className="flex items-center gap-1.5 text-xs text-google-green">
                <CheckCircle size={13} /> {runtime.active_artifact_version}{" "}
                active
              </div>
            )}
          </div>
          {artifacts.length === 0 ? (
            <EmptyState>
              Completed models will appear here after a training run succeeds.
            </EmptyState>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {artifacts.slice(0, 6).map((artifact) => (
                <div
                  key={artifact.id}
                  className="rounded-2xl border border-google-border bg-white/80 p-4"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-google-text">
                      {artifact.artifact_version}
                    </span>
                    <StatusBadge status={artifact.status} />
                  </div>
                  <div className="text-xs text-google-muted">
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
    </div>
  );
}

export default Training;
