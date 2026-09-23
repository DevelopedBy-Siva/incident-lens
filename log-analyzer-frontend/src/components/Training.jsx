import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  CheckCircle,
  ChevronLeft,
  ChevronRight,
  Database,
  Download,
  Eye,
  Loader2,
  MoreVertical,
  Play,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  X,
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

function formatDuration(totalSeconds) {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "—";
  const seconds = Math.round(totalSeconds);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m`;
  return `${seconds}s`;
}

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

function DatasetViewer({
  dataset,
  records,
  selected,
  loading,
  error,
  approving,
  onClose,
  onToggle,
  onSelectAll,
  onClear,
  onApprove,
}) {
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(5);

  useEffect(() => {
    setPage(1);
  }, [dataset?.id, pageSize]);

  if (!dataset) return null;

  const editable = dataset.status === "VALIDATING";
  const totalPages = Math.ceil(records.length / pageSize);
  const visibleRecords = records.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/35 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="dataset-viewer-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-google-border bg-white shadow-google">
        <div className="flex items-start justify-between gap-4 border-b border-google-border px-5 py-4">
          <div>
            <h2
              id="dataset-viewer-title"
              className="text-xl font-semibold tracking-tight text-google-text"
            >
              {dataset.dataset_version}
            </h2>
            <p className="mt-1 text-xs text-google-muted">
              {editable
                ? "Review the records, choose the training data, and approve the dataset."
                : "View the records approved for training."}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-google-muted hover:bg-google-hover hover:text-google-text"
            aria-label="Close dataset"
          >
            <X size={17} />
          </button>
        </div>

        <div className="flex items-center justify-between gap-3 border-b border-google-border bg-google-subtle px-5 py-3">
          <span className="text-xs text-google-muted">
            {selected.size} of {records.length} records selected
          </span>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {editable && (
              <>
                <button
                  type="button"
                  onClick={onSelectAll}
                  disabled={loading || records.length === 0}
                  className="text-xs text-google-blue hover:underline disabled:opacity-50"
                >
                  Select all
                </button>
                <span className="text-google-border">|</span>
                <button
                  type="button"
                  onClick={onClear}
                  disabled={loading || records.length === 0}
                  className="text-xs text-google-muted hover:text-google-text disabled:opacity-50"
                >
                  Clear
                </button>
              </>
            )}
            <label className="ml-2 flex items-center gap-2 text-xs text-google-muted">
              Per page
              <select
                value={pageSize}
                onChange={(event) => setPageSize(Number(event.target.value))}
                className="rounded-lg border border-google-border bg-white px-2 py-1.5 text-xs text-google-text"
              >
                {[5, 10, 20, 50].map((size) => (
                  <option key={size} value={size}>
                    {size}
                  </option>
                ))}
              </select>
            </label>
            <span className="min-w-20 text-center text-xs text-google-muted">
              Page {totalPages ? page : 0} of {totalPages}
            </span>
            <button
              type="button"
              aria-label="Previous dataset page"
              disabled={page <= 1}
              onClick={() => setPage((current) => current - 1)}
              className="rounded-lg border border-google-border bg-white p-1.5 text-google-muted hover:bg-google-hover hover:text-google-text disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              type="button"
              aria-label="Next dataset page"
              disabled={totalPages === 0 || page >= totalPages}
              onClick={() => setPage((current) => current + 1)}
              className="rounded-lg border border-google-border bg-white p-1.5 text-google-muted hover:bg-google-hover hover:text-google-text disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="py-16 text-center">
              <span className="loader" />
            </div>
          ) : error ? (
            <LifecycleError message={error} />
          ) : records.length === 0 ? (
            <EmptyState>This dataset has no records.</EmptyState>
          ) : (
            <div className="space-y-3">
              {visibleRecords.map((record) => {
                const input = record.data?.input || {};
                const incident = input.incident || {};
                const output = record.data?.expected_output || {};
                const logs = Array.isArray(input.logs) ? input.logs : [];
                return (
                  <div
                    key={record.index}
                    className={`rounded-xl border p-4 transition-colors ${
                      selected.has(record.index)
                        ? "border-google-blue/40 bg-google-blue/5"
                        : "border-google-border bg-white hover:bg-google-subtle"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={selected.has(record.index)}
                        onChange={() => onToggle(record.index)}
                        disabled={!editable}
                        className="mt-1 h-4 w-4 accent-google-blue"
                        aria-label={`Select record ${record.index + 1}`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-sm font-medium text-google-text">
                            {output.summary || incident.signature || `Record ${record.index + 1}`}
                          </span>
                          {output.severity && (
                            <span className="rounded-full border border-google-border bg-google-chip px-2 py-0.5 text-[11px] uppercase text-google-muted">
                              {output.severity}
                            </span>
                          )}
                          {output.disposition && (
                            <span className="rounded-full border border-google-border bg-white px-2 py-0.5 text-[11px] text-google-muted">
                              {output.disposition}
                            </span>
                          )}
                        </div>
                        <div className="mt-2 grid gap-2 text-xs text-google-muted sm:grid-cols-2">
                          <div>
                            <span className="text-google-text">Source:</span>{" "}
                            {incident.source || "—"}
                          </div>
                          <div>
                            <span className="text-google-text">Logs:</span>{" "}
                            {logs.length}
                          </div>
                        </div>
                        {logs[0] && (
                          <pre className="mt-3 overflow-x-auto whitespace-pre-wrap rounded-lg bg-google-subtle p-3 font-mono text-[11px] leading-5 text-google-muted">
                            {logs[0]}
                          </pre>
                        )}
                        <details className="mt-3" onClick={(event) => event.stopPropagation()}>
                          <summary className="cursor-pointer text-xs text-google-blue">
                            View full record
                          </summary>
                          <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded-lg bg-google-subtle p-3 font-mono text-[11px] leading-5 text-google-muted">
                            {JSON.stringify(record.data, null, 2)}
                          </pre>
                        </details>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-google-border px-5 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-google-border px-4 py-2 text-xs text-google-muted hover:bg-google-hover hover:text-google-text"
          >
            {editable ? "Cancel" : "Close"}
          </button>
          {editable && (
            <button
              type="button"
              onClick={onApprove}
              disabled={
                loading || Boolean(error) || selected.size === 0 || approving
              }
              className="flex items-center gap-2 rounded-lg bg-google-blue px-4 py-2 text-xs font-medium text-white hover:bg-google-blue-dark disabled:cursor-not-allowed disabled:bg-google-chip disabled:text-google-muted"
            >
              {approving ? (
                <Loader2 size={13} className="animate-spin" />
              ) : (
                <CheckCircle size={13} />
              )}
              {approving ? "Approving..." : `Approve dataset (${selected.size})`}
            </button>
          )}
        </div>
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
  const [viewingDataset, setViewingDataset] = useState(null);
  const [datasetRecords, setDatasetRecords] = useState([]);
  const [selectedRecords, setSelectedRecords] = useState(new Set());
  const [datasetLoading, setDatasetLoading] = useState(false);
  const [datasetViewError, setDatasetViewError] = useState("");
  const [approvingDataset, setApprovingDataset] = useState(false);
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [openDatasetMenu, setOpenDatasetMenu] = useState(null);
  const uploadInputRef = useRef(null);

  useEffect(() => {
    const closeMenu = (event) => {
      if (!event.target.closest("[data-dataset-menu]")) {
        setOpenDatasetMenu(null);
      }
    };
    const closeMenuWithKeyboard = (event) => {
      if (event.key === "Escape") setOpenDatasetMenu(null);
    };
    document.addEventListener("mousedown", closeMenu);
    document.addEventListener("keydown", closeMenuWithKeyboard);
    return () => {
      document.removeEventListener("mousedown", closeMenu);
      document.removeEventListener("keydown", closeMenuWithKeyboard);
    };
  }, []);

  const trainableDatasets = useMemo(
    () =>
      datasets.filter((dataset) => ["READY", "TRAINED"].includes(dataset.status)),
    [datasets],
  );

  useEffect(() => {
    if (
      trainableDatasets.length > 0 &&
      !trainableDatasets.some((dataset) => dataset.id === selectedDataset)
    ) {
      setSelectedDataset(trainableDatasets[0].id);
    }
  }, [trainableDatasets, selectedDataset]);

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
      setMessage(`${response.data.dataset_version} is ready for review.`);
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

  const trainDataset = async (datasetId) => {
    if (!datasetId) return;
    setActionError("");
    setMessage("");
    try {
      const response =
        await modelLifecycleAPI.createTrainingJob(datasetId);
      invalidateModelLifecycleCache();
      await refresh(true);
      await runExistingJob(response.data);
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not start training."),
      );
    }
  };

  const trainAdapter = () => trainDataset(selectedDataset);

  const openDataset = async (dataset) => {
    setViewingDataset(dataset);
    setDatasetRecords([]);
    setSelectedRecords(new Set());
    setDatasetViewError("");
    setDatasetLoading(true);
    try {
      const response = await modelLifecycleAPI.getDatasetRecords(dataset.id);
      const records = response.data.records || [];
      setDatasetRecords(records);
      const storedSelection = response.data.dataset.selected_record_indices;
      setSelectedRecords(
        new Set(
          storedSelection === null
            ? records.map((record) => record.index)
            : storedSelection,
        ),
      );
    } catch (requestError) {
      setDatasetViewError(
        errorMessage(requestError, "Could not load this dataset."),
      );
    } finally {
      setDatasetLoading(false);
    }
  };

  const closeDataset = () => {
    setViewingDataset(null);
    setDatasetRecords([]);
    setSelectedRecords(new Set());
    setDatasetViewError("");
  };

  const toggleDatasetRecord = (index) => {
    setSelectedRecords((current) => {
      const next = new Set(current);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const approveDataset = async () => {
    if (!viewingDataset || selectedRecords.size === 0) return;
    const datasetId = viewingDataset.id;
    const selection = Array.from(selectedRecords).sort((left, right) => left - right);
    setApprovingDataset(true);
    setDatasetViewError("");
    try {
      const response = await modelLifecycleAPI.approveDataset(datasetId, selection);
      closeDataset();
      invalidateModelLifecycleCache();
      await refresh(true);
      setSelectedDataset(datasetId);
      setMessage(
        `${response.data.dataset_version} was approved with ${selection.length} records.`,
      );
    } catch (requestError) {
      setDatasetViewError(
        errorMessage(requestError, "Could not approve this dataset."),
      );
    } finally {
      setApprovingDataset(false);
    }
  };

  const uploadDataset = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploadingDataset(true);
    setActionError("");
    setMessage("");
    try {
      const lines = (await file.text()).split(/\r?\n/);
      const records = lines.flatMap((line, index) => {
        if (!line.trim()) return [];
        try {
          const record = JSON.parse(line);
          if (!record || Array.isArray(record) || typeof record !== "object") {
            throw new Error("must be a JSON object");
          }
          if (
            !("expected_output" in record) &&
            record.output &&
            !Array.isArray(record.output) &&
            typeof record.output === "object"
          ) {
            const { output, ...rest } = record;
            return [{ ...rest, expected_output: output }];
          }
          return [record];
        } catch (error) {
          throw new Error(
            `Invalid JSONL record on line ${index + 1}: ${error.message}`,
          );
        }
      });
      if (records.length === 0) throw new Error("The JSONL file is empty.");
      const response = await modelLifecycleAPI.uploadDataset(records);
      invalidateModelLifecycleCache();
      await refresh(true);
      setMessage(`${response.data.dataset_version} was uploaded and is ready for review.`);
    } catch (requestError) {
      setActionError(
        requestError instanceof SyntaxError || !requestError.response
          ? requestError.message || "Could not read the JSONL file."
          : errorMessage(requestError, "Could not upload this dataset."),
      );
    } finally {
      setUploadingDataset(false);
    }
  };

  const downloadDataset = async (dataset) => {
    setActionError("");
    try {
      const response = await modelLifecycleAPI.downloadDataset(dataset.id);
      const url = URL.createObjectURL(response.data);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${dataset.dataset_version}.jsonl`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not download this dataset."),
      );
    }
  };

  const deleteDataset = async (dataset) => {
    const confirmed = window.confirm(
      `Delete ${dataset.dataset_version}? This cannot be undone.`,
    );
    if (!confirmed) return;

    setActionError("");
    setMessage("");
    try {
      await modelLifecycleAPI.deleteDataset(dataset.id);
      invalidateModelLifecycleCache();
      await refresh(true);
      setMessage(`${dataset.dataset_version} was deleted.`);
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not delete this dataset."),
      );
    }
  };

  const hasExecutingJobs = jobs.some((job) =>
    ["RUNNING", "EVALUATING"].includes(job.status),
  );
  const createBusy = building || uploadingDataset || runningJob !== null || hasActiveJobs;

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

        <section className="soft-card relative z-20 mb-6 overflow-visible p-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-google-text">Datasets</h2>
              <p className="text-xs text-google-muted mt-1">
                Create a reusable set of confirmed incidents for model
                training.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <input
                ref={uploadInputRef}
                type="file"
                accept=".jsonl,application/x-ndjson,application/jsonl"
                onChange={uploadDataset}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => uploadInputRef.current?.click()}
                disabled={building || uploadingDataset}
                className="flex items-center justify-center gap-2 rounded-lg border border-google-border bg-white px-4 py-2 text-xs font-medium text-google-text hover:bg-google-hover disabled:cursor-not-allowed disabled:opacity-50"
              >
                {uploadingDataset ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Upload size={13} />
                )}
                {uploadingDataset ? "Uploading..." : "Upload JSONL"}
              </button>
              <button
                type="button"
                onClick={buildDataset}
                disabled={building || uploadingDataset}
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
            <div className="overflow-visible">
              <table className="w-full text-left text-xs">
                <thead className="text-google-muted border-b border-google-border">
                  <tr>
                    <th className="py-2 pr-4 font-medium">Version</th>
                    <th className="py-2 pr-4 font-medium">Status</th>
                    <th className="py-2 pr-4 font-medium">Records</th>
                    <th className="hidden py-2 pr-4 font-medium sm:table-cell">
                      Created
                    </th>
                    <th className="w-10 py-2 text-right font-medium">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset) => (
                    <tr key={dataset.id} className="border-b border-google-border">
                      <td className="py-3 pr-4 text-google-text">
                        {dataset.dataset_version}
                      </td>
                      <td className="py-3 pr-4">
                        <StatusBadge
                          status={dataset.status}
                          label={
                            dataset.status === "VALIDATING"
                              ? "REVIEW PENDING"
                              : undefined
                          }
                        />
                      </td>
                      <td className="py-3 pr-4 text-google-muted">
                        {dataset.selected_record_indices
                          ? `${dataset.selected_record_indices.length} of ${dataset.record_count}`
                          : dataset.record_count}
                      </td>
                      <td className="hidden py-3 pr-4 text-google-muted sm:table-cell">
                        {formatDate(dataset.created_at)}
                      </td>
                      <td className="relative py-3 text-right">
                        <div
                          className="relative inline-block text-left"
                          data-dataset-menu
                        >
                          <button
                            type="button"
                            onClick={() =>
                              setOpenDatasetMenu((current) =>
                                current === dataset.id ? null : dataset.id,
                              )
                            }
                            className="flex rounded-lg p-2 text-google-muted hover:bg-google-hover hover:text-google-text"
                            aria-label={`Actions for ${dataset.dataset_version}`}
                            aria-expanded={openDatasetMenu === dataset.id}
                          >
                            <MoreVertical size={16} />
                          </button>
                          {openDatasetMenu === dataset.id && (
                          <div className="absolute right-0 top-full z-[80] mt-1 w-44 overflow-hidden rounded-lg border border-google-border bg-white py-1 text-left shadow-google">
                            <button
                              type="button"
                              onClick={() => {
                                setOpenDatasetMenu(null);
                                openDataset(dataset);
                              }}
                              disabled={["FAILED", "TRAINED"].includes(
                                dataset.status,
                              )}
                              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-google-text hover:bg-google-hover disabled:cursor-not-allowed disabled:text-google-muted"
                            >
                              <Eye size={14} /> Review
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setOpenDatasetMenu(null);
                                downloadDataset(dataset);
                              }}
                              disabled={dataset.status === "FAILED"}
                              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-google-text hover:bg-google-hover disabled:cursor-not-allowed disabled:text-google-muted"
                            >
                              <Download size={14} /> Download JSONL
                            </button>
                            <button
                              type="button"
                              onClick={() => {
                                setOpenDatasetMenu(null);
                                deleteDataset(dataset);
                              }}
                              className="flex w-full items-center gap-2 px-3 py-2 text-xs text-google-red hover:bg-red-50"
                            >
                              <Trash2 size={14} /> Delete
                            </button>
                          </div>
                          )}
                        </div>
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
                disabled={createBusy || trainableDatasets.length === 0}
                className="min-w-52 px-3 py-2 bg-google-bg border border-google-border rounded-lg text-xs text-google-text disabled:text-google-muted"
              >
                {trainableDatasets.length === 0 && (
                  <option value="">No datasets ready for training</option>
                )}
                {trainableDatasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.dataset_version} ·{" "}
                    {dataset.selected_record_indices?.length ?? dataset.record_count}{" "}
                    records
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
                const active = ["RUNNING", "EVALUATING"].includes(job.status);
                const elapsedSeconds = job.started_at
                  ? (new Date(job.finished_at || Date.now()).getTime() -
                      new Date(job.started_at).getTime()) /
                    1000
                  : null;
                const hasProgress =
                  job.progress_total_steps > 0 &&
                  job.progress_current_step !== null;
                const progressPercent = hasProgress
                  ? Math.min(
                      100,
                      (job.progress_current_step / job.progress_total_steps) * 100,
                    )
                  : 0;
                const remainingSeconds =
                  active && hasProgress && job.progress_current_step > 0
                    ? (elapsedSeconds / job.progress_current_step) *
                      (job.progress_total_steps - job.progress_current_step)
                    : null;
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
                        {(active || hasProgress) && (
                          <div className="mb-4 max-w-3xl">
                            {hasProgress && (
                              <div className="mb-2 h-2 overflow-hidden rounded-full bg-google-chip">
                                <div
                                  className="h-full rounded-full bg-google-blue transition-[width] duration-500"
                                  style={{ width: `${progressPercent}%` }}
                                />
                              </div>
                            )}
                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-google-muted">
                              {hasProgress && (
                                <span>
                                  {progressPercent.toFixed(1)}% · step{" "}
                                  {job.progress_current_step.toLocaleString()} of{" "}
                                  {job.progress_total_steps.toLocaleString()}
                                </span>
                              )}
                              {elapsedSeconds !== null && (
                                <span>Elapsed {formatDuration(elapsedSeconds)}</span>
                              )}
                              {remainingSeconds !== null && (
                                <span>
                                  About {formatDuration(remainingSeconds)} remaining
                                </span>
                              )}
                              {active && !hasProgress && (
                                <span>Step progress unavailable for this run</span>
                              )}
                            </div>
                          </div>
                        )}
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-xs">
                          <div>
                            <span className="text-google-muted">Dataset</span>
                            <div className="text-google-text mt-1">
                              {dataset?.dataset_version || job.dataset_id}
                            </div>
                            {dataset && (
                              <div className="mt-1 text-[11px] text-google-muted">
                                {(job.selected_record_indices ||
                                  dataset.selected_record_indices)
                                  ? `${(
                                      job.selected_record_indices ||
                                      dataset.selected_record_indices
                                    ).length} of ${dataset.record_count} records`
                                  : `${dataset.record_count} records`}
                              </div>
                            )}
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
      <DatasetViewer
        dataset={viewingDataset}
        records={datasetRecords}
        selected={selectedRecords}
        loading={datasetLoading}
        error={datasetViewError}
        approving={approvingDataset}
        onClose={closeDataset}
        onToggle={toggleDatasetRecord}
        onSelectAll={() =>
          setSelectedRecords(new Set(datasetRecords.map((record) => record.index)))
        }
        onClear={() => setSelectedRecords(new Set())}
        onApprove={approveDataset}
      />
      </div>
    </div>
  );
}

export default Training;
