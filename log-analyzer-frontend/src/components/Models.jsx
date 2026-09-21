import React, { useState } from "react";
import { Check, Cpu, Layers3, Loader2, RefreshCw } from "lucide-react";
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

function score(value) {
  const numeric = Number(value);
  return value === null || value === undefined || !Number.isFinite(numeric)
    ? "—"
    : numeric.toFixed(3);
}

function Models() {
  const { runtime, datasets, artifacts, loading, refreshing, error, refresh } =
    useModelLifecycle();
  const [activating, setActivating] = useState(null);
  const [actionError, setActionError] = useState("");
  const [message, setMessage] = useState("");
  const datasetVersions = Object.fromEntries(
    datasets.map((dataset) => [dataset.id, dataset.dataset_version]),
  );

  const activate = async (artifact) => {
    setActivating(artifact.id);
    setActionError("");
    setMessage("");
    try {
      await modelLifecycleAPI.activateArtifact(artifact.id);
      invalidateModelLifecycleCache();
      await refresh(true);
      setMessage(
        `${artifact.artifact_version} is now serving future incidents.`,
      );
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not activate artifact."),
      );
    } finally {
      setActivating(null);
    }
  };

  return (
    <div className="min-h-screen bg-black">
      <Navbar />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-16">
        <div className="flex items-start justify-between gap-4 mb-8">
          <div>
            <h1 className="text-3xl font-medium text-white">Models</h1>
            <p className="text-sm text-gray-500 mt-1">
              Shared base model and project adapter lifecycle
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

        <LifecycleError message={error || actionError} />
        {message && (
          <div className="mb-5 p-3 rounded-lg border border-green-500/30 bg-green-500/10 text-green-300 text-sm">
            {message}
          </div>
        )}

        <section className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-8">
          <div className="rounded-xl border border-gray-800 p-5">
            <div className="text-xs text-gray-600 mb-2">Project</div>
            <div className="text-lg text-white">
              {runtime?.project_name || "—"}
            </div>
          </div>
          <div className="rounded-xl border border-gray-800 p-5">
            <div className="flex items-center gap-2 text-xs text-gray-600 mb-2">
              <Cpu size={13} /> Base Model
            </div>
            <div className="text-lg text-white capitalize">
              {displayModelName(runtime?.base_model)}
            </div>
            <div className="text-xs text-gray-600 mt-1 font-mono truncate">
              {runtime?.base_model || "—"}
            </div>
          </div>
          <div className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-5">
            <div className="flex items-center gap-2 text-xs text-sky-500 mb-2">
              <Layers3 size={13} /> Current Active Adapter
            </div>
            <div className="text-lg text-white">
              {runtime?.active_artifact_version || "No active adapter"}
            </div>
            <div className="text-xs text-gray-500 mt-1">Local PEFT runtime</div>
          </div>
        </section>

        <section>
          <div className="flex items-end justify-between mb-4">
            <div>
              <h2 className="text-lg font-medium text-white">
                Artifact History
              </h2>
              <p className="text-xs text-gray-500 mt-1">
                READY adapters can be activated without retraining.
              </p>
            </div>
            <span className="text-xs text-gray-600">
              {artifacts.length} artifacts
            </span>
          </div>

          {loading ? (
            <div className="py-16 text-center">
              <span className="loader" />
            </div>
          ) : artifacts.length === 0 ? (
            <EmptyState>
              No model artifacts yet. Build a dataset and train an adapter on
              the Training page.
            </EmptyState>
          ) : (
            <div className="space-y-3">
              {artifacts.map((artifact) => {
                const active = artifact.id === runtime?.active_artifact_id;
                return (
                  <article
                    key={artifact.id}
                    className={`rounded-xl border p-5 ${
                      active
                        ? "border-sky-500/50 bg-sky-500/5"
                        : "border-gray-800 bg-gray-950/30"
                    }`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-base font-medium text-white">
                            {artifact.artifact_version}
                          </span>
                          <StatusBadge status={artifact.status} />
                          {active && (
                            <span className="inline-flex items-center gap-1 text-xs text-sky-400">
                              <Check size={12} /> Active
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-x-8 gap-y-2 text-xs">
                          <div>
                            <span className="text-gray-600">Evaluation</span>
                            <div className="text-gray-300 mt-0.5">
                              {score(artifact.evaluation_score)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Base model</span>
                            <div className="text-gray-300 mt-0.5 truncate">
                              {displayModelName(artifact.base_model)}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Dataset</span>
                            <div className="text-gray-300 mt-0.5 font-mono truncate">
                              {datasetVersions[artifact.dataset_id] ||
                                artifact.dataset_id}
                            </div>
                          </div>
                          <div>
                            <span className="text-gray-600">Created</span>
                            <div className="text-gray-300 mt-0.5">
                              {formatDate(artifact.created_at)}
                            </div>
                          </div>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => activate(artifact)}
                        disabled={
                          active ||
                          artifact.status !== "READY" ||
                          activating !== null
                        }
                        className="shrink-0 px-4 py-2 rounded-lg text-xs font-medium bg-sky-500 text-white hover:bg-sky-600 disabled:bg-gray-800 disabled:text-gray-500 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {activating === artifact.id && (
                          <Loader2 size={12} className="animate-spin" />
                        )}
                        {active ? "Active" : "Activate"}
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default Models;
