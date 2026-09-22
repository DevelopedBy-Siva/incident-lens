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
        `${artifact.artifact_version} will now analyze new incidents.`,
      );
    } catch (requestError) {
      setActionError(
        errorMessage(requestError, "Could not switch to this model."),
      );
    } finally {
      setActivating(null);
    }
  };

  return (
    <div className="app-shell">
      <div className="app-canvas">
        <Navbar />
      <main className="max-w-[96rem] mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-16">
        <div className="flex items-start justify-between gap-4 mb-12">
          <div>
            <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.04em] text-google-text">Models</h1>
            <p className="text-sm text-google-muted mt-2">
              Review trained models and choose which one analyzes new incidents.
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

        <LifecycleError message={error || actionError} />
        {message && (
          <div className="mb-5 p-3 rounded-lg border border-google-green/30 bg-green-50 text-google-green text-sm">
            {message}
          </div>
        )}

        <section className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-8">
          <div className="soft-card p-5">
            <div className="text-xs text-google-muted mb-2">Project</div>
            <div className="text-lg text-google-text">
              {runtime?.project_name || "—"}
            </div>
          </div>
          <div className="soft-card p-5">
            <div className="flex items-center gap-2 text-xs text-google-muted mb-2">
              <Cpu size={13} /> Base Model
            </div>
            <div className="text-lg text-google-text capitalize">
              {displayModelName(runtime?.base_model)}
            </div>
            <div className="text-xs text-google-muted mt-1 font-mono truncate">
              {runtime?.base_model || "—"}
            </div>
          </div>
          <div className="rounded-xl border border-google-blue/30 bg-google-blue/5 p-5">
            <div className="flex items-center gap-2 text-xs text-google-blue mb-2">
              <Layers3 size={13} /> Active Custom Model
            </div>
            <div className="text-lg text-google-text">
              {runtime?.active_artifact_version || "Using base model only"}
            </div>
            <div className="text-xs text-google-muted mt-1">
              Used to analyze new incidents
            </div>
          </div>
        </section>

        <section>
          <div className="flex items-end justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-google-text">
                Trained Model History
              </h2>
              <p className="text-xs text-google-muted mt-1">
                Review trained models and choose which one should analyze new
                incidents.
              </p>
            </div>
            <span className="text-xs text-google-muted">
              {artifacts.length} {artifacts.length === 1 ? "model" : "models"}
            </span>
          </div>

          {loading ? (
            <div className="py-16 text-center">
              <span className="loader" />
            </div>
          ) : artifacts.length === 0 ? (
            <EmptyState>
              No trained models yet. Go to Training to create a dataset and
              train your first model.
            </EmptyState>
          ) : (
            <div className="space-y-3">
              {artifacts.map((artifact) => {
                const active = artifact.id === runtime?.active_artifact_id;
                return (
                  <article
                    key={artifact.id}
                    className={`soft-card p-5 ${
                      active
                        ? "border-google-blue/50 bg-google-blue/5"
                        : "border-google-border bg-white"
                    }`}
                  >
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-base font-medium text-google-text">
                            {artifact.artifact_version}
                          </span>
                          <StatusBadge status={artifact.status} />
                          {active && (
                            <span className="inline-flex items-center gap-1 text-xs text-google-blue">
                              <Check size={12} /> Active
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-x-8 gap-y-2 text-xs">
                          <div>
                            <span className="text-google-muted">Quality score</span>
                            <div className="text-google-text mt-0.5">
                              {score(artifact.evaluation_score)}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Base model</span>
                            <div className="text-google-text mt-0.5 truncate">
                              {displayModelName(artifact.base_model)}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Dataset</span>
                            <div className="text-google-text mt-0.5 font-mono truncate">
                              {datasetVersions[artifact.dataset_id] ||
                                artifact.dataset_id}
                            </div>
                          </div>
                          <div>
                            <span className="text-google-muted">Created</span>
                            <div className="text-google-text mt-0.5">
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
                        className="shrink-0 px-4 py-2 rounded-lg text-xs font-medium bg-google-blue text-white hover:bg-google-blue-dark disabled:bg-google-chip disabled:text-google-muted disabled:cursor-not-allowed flex items-center justify-center gap-2"
                      >
                        {activating === artifact.id && (
                          <Loader2 size={12} className="animate-spin" />
                        )}
                        {active ? "In use" : "Use model"}
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
    </div>
  );
}

export default Models;
