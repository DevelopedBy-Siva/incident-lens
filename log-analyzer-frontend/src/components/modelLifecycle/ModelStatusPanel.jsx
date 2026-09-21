import React from "react";
import { Cpu, Database, Layers3, Workflow } from "lucide-react";
import { displayModelName, formatRelativeDate, StatusBadge } from "./ModelUi";

export default function ModelStatusPanel({
  runtime,
  datasets,
  jobs,
  artifacts,
  loading,
  error,
}) {
  const activeArtifact = artifacts.find(
    (artifact) => artifact.id === runtime?.active_artifact_id,
  );
  const latestJob = jobs[0];
  const items = [
    {
      label: "Base Model",
      value: displayModelName(runtime?.base_model),
      icon: Cpu,
    },
    {
      label: "Current Adapter",
      value: activeArtifact?.artifact_version || "No active adapter",
      icon: Layers3,
    },
    { label: "Inference", value: "Local", icon: Workflow },
    {
      label: "Training Status",
      value: latestJob?.status || "No jobs",
      icon: Workflow,
      status: latestJob?.status,
    },
    { label: "Datasets", value: datasets.length, icon: Database },
    { label: "Training Jobs", value: jobs.length, icon: Workflow },
    {
      label: "Latest Training",
      value: formatRelativeDate(
        latestJob?.finished_at || latestJob?.created_at,
      ),
      icon: Workflow,
    },
  ];

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-950/40 p-5 mb-8">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-medium text-white">AI Model Status</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Current project training and local inference state
          </p>
        </div>
        {activeArtifact ? (
          <StatusBadge status="READY" label="Serving" />
        ) : (
          <StatusBadge status="QUEUED" label="Adapter required" />
        )}
      </div>

      {error ? (
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-xs text-red-300">
          {error}
        </div>
      ) : loading ? (
        <div className="h-20 flex items-center justify-center">
          <span className="loader" />
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {items.map(({ label, value, icon: Icon, status }) => (
            <div key={label} className="rounded-lg border border-gray-800 p-3">
              <Icon size={13} className="text-sky-500 mb-2" />
              {status ? (
                <StatusBadge status={status} />
              ) : (
                <div className="text-sm text-gray-200 truncate" title={value}>
                  {value}
                </div>
              )}
              <div className="text-xs text-gray-600 mt-1">{label}</div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
