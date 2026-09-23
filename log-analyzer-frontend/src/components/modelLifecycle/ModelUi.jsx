import React from "react";

const STATUS_STYLES = {
  READY: "bg-green-500/15 text-google-green border-google-green/30",
  TRAINED: "bg-purple-500/15 text-purple-700 border-purple-300/30",
  PASSED: "bg-green-500/15 text-google-green border-google-green/30",
  RUNNING: "bg-google-blue/15 text-google-blue border-google-blue/30",
  EVALUATING: "bg-google-blue/10 text-google-blue border-google-blue/30",
  QUEUED: "bg-amber-500/15 text-amber-700 border-amber-300/30",
  VALIDATING: "bg-amber-500/15 text-amber-700 border-amber-300/30",
  FAILED: "bg-red-500/15 text-google-red border-google-red/30",
  ARCHIVED: "bg-google-chip text-google-muted border-google-border",
};

export function StatusBadge({ status, label }) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium ${
        STATUS_STYLES[normalized] ||
        "bg-google-chip text-google-muted border-google-border"
      }`}
    >
      {label || normalized}
    </span>
  );
}

export function formatDate(value, fallback = "—") {
  if (!value) return fallback;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return fallback;
  return date.toLocaleString();
}

export function formatRelativeDate(value, fallback = "Never") {
  if (!value) return fallback;
  const timestamp = new Date(value).getTime();
  if (Number.isNaN(timestamp)) return fallback;
  const elapsed = Date.now() - timestamp;
  const minutes = Math.max(0, Math.floor(elapsed / 60000));
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function displayModelName(model) {
  if (!model) return "Not configured";
  const name = model.split("/").pop();
  return name.replace(/-/g, " ");
}

export function LifecycleError({ message }) {
  if (!message) return null;
  return (
    <div className="mb-5 p-3 rounded-lg border border-google-red/30 bg-red-50 text-google-red text-sm">
      {message}
    </div>
  );
}

export function EmptyState({ children }) {
  return (
    <div className="rounded-lg border border-dashed border-google-border py-10 px-4 text-center text-sm text-google-muted">
      {children}
    </div>
  );
}
