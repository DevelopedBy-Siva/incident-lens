import React, { useState } from "react";
import {
  AlertCircle,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronUp,
  Search,
  Shield,
  Zap,
  RotateCcw,
  GitBranch,
  Terminal,
  BrainCircuit,
} from "lucide-react";
import { incidentsAPI } from "../services/api";
import { displayModelName } from "./modelLifecycle/ModelUi";

const SEVERITY_STYLES = {
  critical: "bg-red-500/15 text-google-red border border-google-red/30",
  high: "bg-orange-500/15 text-orange-400 border border-orange-500/30",
  medium: "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  low: "bg-green-500/15 text-google-green border border-google-green/30",
};

const DISPOSITION_STYLES = {
  ESCALATE: "bg-red-500/15 text-google-red",
  NEEDS_ONCALL: "bg-orange-500/15 text-orange-400",
  NEEDS_DEV: "bg-yellow-500/15 text-yellow-400",
  OBSERVE: "bg-google-blue/10 text-google-blue",
  NO_ACTION: "bg-gray-500/15 text-google-muted",
};

const OUTCOME_STYLES = {
  resolved: "text-google-green",
  still_firing: "text-google-red",
  pending: "text-yellow-400",
};

function severityClass(s) {
  return SEVERITY_STYLES[s?.toLowerCase()] || SEVERITY_STYLES.low;
}
function dispositionClass(d) {
  return DISPOSITION_STYLES[d?.toUpperCase()] || "bg-gray-500/15 text-google-muted";
}
function fallbackSeverity(count) {
  if (count >= 10) return "critical";
  if (count >= 5) return "high";
  if (count >= 2) return "medium";
  return "low";
}
function fmt(ts) {
  return new Date(ts).toLocaleTimeString();
}

function TrailSection({ icon: Icon, label, children, accent = "sky" }) {
  const border =
    {
      sky: "border-google-blue/20",
      purple: "border-google-blue/20",
      amber: "border-amber-300/20",
      green: "border-google-green/20",
      blue: "border-google-blue/20",
    }[accent] || "border-google-border";

  const text =
    {
      sky: "text-google-blue",
      purple: "text-google-blue",
      amber: "text-amber-700",
      green: "text-google-green",
      blue: "text-google-blue",
    }[accent] || "text-google-muted";

  return (
    <div className={`border ${border} rounded-lg p-3 mb-2`}>
      <div
        className={`flex items-center gap-1.5 text-xs font-medium ${text} mb-2`}
      >
        <Icon size={12} />
        {label}
      </div>
      {children}
    </div>
  );
}

function AgentTrail({ incidentId, modelInfo, analysisSource }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  React.useEffect(() => {
    let active = true;
    setLoading(true);
    incidentsAPI
      .getInvestigation(incidentId)
      .then((response) => {
        if (active) setData(response.data);
      })
      .catch(() => {
        if (active) setError("Could not load investigation trail");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [incidentId]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-xs text-google-muted py-3">
        <Loader2 size={12} className="animate-spin" /> Loading agent trail...
      </div>
    );
  }
  if (error) return <p className="text-xs text-google-red py-2">{error}</p>;
  if (!data) return null;

  const inv = data.investigation;
  const actions = data.action_log || [];

  return (
    <div className="mt-3 space-y-2">
      <TrailSection icon={BrainCircuit} label="Model information" accent="blue">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-google-muted">Inference model</div>
            <div className="text-google-text mt-0.5 capitalize">
              {displayModelName(modelInfo?.base_model)}
            </div>
          </div>
          <div>
            <div className="text-google-muted">Custom model</div>
            <div className="text-google-text mt-0.5">
              {modelInfo?.active_artifact_version || "Not used"}
            </div>
          </div>
          <div>
            <div className="text-google-muted">Custom model version</div>
            <div className="text-google-text mt-0.5">
              {modelInfo?.active_artifact_version || "—"}
            </div>
          </div>
          <div>
            <div className="text-google-muted">Decision source</div>
            <div className="text-google-text mt-0.5">
              {analysisSource === "runbook"
                ? "Runbook"
                : analysisSource
                  ? "Local Runtime"
                  : "Pending"}
            </div>
          </div>
        </div>
        <p className="text-xs text-google-muted mt-2">
          The model setup used to analyze this incident.
        </p>
      </TrailSection>

      {inv && (
        <TrailSection icon={Search} label="Evidence gathered" accent="sky">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-google-subtle rounded p-2 text-center">
              <div className="text-google-text font-medium">
                {inv.evidence.samples}
              </div>
              <div className="text-google-muted">log samples</div>
            </div>
            <div className="bg-google-subtle rounded p-2 text-center">
              <div className="text-google-text font-medium">
                {inv.evidence.related_incidents}
              </div>
              <div className="text-google-muted">related incidents</div>
            </div>
            <div className="bg-google-subtle rounded p-2 text-center">
              <div className="text-google-text font-medium">
                {inv.evidence.runbook_matched || "none"}
              </div>
              <div className="text-google-muted">runbook</div>
            </div>
          </div>
        </TrailSection>
      )}

      {inv && (
        <TrailSection
          icon={Terminal}
          label="Analysis details"
          accent="purple"
        >
          <div className="text-xs space-y-1">
            <div className="flex justify-between text-google-muted">
              <span>Source</span>
              <span
                className={`px-1.5 py-0.5 rounded text-xs ${
                  inv.agent.analysis_source === "runbook"
                    ? "bg-green-500/15 text-google-green"
                    : "bg-google-blue/10 text-google-blue"
                }`}
              >
                {inv.agent.analysis_source === "runbook"
                  ? "Runbook match"
                  : "AI analysis"}
              </span>
            </div>
            {inv.agent.analysis_source !== "runbook" && (
              <>
                <div className="flex justify-between text-google-muted">
                  <span>Analysis steps</span>
                  <span className="text-google-text">{inv.agent.iterations}</span>
                </div>
                {inv.agent.tool_calls?.length > 0 && (
                  <div className="mt-1.5">
                    <div className="text-google-muted mb-1">
                      Checks performed:
                    </div>
                    {inv.agent.tool_calls.map((tc, i) => (
                      <div
                        key={i}
                        className="bg-google-subtle rounded px-2 py-1 mb-1 font-mono text-xs text-google-text"
                      >
                        {tc.tool}(
                        {Object.entries(tc.args || {})
                          .map(([k, v]) => `${k}=${v}`)
                          .join(", ")}
                        )
                      </div>
                    ))}
                  </div>
                )}
                {inv.agent.fallback_used && (
                  <div className="text-amber-700 text-xs">
                    A simplified analysis was used because the full
                    investigation could not finish.
                  </div>
                )}
              </>
            )}
          </div>
        </TrailSection>
      )}

      {inv && (
        <TrailSection icon={Shield} label="Safety check" accent="amber">
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-google-muted">Decision</span>
              <span
                className={`font-medium ${inv.policy.allowed ? "text-google-green" : "text-google-red"}`}
              >
                {inv.policy.allowed ? "Approved" : "Stopped"}
              </span>
            </div>
            <div className="text-google-muted">{inv.policy.reason}</div>
            {inv.policy.effective_disposition &&
              inv.policy.effective_disposition !== inv.result.disposition && (
                <div className="text-amber-700 text-xs">
                  Action adjusted: {inv.result.disposition} →{" "}
                  {inv.policy.effective_disposition}
                </div>
              )}
            {inv.policy.tags?.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {inv.policy.tags.map((tag, i) => (
                  <span
                    key={i}
                    className="bg-google-chip text-google-muted px-1.5 py-0.5 rounded text-xs"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </TrailSection>
      )}

      {/* Actions taken */}
      {actions.length > 0 && (
        <TrailSection icon={Zap} label="Actions taken" accent="green">
          {actions.map((al, i) => (
            <div key={i} className="text-xs mb-1">
              <div className="flex justify-between text-google-muted">
                <span>{new Date(al.actioned_at).toLocaleTimeString()}</span>
                <span className={OUTCOME_STYLES[al.outcome] || "text-google-muted"}>
                  {al.outcome}
                </span>
              </div>
              {al.actions_taken?.length > 0 ? (
                <div className="flex flex-wrap gap-1 mt-1">
                  {al.actions_taken.map((a, j) => (
                    <span
                      key={j}
                      className="bg-green-50 text-google-green border border-google-green/20 px-1.5 py-0.5 rounded text-xs"
                    >
                      {a}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-google-muted text-xs">
                  no automated actions
                </span>
              )}
            </div>
          ))}
        </TrailSection>
      )}

      {inv?.verifier?.outcome && (
        <TrailSection icon={RotateCcw} label="Follow-up check" accent="blue">
          <div className="text-xs flex justify-between">
            <span className="text-google-muted">
              {inv.verifier.checked_at
                ? `Checked at ${new Date(inv.verifier.checked_at).toLocaleTimeString()}`
                : "Pending verification"}
            </span>
            <span
              className={
                OUTCOME_STYLES[inv.verifier.outcome] || "text-google-muted"
              }
            >
              {inv.verifier.outcome}
            </span>
          </div>
        </TrailSection>
      )}

      {data.root_cause_incident && (
        <TrailSection icon={GitBranch} label="Root cause" accent="amber">
          <div className="text-xs">
            <div className="text-google-muted mb-1">
              {data.root_cause_incident.signature}
            </div>
            {data.cause_explanation && (
              <div className="text-google-muted">{data.cause_explanation}</div>
            )}
          </div>
        </TrailSection>
      )}

      {!inv && (
        <p className="text-xs text-google-muted py-1">
          Analysis has not started yet or is still in progress.
        </p>
      )}
    </div>
  );
}

function IncidentCard({ incident, analysis, modelInfo, onClose, onIgnore }) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [trailOpen, setTrailOpen] = useState(false);

  const displaySeverity =
    analysis?.severity || fallbackSeverity(incident.count);

  const handleClose = async () => {
    setIsProcessing(true);
    try {
      await onClose(incident.id);
    } finally {
      setIsProcessing(false);
    }
  };
  const handleIgnore = async () => {
    setIsProcessing(true);
    try {
      await onIgnore(incident.id);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="soft-card overflow-hidden hover:-translate-y-0.5 hover:shadow-google transition-all">
      {/* Header */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`px-2.5 py-1 rounded-full text-xs font-semibold ${severityClass(displaySeverity)}`}
            >
              {displaySeverity.toUpperCase()}
            </span>
            <span className="px-2.5 py-1 bg-google-chip text-google-muted rounded text-xs">
              {incident.source}
            </span>
            <span className="px-2.5 py-1 bg-google-blue/10 text-google-blue border border-google-blue/20 rounded-full text-xs font-semibold">
              {incident.count}×
            </span>
            {incident.root_cause_incident_id && (
              <span className="px-2 py-1 bg-amber-50 text-amber-700 border border-amber-300/20 rounded text-xs flex items-center gap-1">
                <GitBranch size={10} /> cascade
              </span>
            )}
          </div>
          <div className="flex items-center text-xs text-google-muted gap-1">
            <Clock size={11} />
            {fmt(incident.last_seen)}
          </div>
        </div>

        <div className="bg-google-bg/50 border border-google-border text-google-text p-3 rounded-lg mb-4 overflow-x-auto">
          <code className="text-xs font-mono whitespace-pre-wrap break-all">
            {incident.sample_lines?.[0] || "N/A"}
          </code>
        </div>

        {analysis && (
          <div className="bg-white border border-google-border rounded-lg p-4 mb-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-start gap-2">
                <AlertCircle
                  className="text-google-blue shrink-0 mt-0.5"
                  size={15}
                />
                <p className="text-sm text-google-text leading-snug">
                  {analysis.summary}
                </p>
              </div>
              <span
                className={`ml-3 shrink-0 px-2 py-0.5 rounded text-xs font-medium ${
                  analysis.analysis_source === "runbook"
                    ? "bg-green-500/15 text-google-green"
                    : "bg-google-blue/10 text-google-blue"
                }`}
              >
                {analysis.analysis_source === "runbook" ? "Runbook" : "AI"}
              </span>
            </div>

            <div className="flex items-center gap-2 mb-3">
              <span
                className={`px-2 py-0.5 rounded text-xs font-semibold ${dispositionClass(analysis.disposition)}`}
              >
                {analysis.disposition}
              </span>
              <span className="text-xs text-google-muted">
                {Math.round(analysis.confidence * 100)}% confidence
              </span>
            </div>

            {analysis.next_steps?.length > 0 && (
              <div className="mb-3">
                <p className="text-xs text-google-muted mb-1.5">Next steps</p>
                <ol className="list-decimal list-inside space-y-1">
                  {analysis.next_steps.slice(0, 3).map((step, i) => (
                    <li key={i} className="text-xs text-google-muted">
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {analysis.ticket_title && (
              <div className="bg-google-chip border border-google-border rounded p-3">
                <p className="text-xs text-google-muted mb-1">Ticket draft</p>
                <p className="text-sm text-google-text font-medium mb-1">
                  {analysis.ticket_title}
                </p>
                {analysis.ticket_body && (
                  <p className="text-xs text-google-muted line-clamp-2">
                    {analysis.ticket_body}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {incident.status === "open" ? (
          <div className="flex gap-2">
            <button
              onClick={handleClose}
              disabled={isProcessing}
              className="flex-1 px-3 py-2 bg-google-green text-white rounded-lg hover:bg-green-700 transition-colors text-xs font-medium flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <CheckCircle size={12} />
              )}
              Close
            </button>
            <button
              onClick={handleIgnore}
              disabled={isProcessing}
              className="flex-1 px-3 py-2 bg-white text-google-muted border border-google-border rounded-lg hover:bg-google-hover transition-colors text-xs font-medium flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <Loader2 size={12} className="animate-spin" />
              ) : (
                <XCircle size={12} />
              )}
              Ignore
            </button>
          </div>
        ) : (
          <div className="text-center py-2 px-4 bg-google-chip rounded-lg">
            <span className="text-xs text-google-muted">
              Status: {incident.status.toUpperCase()}
            </span>
          </div>
        )}
      </div>

      <button
        onClick={() => setTrailOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-2.5 bg-google-subtle border-t border-google-border hover:bg-google-hover transition-colors text-xs text-google-muted hover:text-google-text"
      >
        <span className="flex items-center gap-1.5">
          <Shield size={11} />
          Why did the agent do this?
        </span>
        {trailOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {trailOpen && (
        <div className="px-5 pb-5 pt-3 border-t border-google-border bg-google-subtle">
          <AgentTrail
            incidentId={incident.id}
            modelInfo={modelInfo}
            analysisSource={analysis?.analysis_source}
          />
        </div>
      )}
    </div>
  );
}

export default IncidentCard;
