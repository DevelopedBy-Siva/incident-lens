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
  critical: "bg-red-500/15 text-red-400 border border-red-500/30",
  high: "bg-orange-500/15 text-orange-400 border border-orange-500/30",
  medium: "bg-yellow-500/15 text-yellow-400 border border-yellow-500/30",
  low: "bg-green-500/15 text-green-400 border border-green-500/30",
};

const DISPOSITION_STYLES = {
  ESCALATE: "bg-red-500/15 text-red-400",
  NEEDS_ONCALL: "bg-orange-500/15 text-orange-400",
  NEEDS_DEV: "bg-yellow-500/15 text-yellow-400",
  OBSERVE: "bg-blue-500/15 text-blue-400",
  NO_ACTION: "bg-gray-500/15 text-gray-400",
};

const OUTCOME_STYLES = {
  resolved: "text-green-400",
  still_firing: "text-red-400",
  pending: "text-yellow-400",
};

function severityClass(s) {
  return SEVERITY_STYLES[s?.toLowerCase()] || SEVERITY_STYLES.low;
}
function dispositionClass(d) {
  return DISPOSITION_STYLES[d?.toUpperCase()] || "bg-gray-500/15 text-gray-400";
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
      sky: "border-sky-500/20",
      purple: "border-purple-500/20",
      amber: "border-amber-500/20",
      green: "border-green-500/20",
      blue: "border-blue-500/20",
    }[accent] || "border-gray-700";

  const text =
    {
      sky: "text-sky-400",
      purple: "text-purple-400",
      amber: "text-amber-400",
      green: "text-green-400",
      blue: "text-blue-400",
    }[accent] || "text-gray-400";

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
      <div className="flex items-center gap-2 text-xs text-gray-500 py-3">
        <Loader2 size={12} className="animate-spin" /> Loading agent trail...
      </div>
    );
  }
  if (error) return <p className="text-xs text-red-400 py-2">{error}</p>;
  if (!data) return null;

  const inv = data.investigation;
  const actions = data.action_log || [];

  return (
    <div className="mt-3 space-y-2">
      <TrailSection icon={BrainCircuit} label="Model information" accent="blue">
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-gray-600">Inference model</div>
            <div className="text-gray-300 mt-0.5 capitalize">
              {displayModelName(modelInfo?.base_model)}
            </div>
          </div>
          <div>
            <div className="text-gray-600">Adapter</div>
            <div className="text-gray-300 mt-0.5">
              {modelInfo?.active_artifact_version || "No active adapter"}
            </div>
          </div>
          <div>
            <div className="text-gray-600">Artifact version</div>
            <div className="text-gray-300 mt-0.5">
              {modelInfo?.active_artifact_version || "—"}
            </div>
          </div>
          <div>
            <div className="text-gray-600">Decision source</div>
            <div className="text-gray-300 mt-0.5">
              {analysisSource === "runbook"
                ? "Runbook"
                : analysisSource
                  ? "Local Runtime"
                  : "Pending"}
            </div>
          </div>
        </div>
        <p className="text-xs text-gray-600 mt-2">
          Current serving configuration at view time.
        </p>
      </TrailSection>

      {inv && (
        <TrailSection icon={Search} label="Evidence gathered" accent="sky">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-gray-900 rounded p-2 text-center">
              <div className="text-white font-medium">
                {inv.evidence.samples}
              </div>
              <div className="text-gray-500">log samples</div>
            </div>
            <div className="bg-gray-900 rounded p-2 text-center">
              <div className="text-white font-medium">
                {inv.evidence.related_incidents}
              </div>
              <div className="text-gray-500">related incidents</div>
            </div>
            <div className="bg-gray-900 rounded p-2 text-center">
              <div className="text-white font-medium">
                {inv.evidence.runbook_matched || "none"}
              </div>
              <div className="text-gray-500">runbook</div>
            </div>
          </div>
        </TrailSection>
      )}

      {inv && (
        <TrailSection
          icon={Terminal}
          label="Investigation loop"
          accent="purple"
        >
          <div className="text-xs space-y-1">
            <div className="flex justify-between text-gray-400">
              <span>Source</span>
              <span
                className={`px-1.5 py-0.5 rounded text-xs ${
                  inv.agent.analysis_source === "runbook"
                    ? "bg-green-500/15 text-green-400"
                    : "bg-purple-500/15 text-purple-400"
                }`}
              >
                {inv.agent.analysis_source === "runbook"
                  ? "Runbook match"
                  : "LLM investigation"}
              </span>
            </div>
            {inv.agent.analysis_source !== "runbook" && (
              <>
                <div className="flex justify-between text-gray-400">
                  <span>Tool call rounds</span>
                  <span className="text-white">{inv.agent.iterations}</span>
                </div>
                {inv.agent.tool_calls?.length > 0 && (
                  <div className="mt-1.5">
                    <div className="text-gray-500 mb-1">Tools called:</div>
                    {inv.agent.tool_calls.map((tc, i) => (
                      <div
                        key={i}
                        className="bg-gray-900 rounded px-2 py-1 mb-1 font-mono text-xs text-gray-300"
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
                  <div className="text-amber-400 text-xs">
                    ↳ Fell back to single-shot analysis
                  </div>
                )}
              </>
            )}
          </div>
        </TrailSection>
      )}

      {inv && (
        <TrailSection icon={Shield} label="Policy decision" accent="amber">
          <div className="text-xs space-y-1">
            <div className="flex justify-between">
              <span className="text-gray-400">Decision</span>
              <span
                className={`font-medium ${inv.policy.allowed ? "text-green-400" : "text-red-400"}`}
              >
                {inv.policy.allowed ? "Allowed" : "Blocked"}
              </span>
            </div>
            <div className="text-gray-500">{inv.policy.reason}</div>
            {inv.policy.effective_disposition &&
              inv.policy.effective_disposition !== inv.result.disposition && (
                <div className="text-amber-400 text-xs">
                  Downgraded: {inv.result.disposition} →{" "}
                  {inv.policy.effective_disposition}
                </div>
              )}
            {inv.policy.tags?.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {inv.policy.tags.map((tag, i) => (
                  <span
                    key={i}
                    className="bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded text-xs"
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
              <div className="flex justify-between text-gray-400">
                <span>{new Date(al.actioned_at).toLocaleTimeString()}</span>
                <span className={OUTCOME_STYLES[al.outcome] || "text-gray-400"}>
                  {al.outcome}
                </span>
              </div>
              {al.actions_taken?.length > 0 ? (
                <div className="flex flex-wrap gap-1 mt-1">
                  {al.actions_taken.map((a, j) => (
                    <span
                      key={j}
                      className="bg-green-500/10 text-green-400 border border-green-500/20 px-1.5 py-0.5 rounded text-xs"
                    >
                      {a}
                    </span>
                  ))}
                </div>
              ) : (
                <span className="text-gray-600 text-xs">
                  no automated actions
                </span>
              )}
            </div>
          ))}
        </TrailSection>
      )}

      {inv?.verifier?.outcome && (
        <TrailSection icon={RotateCcw} label="Verifier outcome" accent="blue">
          <div className="text-xs flex justify-between">
            <span className="text-gray-400">
              {inv.verifier.checked_at
                ? `Checked at ${new Date(inv.verifier.checked_at).toLocaleTimeString()}`
                : "Pending verification"}
            </span>
            <span
              className={
                OUTCOME_STYLES[inv.verifier.outcome] || "text-gray-400"
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
            <div className="text-gray-400 mb-1">
              {data.root_cause_incident.signature}
            </div>
            {data.cause_explanation && (
              <div className="text-gray-500">{data.cause_explanation}</div>
            )}
          </div>
        </TrailSection>
      )}

      {!inv && (
        <p className="text-xs text-gray-600 py-1">
          No investigation run recorded yet — may still be processing.
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
    <div className="rounded-xl border border-gray-800 bg-gray-950/60 overflow-hidden hover:border-gray-700 transition-colors">
      {/* Header */}
      <div className="px-5 pt-5 pb-4">
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`px-2.5 py-1 rounded-full text-xs font-semibold ${severityClass(displaySeverity)}`}
            >
              {displaySeverity.toUpperCase()}
            </span>
            <span className="px-2.5 py-1 bg-gray-800 text-gray-300 rounded text-xs">
              {incident.source}
            </span>
            <span className="px-2.5 py-1 bg-sky-500/15 text-sky-400 border border-sky-500/20 rounded-full text-xs font-bold">
              {incident.count}×
            </span>
            {incident.root_cause_incident_id && (
              <span className="px-2 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded text-xs flex items-center gap-1">
                <GitBranch size={10} /> cascade
              </span>
            )}
          </div>
          <div className="flex items-center text-xs text-gray-500 gap-1">
            <Clock size={11} />
            {fmt(incident.last_seen)}
          </div>
        </div>

        <div className="bg-black/50 border border-gray-800 text-gray-300 p-3 rounded-lg mb-4 overflow-x-auto">
          <code className="text-xs font-mono whitespace-pre-wrap break-all">
            {incident.sample_lines?.[0] || "N/A"}
          </code>
        </div>

        {analysis && (
          <div className="bg-gray-900/60 border border-gray-800 rounded-lg p-4 mb-4">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-start gap-2">
                <AlertCircle
                  className="text-sky-400 shrink-0 mt-0.5"
                  size={15}
                />
                <p className="text-sm text-gray-300 leading-snug">
                  {analysis.summary}
                </p>
              </div>
              <span
                className={`ml-3 shrink-0 px-2 py-0.5 rounded text-xs font-medium ${
                  analysis.analysis_source === "runbook"
                    ? "bg-green-500/15 text-green-400"
                    : "bg-purple-500/15 text-purple-400"
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
              <span className="text-xs text-gray-500">
                {Math.round(analysis.confidence * 100)}% confidence
              </span>
            </div>

            {analysis.next_steps?.length > 0 && (
              <div className="mb-3">
                <p className="text-xs text-gray-500 mb-1.5">Next steps</p>
                <ol className="list-decimal list-inside space-y-1">
                  {analysis.next_steps.slice(0, 3).map((step, i) => (
                    <li key={i} className="text-xs text-gray-400">
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {analysis.ticket_title && (
              <div className="bg-gray-800/60 border border-gray-700 rounded p-3">
                <p className="text-xs text-gray-500 mb-1">Ticket draft</p>
                <p className="text-sm text-gray-200 font-medium mb-1">
                  {analysis.ticket_title}
                </p>
                {analysis.ticket_body && (
                  <p className="text-xs text-gray-500 line-clamp-2">
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
              className="flex-1 px-3 py-2 bg-green-600/80 text-white rounded-lg hover:bg-green-600 transition-colors text-xs font-medium flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
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
              className="flex-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors text-xs font-medium flex items-center justify-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
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
          <div className="text-center py-2 px-4 bg-gray-800/60 rounded-lg">
            <span className="text-xs text-gray-400">
              Status: {incident.status.toUpperCase()}
            </span>
          </div>
        )}
      </div>

      <button
        onClick={() => setTrailOpen((o) => !o)}
        className="w-full flex items-center justify-between px-5 py-2.5 bg-gray-900/40 border-t border-gray-800 hover:bg-gray-900/70 transition-colors text-xs text-gray-500 hover:text-gray-300"
      >
        <span className="flex items-center gap-1.5">
          <Shield size={11} />
          Why did the agent do this?
        </span>
        {trailOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>

      {trailOpen && (
        <div className="px-5 pb-5 pt-3 border-t border-gray-800 bg-gray-900/20">
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
