import { useState, useEffect, useCallback } from "react";
import { incidentsAPI, logServerAPI, authAPI } from "../services/api";
import Navbar from "./Navbar";
import IncidentCard from "./IncidentCard";
import { useModelLifecycle } from "../hooks/useModelLifecycle";
import {
  AlertTriangle,
  Play,
  Square,
  AlertCircle,
  Zap,
  Activity,
} from "lucide-react";

function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const h = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(h);
  }, [value, delay]);
  return debounced;
}

const SCENARIOS = [
  {
    id: "db_cascade",
    label: "Database Failure",
    description: "Simulates database pressure that disrupts payments",
    duration: "~105s",
  },
  {
    id: "memory_leak",
    label: "Memory Leak",
    description: "Simulates rising memory use until a service restarts",
    duration: "~185s",
  },
  {
    id: "deployment_gone_wrong",
    label: "Bad Deploy",
    description: "Simulates a deployment that reduces capacity and fails",
    duration: "~85s",
  },
  {
    id: "auth_cascade",
    label: "Authentication Failure",
    description: "Simulates an authentication outage and rate limiting",
    duration: "~75s",
  },
];

function ScenarioButton({ scenario, onRun, running }) {
  return (
    <button
      onClick={() => onRun(scenario.id)}
      disabled={running === scenario.id}
      className="flex flex-col items-start px-3 py-2.5 rounded-lg border border-google-border hover:border-google-blue/40 hover:bg-google-blue/5 transition-all text-left disabled:opacity-50 disabled:cursor-not-allowed group"
    >
      <div className="flex items-center gap-1.5 mb-0.5">
        {running === scenario.id ? (
          <Activity size={11} className="text-google-blue animate-pulse" />
        ) : (
          <Zap
            size={11}
            className="text-google-muted group-hover:text-google-blue transition-colors"
          />
        )}
        <span className="text-xs font-medium text-google-text">
          {scenario.label}
        </span>
        <span className="text-xs text-google-muted">{scenario.duration}</span>
      </div>
      <p className="text-xs text-google-muted leading-tight">
        {scenario.description}
      </p>
    </button>
  );
}

function Dashboard() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isTest, setIsTest] = useState(false);
  const [logStatus, setLogStatus] = useState("unknown");
  const [logServerError, setLogServerError] = useState("");
  const [runningScenario, setRunningScenario] = useState(null);
  const [scenarioMsg, setScenarioMsg] = useState("");
  const [filters, setFilters] = useState({
    status: "open",
    severity: "",
    ticket_title: "",
  });
  const modelLifecycle = useModelLifecycle({ pollWhileActive: true });

  const debouncedFilters = useDebounce(filters, 500);

  useEffect(() => {
    authAPI
      .getMe()
      .then((res) => setIsTest(res.data?.is_test ?? false))
      .catch(() => {});
  }, []);

  const fetchIncidents = useCallback(async (f) => {
    try {
      const res = await incidentsAPI.list(f);
      setIncidents(res.data);
    } catch (e) {
      console.error("fetch incidents:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchIncidents(debouncedFilters);
  }, [debouncedFilters, fetchIncidents]);

  useEffect(() => {
    const t = setInterval(() => fetchIncidents(debouncedFilters), 5000);
    return () => clearInterval(t);
  }, [debouncedFilters, fetchIncidents]);

  useEffect(() => {
    if (!isTest) return;
    let alive = true;
    const poll = async () => {
      try {
        const res = await logServerAPI.status();
        if (alive) setLogStatus(res.data?.status || "unknown");
      } catch {
        if (alive) setLogStatus("unknown");
      }
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [isTest]);

  const handleStart = async () => {
    setLogServerError("");
    try {
      await logServerAPI.start();
      setLogStatus("running");
    } catch (err) {
      setLogServerError(
        err.response?.data?.detail || "Failed to start log server.",
      );
      setLogStatus("idle");
    }
  };

  const handleStop = async () => {
    setLogServerError("");
    try {
      await logServerAPI.stop();
      setLogStatus("idle");
    } catch {
      setLogServerError("Failed to stop log server.");
    }
  };

  const handleRunScenario = async (name) => {
    setRunningScenario(name);
    setScenarioMsg("");
    try {
      const res = await logServerAPI.runScenario(name);
      const est = res.data?.estimated_duration_seconds;
      setScenarioMsg(
        `Scenario "${name}" started — ${res.data?.steps} log entries over ~${est}s. Watch incidents appear below.`,
      );
      setTimeout(() => setRunningScenario(null), (est || 120) * 1000);
    } catch (err) {
      setScenarioMsg(err.response?.data?.detail || "Failed to run scenario.");
      setRunningScenario(null);
    }
  };

  const handleClose = async (id) => {
    await incidentsAPI.close(id);
    fetchIncidents(debouncedFilters);
  };
  const handleIgnore = async (id) => {
    await incidentsAPI.ignore(id);
    fetchIncidents(debouncedFilters);
  };

  const stats = {
    total: incidents.length,
    totalEvents: incidents.reduce((s, i) => s + i.count, 0),
    highFreq: incidents.filter((i) => i.count >= 5).length,
    analyzed: incidents.filter((i) => i.analysis).length,
  };

  return (
    <div className="app-shell">
      <div className="app-canvas">
      <Navbar />
      <div className="max-w-[96rem] mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-16">
        {/* Title */}
        <div className="mb-12">
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.04em] text-google-text">
            Incident Dashboard
          </h1>
          <p className="text-sm text-google-muted mt-2">
            Monitor active incidents, investigate their causes, and track the
            response.
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
          {[
            { label: "Active incidents", value: stats.total },
            { label: "Total events", value: stats.totalEvents },
            { label: "High frequency", value: stats.highFreq },
            { label: "Agent analyzed", value: stats.analyzed },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="soft-card py-6 px-6"
            >
              <div className="text-4xl sm:text-5xl font-semibold tracking-[-0.04em] text-google-text">
                {value}
              </div>
              <div className="text-xs font-medium text-google-muted mt-3">
                {label}
              </div>
            </div>
          ))}
        </div>

        {/* Controls */}
        <div className="soft-card p-5 mb-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs text-google-muted mb-1.5">
                Status
              </label>
              <select
                value={filters.status}
                onChange={(e) =>
                  setFilters({ ...filters, status: e.target.value })
                }
                className="text-sm bg-white w-full px-3 py-2 border border-google-border rounded-lg text-google-muted focus:ring-1 focus:ring-google-blue"
              >
                <option value="" className="bg-google-bg">
                  All
                </option>
                <option value="open" className="bg-google-bg">
                  Open
                </option>
                <option value="closed" className="bg-google-bg">
                  Closed
                </option>
                <option value="ignored" className="bg-google-bg">
                  Ignored
                </option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-google-muted mb-1.5">
                Severity
              </label>
              <select
                value={filters.severity}
                onChange={(e) =>
                  setFilters({ ...filters, severity: e.target.value })
                }
                className="text-sm bg-white w-full px-3 py-2 border border-google-border rounded-lg text-google-muted focus:ring-1 focus:ring-google-blue"
              >
                <option value="" className="bg-google-bg">
                  All
                </option>
                <option value="critical" className="bg-google-bg">
                  Critical
                </option>
                <option value="high" className="bg-google-bg">
                  High
                </option>
                <option value="medium" className="bg-google-bg">
                  Medium
                </option>
                <option value="low" className="bg-google-bg">
                  Low
                </option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-google-muted mb-1.5">
                Search
              </label>
              <input
                type="text"
                value={filters.ticket_title}
                onChange={(e) =>
                  setFilters({ ...filters, ticket_title: e.target.value })
                }
                placeholder="Search incident titles"
                className="text-sm bg-white w-full px-3 py-2 border border-google-border rounded-lg text-google-muted placeholder:text-google-muted focus:ring-1 focus:ring-google-blue"
              />
            </div>
            {isTest && (
              <div className="flex items-end gap-2">
                <button
                  onClick={handleStart}
                  disabled={logStatus === "running"}
                  className="flex-1 px-3 py-2 text-sm bg-google-blue text-white rounded-lg flex items-center justify-center gap-1.5 hover:bg-google-blue-dark disabled:bg-google-border disabled:cursor-not-allowed transition-colors"
                >
                  <Play size={13} /> Start
                </button>
                <button
                  onClick={handleStop}
                  disabled={logStatus !== "running"}
                  className="flex-1 px-3 py-2 text-sm bg-google-red text-white rounded-lg flex items-center justify-center gap-1.5 hover:bg-red-700 disabled:bg-google-border disabled:cursor-not-allowed transition-colors"
                >
                  <Square size={13} /> Stop
                </button>
              </div>
            )}
          </div>

          {/* Scenario buttons */}
          {isTest && (
            <div>
              <p className="text-xs text-google-muted mb-2">
                Test with a simulated incident
              </p>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {SCENARIOS.map((s) => (
                  <ScenarioButton
                    key={s.id}
                    scenario={s}
                    onRun={handleRunScenario}
                    running={runningScenario}
                  />
                ))}
              </div>
              {scenarioMsg && (
                <div className="mt-3 p-2.5 bg-google-blue/10 border border-google-blue/20 rounded-lg text-xs text-google-blue">
                  {scenarioMsg}
                </div>
              )}
            </div>
          )}

          {logServerError && (
            <div className="flex items-start gap-3 p-3 bg-red-50 border border-google-red/30 rounded-lg">
              <AlertCircle className="text-google-red shrink-0 mt-0.5" size={15} />
              <p className="text-google-red text-xs">{logServerError}</p>
            </div>
          )}
        </div>

        {/* Incident list */}
        {loading ? (
          <div className="text-center py-12">
            <span className="loader" />
            <p className="text-google-muted text-sm mt-4">Loading incidents...</p>
          </div>
        ) : incidents.length === 0 ? (
          <div className="text-center py-16 rounded-lg border border-google-border">
            <AlertTriangle className="text-google-muted mx-auto mb-3" size={32} />
            <p className="text-google-muted text-sm">
              {filters.status || filters.severity || filters.ticket_title
                ? "No incidents match your filters"
                : "No incidents yet — start the log server or run a scenario"}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4">
            {incidents.map((incident) => (
              <IncidentCard
                key={incident.id}
                incident={incident}
                analysis={incident.analysis}
                modelInfo={modelLifecycle.runtime}
                onClose={handleClose}
                onIgnore={handleIgnore}
              />
            ))}
          </div>
        )}

      </div>
      </div>
    </div>
  );
}

export default Dashboard;
