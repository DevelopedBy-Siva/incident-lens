import { useState, useEffect, useCallback } from "react";
import { incidentsAPI, logServerAPI, authAPI } from "../services/api";
import Navbar from "./Navbar";
import IncidentCard from "./IncidentCard";
import ModelStatusPanel from "./modelLifecycle/ModelStatusPanel";
import { useModelLifecycle } from "../hooks/useModelLifecycle";
import {
  RefreshCw,
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
    label: "DB Cascade",
    description: "Pool exhaustion → payment timeout → NPE",
    duration: "~105s",
  },
  {
    id: "memory_leak",
    label: "Memory Leak",
    description: "Heap grows to OOM kill, pod restarts",
    duration: "~185s",
  },
  {
    id: "deployment_gone_wrong",
    label: "Bad Deploy",
    description: "Config change → pool halved → OOM",
    duration: "~85s",
  },
  {
    id: "auth_cascade",
    label: "Auth Cascade",
    description: "Redis down → JWT invalid → rate limit",
    duration: "~75s",
  },
];

function ScenarioButton({ scenario, onRun, running }) {
  return (
    <button
      onClick={() => onRun(scenario.id)}
      disabled={running === scenario.id}
      className="flex flex-col items-start px-3 py-2.5 rounded-lg border border-gray-700 hover:border-sky-500/40 hover:bg-sky-500/5 transition-all text-left disabled:opacity-50 disabled:cursor-not-allowed group"
    >
      <div className="flex items-center gap-1.5 mb-0.5">
        {running === scenario.id ? (
          <Activity size={11} className="text-sky-400 animate-pulse" />
        ) : (
          <Zap
            size={11}
            className="text-gray-500 group-hover:text-sky-400 transition-colors"
          />
        )}
        <span className="text-xs font-medium text-gray-300">
          {scenario.label}
        </span>
        <span className="text-xs text-gray-600">{scenario.duration}</span>
      </div>
      <p className="text-xs text-gray-600 leading-tight">
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
    <div className="min-h-screen bg-black">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 pb-16">
        {/* Title */}
        <div className="mb-8">
          <h1 className="text-3xl font-medium text-white">
            Incident Dashboard
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            IncidentLens — policy-bound autonomous observability agent
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
              className="rounded-lg border border-gray-800 py-5 px-5"
            >
              <div className="text-3xl font-semibold text-white">{value}</div>
              <div className="text-xs text-gray-500 mt-1">{label}</div>
            </div>
          ))}
        </div>

        <ModelStatusPanel
          runtime={modelLifecycle.runtime}
          datasets={modelLifecycle.datasets}
          jobs={modelLifecycle.jobs}
          artifacts={modelLifecycle.artifacts}
          loading={modelLifecycle.loading}
          error={modelLifecycle.error}
        />

        {/* Controls */}
        <div className="rounded-lg border border-gray-800 p-5 mb-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Status
              </label>
              <select
                value={filters.status}
                onChange={(e) =>
                  setFilters({ ...filters, status: e.target.value })
                }
                className="text-sm bg-transparent w-full px-3 py-2 border border-gray-700 rounded-lg text-gray-400 focus:ring-1 focus:ring-sky-500"
              >
                <option value="" className="bg-black">
                  All
                </option>
                <option value="open" className="bg-black">
                  Open
                </option>
                <option value="closed" className="bg-black">
                  Closed
                </option>
                <option value="ignored" className="bg-black">
                  Ignored
                </option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Severity
              </label>
              <select
                value={filters.severity}
                onChange={(e) =>
                  setFilters({ ...filters, severity: e.target.value })
                }
                className="text-sm bg-transparent w-full px-3 py-2 border border-gray-700 rounded-lg text-gray-400 focus:ring-1 focus:ring-sky-500"
              >
                <option value="" className="bg-black">
                  All
                </option>
                <option value="critical" className="bg-black">
                  Critical
                </option>
                <option value="high" className="bg-black">
                  High
                </option>
                <option value="medium" className="bg-black">
                  Medium
                </option>
                <option value="low" className="bg-black">
                  Low
                </option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1.5">
                Search
              </label>
              <input
                type="text"
                value={filters.ticket_title}
                onChange={(e) =>
                  setFilters({ ...filters, ticket_title: e.target.value })
                }
                placeholder="Ticket title..."
                className="text-sm bg-transparent w-full px-3 py-2 border border-gray-700 rounded-lg text-gray-400 placeholder:text-gray-600 focus:ring-1 focus:ring-sky-500"
              />
            </div>
            {isTest && (
              <div className="flex items-end gap-2">
                <button
                  onClick={handleStart}
                  disabled={logStatus === "running"}
                  className="flex-1 px-3 py-2 text-sm bg-sky-500 text-white rounded-lg flex items-center justify-center gap-1.5 hover:bg-sky-600 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
                >
                  <Play size={13} /> Start
                </button>
                <button
                  onClick={handleStop}
                  disabled={logStatus !== "running"}
                  className="flex-1 px-3 py-2 text-sm bg-red-500 text-white rounded-lg flex items-center justify-center gap-1.5 hover:bg-red-600 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
                >
                  <Square size={13} /> Stop
                </button>
              </div>
            )}
          </div>

          {/* Scenario buttons */}
          {isTest && (
            <div>
              <p className="text-xs text-gray-500 mb-2">Demo scenarios</p>
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
                <div className="mt-3 p-2.5 bg-sky-500/10 border border-sky-500/20 rounded-lg text-xs text-sky-300">
                  {scenarioMsg}
                </div>
              )}
            </div>
          )}

          {logServerError && (
            <div className="flex items-start gap-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <AlertCircle className="text-red-400 shrink-0 mt-0.5" size={15} />
              <p className="text-red-300 text-xs">{logServerError}</p>
            </div>
          )}
        </div>

        {/* Incident list */}
        {loading ? (
          <div className="text-center py-12">
            <span className="loader" />
            <p className="text-gray-500 text-sm mt-4">Loading incidents...</p>
          </div>
        ) : incidents.length === 0 ? (
          <div className="text-center py-16 rounded-lg border border-gray-800">
            <AlertTriangle className="text-gray-600 mx-auto mb-3" size={32} />
            <p className="text-gray-500 text-sm">
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

        <div className="fixed bottom-0 left-0 right-0 border-t border-white/5 bg-black/80 backdrop-blur px-4 py-3 text-center text-xs text-gray-600">
          <RefreshCw size={12} className="inline mr-1.5" />
          Auto-refreshing every 5 seconds
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
