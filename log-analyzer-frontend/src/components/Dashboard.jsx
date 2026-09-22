import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { incidentsAPI, authAPI } from "../services/api";
import Navbar from "./Navbar";
import IncidentCard from "./IncidentCard";
import { useModelLifecycle } from "../hooks/useModelLifecycle";
import {
  AlertTriangle,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";

const SEVERITY_OPTIONS = ["critical", "high", "medium", "low"];

function SeveritySelect({ value, onChange }) {
  const toggle = (severity) => {
    onChange(
      value.includes(severity)
        ? value.filter((item) => item !== severity)
        : [...value, severity],
    );
  };

  const selectionLabel =
    value.length === 0 || value.length === SEVERITY_OPTIONS.length
      ? "All severities"
      : `${value.length} selected`;

  return (
    <details className="group relative">
      <summary className="flex w-full cursor-pointer list-none items-center justify-between rounded-lg border border-google-border bg-white px-3 py-2 text-sm text-google-muted focus:ring-1 focus:ring-google-blue [&::-webkit-details-marker]:hidden">
        <span>{selectionLabel}</span>
        <ChevronDown
          size={14}
          className="transition-transform group-open:rotate-180"
        />
      </summary>
      <div className="absolute z-50 mt-1 w-full rounded-lg border border-google-border bg-white p-2 shadow-google">
        {SEVERITY_OPTIONS.map((severity) => (
          <label
            key={severity}
            className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 text-sm text-google-text hover:bg-google-hover"
          >
            <input
              type="checkbox"
              checked={value.includes(severity)}
              onChange={() => toggle(severity)}
              className="h-4 w-4 rounded border-google-border text-google-blue focus:ring-google-blue"
            />
            <span className="capitalize">{severity}</span>
          </label>
        ))}
      </div>
    </details>
  );
}

function useDebounce(value, delay) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const h = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(h);
  }, [value, delay]);
  return debounced;
}

function Dashboard() {
  const [incidents, setIncidents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [datadogConfigured, setDatadogConfigured] = useState(null);
  const [filters, setFilters] = useState({
    status: "open",
    severity: ["medium", "high", "critical"],
    search: "",
  });
  const [pagination, setPagination] = useState({
    page: 1,
    pageSize: 10,
    total: 0,
    totalPages: 0,
  });
  const modelLifecycle = useModelLifecycle({ pollWhileActive: true });

  const debouncedFilters = useDebounce(filters, 500);

  useEffect(() => {
    authAPI
      .getMe()
      .then((res) =>
        setDatadogConfigured(res.data?.setup_status?.datadog ?? false),
      )
      .catch(() => {});
  }, []);

  const fetchIncidents = useCallback(async (f, page, pageSize) => {
    try {
      const res = await incidentsAPI.list({
        ...f,
        severity: f.severity.length ? f.severity.join(",") : undefined,
        page,
        page_size: pageSize,
      });
      setIncidents(res.data.items);
      setPagination((current) => ({
        ...current,
        page: res.data.page,
        pageSize: res.data.page_size,
        total: res.data.total,
        totalPages: res.data.total_pages,
      }));
    } catch (e) {
      console.error("fetch incidents:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    fetchIncidents(
      debouncedFilters,
      pagination.page,
      pagination.pageSize,
    );
  }, [
    debouncedFilters,
    pagination.page,
    pagination.pageSize,
    fetchIncidents,
  ]);

  useEffect(() => {
    const t = setInterval(
      () =>
        fetchIncidents(
          debouncedFilters,
          pagination.page,
          pagination.pageSize,
        ),
      5000,
    );
    return () => clearInterval(t);
  }, [
    debouncedFilters,
    pagination.page,
    pagination.pageSize,
    fetchIncidents,
  ]);

  const updateFilters = (changes) => {
    setFilters((current) => ({ ...current, ...changes }));
    setPagination((current) => ({ ...current, page: 1 }));
  };

  const handleClose = async (id) => {
    await incidentsAPI.close(id);
    fetchIncidents(debouncedFilters, pagination.page, pagination.pageSize);
  };
  const handleIgnore = async (id) => {
    await incidentsAPI.ignore(id);
    fetchIncidents(debouncedFilters, pagination.page, pagination.pageSize);
  };

  const stats = {
    total: pagination.total,
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

        {datadogConfigured === false && (
          <div className="mb-8 flex flex-col gap-4 rounded-lg border border-amber-300 bg-amber-50 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-start gap-3">
              <AlertTriangle
                className="mt-0.5 shrink-0 text-amber-700"
                size={18}
              />
              <div>
                <p className="text-sm font-medium text-amber-800">
                  Configuration required
                </p>
                <p className="mt-0.5 text-xs text-amber-700">
                  Go to Settings and configure your log connection to start
                  monitoring incidents.
                </p>
              </div>
            </div>
            <Link
              to="/settings"
              className="shrink-0 rounded-lg border border-amber-400 bg-white px-4 py-2 text-center text-xs font-medium text-amber-800 transition-colors hover:bg-google-bg"
            >
              Configure
            </Link>
          </div>
        )}

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
        <div className="soft-card relative z-20 overflow-visible p-5 mb-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs text-google-muted mb-1.5">
                Status
              </label>
              <select
                value={filters.status}
                onChange={(e) =>
                  updateFilters({ status: e.target.value })
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
              <SeveritySelect
                value={filters.severity}
                onChange={(severity) =>
                  updateFilters({ severity })
                }
              />
            </div>
            <div>
              <label className="block text-xs text-google-muted mb-1.5">
                Search incidents
              </label>
              <input
                type="text"
                value={filters.search}
                onChange={(e) =>
                  updateFilters({ search: e.target.value })
                }
                placeholder="Search incidents"
                className="text-sm bg-white w-full px-3 py-2 border border-google-border rounded-lg text-google-muted placeholder:text-google-muted focus:ring-1 focus:ring-google-blue"
              />
            </div>
          </div>
        </div>

        {/* Incident list */}
        <section className="relative z-0 rounded-2xl border border-google-border bg-google-subtle/90 p-4 sm:p-5">
          <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs text-google-muted">
                {pagination.total} matching incidents
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2 sm:justify-end">
              <label className="flex items-center gap-2 text-xs text-google-muted">
                Per page
                <select
                  value={pagination.pageSize}
                  onChange={(event) =>
                    setPagination((current) => ({
                      ...current,
                      page: 1,
                      pageSize: Number(event.target.value),
                    }))
                  }
                  className="rounded-lg border border-google-border bg-white px-2 py-1.5 text-xs text-google-text focus:ring-1 focus:ring-google-blue"
                >
                  {[5, 10, 20, 50].map((size) => (
                    <option key={size} value={size}>
                      {size}
                    </option>
                  ))}
                </select>
              </label>

              <span className="min-w-20 text-center text-xs text-google-muted">
                Page {pagination.totalPages ? pagination.page : 0} of{" "}
                {pagination.totalPages}
              </span>

              <button
                type="button"
                aria-label="Previous page"
                disabled={pagination.page <= 1}
                onClick={() =>
                  setPagination((current) => ({
                    ...current,
                    page: current.page - 1,
                  }))
                }
                className="rounded-lg border border-google-border bg-white p-1.5 text-google-muted transition-colors hover:bg-google-hover hover:text-google-text disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ChevronLeft size={14} />
              </button>
              <button
                type="button"
                aria-label="Next page"
                disabled={
                  pagination.totalPages === 0 ||
                  pagination.page >= pagination.totalPages
                }
                onClick={() =>
                  setPagination((current) => ({
                    ...current,
                    page: current.page + 1,
                  }))
                }
                className="rounded-lg border border-google-border bg-white p-1.5 text-google-muted transition-colors hover:bg-google-hover hover:text-google-text disabled:cursor-not-allowed disabled:opacity-40"
              >
                <ChevronRight size={14} />
              </button>
            </div>
          </div>

          {loading ? (
            <div className="rounded-xl border border-google-border bg-white py-12 text-center">
              <span className="loader" />
              <p className="mt-4 text-sm text-google-muted">
                Loading incidents...
              </p>
            </div>
          ) : incidents.length === 0 ? (
            <div className="rounded-xl border border-google-border bg-white py-16 text-center">
              <AlertTriangle
                className="mx-auto mb-3 text-google-muted"
                size={32}
              />
              <p className="text-sm text-google-muted">
                {filters.status || filters.severity.length || filters.search
                  ? "No incidents match your filters"
                  : "No incidents yet"}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-3">
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
        </section>

      </div>
      </div>
    </div>
  );
}

export default Dashboard;
