import React, { useState, useEffect } from "react";
import { authAPI, modelLifecycleAPI } from "../services/api";
import Navbar from "./Navbar";
import {
  Save,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  Cpu,
} from "lucide-react";

const HIDDEN_MARKER = "HIDDEN: TEST CREDENTIAL";

function SectionHeader({ title, description, configured, open, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="w-full flex items-center justify-between py-3 text-left"
    >
      <div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200">{title}</span>
          {configured ? (
            <span className="text-xs px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full border border-green-500/30">
              Configured
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 bg-amber-500/20 text-amber-400 rounded-full border border-amber-500/30">
              Not set
            </span>
          )}
        </div>
        {description && (
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        )}
      </div>
      {open ? (
        <ChevronUp size={16} className="text-gray-400" />
      ) : (
        <ChevronDown size={16} className="text-gray-400" />
      )}
    </button>
  );
}

function SecretInput({
  label,
  name,
  value,
  onChange,
  placeholder,
  disabled,
  hint,
}) {
  const [show, setShow] = useState(false);
  const isHidden = value === HIDDEN_MARKER || value === "••••••";

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1.5">{label}</label>
      <div className="relative">
        <input
          type={show && !isHidden ? "text" : "password"}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full px-4 py-2.5 pr-10 bg-transparent text-gray-100 border border-gray-700 rounded-lg text-sm disabled:text-gray-500 disabled:cursor-not-allowed placeholder:text-gray-600 focus:ring-1 focus:ring-sky-500 focus:border-sky-500"
        />
        {!disabled && (
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
          >
            {show ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        )}
      </div>
      {hint && <p className="text-xs text-gray-600 mt-1">{hint}</p>}
    </div>
  );
}

function PlainInput({
  label,
  name,
  value,
  onChange,
  placeholder,
  disabled,
  hint,
  type = "text",
}) {
  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1.5">{label}</label>
      <input
        type={type}
        name={name}
        value={value || ""}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full px-4 py-2.5 bg-transparent text-gray-100 border border-gray-700 rounded-lg text-sm disabled:text-gray-500 disabled:cursor-not-allowed placeholder:text-gray-600 focus:ring-1 focus:ring-sky-500 focus:border-sky-500"
      />
      {hint && <p className="text-xs text-gray-600 mt-1">{hint}</p>}
    </div>
  );
}

function RuntimeValue({ label, value, mono = false }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-950/50 p-3">
      <div className="text-xs text-gray-600">{label}</div>
      <div
        className={`text-sm text-gray-300 mt-1 break-all ${mono ? "font-mono" : ""}`}
      >
        {value || "—"}
      </div>
    </div>
  );
}

function Settings() {
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [isTest, setIsTest] = useState(false);
  const [runtime, setRuntime] = useState(null);
  const [openSections, setOpenSections] = useState({
    datadog: true,
    runtime: true,
    observability: false,
    notifications: false,
    security: false,
  });
  const [setupStatus, setSetupStatus] = useState({});

  const [form, setForm] = useState({
    datadog_api_key: "",
    datadog_app_key: "",
    datadog_site: "datadoghq.com",
    datadog_query: "status:(error OR warn OR critical)",
    datadog_environment: "prod",
    datadog_service: "",
    langfuse_public_key: "",
    langfuse_secret_key: "",
    langfuse_host: "",
    user_email: "",
    discord_webhook_escalate: "",
    discord_webhook_dev: "",
    password: "",
  });

  useEffect(() => {
    const load = async () => {
      try {
        const [projectResponse, runtimeResponse] = await Promise.all([
          authAPI.getMe(),
          modelLifecycleAPI.getRuntime(),
        ]);
        const p = projectResponse.data;
        setRuntime(runtimeResponse.data);
        setIsTest(p.is_test);
        setSetupStatus(p.setup_status || {});
        setForm({
          datadog_api_key: p.datadog_api_key || "",
          datadog_app_key: p.datadog_app_key || "",
          datadog_site: p.datadog_site || "datadoghq.com",
          datadog_query:
            p.datadog_query || "status:(error OR warn OR critical)",
          datadog_environment: p.datadog_environment || "prod",
          datadog_service: p.datadog_service || "",
          langfuse_public_key: p.langfuse_public_key || "",
          langfuse_secret_key: p.langfuse_secret_key || "",
          langfuse_host: p.langfuse_host || "https://cloud.langfuse.com",
          user_email: p.user_email || "",
          discord_webhook_escalate: p.discord_webhook_escalate || "",
          discord_webhook_dev: p.discord_webhook_dev || "",
          password: "",
        });
      } catch (e) {
        console.error("Failed to load settings:", e);
      } finally {
        setFetching(false);
      }
    };
    load();
  }, []);

  const handleChange = (e) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const toggleSection = (key) => {
    setOpenSections((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess(false);
    setLoading(true);

    // Only send non-empty, non-masked values
    const payload = {};
    Object.entries(form).forEach(([k, v]) => {
      const isVisibleValue = v !== "••••••" && v !== HIDDEN_MARKER;
      if (isVisibleValue && (v || k === "datadog_service")) {
        payload[k] = v;
      }
    });

    try {
      const res = await authAPI.updateSettings(payload);
      setSetupStatus(res.data.project?.setup_status || {});
      setSuccess(true);
      localStorage.setItem("project", JSON.stringify(res.data.project));
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string" ? detail : "Failed to update settings.",
      );
    } finally {
      setLoading(false);
    }
  };

  if (fetching) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <span className="loader" />
        </div>
      </div>
    );
  }

  const disabled = isTest;

  return (
    <div className="min-h-screen">
      <Navbar />
      <div className="max-w-2xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-semibold text-white mb-1">
          Project Settings
        </h1>
        <p className="text-sm text-gray-400 mb-8">
          Configure your credentials to start monitoring.
        </p>

        {isTest && (
          <div className="mb-6 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-start">
            <AlertCircle
              className="text-amber-400 mr-3 shrink-0 mt-0.5"
              size={16}
            />
            <p className="text-amber-300 text-sm">
              This is a test project. Credentials are hidden and settings are
              read-only.
            </p>
          </div>
        )}

        {!isTest && !setupStatus.datadog && (
          <div className="mb-6 p-4 bg-sky-500/10 border border-sky-500/30 rounded-lg flex items-start">
            <AlertCircle
              className="text-sky-400 mr-3 shrink-0 mt-0.5"
              size={16}
            />
            <div>
              <p className="text-sky-300 text-sm font-medium">
                Setup incomplete
              </p>
              <p className="text-sky-400/70 text-xs mt-0.5">
                Configure Datadog credentials to start ingestion. Local AI
                inference becomes ready when the project has an active adapter.
              </p>
            </div>
          </div>
        )}

        {success && (
          <div className="mb-6 p-4 bg-green-500/10 border border-green-500/30 rounded-lg flex items-center">
            <CheckCircle className="text-green-400 mr-3 shrink-0" size={16} />
            <p className="text-green-300 text-sm">
              Settings saved successfully.
            </p>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center">
            <AlertCircle className="text-red-400 mr-3 shrink-0" size={16} />
            <p className="text-red-300 text-sm">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-2">
          {/* Datadog Configuration */}
          <div className="border border-gray-800 rounded-xl px-5">
            <SectionHeader
              title="Datadog Configuration"
              description="Log ingestion source — required to start monitoring"
              configured={setupStatus.datadog}
              open={openSections.datadog}
              onToggle={() => toggleSection("datadog")}
            />
            {openSections.datadog && (
              <div className="pb-5 space-y-4">
                <SecretInput
                  label="Datadog API Key"
                  name="datadog_api_key"
                  value={form.datadog_api_key}
                  onChange={handleChange}
                  placeholder="API key"
                  disabled={disabled}
                  hint="Organization Settings → API Keys"
                />
                <SecretInput
                  label="Datadog Application Key"
                  name="datadog_app_key"
                  value={form.datadog_app_key}
                  onChange={handleChange}
                  placeholder="Application key"
                  disabled={disabled}
                  hint="Must have the logs_read_data permission"
                />
                <PlainInput
                  label="Datadog Site"
                  name="datadog_site"
                  value={form.datadog_site}
                  onChange={handleChange}
                  placeholder="datadoghq.com"
                  disabled={disabled}
                  hint="For example: datadoghq.com, datadoghq.eu, or us5.datadoghq.com"
                />
                <PlainInput
                  label="Log Query"
                  name="datadog_query"
                  value={form.datadog_query}
                  onChange={handleChange}
                  placeholder="status:(error OR warn OR critical)"
                  disabled={disabled}
                  hint="Datadog Logs search syntax. Environment and service filters are added automatically."
                />
                <PlainInput
                  label="Environment"
                  name="datadog_environment"
                  value={form.datadog_environment}
                  onChange={handleChange}
                  placeholder="prod"
                  disabled={disabled}
                  hint="Matches the env tag and becomes the incident environment"
                />
                <PlainInput
                  label="Service Filter (optional)"
                  name="datadog_service"
                  value={form.datadog_service}
                  onChange={handleChange}
                  placeholder="log-server"
                  disabled={disabled}
                  hint="Leave blank to ingest matching logs from every service"
                />
              </div>
            )}
          </div>

          {/* Local AI runtime */}
          <div className="border border-gray-800 rounded-xl px-5">
            <SectionHeader
              title="AI Runtime"
              description="Read-only local model and storage configuration"
              configured={Boolean(runtime)}
              open={openSections.runtime}
              onToggle={() => toggleSection("runtime")}
            />
            {openSections.runtime && (
              <div className="pb-5">
                <div className="flex items-center gap-2 mb-4 text-xs text-sky-400">
                  <Cpu size={14} />
                  Current Runtime: Local
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <RuntimeValue
                    label="Base Model"
                    value={runtime?.base_model}
                  />
                  <RuntimeValue
                    label="Inference Device"
                    value={runtime?.device?.toUpperCase()}
                  />
                  <RuntimeValue
                    label="Model Path"
                    value={runtime?.model_path}
                    mono
                  />
                  <RuntimeValue
                    label="Active Adapter"
                    value={
                      runtime?.active_artifact_version || "No active adapter"
                    }
                  />
                  <RuntimeValue
                    label="Artifact Storage"
                    value={runtime?.artifact_storage}
                    mono
                  />
                  <RuntimeValue
                    label="Dataset Storage"
                    value={runtime?.dataset_storage}
                    mono
                  />
                </div>
                <p className="text-xs text-gray-600 mt-3">
                  Runtime values are configured by the backend deployment and
                  cannot be changed per project.
                </p>
              </div>
            )}
          </div>

          {/* Langfuse */}
          <div className="border border-gray-800 rounded-xl px-5">
            <SectionHeader
              title="Langfuse Observability"
              description="LLM tracing, cost tracking, and latency metrics — optional but recommended"
              configured={setupStatus.observability}
              open={openSections.observability}
              onToggle={() => toggleSection("observability")}
            />
            {openSections.observability && (
              <div className="pb-5 space-y-4">
                <SecretInput
                  label="Public Key"
                  name="langfuse_public_key"
                  value={form.langfuse_public_key}
                  onChange={handleChange}
                  placeholder="pk-lf-..."
                  disabled={disabled}
                  hint="From cloud.langfuse.com → project → Settings"
                />
                <SecretInput
                  label="Secret Key"
                  name="langfuse_secret_key"
                  value={form.langfuse_secret_key}
                  onChange={handleChange}
                  placeholder="sk-lf-..."
                  disabled={disabled}
                />
                <PlainInput
                  label="Host"
                  name="langfuse_host"
                  value={form.langfuse_host}
                  onChange={handleChange}
                  placeholder="https://cloud.langfuse.com"
                  disabled={disabled}
                  hint="Only change if self-hosting Langfuse"
                />
              </div>
            )}
          </div>

          {/* Notifications */}
          <div className="border border-gray-800 rounded-xl px-5">
            <SectionHeader
              title="Notifications"
              description="Discord webhooks and email for incident alerts — optional"
              configured={setupStatus.notifications}
              open={openSections.notifications}
              onToggle={() => toggleSection("notifications")}
            />
            {openSections.notifications && (
              <div className="pb-5 space-y-4">
                <PlainInput
                  label="On-call Email"
                  name="user_email"
                  value={form.user_email}
                  onChange={handleChange}
                  placeholder="oncall@example.com"
                  type="email"
                  disabled={disabled}
                  hint="Receives NEEDS_ONCALL incidents"
                />
                <PlainInput
                  label="Discord Webhook — Critical"
                  name="discord_webhook_escalate"
                  value={form.discord_webhook_escalate}
                  onChange={handleChange}
                  placeholder="https://discord.com/api/webhooks/..."
                  disabled={disabled}
                  hint="Receives ESCALATE incidents"
                />
                <PlainInput
                  label="Discord Webhook — Dev Team"
                  name="discord_webhook_dev"
                  value={form.discord_webhook_dev}
                  onChange={handleChange}
                  placeholder="https://discord.com/api/webhooks/..."
                  disabled={disabled}
                  hint="Receives NEEDS_DEV incidents"
                />
              </div>
            )}
          </div>

          {/* Security */}
          <div className="border border-gray-800 rounded-xl px-5">
            <SectionHeader
              title="Security"
              description="Change your project password"
              configured={true}
              open={openSections.security}
              onToggle={() => toggleSection("security")}
            />
            {openSections.security && (
              <div className="pb-5">
                <SecretInput
                  label="New Password (leave blank to keep current)"
                  name="password"
                  value={form.password}
                  onChange={handleChange}
                  placeholder="••••••••"
                  disabled={disabled}
                  hint="Minimum 8 characters"
                />
              </div>
            )}
          </div>

          <div className="pt-4">
            <button
              type="submit"
              disabled={disabled || loading}
              className="w-full py-3 rounded-lg text-sm font-medium text-white flex items-center justify-center gap-2 bg-sky-500 hover:bg-sky-600 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
            >
              <Save size={15} />
              {loading ? "Saving..." : "Save Settings"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Settings;
