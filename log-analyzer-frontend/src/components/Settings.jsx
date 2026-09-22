import React, { useState, useEffect } from "react";
import { authAPI } from "../services/api";
import Navbar from "./Navbar";
import {
  Save,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  ChevronDown,
  ChevronUp,
  Info,
} from "lucide-react";

const HIDDEN_MARKER = "HIDDEN: TEST CREDENTIAL";

function SectionHeader({ title, description, configured, open, onToggle }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className="w-full flex items-center justify-between pt-4 pb-5 pr-2 text-left"
    >
      <div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-google-text">{title}</span>
          {configured ? (
            <span className="text-[10px] leading-4 px-1.5 py-px bg-green-500/20 text-google-green rounded-full border border-google-green/30">
              Configured
            </span>
          ) : (
            <span className="text-[10px] leading-4 px-1.5 py-px bg-amber-500/20 text-amber-700 rounded-full border border-amber-300/30">
              Not set
            </span>
          )}
        </div>
        {description && (
          <p className="text-xs text-google-muted mt-0.5">{description}</p>
        )}
      </div>
      {open ? (
        <ChevronUp size={16} className="text-google-muted" />
      ) : (
        <ChevronDown size={16} className="text-google-muted" />
      )}
    </button>
  );
}

function FieldLabel({ label, hint }) {
  return (
    <div className="flex items-center gap-1.5 mb-1.5">
      <label className="block text-xs text-google-muted">{label}</label>
      {hint && (
        <span className="relative inline-flex group">
          <button
            type="button"
            aria-label={`${label} information`}
            className="text-google-muted hover:text-google-blue focus:text-google-blue focus:outline-none"
          >
            <Info size={13} />
          </button>
          <span
            role="tooltip"
            className="pointer-events-none invisible absolute left-1/2 bottom-full z-20 mb-2 w-64 -translate-x-1/2 rounded-lg bg-google-text px-3 py-2 text-xs leading-relaxed text-white opacity-0 shadow-lg transition-opacity group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
          >
            {hint}
          </span>
        </span>
      )}
    </div>
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
      <FieldLabel label={label} hint={hint} />
      <div className="relative">
        <input
          type={show && !isHidden ? "text" : "password"}
          name={name}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          className="w-full px-4 py-2.5 pr-10 bg-white text-google-text border border-google-border rounded-lg text-sm disabled:text-google-muted disabled:cursor-not-allowed placeholder:text-google-muted focus:ring-1 focus:ring-google-blue focus:border-google-blue"
        />
        {!disabled && (
          <button
            type="button"
            onClick={() => setShow((s) => !s)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-google-muted hover:text-google-text"
          >
            {show ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        )}
      </div>
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
      <FieldLabel label={label} hint={hint} />
      <input
        type={type}
        name={name}
        value={value || ""}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
        className="w-full px-4 py-2.5 bg-white text-google-text border border-google-border rounded-lg text-sm disabled:text-google-muted disabled:cursor-not-allowed placeholder:text-google-muted focus:ring-1 focus:ring-google-blue focus:border-google-blue"
      />
    </div>
  );
}

function Settings() {
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const [isTest, setIsTest] = useState(false);
  const [openSections, setOpenSections] = useState({
    datadog: true,
    notifications: false,
    security: false,
  });
  const [setupStatus, setSetupStatus] = useState({});

  const [form, setForm] = useState({
    datadog_api_key: "",
    datadog_app_key: "",
    datadog_site: "datadoghq.com",
    user_email: "",
    discord_webhook_escalate: "",
    discord_webhook_dev: "",
    password: "",
  });

  useEffect(() => {
    const load = async () => {
      try {
        const projectResponse = await authAPI.getMe();
        const p = projectResponse.data;
        setIsTest(p.is_test);
        setSetupStatus(p.setup_status || {});
        setForm({
          datadog_api_key: p.datadog_api_key || "",
          datadog_app_key: p.datadog_app_key || "",
          datadog_site: p.datadog_site || "datadoghq.com",
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
      if (isVisibleValue && v) {
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
      <div className="app-shell">
        <div className="app-canvas min-h-full flex flex-col">
        <Navbar />
        <div className="flex-1 flex items-center justify-center">
          <span className="loader" />
        </div>
        </div>
      </div>
    );
  }

  const disabled = isTest;

  return (
    <div className="app-shell">
      <div className="app-canvas">
      <Navbar />
      <div className="max-w-2xl mx-auto px-4 pt-16 pb-16">
        <div className="mb-12">
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-[-0.04em] text-google-text">
            Project Settings
          </h1>
          <p className="text-sm text-google-muted mt-2">
            Connect your log source, choose alert recipients, and manage access.
          </p>
        </div>

        {isTest && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-300/30 rounded-lg flex items-start">
            <AlertCircle
              className="text-amber-700 mr-3 shrink-0 mt-0.5"
              size={16}
            />
            <p className="text-amber-700 text-sm">
              This is a test project. Credentials are hidden and settings are
              read-only.
            </p>
          </div>
        )}

        {!isTest && !setupStatus.datadog && (
          <div className="mb-6 p-4 bg-google-blue/10 border border-google-blue/30 rounded-lg flex items-start">
            <AlertCircle
              className="text-google-blue mr-3 shrink-0 mt-0.5"
              size={16}
            />
            <div>
              <p className="text-google-blue text-sm font-medium">
                Setup incomplete
              </p>
              <p className="text-google-blue/70 text-xs mt-0.5">
                Add your Datadog credentials below so IncidentLens can start
                monitoring your logs.
              </p>
            </div>
          </div>
        )}

        {success && (
          <div className="mb-6 p-4 bg-green-50 border border-google-green/30 rounded-lg flex items-center">
            <CheckCircle className="text-google-green mr-3 shrink-0" size={16} />
            <p className="text-google-green text-sm">
              Settings saved successfully.
            </p>
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-google-red/30 rounded-lg flex items-center">
            <AlertCircle className="text-google-red mr-3 shrink-0" size={16} />
            <p className="text-google-red text-sm">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Datadog Configuration */}
          <div className="soft-card px-5">
            <SectionHeader
              title="Datadog Connection"
              description="Connect Datadog so IncidentLens can read and analyze your logs."
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
                  hint="In Datadog, open Organization Settings → API Keys, then copy an existing key or create a new one."
                />
                <SecretInput
                  label="Datadog Application Key"
                  name="datadog_app_key"
                  value={form.datadog_app_key}
                  onChange={handleChange}
                  placeholder="Application key"
                  disabled={disabled}
                  hint="Create an application key in Datadog with the logs_read_data permission so IncidentLens can read your logs."
                />
                <PlainInput
                  label="Datadog Site"
                  name="datadog_site"
                  value={form.datadog_site}
                  onChange={handleChange}
                  placeholder="datadoghq.com"
                  disabled={disabled}
                  hint="Use the site from your Datadog URL, such as datadoghq.com, datadoghq.eu, or us5.datadoghq.com."
                />
              </div>
            )}
          </div>

          {/* Notifications */}
          <div className="soft-card px-5">
            <SectionHeader
              title="Notifications"
              description="Choose who should be notified when an incident needs attention."
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
                  hint="IncidentLens emails this address when an incident needs an on-call response."
                />
                <PlainInput
                  label="Discord Webhook — Critical"
                  name="discord_webhook_escalate"
                  value={form.discord_webhook_escalate}
                  onChange={handleChange}
                  placeholder="https://discord.com/api/webhooks/..."
                  disabled={disabled}
                  hint="Urgent incidents that need immediate escalation are sent to this Discord channel."
                />
                <PlainInput
                  label="Discord Webhook — Dev Team"
                  name="discord_webhook_dev"
                  value={form.discord_webhook_dev}
                  onChange={handleChange}
                  placeholder="https://discord.com/api/webhooks/..."
                  disabled={disabled}
                  hint="Incidents that need engineering investigation are sent to this Discord channel."
                />
              </div>
            )}
          </div>

          {/* Security */}
          <div className="soft-card px-5">
            <SectionHeader
              title="Security"
              description="Update the password used to sign in to this project."
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
                  hint="Use at least 8 characters. Leave this field empty to keep your current password."
                />
              </div>
            )}
          </div>

          <div className="pt-4">
            <button
              type="submit"
              disabled={disabled || loading}
              className="w-full py-3 rounded-lg text-sm font-medium text-white flex items-center justify-center gap-2 bg-google-blue hover:bg-google-blue-dark disabled:bg-google-border disabled:cursor-not-allowed transition-colors"
            >
              <Save size={15} />
              {loading ? "Saving..." : "Save Settings"}
            </button>
          </div>
        </form>
      </div>
      </div>
    </div>
  );
}

export default Settings;
