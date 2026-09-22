import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authAPI } from "../services/api";
import { AlertCircle, Activity, ArrowRight } from "lucide-react";

function Register() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [formData, setFormData] = useState({ name: "", password: "" });

  const handleChange = (e) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const response = await authAPI.register(formData);
      const { access_token, project } = response.data;
      localStorage.setItem("token", access_token);
      localStorage.setItem("project", JSON.stringify(project));
      navigate("/settings"); // send straight to Settings to complete setup
    } catch (err) {
      const detail = err.response?.data?.detail;
      if (typeof detail === "string") {
        setError(detail);
      } else if (detail?.errors) {
        setError(detail.errors.map((e) => e.message).join(", "));
      } else {
        setError("Registration failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell min-h-screen flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="flex items-center justify-center mb-8">
          <Activity className="text-google-text mr-2" size={30} strokeWidth={2.4} />
          <span className="text-xl font-semibold tracking-tight text-google-text">IncidentLens</span>
        </div>

        <div className="auth-card p-8">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-semibold tracking-[-0.04em] text-google-text mb-1">
              Create Project
            </h1>
            <p className="text-sm text-google-muted mt-2">
              Create a workspace for monitoring and investigating incidents.
            </p>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-google-red/30 rounded-lg flex items-start">
              <AlertCircle className="text-google-red mr-3 shrink-0" size={18} />
              <p className="text-google-red text-sm">{error}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm text-google-muted mb-2">
                Project Name
              </label>
              <input
                type="text"
                name="name"
                value={formData.name}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-white text-google-text border border-google-border rounded-lg focus:ring-2 focus:ring-google-blue focus:border-google-blue placeholder:text-google-muted text-sm"
                placeholder="my-project"
                required
                minLength={3}
                maxLength={50}
              />
              <p className="text-xs text-google-muted mt-1">
                Use 3–50 letters, numbers, hyphens, or underscores.
              </p>
            </div>

            <div>
              <label className="block text-sm text-google-muted mb-2">
                Password
              </label>
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleChange}
                className="w-full px-4 py-3 bg-white text-google-text border border-google-border rounded-lg focus:ring-2 focus:ring-google-blue focus:border-google-blue placeholder:text-google-muted text-sm"
                placeholder="••••••••"
                required
                minLength={8}
                autoComplete="new-password"
              />
              <p className="text-xs text-google-muted mt-1">
                Use at least 8 characters.
              </p>
            </div>

            <button
              type="submit"
              disabled={loading || !formData.name || !formData.password}
              className="w-full py-3 rounded-lg text-sm font-medium text-white transition-colors flex items-center justify-center gap-2 bg-google-blue hover:bg-google-blue-dark disabled:bg-google-border disabled:text-google-muted disabled:cursor-not-allowed"
            >
              {loading ? (
                "Creating..."
              ) : (
                <>
                  <span>Create Project</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          <p className="text-center text-sm text-google-muted mt-6">
            Already have a project?{" "}
            <Link
              to="/login"
              className="text-google-blue hover:text-google-blue font-medium"
            >
              Login here
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Register;
