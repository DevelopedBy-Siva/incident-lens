import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { authAPI } from "../services/api";
import { AlertCircle } from "lucide-react";
import { Activity } from "lucide-react";

function Login() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [formData, setFormData] = useState({
    name: "",
    password: "",
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await authAPI.login(formData);
      const { access_token, project } = response.data;

      localStorage.setItem("token", access_token);
      localStorage.setItem("project", JSON.stringify(project));

      navigate("/dashboard");
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Login failed. Please check your credentials.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-shell min-h-screen flex flex-col items-center justify-center p-4 relative">
      <div className="flex items-center mb-8">
        <Activity className="text-google-text mr-2" size={30} strokeWidth={2.4} />
        <span className="text-xl font-semibold tracking-tight text-google-text">IncidentLens</span>
      </div>
      <div className="auth-card w-full max-w-md p-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-semibold tracking-[-0.04em] text-google-text mb-2">
            Login
          </h1>
          <p className="text-sm text-google-muted">
            Sign in to monitor and investigate your project’s incidents.
          </p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start">
            <AlertCircle
              className="text-google-red mr-3 flex-shrink-0"
              size={20}
            />
            <p className="text-google-red text-sm">{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-normal text-google-muted mb-2">
              Project Name
            </label>
            <input
              type="text"
              name="name"
              value={formData.name}
              onChange={handleChange}
              className="text-sm w-full px-4 py-3 bg-white text-google-text border border-google-border rounded-lg focus:ring-2 focus:ring-google-blue focus:border-google-blue placeholder:text-google-muted"
              placeholder="my-awesome-project"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-normal text-google-muted mb-2">
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-3 bg-white text-google-text border border-google-border rounded-lg focus:ring-2 focus:ring-google-blue focus:border-google-blue placeholder:text-google-muted"
              placeholder="••••••••"
              required
              autoComplete="new-password"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3 rounded-lg text-sm font-medium transition-colors ${
              loading
                ? "bg-google-border text-google-muted cursor-not-allowed"
                : "bg-google-blue text-white hover:bg-google-blue-dark"
            }`}
          >
            {loading ? "Logging in..." : "Login"}
          </button>
        </form>

        <p className="text-center text-sm text-google-muted mt-6">
          Don't have a project?{" "}
          <Link
            to="/register"
            className="text-google-blue hover:text-google-muted font-medium"
          >
            Create one here
          </Link>
        </p>
      </div>
    </div>
  );
}

export default Login;
