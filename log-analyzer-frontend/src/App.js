import React, { useEffect, useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import {
  API_BASE_URL,
  isAuthenticated,
  TEST_LOG_SERVER_URL,
} from "./services/api";
import Login from "./components/Login";
import Register from "./components/Register";
import Dashboard from "./components/Dashboard";
import Models from "./components/Models";
import Training from "./components/Training";
import Settings from "./components/Settings";
import axios from "axios";
import { MdError } from "react-icons/md";

function ProtectedRoute({ children }) {
  return isAuthenticated() ? children : <Navigate to="/login" />;
}

function PublicRoute({ children }) {
  return !isAuthenticated() ? children : <Navigate to="/dashboard" />;
}

function App() {
  const [status, setStatus] = useState({
    serverA: "pending",
    serverB: "pending",
  });

  useEffect(() => {
    const wake = async (name, url) => {
      try {
        await axios.get(url, { timeout: 120000 });
        setStatus((s) => ({ ...s, [name]: "up" }));
        return true;
      } catch {
        setStatus((s) => ({ ...s, [name]: "down" }));
        return false;
      }
    };

    const wakeAll = async () => {
      await Promise.all([
        wake("serverA", `${API_BASE_URL}/health`),
        wake("serverB", `${TEST_LOG_SERVER_URL}/health`),
      ]);
    };

    wakeAll();
  }, []);

  return status.serverA !== "up" && status.serverB !== "up" ? (
    <div className="server-loading">
      {status.serverA === "pending" || status.serverB === "pending" ? (
        <>
          <span className="loader"></span>
          <p>
            Please allow a few seconds for everything to initialize, as the
            servers are on free instances.
          </p>
        </>
      ) : status.serverA === "down" || status.serverB === "down" ? (
        <>
          <MdError />
          <p>Failed to initialize the server. Please try again later. </p>
        </>
      ) : (
        <p>Initialize Successful </p>
      )}
      <p></p>
    </div>
  ) : (
    <Router>
      <Routes>
        <Route
          path="/login"
          element={
            <PublicRoute>
              <Login />
            </PublicRoute>
          }
        />
        <Route
          path="/register"
          element={
            <PublicRoute>
              <Register />
            </PublicRoute>
          }
        />

        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/models"
          element={
            <ProtectedRoute>
              <Models />
            </ProtectedRoute>
          }
        />
        <Route
          path="/training"
          element={
            <ProtectedRoute>
              <Training />
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <Settings />
            </ProtectedRoute>
          }
        />

        <Route path="/" element={<Navigate to="/dashboard" />} />

        <Route
          path="*"
          element={
            <div className="min-h-screen flex items-center justify-center">
              <div className="text-center">
                <h1 className="text-7xl font-semibold text-google-muted mb-4">
                  404
                </h1>
                <p className="text-google-muted mb-10">Page not available.</p>
                <a
                  href="/dashboard"
                  className="text-sm px-8 py-3 bg-google-blue text-white font-normal rounded-lg hover:bg-google-blue-dark"
                >
                  Return to Home
                </a>
              </div>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
