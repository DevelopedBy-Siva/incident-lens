import axios from "axios";

export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

export const authAPI = {
  register: (data) => api.post("/api/auth/register", data),
  login: (data) => api.post("/api/auth/login", data),
  getMe: () => api.get("/api/auth/me"),
  updateSettings: (data) => api.put("/api/auth/settings", data),
  verifyDatadogSettings: (data) =>
    api.post("/api/auth/settings/datadog/verify", data),
  deleteProject: () => api.delete("/api/auth/project"),
  settingsStatus: () => api.get("/api/auth/settings/status"),
};

export const logServerAPI = {
  start: () => api.post("/api/log-server/start"),
  stop: () => api.post("/api/log-server/stop"),
};

export const incidentsAPI = {
  list: (params) => api.get("/api/incidents", { params }),
  get: (id) => api.get(`/api/incidents/${id}`),
  clear: () => api.delete("/api/incidents"),
  close: (id) => api.post(`/api/incidents/${id}/close`),
  ignore: (id) => api.post(`/api/incidents/${id}/ignore`),
  // Agent visibility — new
  getEvidence: (id) => api.get(`/api/incidents/${id}/evidence`),
  getActions: (id) => api.get(`/api/incidents/${id}/actions`),
  getInvestigation: (id) => api.get(`/api/incidents/${id}/investigation`),
};

export const modelLifecycleAPI = {
  getRuntime: () => api.get("/api/model-runtime"),
  listDatasets: () => api.get("/api/datasets"),
  getDataset: (id) => api.get(`/api/datasets/${id}`),
  buildDataset: () => api.post("/api/datasets/build"),
  listTrainingJobs: () => api.get("/api/training-jobs"),
  getTrainingJob: (id) => api.get(`/api/training-jobs/${id}`),
  createTrainingJob: (datasetId) =>
    api.post("/api/training-jobs", { dataset_id: datasetId }),
  runTrainingJob: (id) => api.post(`/api/training-jobs/${id}/run`),
  listArtifacts: () => api.get("/api/model-artifacts"),
  getArtifact: (id) => api.get(`/api/model-artifacts/${id}`),
  activateArtifact: (id) => api.post(`/api/model-artifacts/${id}/activate`),
};

export const isAuthenticated = () => !!localStorage.getItem("token");

export const logout = () => {
  localStorage.removeItem("token");
  localStorage.removeItem("project");
  window.location.href = "/login";
};

export default api;
