import { useCallback, useEffect, useMemo, useState } from "react";
import { modelLifecycleAPI } from "../services/api";

const CACHE_TTL_MS = 10000;
const ACTIVE_JOB_STATUSES = new Set(["QUEUED", "RUNNING", "EVALUATING"]);

let cachedSnapshot = null;
let cachedAt = 0;
let pendingRequest = null;

function errorMessage(error, fallback) {
  const detail = error?.response?.data?.detail;
  return typeof detail === "string" ? detail : fallback;
}

async function requestSnapshot(force = false) {
  const cacheIsFresh = Date.now() - cachedAt < CACHE_TTL_MS;
  if (!force && cachedSnapshot && cacheIsFresh) return cachedSnapshot;
  if (pendingRequest) return pendingRequest;

  pendingRequest = Promise.all([
    modelLifecycleAPI.getRuntime(),
    modelLifecycleAPI.listDatasets(),
    modelLifecycleAPI.listTrainingJobs(),
    modelLifecycleAPI.listArtifacts(),
  ])
    .then(([runtime, datasets, jobs, artifacts]) => {
      cachedSnapshot = {
        runtime: runtime.data,
        datasets: datasets.data,
        jobs: jobs.data,
        artifacts: artifacts.data,
      };
      cachedAt = Date.now();
      return cachedSnapshot;
    })
    .finally(() => {
      pendingRequest = null;
    });

  return pendingRequest;
}

export function invalidateModelLifecycleCache() {
  cachedSnapshot = null;
  cachedAt = 0;
}

export function useModelLifecycle({ pollWhileActive = false } = {}) {
  const [snapshot, setSnapshot] = useState(cachedSnapshot);
  const [loading, setLoading] = useState(!cachedSnapshot);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(async (force = true) => {
    force ? setRefreshing(true) : setLoading(true);
    setError("");
    try {
      const next = await requestSnapshot(force);
      setSnapshot(next);
      return next;
    } catch (requestError) {
      setError(
        errorMessage(requestError, "Could not load model lifecycle data."),
      );
      throw requestError;
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    requestSnapshot(false)
      .then((next) => {
        if (active) setSnapshot(next);
      })
      .catch((requestError) => {
        if (active) {
          setError(
            errorMessage(requestError, "Could not load model lifecycle data."),
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const hasActiveJobs = useMemo(
    () =>
      (snapshot?.jobs || []).some((job) => ACTIVE_JOB_STATUSES.has(job.status)),
    [snapshot?.jobs],
  );

  useEffect(() => {
    if (!pollWhileActive || !hasActiveJobs) return undefined;
    const timer = setInterval(() => refresh(true).catch(() => {}), 4000);
    return () => clearInterval(timer);
  }, [hasActiveJobs, pollWhileActive, refresh]);

  return {
    runtime: snapshot?.runtime || null,
    datasets: snapshot?.datasets || [],
    jobs: snapshot?.jobs || [],
    artifacts: snapshot?.artifacts || [],
    loading,
    refreshing,
    error,
    hasActiveJobs,
    refresh,
  };
}

export { ACTIVE_JOB_STATUSES, errorMessage };
