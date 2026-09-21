"""
app/main.py  —  IncidentLens
"""

import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_auth,
    routes_incidents,
    routes_ingest,
    routes_investigation,
    routes_model_management,
)
from app.control.maintenance import cleanup_all_data
from app.shared.database import init_db

load_dotenv()


def _should_reset_data_on_startup() -> bool:
    value = os.getenv("RESET_DATA_ON_STARTUP", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def start_worker():
    import traceback

    try:
        from app.data.ingestion.loki_watcher import run as run_worker

        run_worker()
    except Exception as e:
        print(f"[MAIN] Loki watcher failed: {e}")
        traceback.print_exc()


def start_verifier():
    import traceback

    try:
        from app.serving.verification import run as run_verifier

        run_verifier()
    except Exception as e:
        print(f"[MAIN] Verifier failed: {e}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting IncidentLens...")
    init_db()
    if _should_reset_data_on_startup():
        cleanup_all_data()
        print("[MAIN] RESET_DATA_ON_STARTUP enabled — cleared incident-processing data")

    threading.Thread(target=start_worker, daemon=True).start()
    threading.Thread(target=start_verifier, daemon=True).start()
    print("[MAIN] Worker + verifier threads started")

    yield
    print("[MAIN] Shutting down")


app = FastAPI(title="IncidentLens", version="1.0.0", lifespan=lifespan)

cors_origins = os.getenv("CORS_ORIGINS", "")
origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(routes_ingest.router, prefix="/api", tags=["ingest"])
app.include_router(routes_incidents.router, prefix="/api", tags=["incidents"])
app.include_router(routes_investigation.router, prefix="/api", tags=["investigation"])
app.include_router(
    routes_model_management.router, prefix="/api", tags=["model-management"]
)


@app.get("/")
def root():
    return {
        "service": "IncidentLens",
        "version": "1.0.0",
        "status": "running",
        "description": "Policy-bound autonomous observability agent",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
