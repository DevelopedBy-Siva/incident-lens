"""
app/main.py  —  IncidentLens
"""

import os
import threading
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_auth,
    routes_incidents,
    routes_ingest,
    routes_investigation,
    routes_model_management,
)
from app.control.maintenance import cleanup_all_data
from app.serving.model_runtime import get_model_runtime
from app.shared.database import SessionLocal, init_db


def _validate_production_config() -> None:
    if os.getenv("APP_ENV", "development").strip().lower() not in {
        "prod",
        "production",
    }:
        return

    required = ("DATABASE_URL", "CORS_ORIGINS", "SECRET_KEY", "S3_BUCKET")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(
            "Missing required production configuration: " + ", ".join(missing)
        )
    if os.environ["SECRET_KEY"] == "your-secret-key-change-in-production":
        raise RuntimeError("SECRET_KEY must be changed in production")


load_dotenv()


def _configure_logging() -> None:
    """Ensure background worker lifecycle logs reach container stdout."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_configure_logging()
logger = logging.getLogger(__name__)


def _should_reset_data_on_startup() -> bool:
    value = os.getenv("RESET_DATA_ON_STARTUP", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def start_worker():
    try:
        from app.data.ingestion.log_source_watcher import run as run_worker

        run_worker()
    except Exception:
        logger.exception("[MAIN] Log source watcher stopped unexpectedly")


def start_verifier():
    try:
        from app.serving.verification import run as run_verifier

        run_verifier()
    except Exception:
        logger.exception("[MAIN] Verifier stopped unexpectedly")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[MAIN] Starting IncidentLens")
    _validate_production_config()
    init_db()
    get_model_runtime().initialize()
    logger.info("[MODEL] Shared local base model loaded")
    if _should_reset_data_on_startup():
        cleanup_all_data()
        logger.warning("[MAIN] RESET_DATA_ON_STARTUP enabled; cleared incident-processing data")

    threading.Thread(target=start_worker, daemon=True).start()
    threading.Thread(target=start_verifier, daemon=True).start()
    logger.info("[MAIN] Worker and verifier threads started")

    yield
    logger.info("[MAIN] Shutting down")


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


@app.get("/ready")
def ready():
    """Readiness probe that confirms the external database is reachable."""
    from sqlalchemy import text

    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    finally:
        db.close()
