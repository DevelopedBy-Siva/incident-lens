import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://log_user:password@localhost:5432/log_analyzer"
)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL.removeprefix("postgres://")

# Neon may suspend idle compute and close old connections. These settings make
# SQLAlchemy validate pooled connections and periodically replace them.
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    # Import plane-owned models so SQLAlchemy registers them on shared metadata.
    from app.control import models as control_models  # noqa: F401
    from app.data import models as data_models  # noqa: F401
    from app.serving import models as serving_models  # noqa: F401
    from app.shared.migrations.runner import run_migrations
    from app.training import models as training_models  # noqa: F401

    run_migrations(engine)
    Base.metadata.create_all(bind=engine)
    _ensure_action_log_columns()
    print("[DB] Tables initialised")


def _ensure_action_log_columns():
    """Best-effort additive migration for deployments using create_all."""
    try:
        inspector = inspect(engine)
        if not inspector.has_table("action_logs"):
            return

        existing = {column["name"] for column in inspector.get_columns("action_logs")}
        missing = {
            "requested_actions": "JSON",
            "allowed_actions": "JSON",
            "blocked_actions": "JSON",
            "policy_reason": "VARCHAR",
        }

        with engine.begin() as conn:
            for column_name, column_type in missing.items():
                if column_name not in existing:
                    conn.execute(
                        text(
                            f"ALTER TABLE action_logs ADD COLUMN {column_name} {column_type}"
                        )
                    )
    except Exception as e:
        print(f"[DB] ActionLog column check skipped: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
