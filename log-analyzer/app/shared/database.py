from sqlalchemy import create_engine, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://log_user:password@localhost:5432/log_analyzer"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    # Import plane-owned models so SQLAlchemy registers them on shared metadata.
    from app.control import models as control_models  # noqa: F401
    from app.data import models as data_models  # noqa: F401
    from app.serving import models as serving_models  # noqa: F401

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
