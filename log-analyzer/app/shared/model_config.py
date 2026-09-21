import os

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL_ENV = "LORA_BASE_MODEL_ID"


def configured_base_model() -> str:
    """Return the one shared base model used by every project adapter."""
    return os.getenv(BASE_MODEL_ENV, DEFAULT_BASE_MODEL).strip() or DEFAULT_BASE_MODEL
