import os
from dataclasses import dataclass

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MODEL_PROVIDER = "local"
DEFAULT_MODEL_DEVICE = "cpu"
DEFAULT_MODEL_DTYPE = "auto"


@dataclass(frozen=True)
class RuntimeModelSettings:
    provider: str = DEFAULT_MODEL_PROVIDER
    base_model: str = DEFAULT_BASE_MODEL
    device: str = DEFAULT_MODEL_DEVICE
    dtype: str = DEFAULT_MODEL_DTYPE
    max_new_tokens: int = 1500

    def __post_init__(self) -> None:
        if self.provider != "local":
            raise ValueError("MODEL_PROVIDER must be 'local'")
        if not self.base_model:
            raise ValueError("BASE_MODEL cannot be empty")
        if not self.device:
            raise ValueError("DEVICE cannot be empty")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("DTYPE must be one of: auto, float32, float16, bfloat16")
        if self.max_new_tokens <= 0:
            raise ValueError("Maximum generated tokens must be positive")


def configured_base_model() -> str:
    """Return the one shared base model used by every project adapter."""
    return os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL).strip() or DEFAULT_BASE_MODEL


def configured_runtime_settings() -> RuntimeModelSettings:
    """Load the single supported local-inference configuration."""
    return RuntimeModelSettings(
        provider=os.getenv("MODEL_PROVIDER", DEFAULT_MODEL_PROVIDER).strip().lower(),
        base_model=configured_base_model(),
        device=os.getenv("DEVICE", DEFAULT_MODEL_DEVICE).strip().lower(),
        dtype=os.getenv("DTYPE", DEFAULT_MODEL_DTYPE).strip().lower(),
    )
