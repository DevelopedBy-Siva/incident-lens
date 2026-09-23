import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LoraTrainingProfile:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    epochs: float = 1.0
    learning_rate: float = 0.0001
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_sequence_length: int = 1024
    validation_fraction: float = 0.1
    seed: int = 42
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be between 0 and 1")
        if self.epochs <= 0:
            raise ValueError("Training epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be positive")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("Validation fraction must be between 0 and 1")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required")

    def as_metadata(self) -> dict[str, Any]:
        profile = asdict(self)
        profile["target_modules"] = list(self.target_modules)
        return profile


def configured_training_profile() -> LoraTrainingProfile:
    target_modules = tuple(
        module.strip()
        for module in os.getenv(
            "LORA_TARGET_MODULES",
            (
                "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,"
                "in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj"
            ),
        ).split(",")
        if module.strip()
    )
    return LoraTrainingProfile(
        rank=_integer("LORA_RANK", 8),
        alpha=_integer("LORA_ALPHA", 16),
        dropout=_floating("LORA_DROPOUT", 0.05),
        epochs=_floating("LORA_EPOCHS", 1.0),
        learning_rate=_floating("LORA_LEARNING_RATE", 0.0001),
        batch_size=_integer("LORA_BATCH_SIZE", 1),
        gradient_accumulation_steps=_integer("LORA_GRADIENT_ACCUMULATION_STEPS", 1),
        max_sequence_length=_integer("LORA_MAX_SEQUENCE_LENGTH", 1024),
        validation_fraction=_floating("LORA_VALIDATION_FRACTION", 0.1),
        seed=_integer("LORA_SEED", 42),
        target_modules=target_modules,
    )


def _integer(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _floating(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
