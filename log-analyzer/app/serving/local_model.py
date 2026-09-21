import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.shared.model_config import RuntimeModelSettings


class BaseModelLoadError(RuntimeError):
    pass


class AdapterLoadError(RuntimeError):
    pass


class LocalInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedBaseModel:
    model_id: str
    tokenizer: Any
    model: Any


@dataclass(frozen=True)
class LoadedAdapter:
    artifact_id: str
    project_id: str
    adapter_name: str
    adapter_path: str


@dataclass(frozen=True)
class LocalGeneration:
    content: str
    input_tokens: int
    output_tokens: int


class BaseModelLoader:
    """Load the configured shared causal language model and tokenizer."""

    def __init__(self, auto_model=None, auto_tokenizer=None, torch_module=None):
        self.auto_model = auto_model
        self.auto_tokenizer = auto_tokenizer
        self.torch = torch_module

    def load(self, settings: RuntimeModelSettings) -> LoadedBaseModel:
        try:
            auto_model, auto_tokenizer, torch_module = self._dependencies()
            token = os.getenv("HF_TOKEN", "").strip() or None
            tokenizer = auto_tokenizer.from_pretrained(
                settings.base_model,
                token=token,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = auto_model.from_pretrained(
                settings.base_model,
                token=token,
                dtype=self._dtype(torch_module, settings.dtype),
                low_cpu_mem_usage=True,
            )
            model.to(settings.device)
            model.eval()
            return LoadedBaseModel(settings.base_model, tokenizer, model)
        except Exception as exc:
            if isinstance(exc, BaseModelLoadError):
                raise
            raise BaseModelLoadError(
                f"Unable to load shared base model {settings.base_model!r} "
                f"on {settings.device!r}: {exc}"
            ) from exc

    def _dependencies(self):
        if self.auto_model is not None:
            return self.auto_model, self.auto_tokenizer, self.torch
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BaseModelLoadError(
                "Local inference dependencies are unavailable"
            ) from exc
        return AutoModelForCausalLM, AutoTokenizer, torch

    @staticmethod
    def _dtype(torch_module, configured: str):
        if configured == "auto":
            return "auto"
        return getattr(torch_module, configured)


class AdapterLoader:
    """Validate and attach immutable PEFT adapters to the shared base model."""

    def __init__(self, peft_model=None, peft_config=None):
        self.peft_model = peft_model
        self.peft_config = peft_config

    def load_first(
        self,
        base_model,
        *,
        adapter_path: str,
        adapter_name: str,
        expected_base_model: str,
    ):
        peft_model, _ = self._dependencies()
        path = self._validate(adapter_path, expected_base_model)
        try:
            return peft_model.from_pretrained(
                base_model,
                str(path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        except Exception as exc:
            raise AdapterLoadError(
                f"Unable to load LoRA adapter from {path}: {exc}"
            ) from exc

    def load_additional(
        self,
        model,
        *,
        adapter_path: str,
        adapter_name: str,
        expected_base_model: str,
    ) -> None:
        path = self._validate(adapter_path, expected_base_model)
        try:
            model.load_adapter(
                str(path),
                adapter_name=adapter_name,
                is_trainable=False,
            )
        except Exception as exc:
            raise AdapterLoadError(
                f"Unable to load LoRA adapter from {path}: {exc}"
            ) from exc

    def _validate(self, adapter_path: str, expected_base_model: str) -> Path:
        _, peft_config = self._dependencies()
        path = Path(adapter_path).expanduser().resolve()
        config_path = path / "adapter_config.json"
        weights_path = path / "adapter_model.safetensors"
        if not path.is_dir():
            raise AdapterLoadError(f"Adapter directory does not exist: {path}")
        if not config_path.is_file():
            raise AdapterLoadError(f"Adapter configuration is missing: {config_path}")
        if not weights_path.is_file() or weights_path.stat().st_size == 0:
            raise AdapterLoadError(
                f"Adapter weights are missing or empty: {weights_path}"
            )
        try:
            config = peft_config.from_pretrained(str(path))
        except Exception as exc:
            raise AdapterLoadError(
                f"Adapter configuration could not be loaded from {path}: {exc}"
            ) from exc
        actual_base_model = str(config.base_model_name_or_path)
        if actual_base_model != expected_base_model:
            raise AdapterLoadError(
                f"Adapter base model {actual_base_model!r} does not match "
                f"configured base model {expected_base_model!r}"
            )
        return path

    def _dependencies(self):
        if self.peft_model is not None:
            return self.peft_model, self.peft_config
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise AdapterLoadError("PEFT is unavailable") from exc
        return PeftModel, PeftConfig


class LocalModelCache:
    """Cache one base model and named project adapters for local generation."""

    def __init__(
        self,
        settings: RuntimeModelSettings,
        base_loader: BaseModelLoader | None = None,
        adapter_loader: AdapterLoader | None = None,
    ):
        self.settings = settings
        self.base_loader = base_loader or BaseModelLoader()
        self.adapter_loader = adapter_loader or AdapterLoader()
        self._lock = threading.RLock()
        self._base: LoadedBaseModel | None = None
        self._model = None
        self._adapters: dict[str, LoadedAdapter] = {}
        self._project_artifacts: dict[str, str] = {}

    @property
    def initialized(self) -> bool:
        return self._base is not None

    @property
    def loaded_artifact_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._adapters)

    def active_artifact_for(self, project_id: str) -> str | None:
        with self._lock:
            return self._project_artifacts.get(project_id)

    def initialize(self) -> None:
        with self._lock:
            if self._base is None:
                self._base = self.base_loader.load(self.settings)

    def generate(
        self,
        *,
        project_id: str,
        artifact_id: str,
        adapter_path: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_new_tokens: int,
    ) -> LocalGeneration:
        with self._lock:
            self.initialize()
            model = self._activate_adapter(
                project_id=project_id,
                artifact_id=artifact_id,
                adapter_path=adapter_path,
            )
            try:
                template_args = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                }
                if tools:
                    template_args["tools"] = tools
                inputs = self._base.tokenizer.apply_chat_template(
                    messages,
                    **template_args,
                )
                inputs = inputs.to(self.settings.device)
                input_tokens = int(inputs["input_ids"].shape[-1])
                generation_args = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "pad_token_id": self._base.tokenizer.pad_token_id,
                    "eos_token_id": self._base.tokenizer.eos_token_id,
                }
                if temperature > 0:
                    generation_args["temperature"] = temperature

                import torch

                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_args)
                output_ids = generated[0][input_tokens:]
                content = self._base.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
                return LocalGeneration(
                    content=content,
                    input_tokens=input_tokens,
                    output_tokens=len(output_ids),
                )
            except Exception as exc:
                if isinstance(exc, (BaseModelLoadError, AdapterLoadError)):
                    raise
                raise LocalInferenceError(
                    f"Local inference failed for project {project_id!r} with "
                    f"artifact {artifact_id!r}: {exc}"
                ) from exc

    def _activate_adapter(
        self,
        *,
        project_id: str,
        artifact_id: str,
        adapter_path: str,
    ):
        adapter = self._adapters.get(artifact_id)
        if adapter is None:
            adapter_name = self._adapter_name(project_id, artifact_id)
            if self._model is None:
                self._model = self.adapter_loader.load_first(
                    self._base.model,
                    adapter_path=adapter_path,
                    adapter_name=adapter_name,
                    expected_base_model=self.settings.base_model,
                )
            else:
                self.adapter_loader.load_additional(
                    self._model,
                    adapter_path=adapter_path,
                    adapter_name=adapter_name,
                    expected_base_model=self.settings.base_model,
                )
            adapter = LoadedAdapter(
                artifact_id=artifact_id,
                project_id=project_id,
                adapter_name=adapter_name,
                adapter_path=str(Path(adapter_path).expanduser().resolve()),
            )
            self._adapters[artifact_id] = adapter

        if adapter.project_id != project_id:
            raise AdapterLoadError(
                f"Artifact {artifact_id!r} is already owned by another project"
            )
        if adapter.adapter_path != str(Path(adapter_path).expanduser().resolve()):
            raise AdapterLoadError(
                f"Artifact {artifact_id!r} resolved to a different immutable path"
            )
        try:
            self._model.set_adapter(adapter.adapter_name)
            self._model.eval()
        except Exception as exc:
            raise AdapterLoadError(
                f"Unable to activate adapter {artifact_id!r}: {exc}"
            ) from exc
        self._project_artifacts[project_id] = artifact_id
        return self._model

    @staticmethod
    def _adapter_name(project_id: str, artifact_id: str) -> str:
        safe_project = "".join(
            character if character.isalnum() else "_" for character in project_id
        )
        safe_artifact = "".join(
            character if character.isalnum() else "_" for character in artifact_id
        )
        return f"project_{safe_project}__artifact_{safe_artifact}"
