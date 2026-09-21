import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.control.models import DEFAULT_BASE_MODEL, Project
from app.control.repositories import ProjectRepository
from app.serving.model_provider import (
    GroqProvider,
    ModelProvider,
    ProviderResponse,
)
from app.shared.database import SessionLocal
from app.training.models import ModelArtifact, ModelArtifactStatus
from app.training.repositories import ModelArtifactRepository


@dataclass(frozen=True)
class ResolvedArtifact:
    id: str
    version: str
    base_model: str
    adapter_path: str | None
    dataset_id: str
    evaluation_score: float | None
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class ArtifactResolution:
    artifact: ResolvedArtifact | None
    warnings: tuple[str, ...] = ()


class BaseModelResolver:
    def resolve(self, project: Project) -> str:
        return project.base_model or DEFAULT_BASE_MODEL


class ArtifactResolver:
    def __init__(self, db):
        self.artifacts = ModelArtifactRepository(db)

    def resolve(self, project: Project, base_model: str) -> ArtifactResolution:
        if not project.active_artifact_id:
            return ArtifactResolution(None)

        artifact = self.artifacts.get_for_project(
            project.active_artifact_id, project.id
        )
        if not artifact:
            return ArtifactResolution(
                None,
                ("Configured active artifact does not exist for this project",),
            )
        if artifact.status != ModelArtifactStatus.READY:
            return ArtifactResolution(
                None,
                (f"Active artifact is not READY: {artifact.status.value}",),
            )
        if artifact.base_model != base_model:
            warning = (
                f"Active artifact base model {artifact.base_model!r} does not "
                f"match project base model {base_model!r}"
            )
            return ArtifactResolution(
                None,
                (warning,),
            )

        metadata, metadata_warning = self._load_metadata(artifact)
        warnings = (metadata_warning,) if metadata_warning else ()
        return ArtifactResolution(
            ResolvedArtifact(
                id=artifact.id,
                version=artifact.artifact_version,
                base_model=artifact.base_model,
                adapter_path=artifact.adapter_path,
                dataset_id=artifact.dataset_id,
                evaluation_score=artifact.evaluation_score,
                metadata=metadata,
            ),
            warnings,
        )

    @staticmethod
    def _load_metadata(
        artifact: ModelArtifact,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not artifact.adapter_path:
            return None, "Active artifact has no adapter path"

        metadata_path = Path(artifact.adapter_path) / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None, f"Artifact metadata was not found at {metadata_path}"
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"Artifact metadata could not be loaded: {exc}"
        if not isinstance(metadata, dict):
            return None, "Artifact metadata must be a JSON object"

        expected = {
            "artifact_id": artifact.id,
            "project_id": artifact.project_id,
            "base_model": artifact.base_model,
        }
        mismatches = [
            key for key, value in expected.items() if metadata.get(key) != value
        ]
        if mismatches:
            return None, (
                "Artifact metadata does not match database fields: "
                + ", ".join(mismatches)
            )
        return metadata, None


@dataclass(frozen=True)
class ModelRuntimeSession:
    project_id: str
    project_name: str
    base_model: str
    active_artifact: ResolvedArtifact | None
    adapter_path: str | None
    runtime_type: str
    provider: str
    capabilities: dict[str, bool]
    model_candidates: tuple[str, ...]
    default_model: str
    _provider_impl: ModelProvider = field(repr=False, compare=False)
    _provider_api_key: str | None = field(repr=False, compare=False, default=None)
    validation_warnings: tuple[str, ...] = ()

    @property
    def provider_available(self) -> bool:
        return self._provider_impl.is_available(self)

    def complete(
        self,
        *,
        model: str,
        messages: Any,
        temperature: float,
    ) -> ProviderResponse:
        return self._provider_impl.complete(
            self,
            model=model,
            messages=messages,
            temperature=temperature,
        )

    def complete_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        return self._provider_impl.complete_with_tools(
            self,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class ModelRuntime:
    """Resolve project model metadata and expose provider-neutral inference."""

    def __init__(self, session_factory=SessionLocal, provider=None):
        self.session_factory = session_factory
        self.provider = provider or GroqProvider()
        self.base_models = BaseModelResolver()

    def resolve_project_model(
        self,
        project_id: str | None,
        *,
        project: Project | None = None,
    ) -> ModelRuntimeSession:
        db = None
        try:
            if (
                project is not None
                and project_id is not None
                and project.id != project_id
            ):
                raise ValueError("The supplied project does not match project_id")
            if project is None and project_id is not None:
                db = self.session_factory()
                project = ProjectRepository(db).get(project_id)
                if project is None:
                    raise LookupError(f"Project {project_id!r} was not found")
            if project is None:
                project = self._anonymous_project()

            base_model = self.base_models.resolve(project)

            artifact_resolution = ArtifactResolution(None)
            if project.active_artifact_id:
                db = db or self.session_factory()
                artifact_resolution = ArtifactResolver(db).resolve(project, base_model)
            warnings = artifact_resolution.warnings
            artifact = artifact_resolution.artifact
            capabilities = {
                **self.provider.capabilities(),
                "adapter_resolution": True,
                "adapter_metadata_available": bool(
                    artifact and artifact.metadata is not None
                ),
                "adapter_loading": False,
                "weights_loaded": False,
            }
            return ModelRuntimeSession(
                project_id=project.id,
                project_name=project.name,
                base_model=base_model,
                active_artifact=artifact,
                adapter_path=artifact.adapter_path if artifact else None,
                runtime_type=self.provider.runtime_type,
                provider=self.provider.name,
                capabilities=capabilities,
                model_candidates=self.provider.model_candidates(),
                default_model=self.provider.default_model,
                validation_warnings=warnings,
                _provider_impl=self.provider,
                _provider_api_key=project.groq_api_key,
            )
        finally:
            if db is not None:
                db.close()

    @staticmethod
    def _anonymous_project():
        return type(
            "AnonymousRuntimeProject",
            (),
            {
                "id": "system",
                "name": "system",
                "base_model": DEFAULT_BASE_MODEL,
                "active_artifact_id": None,
                "groq_api_key": None,
            },
        )()


_model_runtime: ModelRuntime | None = None


def get_model_runtime() -> ModelRuntime:
    global _model_runtime
    if _model_runtime is None:
        _model_runtime = ModelRuntime()
    return _model_runtime
