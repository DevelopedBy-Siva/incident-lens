import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactMetadataAlreadyExistsError(FileExistsError):
    pass


class ArtifactMetadataWriter(ABC):
    @abstractmethod
    def artifact_path(self, project_id: str, artifact_version: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        project_id: str,
        artifact_version: str,
        metadata: dict[str, Any],
    ) -> None:
        raise NotImplementedError


class LocalArtifactMetadataWriter(ArtifactMetadataWriter):
    """Write artifact metadata only; no model weights are created."""

    def __init__(self, root: str | Path):
        self.configured_root = Path(root)
        self.root = self.configured_root.expanduser().resolve()

    def artifact_path(self, project_id: str, artifact_version: str) -> str:
        return f"{(self.configured_root / project_id / artifact_version).as_posix()}/"

    def write(
        self,
        project_id: str,
        artifact_version: str,
        metadata: dict[str, Any],
    ) -> None:
        artifact_directory = (self.root / project_id / artifact_version).resolve()
        if self.root not in artifact_directory.parents:
            raise ValueError("Artifact path escapes configured root")

        try:
            artifact_directory.mkdir(parents=True, exist_ok=False)
            metadata_path = artifact_directory / "metadata.json"
            with metadata_path.open("x", encoding="utf-8") as metadata_file:
                json.dump(
                    metadata,
                    metadata_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                metadata_file.write("\n")
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
        except FileExistsError as exc:
            raise ArtifactMetadataAlreadyExistsError(
                f"Artifact metadata already exists for {artifact_version}"
            ) from exc


def configured_artifact_metadata_writer() -> ArtifactMetadataWriter:
    return LocalArtifactMetadataWriter(os.getenv("ARTIFACT_STORAGE_PATH", "artifacts"))
