import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any


class ArtifactMetadataAlreadyExistsError(FileExistsError):
    pass


class ArtifactMetadataWriter(ABC):
    @abstractmethod
    def artifact_path(self, project_id: str, artifact_version: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, project_id: str, artifact_version: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        project_id: str,
        artifact_version: str,
        metadata: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, project_id: str, artifact_version: str) -> None:
        raise NotImplementedError

    def ensure_local(
        self, project_id: str, artifact_version: str, current_path: str
    ) -> str:
        """Return a local artifact directory, restoring it when necessary."""
        return current_path


class LocalArtifactMetadataWriter(ArtifactMetadataWriter):
    """Own immutable local adapter directories and their metadata file."""

    def __init__(self, root: str | Path):
        self.configured_root = Path(root)
        self.root = self.configured_root.expanduser().resolve()

    def artifact_path(self, project_id: str, artifact_version: str) -> str:
        return f"{self._directory(project_id, artifact_version).as_posix()}/"

    def prepare(self, project_id: str, artifact_version: str) -> str:
        artifact_directory = self._directory(project_id, artifact_version)
        try:
            artifact_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ArtifactMetadataAlreadyExistsError(
                f"Artifact directory already exists for {artifact_version}"
            ) from exc
        return self.artifact_path(project_id, artifact_version)

    def write(
        self,
        project_id: str,
        artifact_version: str,
        metadata: dict[str, Any],
    ) -> None:
        artifact_directory = self._directory(project_id, artifact_version)

        try:
            artifact_directory.mkdir(parents=True, exist_ok=True)
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

    def remove(self, project_id: str, artifact_version: str) -> None:
        artifact_directory = self._directory(project_id, artifact_version)
        if artifact_directory.exists():
            shutil.rmtree(artifact_directory)

    def _directory(self, project_id: str, artifact_version: str) -> Path:
        artifact_directory = (self.root / project_id / artifact_version).resolve()
        if self.root not in artifact_directory.parents:
            raise ValueError("Artifact path escapes configured root")
        return artifact_directory


class S3ArtifactMetadataWriter(LocalArtifactMetadataWriter):
    """Train from a local cache and persist completed adapters in S3."""

    def __init__(
        self,
        root: str | Path,
        bucket: str,
        prefix: str = "artifacts",
        client=None,
    ):
        super().__init__(root)
        if not bucket:
            raise ValueError("S3_BUCKET is required for production storage")
        if client is None:
            import boto3

            client = boto3.client(
                "s3", region_name=os.getenv("AWS_REGION", "").strip() or None
            )
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def write(
        self,
        project_id: str,
        artifact_version: str,
        metadata: dict[str, Any],
    ) -> None:
        super().write(project_id, artifact_version, metadata)
        directory = self._directory(project_id, artifact_version)
        for path in sorted(entry for entry in directory.rglob("*") if entry.is_file()):
            self.client.upload_file(
                str(path),
                self.bucket,
                self._key(project_id, artifact_version, path.relative_to(directory)),
            )

    def ensure_local(
        self, project_id: str, artifact_version: str, current_path: str
    ) -> str:
        directory = self._directory(project_id, artifact_version)
        if (directory / "metadata.json").is_file():
            return self.artifact_path(project_id, artifact_version)

        prefix = self._key(project_id, artifact_version, "")
        paginator = self.client.get_paginator("list_objects_v2")
        found = False
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                relative = key.removeprefix(prefix)
                if not relative:
                    continue
                target = (directory / PurePosixPath(relative)).resolve()
                if directory != target and directory not in target.parents:
                    raise ValueError("S3 artifact key escapes local cache root")
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, key, str(target))
                found = True
        if not found:
            return current_path
        return self.artifact_path(project_id, artifact_version)

    def remove(self, project_id: str, artifact_version: str) -> None:
        super().remove(project_id, artifact_version)
        prefix = self._key(project_id, artifact_version, "")
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                self.client.delete_objects(
                    Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True}
                )

    def _key(
        self, project_id: str, artifact_version: str, relative: str | PurePosixPath
    ) -> str:
        base = PurePosixPath(self.prefix) / project_id / artifact_version
        key = base / PurePosixPath(relative) if str(relative) else base
        value = key.as_posix()
        return f"{value}/" if not str(relative) else value


def configured_artifact_metadata_writer() -> ArtifactMetadataWriter:
    root = os.getenv("ARTIFACT_STORAGE_PATH", "artifacts")
    if os.getenv("APP_ENV", "development").strip().lower() in {"prod", "production"}:
        return S3ArtifactMetadataWriter(
            root=root,
            bucket=os.getenv("S3_BUCKET", "").strip(),
            prefix=os.getenv("S3_ARTIFACT_PREFIX", "artifacts"),
        )
    return LocalArtifactMetadataWriter(root)
