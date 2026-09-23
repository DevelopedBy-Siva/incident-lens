import os
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath

from botocore.exceptions import ClientError


class DatasetAlreadyExistsError(FileExistsError):
    pass


class DatasetStorage(ABC):
    """Storage boundary for immutable serialized datasets."""

    @abstractmethod
    def storage_key(
        self, project_id: str, dataset_version: str, file_extension: str
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def save(self, storage_key: str, content: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, storage_key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_key: str) -> None:
        raise NotImplementedError


class LocalDatasetStorage(DatasetStorage):
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def storage_key(
        self, project_id: str, dataset_version: str, file_extension: str
    ) -> str:
        return str(
            PurePosixPath("projects")
            / project_id
            / f"{dataset_version}.{file_extension}"
        )

    def save(self, storage_key: str, content: bytes) -> None:
        target = self._resolve(storage_key)

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as dataset_file:
                dataset_file.write(content)
                dataset_file.flush()
                os.fsync(dataset_file.fileno())
        except FileExistsError as exc:
            raise DatasetAlreadyExistsError(
                f"Dataset already exists at {storage_key}"
            ) from exc
        except Exception:
            if target.exists():
                target.unlink()
            raise

    def load(self, storage_key: str) -> bytes:
        return self._resolve(storage_key).read_bytes()

    def delete(self, storage_key: str) -> None:
        target = self._resolve(storage_key)
        if target.exists():
            target.unlink()

    def _resolve(self, storage_key: str) -> Path:
        target = (self.root / PurePosixPath(storage_key)).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError("Dataset storage key escapes configured root")
        return target


class S3DatasetStorage(DatasetStorage):
    """Immutable dataset storage backed by a private S3 bucket."""

    def __init__(self, bucket: str, prefix: str = "datasets", client=None):
        if not bucket:
            raise ValueError("S3_BUCKET is required for S3 dataset storage")
        if client is None:
            import boto3

            client = boto3.client(
                "s3", region_name=os.getenv("AWS_REGION", "").strip() or None
            )
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def storage_key(
        self, project_id: str, dataset_version: str, file_extension: str
    ) -> str:
        relative = str(
            PurePosixPath("projects")
            / project_id
            / f"{dataset_version}.{file_extension}"
        )
        return str(PurePosixPath(self.prefix) / relative) if self.prefix else relative

    def save(self, storage_key: str, content: bytes) -> None:
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=storage_key,
                Body=content,
                ContentType="application/x-ndjson",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            code = exc.response.get("Error", {}).get("Code")
            if status == 412 or code in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise DatasetAlreadyExistsError(
                    f"Dataset already exists at {storage_key}"
                ) from exc
            raise

    def load(self, storage_key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=storage_key)
        return response["Body"].read()

    def delete(self, storage_key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=storage_key)


def configured_dataset_storage() -> DatasetStorage:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if bucket:
        return S3DatasetStorage(
            bucket=bucket,
            prefix=os.getenv("S3_DATASET_PREFIX", "datasets"),
        )
    return LocalDatasetStorage(os.getenv("DATASET_STORAGE_PATH", "datasets"))


def configured_dataset_storage_location() -> str:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if bucket:
        prefix = os.getenv("S3_DATASET_PREFIX", "datasets").strip("/")
        return f"s3://{bucket}/{prefix}" if prefix else f"s3://{bucket}"
    return str(
        Path(os.getenv("DATASET_STORAGE_PATH", "datasets")).expanduser().resolve()
    )
