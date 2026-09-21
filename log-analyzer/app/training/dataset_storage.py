import os
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath


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
        target = (self.root / PurePosixPath(storage_key)).resolve()
        if self.root != target and self.root not in target.parents:
            raise ValueError("Dataset storage key escapes configured root")

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


def configured_dataset_storage() -> DatasetStorage:
    return LocalDatasetStorage(os.getenv("DATASET_STORAGE_PATH", "datasets"))
