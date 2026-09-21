from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control.model_management import ModelManagementService
from app.control.repositories import ProjectRepository
from app.training.dataset_builder import DatasetBuilder, DatasetEligibilityRules
from app.training.dataset_serializer import JsonLinesDatasetSerializer
from app.training.dataset_storage import DatasetStorage, configured_dataset_storage
from app.training.models import Dataset
from app.training.repositories import DatasetRepository, DatasetSourceRepository


class NoEligibleIncidentsError(ValueError):
    pass


class DatasetBuildCommand:
    """Synchronously build, store, and register one immutable dataset version."""

    VERSION_RESERVATION_ATTEMPTS = 3

    def __init__(
        self,
        db: Session,
        storage: DatasetStorage | None = None,
        serializer: JsonLinesDatasetSerializer | None = None,
        eligibility_rules: DatasetEligibilityRules | None = None,
    ):
        self.db = db
        self.storage = storage or configured_dataset_storage()
        self.serializer = serializer or JsonLinesDatasetSerializer()
        self.builder = DatasetBuilder(
            DatasetSourceRepository(db), rules=eligibility_rules
        )
        self.datasets = DatasetRepository(db)
        self.model_management = ModelManagementService(db)

    def execute(self, project_id: str) -> Dataset:
        if not ProjectRepository(self.db).get(project_id):
            raise ValueError("Project not found")

        examples = self.builder.build(project_id)
        if not examples:
            raise NoEligibleIncidentsError(
                "No incidents satisfy the dataset eligibility rules"
            )
        serialized = self.serializer.serialize(examples)
        dataset = self._reserve_version(project_id)

        try:
            self.storage.save(dataset.storage_key, serialized)
        except Exception:
            self.model_management.mark_dataset_failed(project_id, dataset.id)
            raise

        return self.model_management.mark_dataset_ready(
            project_id, dataset.id, record_count=len(examples)
        )

    def _reserve_version(self, project_id: str) -> Dataset:
        for attempt in range(self.VERSION_RESERVATION_ATTEMPTS):
            dataset_version = self.datasets.next_version(project_id)
            storage_key = self.storage.storage_key(
                project_id,
                dataset_version,
                self.serializer.file_extension,
            )
            try:
                return self.model_management.create_dataset_metadata(
                    project_id=project_id,
                    dataset_version=dataset_version,
                    storage_key=storage_key,
                )
            except IntegrityError:
                if attempt == self.VERSION_RESERVATION_ATTEMPTS - 1:
                    raise

        raise RuntimeError("Unable to reserve a dataset version")
