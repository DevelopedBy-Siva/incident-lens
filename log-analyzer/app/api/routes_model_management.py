from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.routes_auth import get_current_project
from app.control.dataset_management import (
    DatasetBuildCommand,
    DatasetImportCommand,
    NoEligibleIncidentsError,
)
from app.control.model_management import ModelManagementService
from app.control.models import Project
from app.shared.database import get_db
from app.shared.model_config import configured_runtime_settings
from app.training.artifact_metadata import configured_artifact_storage_location
from app.training.dataset_serializer import (
    DatasetValidationError,
    JsonLinesDatasetSerializer,
)
from app.training.dataset_storage import (
    DatasetAlreadyExistsError,
    configured_dataset_storage,
    configured_dataset_storage_location,
)
from app.training.models import (
    DatasetStatus,
    ModelArtifact,
    ModelArtifactStatus,
    TrainingJob,
    TrainingJobStatus,
)
from app.training.repositories import (
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)
from app.training.worker import (
    TrainingJobNotFoundError,
    TrainingJobStateError,
    TrainingWorker,
)

router = APIRouter()


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_version: str
    storage_key: str
    record_count: int
    selected_record_indices: list[int] | None
    status: DatasetStatus
    created_at: datetime


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_id: str
    status: TrainingJobStatus
    artifact_id: str | None
    ec2_instance_id: str | None  # EC2 instance ID if training uses remote GPU
    selected_record_indices: list[int] | None
    progress_current_step: int | None
    progress_total_steps: int | None
    progress_updated_at: datetime | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TrainingJobCreate(BaseModel):
    dataset_id: str
    selected_record_indices: list[int] | None = None


class DatasetRecordResponse(BaseModel):
    index: int
    data: dict[str, Any]


class DatasetContentsResponse(BaseModel):
    dataset: DatasetResponse
    records: list[DatasetRecordResponse]


class DatasetUploadRequest(BaseModel):
    records: list[dict[str, Any]]


class DatasetApprovalRequest(BaseModel):
    selected_record_indices: list[int]


class ModelArtifactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    artifact_version: str
    base_model: str
    adapter_path: str | None
    dataset_id: str
    evaluation_score: float | None
    status: ModelArtifactStatus
    created_at: datetime


class ModelRuntimeResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    project_id: str
    project_name: str
    provider: str
    runtime_type: str
    base_model: str
    model_path: str
    device: str
    dtype: str
    active_artifact_id: str | None
    active_artifact_version: str | None
    adapter_path: str | None
    artifact_storage: str
    dataset_storage: str


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    return DatasetRepository(db).list_for_project(project.id)


@router.post("/datasets/build", response_model=DatasetResponse, status_code=201)
def build_dataset(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        return DatasetBuildCommand(db).execute(project.id)
    except NoEligibleIncidentsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DatasetAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/datasets/upload", response_model=DatasetResponse, status_code=201)
def upload_dataset(
    request: DatasetUploadRequest,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    try:
        return DatasetImportCommand(db).execute(project.id, request.records)
    except DatasetValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DatasetAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(
    dataset_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/records", response_model=DatasetContentsResponse)
def get_dataset_records(
    dataset_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status not in {DatasetStatus.VALIDATING, DatasetStatus.READY}:
        raise HTTPException(status_code=409, detail="Dataset is not available to view")

    try:
        records = JsonLinesDatasetSerializer().deserialize(
            configured_dataset_storage().load(dataset.storage_key)
        )
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="Dataset contents are not available"
        ) from exc

    return DatasetContentsResponse(
        dataset=DatasetResponse.model_validate(dataset),
        records=[
            DatasetRecordResponse(index=index, data=record)
            for index, record in enumerate(records)
        ],
    )


@router.get("/datasets/{dataset_id}/download")
def download_dataset(
    dataset_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status not in {
        DatasetStatus.VALIDATING,
        DatasetStatus.READY,
        DatasetStatus.TRAINED,
    }:
        raise HTTPException(status_code=409, detail="Dataset is not available")
    try:
        content = configured_dataset_storage().load(dataset.storage_key)
        JsonLinesDatasetSerializer().deserialize(content)
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=409, detail="Dataset contents are not available"
        ) from exc

    return Response(
        content=content,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{dataset.dataset_version}.jsonl"'
            )
        },
    )


@router.post("/datasets/{dataset_id}/approve", response_model=DatasetResponse)
def approve_dataset(
    dataset_id: str,
    request: DatasetApprovalRequest,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status != DatasetStatus.VALIDATING:
        raise HTTPException(status_code=409, detail="Dataset is not pending review")

    selected_indices = request.selected_record_indices
    if not selected_indices:
        raise HTTPException(
            status_code=422, detail="Select at least one dataset record"
        )
    if len(selected_indices) != len(set(selected_indices)):
        raise HTTPException(
            status_code=422, detail="Dataset record selection contains duplicates"
        )
    if min(selected_indices) < 0 or max(selected_indices) >= dataset.record_count:
        raise HTTPException(
            status_code=422, detail="Dataset record selection is out of range"
        )

    try:
        return ModelManagementService(db).mark_dataset_ready(
            project.id,
            dataset.id,
            dataset.record_count,
            sorted(selected_indices),
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    is_referenced = (
        db.query(TrainingJob.id)
        .filter(
            TrainingJob.project_id == project.id,
            TrainingJob.dataset_id == dataset.id,
        )
        .first()
        is not None
        or db.query(ModelArtifact.id)
        .filter(
            ModelArtifact.project_id == project.id,
            ModelArtifact.dataset_id == dataset.id,
        )
        .first()
        is not None
    )
    if is_referenced:
        raise HTTPException(
            status_code=409,
            detail="This dataset is used by training history and cannot be deleted",
        )

    try:
        configured_dataset_storage().delete(dataset.storage_key)
        db.delete(dataset)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/training-jobs", response_model=list[TrainingJobResponse])
def list_training_jobs(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    return TrainingJobRepository(db).list_for_project(project.id)


@router.post("/training-jobs", response_model=TrainingJobResponse, status_code=201)
def create_training_job(
    request: TrainingJobCreate,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    dataset = DatasetRepository(db).get_for_project(request.dataset_id, project.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if dataset.status not in {DatasetStatus.READY, DatasetStatus.TRAINED}:
        raise HTTPException(status_code=409, detail="Dataset is not ready for training")

    selected_indices = request.selected_record_indices
    if selected_indices is not None:
        if not selected_indices:
            raise HTTPException(
                status_code=422, detail="Select at least one dataset record"
            )
        if len(selected_indices) != len(set(selected_indices)):
            raise HTTPException(
                status_code=422, detail="Dataset record selection contains duplicates"
            )
        if min(selected_indices) < 0 or max(selected_indices) >= dataset.record_count:
            raise HTTPException(
                status_code=422, detail="Dataset record selection is out of range"
            )
        selected_indices = sorted(selected_indices)
    return ModelManagementService(db).create_training_job(
        project.id, request.dataset_id, selected_indices
    )


@router.get("/training-jobs/{job_id}", response_model=TrainingJobResponse)
def get_training_job(
    job_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    job = TrainingJobRepository(db).get_for_project(job_id, project.id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.post("/training-jobs/{job_id}/run")
def run_training_job(
    job_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Execute a training job.
    
    Returns:
    - 200 OK with completed job if local/synchronous execution (development mode)
    - 202 Accepted with running job if EC2 remote execution (production mode)
    
    The response status code indicates the execution mode:
    - 200: Job execution started and completed immediately (local mode)
    - 202: Job execution delegated to EC2 GPU instance (remote mode, async)
    """
    try:
        worker = TrainingWorker(db)
        job = worker.run(job_id, project.id)
        job_response = TrainingJobResponse.model_validate(job)
        
        # Determine response status based on execution mode
        if worker.use_ec2_remote and job.status == TrainingJobStatus.RUNNING:
            # EC2 remote mode: return 202 Accepted with running job
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=job_response.model_dump(),
            )
        else:
            # Local mode: return 200 OK with completed job
            return job_response
            
    except TrainingJobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TrainingJobStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/model-artifacts", response_model=list[ModelArtifactResponse])
def list_model_artifacts(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    return ModelArtifactRepository(db).list_for_project(project.id)


@router.get("/model-runtime", response_model=ModelRuntimeResponse)
def get_model_runtime_status(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    """Expose the existing local runtime configuration to the dashboard."""
    settings = configured_runtime_settings()
    artifact = None
    if project.active_artifact_id:
        artifact = ModelArtifactRepository(db).get_for_project(
            project.active_artifact_id, project.id
        )
    return ModelRuntimeResponse(
        project_id=project.id,
        project_name=project.name,
        provider=settings.provider,
        runtime_type="local_peft",
        base_model=settings.base_model,
        model_path=settings.base_model,
        device=settings.device,
        dtype=settings.dtype,
        active_artifact_id=artifact.id if artifact else None,
        active_artifact_version=artifact.artifact_version if artifact else None,
        adapter_path=artifact.adapter_path if artifact else None,
        artifact_storage=configured_artifact_storage_location(),
        dataset_storage=configured_dataset_storage_location(),
    )


@router.get("/model-artifacts/{artifact_id}", response_model=ModelArtifactResponse)
def get_model_artifact(
    artifact_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    artifact = ModelArtifactRepository(db).get_for_project(artifact_id, project.id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    return artifact


@router.post(
    "/model-artifacts/{artifact_id}/activate",
    response_model=ModelArtifactResponse,
)
def activate_model_artifact(
    artifact_id: str,
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    artifact = ModelArtifactRepository(db).get_for_project(artifact_id, project.id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Model artifact not found")
    if artifact.status != ModelArtifactStatus.READY:
        raise HTTPException(
            status_code=409, detail="Only READY artifacts can be activated"
        )
    ModelManagementService(db).update_active_artifact(project.id, artifact.id)
    return artifact
