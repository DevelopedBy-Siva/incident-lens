from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.api.routes_auth import get_current_project
from app.control.dataset_management import DatasetBuildCommand, NoEligibleIncidentsError
from app.control.models import Project
from app.shared.database import get_db
from app.training.dataset_storage import DatasetAlreadyExistsError
from app.training.models import (
    DatasetStatus,
    ModelArtifactStatus,
    TrainingJobStatus,
)
from app.training.repositories import (
    DatasetRepository,
    ModelArtifactRepository,
    TrainingJobRepository,
)

router = APIRouter()


class DatasetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_version: str
    storage_key: str
    record_count: int
    status: DatasetStatus
    created_at: datetime


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    project_id: str
    dataset_id: str
    status: TrainingJobStatus
    artifact_id: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


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


@router.get("/training-jobs", response_model=list[TrainingJobResponse])
def list_training_jobs(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    return TrainingJobRepository(db).list_for_project(project.id)


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


@router.get("/model-artifacts", response_model=list[ModelArtifactResponse])
def list_model_artifacts(
    project: Project = Depends(get_current_project),
    db: Session = Depends(get_db),
):
    return ModelArtifactRepository(db).list_for_project(project.id)


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
