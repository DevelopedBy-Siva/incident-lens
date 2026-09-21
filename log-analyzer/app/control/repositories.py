from sqlalchemy.orm import Session

from app.control.models import Project


class ProjectRepository:
    """Persistence operations for projects."""

    def __init__(self, db: Session):
        self.db = db

    def get(self, project_id: str) -> Project | None:
        return self.db.query(Project).filter(Project.id == project_id).first()

    def add(self, project: Project) -> Project:
        self.db.add(project)
        return project

    def set_active_artifact(self, project: Project, artifact_id: str | None) -> Project:
        project.active_artifact_id = artifact_id
        return project
