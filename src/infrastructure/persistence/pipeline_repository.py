import json
from pathlib import Path
from typing import Optional
from uuid import UUID
from datetime import datetime

from src.application.ports import PipelineRepository
from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import PipelineStatus, StepStatus, StepConfig

class PipelineMapper:
    @staticmethod
    def to_dict(pipeline: PipelineAggregate) -> dict:
        return {
            "id": str(pipeline.id),
            "name": pipeline.name,
            "status": pipeline.status.value,
            "source_path": str(pipeline.source_path),
            "output_path": str(pipeline.output_path),
            "created_at": pipeline.created_at.isoformat(),
            "updated_at": pipeline.updated_at.isoformat(),
            "steps": [
                {
                    "id": str(step.id),
                    "step_number": step.step_number,
                    "step_name": step.step_name,
                    "status": step.status.value,
                    "config": {"params": step.config.params},
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error": step.error,
                    "created_at": step.created_at.isoformat(),
                    "updated_at": step.updated_at.isoformat()
                } for step in pipeline.steps
            ]
        }

    @staticmethod
    def from_dict(data: dict) -> PipelineAggregate:
        steps = [
            PipelineStep(
                id=UUID(step_data["id"]),
                step_number=step_data["step_number"],
                step_name=step_data["step_name"],
                status=StepStatus(step_data["status"]),
                config=StepConfig(params=step_data["config"].get("params", {})),
                started_at=datetime.fromisoformat(step_data["started_at"]) if step_data.get("started_at") else None,
                completed_at=datetime.fromisoformat(step_data["completed_at"]) if step_data.get("completed_at") else None,
                error=step_data.get("error"),
                created_at=datetime.fromisoformat(step_data["created_at"]),
                updated_at=datetime.fromisoformat(step_data["updated_at"])
            ) for step_data in data.get("steps", [])
        ]
        return PipelineAggregate(
            name=data["name"],
            id=UUID(data["id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=PipelineStatus(data["status"]),
            steps=steps,
            source_path=Path(data["source_path"]),
            output_path=Path(data["output_path"]),
        )

class JsonPipelineRepository(PipelineRepository):
    def __init__(self, storage_dir: Path):
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, pipeline_id: UUID) -> Path:
        return self._storage_dir / f"pipeline_{pipeline_id}.json"

    def save(self, pipeline: PipelineAggregate) -> None:
        data = PipelineMapper.to_dict(pipeline)
        file_path = self._get_file_path(pipeline.id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def find_by_id(self, id: UUID) -> Optional[PipelineAggregate]:
        file_path = self._get_file_path(id)
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PipelineMapper.from_dict(data)