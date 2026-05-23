import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import PipelineStatus, StepStatus, StepConfig


class PipelineSerializer(ABC):
    @abstractmethod
    def serialize(self, pipeline: PipelineAggregate) -> str: ...

    @abstractmethod
    def deserialize(self, raw: str) -> PipelineAggregate: ...


class JsonPipelineSerializer(PipelineSerializer):
    def serialize(self, pipeline: PipelineAggregate) -> str:
        data: Dict[str, Any] = {
            "id": str(pipeline.id),
            "name": pipeline.name,
            "status": pipeline.status.value,
            "source_directory": str(pipeline.source_directory),
            "output_directory": str(pipeline.output_directory),
            "created_at": pipeline.created_at.isoformat(),
            "updated_at": pipeline.updated_at.isoformat(),
            "steps": [
                {
                    "id": str(step.id),
                    "sequence_number": step.sequence_number,
                    "name": step.name,
                    "status": step.status.value,
                    "config": {"params": step.config.params},
                    "started_at": step.started_at.isoformat() if step.started_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error": step.error,
                    "created_at": step.created_at.isoformat(),
                    "updated_at": step.updated_at.isoformat(),
                }
                for step in pipeline.steps
            ],
        }
        return json.dumps(data, indent=4, ensure_ascii=False)

    def deserialize(self, raw: str) -> PipelineAggregate:
        data = json.loads(raw)
        steps = [
            PipelineStep(
                id=UUID(step_data["id"]),
                sequence_number=step_data["sequence_number"],
                name=step_data["name"],
                status=StepStatus(step_data["status"]),
                config=StepConfig(params=step_data["config"].get("params", {})),
                started_at=datetime.fromisoformat(step_data["started_at"]) if step_data.get("started_at") else None,
                completed_at=datetime.fromisoformat(step_data["completed_at"]) if step_data.get("completed_at") else None,
                error=step_data.get("error"),
                created_at=datetime.fromisoformat(step_data["created_at"]),
                updated_at=datetime.fromisoformat(step_data["updated_at"]),
            )
            for step_data in data.get("steps", [])
        ]
        return PipelineAggregate(
            name=data["name"],
            id=UUID(data["id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=PipelineStatus(data["status"]),
            steps=steps,
            source_directory=Path(data["source_directory"]),
            output_directory=Path(data["output_directory"]),
        )