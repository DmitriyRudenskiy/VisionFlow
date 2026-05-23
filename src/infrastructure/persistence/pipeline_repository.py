import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

from src.application.ports import PipelineRepository
from src.domain.pipeline.entities import PipelineAggregate
from src.infrastructure.persistence.pipeline_serializer import PipelineSerializer

logger = logging.getLogger(__name__)


class JsonPipelineRepository(PipelineRepository):
    def __init__(self, storage_dir: Path, serializer: PipelineSerializer):
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._serializer = serializer

    def _get_file_path(self, pipeline_id: UUID) -> Path:
        return self._storage_dir / f"pipeline_{pipeline_id}.json"

    def save(self, pipeline: PipelineAggregate) -> None:
        file_path = self._get_file_path(pipeline.id)
        tmp_path = file_path.with_suffix(".tmp")
        try:
            raw = self._serializer.serialize(pipeline)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(raw)
            tmp_path.replace(file_path)
        except OSError as e:
            logger.error(f"Failed to save pipeline {pipeline.id}: {e}")
            raise

    def find_by_id(self, id: UUID) -> Optional[PipelineAggregate]:
        file_path = self._get_file_path(id)
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
            return self._serializer.deserialize(raw)
        except Exception as e:
            logger.error(f"Failed to load pipeline {id}: {e}")
            return None