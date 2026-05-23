import pytest
from pathlib import Path

from src.infrastructure.file_system import FileSystemStorage
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.infrastructure.persistence.pipeline_serializer import JsonPipelineSerializer


@pytest.fixture
def file_storage() -> FileSystemStorage:
    return FileSystemStorage()


@pytest.fixture
def repository(tmp_path: Path) -> JsonPipelineRepository:
    return JsonPipelineRepository(
        storage_dir=tmp_path / "pipelines",
        serializer=JsonPipelineSerializer(),
    )