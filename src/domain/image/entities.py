from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from src.domain.image.value_objects import FilePath, ImageMetadata
from src.domain.image.exceptions import InvalidImageStateTransition


@dataclass
class ImageFile:
    file_path: FilePath
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[ImageMetadata] = None
    status: str = "active"
    processing_status: str = "pending"
    error_message: Optional[str] = None

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageFile):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def mark_as_processing(self) -> None:
        if self.processing_status != "pending":
            raise InvalidImageStateTransition(
                f"Cannot start processing image in status '{self.processing_status}'"
            )
        self.processing_status = "processing"
        self.touch()

    def mark_as_processed(self) -> None:
        if self.processing_status != "processing":
            raise InvalidImageStateTransition(
                f"Cannot complete processing image in status '{self.processing_status}'"
            )
        self.processing_status = "processed"
        self.touch()

    def mark_as_failed(self, error: str) -> None:
        if self.processing_status != "processing":
            raise InvalidImageStateTransition(
                f"Cannot fail processing image in status '{self.processing_status}'"
            )
        self.processing_status = "failed"
        self.error_message = error
        self.touch()
