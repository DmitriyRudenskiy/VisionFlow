from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

from src.domain.image.value_objects import FilePath
from src.domain.metadata.value_objects import ColorEntry, PoseKeypoint


@dataclass
class ColorPalette:
    file_path: FilePath
    colors: List[ColorEntry] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColorPalette):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def total_colors(self) -> int:
        return len(self.colors)

    def add_color(self, color: ColorEntry) -> None:
        self.colors.append(color)
        self.colors.sort(key=lambda c: c.percentage, reverse=True)


@dataclass
class PoseData:
    file_path: FilePath
    keypoints_body: List[PoseKeypoint] = field(default_factory=list)
    keypoints_face: List[PoseKeypoint] = field(default_factory=list)
    keypoints_left_hand: List[PoseKeypoint] = field(default_factory=list)
    keypoints_right_hand: List[PoseKeypoint] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoseData):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def total_keypoints(self) -> int:
        return (
            len(self.keypoints_body)
            + len(self.keypoints_face)
            + len(self.keypoints_left_hand)
            + len(self.keypoints_right_hand)
        )
