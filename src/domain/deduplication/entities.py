from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID, uuid4

from src.domain.deduplication.value_objects import FileHash
from src.domain.image.value_objects import FilePath


@dataclass
class HashEntry:
    file_hash: FileHash
    file_path: FilePath
    file_size: int
    modified_at: float = 0.0
    is_original: bool = False
    is_duplicate: bool = False
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HashEntry):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class DuplicateGroup:
    group_id: str
    entries: List[HashEntry] = field(default_factory=list)
    original_entry: Optional[HashEntry] = None
    duplicates: List[HashEntry] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DuplicateGroup):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def add_entry(self, entry: HashEntry) -> None:
        self.entries.append(entry)
        self.touch()

    def resolve_original_by_size(self) -> None:
        if not self.entries:
            return
        sorted_entries = sorted(
            self.entries, key=lambda e: (e.file_size, e.modified_at), reverse=True
        )
        original = sorted_entries[0]
        original.is_original = True
        original.is_duplicate = False
        self.original_entry = original

        self.duplicates = []
        for dup in sorted_entries[1:]:
            dup.is_original = False
            dup.is_duplicate = True
            self.duplicates.append(dup)
        self.touch()
