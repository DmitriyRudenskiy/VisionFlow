from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import Any

@dataclass
class BaseEntity:
    """Базовый класс для всех сущностей (Entity)."""
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

@dataclass(frozen=True)
class BaseValueObject:
    """Базовый класс для объектов-значений (Value Objects).
    Immutable (frozen=True). Сравнение происходит автоматически по всем полям."""
    pass