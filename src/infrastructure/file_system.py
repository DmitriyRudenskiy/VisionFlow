# src/infrastructure/file_system.py
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, List

from src.application.ports import StoragePort

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".avif"}
)


class FileSystemStorage(StoragePort):
    def scan_directory(self, path: Path, recursive: bool = True) -> List[Path]:
        """Возвращает список файлов изображений."""
        if not path.is_dir():
            return []
        pattern = "**/*" if recursive else "*"
        return sorted([
            p for p in path.glob(pattern)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

    def get_all_files(self, path: Path, recursive: bool = True) -> List[Path]:
        """Возвращает список всех файлов в директории без фильтрации по расширению."""
        if not path.is_dir():
            return []
        pattern = "**/*" if recursive else "*"
        return sorted([p for p in path.glob(pattern) if p.is_file()])

    def move_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination != source:
            if destination.is_file():
                destination.unlink(missing_ok=True)
            else:
                raise IsADirectoryError(
                    f"Cannot move file over directory: {destination}"
                )
        shutil.move(str(source), str(destination))

    def copy_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))

    def create_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    def get_file_hash(self, path: Path, algorithm: str = "md5") -> str:
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_file_size(self, path: Path) -> int:
        return path.stat().st_size

    def get_file_modified_time(self, path: Path) -> float:
        return path.stat().st_mtime

    def persist_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

    def load_text(self, path: Path, encoding: str = "utf-8") -> str:
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def persist_json(self, path: Path, data: Any, encoding: str = "utf-8") -> None:
        self.persist_text(path, json.dumps(data, ensure_ascii=False, indent=2), encoding)

    def load_json(self, path: Path, encoding: str = "utf-8") -> Any:
        return json.loads(self.load_text(path, encoding))

    def path_exists(self, path: Path) -> bool:
        return path.exists()