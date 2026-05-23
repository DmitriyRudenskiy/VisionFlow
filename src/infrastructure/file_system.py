# infrastructure/file_system.py
import hashlib
import os
import shutil
from pathlib import Path
from typing import List

from src.application.ports import FileSystemServicePort


class FileSystemService(FileSystemServicePort):
    def scan_directory(self, path: Path, recursive: bool = True) -> List[Path]:
        if not path.is_dir():
            return []
        pattern = "**/*" if recursive else "*"
        files = []
        for p in path.glob(pattern):
            if p.is_file():
                files.append(p)
        return files

    def move_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
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