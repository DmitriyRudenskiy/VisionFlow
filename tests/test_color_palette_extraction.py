# tests/test_color_palette_extraction.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from src.application.pipeline.dto import StepConfigDTO
from src.application.pipeline.steps.color_palette_extraction_step import ColorPaletteExtractionStep
from src.infrastructure.ai.color_client import ColorExtractorClient
from src.infrastructure.file_system import FileSystemStorage

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def color_sample_dir(tmp_path: Path) -> Path:
    """Копирует эталонные изображения во временную изолированную директорию."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Fixtures directory not found: {FIXTURES_DIR}")

    target = tmp_path / "samples"
    target.mkdir(parents=True, exist_ok=True)

    image_files: list[Path] = []
    for ext in sorted([".jpg", ".jpeg", ".png", ".webp", ".avif", ".gif", ".bmp", ".tiff"]):
        image_files.extend(FIXTURES_DIR.glob(f"*{ext}"))

    if not image_files:
        pytest.skip(f"No supported images found in {FIXTURES_DIR}")

    for img in image_files:
        shutil.copy(img, target / img.name)

    return target


@pytest.fixture
def expected_palette() -> list[dict[str, Any]]:
    """Загружает эталонный JSON с ожидаемыми палитрами."""
    json_files = sorted(FIXTURES_DIR.glob("*.json"))
    if not json_files:
        pytest.skip(f"Expected colors JSON not found in {FIXTURES_DIR}")

    all_expected: list[dict[str, Any]] = []
    for json_file in json_files:
        if json_file.stat().st_size == 0:
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "filename" in data:
                all_expected.append(data)
            elif "file" in data:
                all_expected.append({
                    "filename": data["file"],
                    "colors": data["colors"],
                })

    if not all_expected:
        pytest.skip(f"No valid expected colors found in {FIXTURES_DIR}")

    return all_expected


class TestColorPaletteExtractionIntegration:
    """Изолированный интеграционный тест шага извлечения цветов."""

    def test_extracted_colors_match_expected(
        self,
        color_sample_dir: Path,
        expected_palette: list[dict[str, Any]],
        file_storage: FileSystemStorage,
    ) -> None:
        extractor = ColorExtractorClient()
        step = ColorPaletteExtractionStep(file_storage, extractor)

        step.prepare()

        config = StepConfigDTO(
            sequence_number=7,
            params={
                "source_path": str(color_sample_dir),
                "output_path": str(color_sample_dir),
                "num_colors": 20,
            },
        )

        result = step.execute(config)

        assert result.status == "COMPLETED"
        expected_count = len(expected_palette)
        if expected_count > 0:
             assert result.processed_count >= expected_count, \
                 f"Expected at least {expected_count} processed files, got {result.processed_count}"

        colors_dir = color_sample_dir / "_colors"
        assert colors_dir.is_dir()

        generated_files = sorted(colors_dir.glob("*.json"))
        assert len(generated_files) == result.processed_count

        expected_by_stem: dict[str, list[dict[str, Any]]] = {}
        for item in expected_palette:
            stem = Path(item["filename"]).stem
            expected_by_stem[stem] = item["colors"]

        for gen_file in generated_files:
            with open(gen_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            source_name = data["file"]
            source_stem = Path(source_name).stem

            expected_colors = expected_by_stem.get(source_stem)
            if expected_colors is None:
                continue

            actual_colors = data["colors"]
            assert len(actual_colors) == len(expected_colors), (
                f"Color count mismatch for {source_name}: "
                f"got {len(actual_colors)}, expected {len(expected_colors)}"
            )

            for actual, expected in zip(actual_colors, expected_colors):
                assert actual["rgb"] == expected["rgb"], (
                    f"RGB mismatch for {source_name}: {actual['rgb']} != {expected['rgb']}"
                )
                assert actual["hex"].lower() == expected["hex"].lower(), (
                    f"Hex mismatch for {source_name}: {actual['hex']} != {expected['hex']}"
                )
                # Поддерживаем оба варианта именования в эталоне для обратной совместимости
                expected_pct = expected.get("percentage", expected.get("percent", 0))
                assert actual["percentage"] == pytest.approx(expected_pct, abs=5.0), (
                    f"Percentage mismatch for {source_name}: "
                    f"{actual['percentage']} vs expected {expected_pct}"
                )