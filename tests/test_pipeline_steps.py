# tests/test_pipeline_steps.py
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.application.pipeline.dto import StepConfigDTO
from src.application.pipeline.steps.flatten_directories_step import FlattenDirectoriesStep
from src.application.pipeline.steps.prepare_images_step import PrepareImagesStep
from src.application.pipeline.steps.exact_deduplication_step import ExactDeduplicationStep
from src.application.pipeline.steps.visual_deduplication_step import VisualDeduplicationStep
from src.application.pipeline.steps.smart_crop_step import SmartCropStep
from src.application.pipeline.steps.embedding_extraction_step import EmbeddingExtractionStep
from src.application.pipeline.steps.pose_extraction_step import PoseExtractionStep
from src.application.pipeline.steps.color_palette_extraction_step import ColorPaletteExtractionStep
from src.application.pipeline.steps.content_safety_step import ContentSafetyClassificationStep
from src.infrastructure.file_system import FileSystemStorage


def make_step_config(sequence_number: int, source: Path) -> StepConfigDTO:
    return StepConfigDTO(
        sequence_number=sequence_number,
        params={"source_path": str(source), "output_path": str(source)},
    )


def create_test_file(path: Path, name: str, content: bytes = b"img") -> Path:
    file_path = path / name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(content)
    return file_path


class TestFlattenDirectoriesStep:
    def test_flatten_moves_files_to_root(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        nested = src / "sub"
        nested.mkdir()
        create_test_file(nested, "a.jpg")
        create_test_file(src, "b.png")
        create_test_file(src, "c.jpg")
        create_test_file(nested, "c.jpg")

        step = FlattenDirectoriesStep(file_storage)
        config = make_step_config(0, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.message.startswith("Flattened directory")
        root_files = [f for f in src.iterdir() if f.is_file()]
        assert len(root_files) == 4
        assert (src / "a.jpg").exists()
        assert (src / "b.png").exists()
        assert (src / "c.jpg").exists()
        assert any(f.name.startswith("c_") for f in root_files)


class TestPrepareImagesStep:
    def test_backup_and_rename(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "photo.jpg")
        create_test_file(src, "image.png")
        (src / "readme.txt").write_text("hello")

        step = PrepareImagesStep(file_storage)
        config = make_step_config(1, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_originals").is_dir()
        assert (src / "_originals" / "photo.jpg").exists()
        assert (src / "_originals" / "image.png").exists()
        assert not (src / "_originals" / "readme.txt").exists()

        assert len(list((src / "_originals").iterdir())) == 2

        root_files = [f for f in src.iterdir() if f.is_file()]
        assert len(root_files) == 3
        assert (src / "readme.txt").exists()

    def test_ignore_non_images(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "data.bin")

        step = PrepareImagesStep(file_storage)
        config = make_step_config(1, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert not (src / "_originals" / "data.bin").exists()
        assert (src / "data.bin").exists()


class TestExactDeduplicationStep:
    def test_detect_exact_duplicates(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        content = b"duplicate content"
        create_test_file(src, "orig.jpg", content)
        create_test_file(src, "copy.jpg", content)
        create_test_file(src, "unique.png", b"different")

        step = ExactDeduplicationStep(file_storage)
        config = make_step_config(2, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        dup_dir = src / "_duplicates"
        assert dup_dir.is_dir()
        duplicates = list(dup_dir.iterdir())
        assert len(duplicates) == 1

        remaining_dups = [f for f in src.iterdir() if f.is_file() and f.name in ("orig.jpg", "copy.jpg")]
        assert len(remaining_dups) == 1
        assert (src / "unique.png").exists()


class TestVisualDeduplicationStep:
    def test_visual_duplicates_creates_report(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "a.jpg")
        create_test_file(src, "b.png")

        detector = MagicMock()
        detector.calculate_phash.side_effect = ["hash_a", "hash_b"]

        step = VisualDeduplicationStep(file_storage, detector=detector)
        config = make_step_config(3, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert (src / "_visual_duplicates").is_dir()
        report = src / "visual_dups_report.html"
        assert report.exists()
        assert "Visual Duplicates Report" in report.read_text()
        assert len(list((src / "_visual_duplicates").iterdir())) == 0

    def test_visual_duplicates_moves_files(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "a.jpg", b"1")
        create_test_file(src, "b.png", b"22")

        detector = MagicMock()
        detector.calculate_phash.return_value = "same_hash"

        step = VisualDeduplicationStep(file_storage, detector=detector)
        config = make_step_config(3, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert len(list((src / "_visual_duplicates").iterdir())) == 1
        assert (src / "b.png").exists()


class TestSmartCropStep:
    def test_smart_crop_keeps_files_in_root(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pic.jpg")
        create_test_file(src, "img.png")

        segmenter = MagicMock()
        segmenter.crop_image.return_value = src / "pic.jpg"

        step = SmartCropStep(file_storage, segmenter=segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.processed_count == 2
        assert (src / "pic.jpg").exists()
        assert (src / "img.png").exists()

    def test_smart_crop_handles_temp_file(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pic2.jpg")

        temp_crop = tmp_path / "temp_crop.jpg"
        temp_crop.write_bytes(b"cropped")

        segmenter = MagicMock()
        segmenter.crop_image.return_value = temp_crop

        step = SmartCropStep(file_storage, segmenter=segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.processed_count == 1
        assert not temp_crop.exists()
        # Оригинал должен быть заменён кропнутой версией
        assert (src / "pic2.jpg").exists()
        assert (src / "pic2.jpg").read_bytes() == b"cropped"

    def test_smart_crop_reports_errors(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "fail.jpg")

        segmenter = MagicMock()
        segmenter.crop_image.side_effect = RuntimeError("Segmentation failed")

        step = SmartCropStep(file_storage, segmenter=segmenter)
        config = make_step_config(4, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        assert result.processed_count == 0
        assert result.errors is not None
        assert any("Segmentation failed" in err for err in result.errors)


class TestEmbeddingExtractionStep:
    def test_creates_vector_json(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "img.jpg")

        vectorizer = MagicMock()
        vectorizer.get_embedding.return_value = [0.1] * 512

        step = EmbeddingExtractionStep(file_storage, extractor=vectorizer)
        config = make_step_config(5, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        vec_dir = src / "_vectors"
        assert vec_dir.is_dir()
        json_files = list(vec_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "vector" in data
        assert data["model"] == "Qwen-VL"


class TestPoseExtractionStep:
    def test_creates_pose_json(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "pose.jpg")

        pose_ext = MagicMock()
        pose_ext.extract_keypoints.return_value = {
            "body": [{"x": 0, "y": 0, "confidence": 1.0, "name": "nose"}],
            "face": [], "left_hand": [], "right_hand": []
        }

        step = PoseExtractionStep(file_storage, extractor=pose_ext)
        config = make_step_config(6, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        pose_dir = src / "_poses"
        assert pose_dir.is_dir()
        json_files = list(pose_dir.glob("*_pose.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "total_keypoints" in data


class TestColorPaletteExtractionStep:
    def test_creates_colors_json(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "colorful.jpg")

        color_ext = MagicMock()
        color_ext.extract_palette.return_value = [
            {"rgb": [1, 2, 3], "hex": "#010203", "percentage": 50.0}
        ]

        step = ColorPaletteExtractionStep(file_storage, extractor=color_ext)
        config = make_step_config(7, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        colors_dir = src / "_colors"
        assert colors_dir.is_dir()
        json_files = list(colors_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "colors" in data


class TestContentSafetyClassificationStep:
    def test_creates_nsfw_json(self, tmp_path: Path, file_storage: FileSystemStorage) -> None:
        src = tmp_path / "source"
        src.mkdir()
        create_test_file(src, "safe.jpg")

        nsfw_clf = MagicMock()
        nsfw_clf.classify.return_value = (0.1, 0.9)

        step = ContentSafetyClassificationStep(file_storage, classifier=nsfw_clf)
        config = make_step_config(8, src)
        result = step.execute(config)

        assert result.status == "COMPLETED"
        nsfw_dir = src / "_nsfw"
        assert nsfw_dir.is_dir()
        json_files = list(nsfw_dir.glob("*_nsfw.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "safe" in data