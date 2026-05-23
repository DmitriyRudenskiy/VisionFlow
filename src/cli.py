import argparse
import logging
import sys
from pathlib import Path
from uuid import UUID

from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO
from src.infrastructure.file_system import FileSystemStorage
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.infrastructure.persistence.pipeline_serializer import JsonPipelineSerializer
from src.infrastructure.ai.vit_client import VisionTransformerClient
from src.infrastructure.ai.sam3_client import SAM3Client
from src.infrastructure.ai.qwen_client import QwenVLClient
from src.infrastructure.ai.dwpose_client import DWPoseClient
from src.infrastructure.ai.nsfw_client import NSFWClient
from src.infrastructure.ai.color_client import ColorExtractorClient
from src.application.pipeline.steps.flatten_directories_step import FlattenDirectoriesStep
from src.application.pipeline.steps.prepare_images_step import PrepareImagesStep
from src.application.pipeline.steps.exact_deduplication_step import ExactDeduplicationStep
from src.application.pipeline.steps.visual_deduplication_step import VisualDeduplicationStep
from src.application.pipeline.steps.smart_crop_step import SmartCropStep
from src.application.pipeline.steps.embedding_extraction_step import EmbeddingExtractionStep
from src.application.pipeline.steps.pose_extraction_step import PoseExtractionStep
from src.application.pipeline.steps.color_palette_extraction_step import ColorPaletteExtractionStep
from src.application.pipeline.steps.content_safety_step import ContentSafetyClassificationStep


def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def compose_orchestrator() -> PipelineOrchestrator:
    repo = JsonPipelineRepository(
        storage_dir=Path("./data/pipelines"),
        serializer=JsonPipelineSerializer(),
    )
    file_storage = FileSystemStorage()
    logger.info("Initializing AI Clients...")
    duplicate_detector = VisionTransformerClient()
    image_segmenter = SAM3Client()
    embedding_extractor = QwenVLClient()
    pose_extractor = DWPoseClient()
    safety_classifier = NSFWClient()
    palette_extractor = ColorExtractorClient()

    registry = StepRegistry()
    registry.register(0, FlattenDirectoriesStep(file_storage))
    registry.register(1, PrepareImagesStep(file_storage))
    registry.register(2, ExactDeduplicationStep(file_storage))
    registry.register(3, VisualDeduplicationStep(file_storage, duplicate_detector))
    registry.register(4, SmartCropStep(file_storage, image_segmenter))
    registry.register(5, EmbeddingExtractionStep(file_storage, embedding_extractor))
    registry.register(6, PoseExtractionStep(file_storage, pose_extractor))
    registry.register(7, ColorPaletteExtractionStep(file_storage, palette_extractor))
    registry.register(8, ContentSafetyClassificationStep(file_storage, safety_classifier))

    return PipelineOrchestrator(registry, repo)


def validate_source_directory(source: Path) -> None:
    if not source.exists():
        logger.error(f"Исходная директория не найдена: {source}")
        sys.exit(1)
    if not source.is_dir():
        logger.error(f"Указанный путь не является директорией: {source}")
        sys.exit(1)


def validate_requested_steps(requested_steps, registered_steps):
    if not requested_steps:
        return
    invalid_steps = set(requested_steps) - set(registered_steps)
    if invalid_steps:
        logger.error(
            f"Неизвестные шаги: {invalid_steps}. Доступные: {registered_steps}"
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="VisionFlow: Intelligent Image Processing Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli ./input_images ./output_dir
  python -m src.cli '/Users/user/Downloads/Новая папка' '/Users/user/Downloads/Новая папка' --steps 0
  python -m src.cli ./input_images ./output_dir --pipeline-id 123e4567-e89b-12d3-a456-426614174000
        """,
    )
    parser.add_argument(
        "source", type=Path, help="Путь к директории с исходными изображениями"
    )
    parser.add_argument("output", type=Path, help="Путь к выходной директории")
    parser.add_argument("--steps", nargs="+", type=int, help="Список номеров шагов")
    parser.add_argument("--pipeline-id", type=UUID, help="UUID существующего пайплайна")
    parser.add_argument("--no-stop-on-error", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(args.verbose)
    validate_source_directory(args.source)

    args.output.mkdir(parents=True, exist_ok=True)

    orchestrator = compose_orchestrator()
    registered_steps = orchestrator.get_registered_sequence_numbers()
    validate_requested_steps(args.steps, registered_steps)

    config = PipelineConfigDTO(
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
        steps_to_run=args.steps,
        halt_on_failure=not args.no_stop_on_error,
    )

    logger.info("Запуск VisionFlow Pipeline...")
    logger.info(f"Source: {config.source_path}")
    logger.info(f"Output: {config.output_path}")
    if args.pipeline_id:
        logger.info(f"Resume Mode: ID={args.pipeline_id}")

    try:
        results = list(orchestrator.execute(config, pipeline_id=args.pipeline_id))
        _print_execution_summary(results)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Critical unhandled error during pipeline execution.")
        sys.exit(1)


def _print_execution_summary(results):
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 50)
    success_count = skip_count = fail_count = 0
    for res in results:
        if res.status == "COMPLETED":
            icon = "✅"
            success_count += 1
        elif res.status == "SKIPPED":
            icon = "⏭️"
            skip_count += 1
        else:
            icon = "❌"
            fail_count += 1
        msg = f"{icon} Step {res.sequence_number}: {res.status}"
        if res.message:
            msg += f" | {res.message}"
        if res.status == "FAILED":
            logger.warning(msg)
            if res.errors:
                for err in res.errors:
                    logger.error(f"   └── Error: {err}")
        else:
            logger.info(msg)
    logger.info("-" * 50)
    logger.info(
        f"Total: {success_count} succeeded, {skip_count} skipped, {fail_count} failed."
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()