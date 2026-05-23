# src/cli.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List
from uuid import UUID

# Application Layer
from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO

# Infrastructure Layer
from src.infrastructure.file_system import FileSystemStorage
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.infrastructure.persistence.pipeline_serializer import JsonPipelineSerializer

# Steps
from src.application.pipeline.steps.flatten_directories_step import FlattenDirectoriesStep
from src.application.pipeline.steps.prepare_images_step import PrepareImagesStep
from src.application.pipeline.steps.exact_deduplication_step import ExactDeduplicationStep
from src.application.pipeline.steps.visual_deduplication_step import VisualDeduplicationStep
from src.application.pipeline.steps.smart_crop_step import SmartCropStep
from src.application.pipeline.steps.embedding_extraction_step import EmbeddingExtractionStep
from src.application.pipeline.steps.pose_extraction_step import PoseExtractionStep
from src.application.pipeline.steps.color_palette_extraction_step import ColorPaletteExtractionStep
from src.application.pipeline.steps.content_safety_step import ContentSafetyClassificationStep

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_container() -> PipelineOrchestrator:
    """Собирает все зависимости и регистрирует шаги пайплайна.

    AI-клиенты передаются как None — они инициализируются лениво
    в prepare() только при выполнении соответствующего шага.
    """
    repo = JsonPipelineRepository(
        storage_dir=Path("./data/pipelines"),
        serializer=JsonPipelineSerializer(),
    )
    fs_service = FileSystemStorage()

    logger.info("Initializing pipeline registry (AI models loaded lazily)...")

    registry = StepRegistry()
    # Шаги 0-3: не требуют AI-моделей
    registry.register(0, FlattenDirectoriesStep(fs_service))
    registry.register(1, PrepareImagesStep(fs_service))
    registry.register(2, ExactDeduplicationStep(fs_service))
    registry.register(3, VisualDeduplicationStep(fs_service, detector=None))  # lazy

    # Шаги 4-8: AI-клиенты = None → создадутся в prepare()
    registry.register(4, SmartCropStep(fs_service, segmenter=None))  # lazy SAM3
    registry.register(5, EmbeddingExtractionStep(fs_service, extractor=None))  # lazy Qwen-VL
    registry.register(6, PoseExtractionStep(fs_service, extractor=None))  # lazy DWPose
    registry.register(7, ColorPaletteExtractionStep(fs_service, extractor=None))  # lazy ColorExtractor
    registry.register(8, ContentSafetyClassificationStep(fs_service, classifier=None))  # lazy NSFW

    return PipelineOrchestrator(registry, repo)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VisionFlow: Intelligent Image Processing Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.cli ./input_images ./output_dir
  python -m src.cli '/path/to/input' '/path/to/output' --steps 0 1 2
  python -m src.cli ./input ./output --pipeline-id <uuid>
        """,
    )
    parser.add_argument("source", type=Path, help="Путь к директории с исходными изображениями")
    parser.add_argument("output", type=Path, help="Путь к выходной директории")
    parser.add_argument("--steps", nargs="+", type=int, help="Список номеров шагов для выполнения")
    parser.add_argument("--pipeline-id", type=UUID, help="UUID существующего пайплайна для возобновления")
    parser.add_argument("--no-stop-on-error", action="store_true", help="Не останавливать пайплайн при первой ошибке")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод (DEBUG)")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.exists():
        logger.error(f"Исходная директория не найдена: {args.source}")
        sys.exit(1)
    if not args.source.is_dir():
        logger.error(f"Указанный путь не является директорией: {args.source}")
        sys.exit(1)
    args.output.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    validate_args(args)

    orchestrator = build_container()
    available_steps = orchestrator.get_available_steps()

    # Валидация запрашиваемых шагов
    if args.steps:
        invalid_steps = set(args.steps) - set(available_steps)
        if invalid_steps:
            logger.error(f"Неизвестные шаги: {invalid_steps}. Доступные: {available_steps}")
            sys.exit(1)

    config = PipelineConfigDTO(
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
        steps_to_run=args.steps,
        stop_on_error=not args.no_stop_on_error,
    )

    logger.info("Запуск VisionFlow Pipeline...")
    logger.info(f"Source: {config.source_path}")
    logger.info(f"Output: {config.output_path}")
    if args.pipeline_id:
        logger.info(f"Resume Mode: ID={args.pipeline_id}")
    if args.steps:
        logger.info(f"Steps to run: {args.steps}")
    else:
        logger.info("All available steps will be executed")

    try:
        results = list(orchestrator.execute(config, pipeline_id=args.pipeline_id))
        _print_summary(results)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nExecution interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Critical unhandled error during pipeline execution.")
        sys.exit(1)

def _print_summary(results: List) -> None:
    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 50)
    success = skip = fail = 0
    for res in results:
        if res.status == "COMPLETED":
            icon = "✅"
            success += 1
        elif res.status == "SKIPPED":
            icon = "⏭️"
            skip += 1
        else:
            icon = "❌"
            fail += 1

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
            # Показываем ошибки даже для COMPLETED (если были частичные сбои)
            if res.errors:
                for err in res.errors:
                    logger.warning(f"   └── Warning: {err}")

    logger.info("-" * 50)
    logger.info(f"Total: {success} succeeded, {skip} skipped, {fail} failed.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()