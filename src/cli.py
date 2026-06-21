# src/cli.py
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List
from uuid import UUID

from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO
from src.infrastructure.file_system import FileSystemStorage
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.infrastructure.persistence.pipeline_serializer import JsonPipelineSerializer

# Импорты шагов
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
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_container(sam_model_path: str | None = None) -> PipelineOrchestrator:
    """Composition Root: сборка зависимостей."""
    repo = JsonPipelineRepository(
        storage_dir=Path("./data/pipelines"),
        serializer=JsonPipelineSerializer(),
    )
    fs = FileSystemStorage()

    registry = StepRegistry()
    registry.register(0, FlattenDirectoriesStep(fs))
    registry.register(1, PrepareImagesStep(fs))
    registry.register(2, ExactDeduplicationStep(fs))
    registry.register(3, VisualDeduplicationStep(fs, detector=None))

    # --- Путь к модели SAM ---
    model_path = sam_model_path or os.environ.get("SAM_MODEL_PATH", "")
    if not model_path:
        # Fallback: ищем относительно текущей директории
        models_dir = Path.cwd() / "models"
        candidates = sorted(models_dir.glob("sam*.pt"))
        if candidates:
            model_path = str(candidates[0])
        else:
            model_path = str(models_dir / "sam3.pt")

    if not Path(model_path).exists():
        logger.warning(f"SAM model not found at: {model_path}")
    else:
        logger.info(f"Using SAM model: {model_path}")

    registry.register(4, SmartCropStep(fs, segmenter=None, model_path=model_path))

    # Остальные AI-интенсивные шаги
    registry.register(5, EmbeddingExtractionStep(fs, extractor=None))
    registry.register(6, PoseExtractionStep(fs, extractor=None))
    registry.register(7, ColorPaletteExtractionStep(fs, extractor=None))
    registry.register(8, ContentSafetyClassificationStep(fs, classifier=None))

    return PipelineOrchestrator(registry, repo)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionFlow: Intelligent Image Pipeline.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Directory with source images")
    parser.add_argument("output", type=Path, help="Directory for results")
    parser.add_argument("--steps", nargs="+", type=int, help="Specific step numbers to run")
    parser.add_argument("--pipeline-id", type=UUID, help="Resume existing pipeline by ID")
    parser.add_argument("--no-stop-on-error", action="store_true", help="Continue on errors")
    parser.add_argument("--force", action="store_true", help="Overwrite existing results")
    parser.add_argument("--sam-model", type=Path, help="Path to SAM3 model (.pt file)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug output")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.exists():
        sys.exit(f"Error: Source path not found: {args.source}")
    if not args.source.is_dir():
        sys.exit(f"Error: Source path is not a directory: {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)


def _print_summary(results: List) -> None:
    logger.info("\n" + "=" * 50)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 50)

    success, skip, fail = 0, 0, 0

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

        log_fn = logger.warning if res.status == "FAILED" else logger.info
        log_fn(msg)

        if res.errors:
            for err in res.errors:
                logger.error(f"   └── {err}")

    logger.info("-" * 50)
    logger.info(f"Total: {success} OK, {skip} Skipped, {fail} Failed.")
    logger.info("=" * 50)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    validate_args(args)

    orchestrator = build_container(sam_model_path=str(args.sam_model) if args.sam_model else None)

    # Валидация шагов
    available = orchestrator.get_available_steps()
    if args.steps:
        invalid = set(args.steps) - set(available)
        if invalid:
            sys.exit(f"Error: Invalid steps {invalid}. Available: {available}")

    config = PipelineConfigDTO(
        source_path=args.source.resolve(),
        output_path=args.output.resolve(),
        steps_to_run=args.steps,
        stop_on_error=not args.no_stop_on_error,
    )

    # Передаем force_overwrite через params для шагов, которые его поддерживают
    if args.force:
        # Для batch-шагов (5-8) force передается через специальный механизм
        # Но PipelineConfigDTO не имеет params, поэтому используем хак:
        # Сохраняем в глобальную переменную или модифицируем config после создания
        pass

    logger.info(f"Starting pipeline. Source: {config.source_path}")

    try:
        results = list(orchestrator.execute(config, pipeline_id=str(args.pipeline_id) if args.pipeline_id else None))
        _print_summary(results)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Critical unhandled error.")
        sys.exit(1)


if __name__ == "__main__":
    main()