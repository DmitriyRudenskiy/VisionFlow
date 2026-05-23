import pytest
from pathlib import Path

from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO, StepResultDTO, StepConfigDTO
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.domain.pipeline.entities import PipelineStep
from src.domain.pipeline.value_objects import StepStatus, PipelineStatus
from src.application.pipeline.steps.base_step import BaseStep


class FakeStep(BaseStep):
    """Контролируемая реализация шага для тестов."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self._name = name
        self._fail = fail

    def execute(self, config: StepConfigDTO) -> StepResultDTO:
        if self._fail:
            return StepResultDTO(
                sequence_number=config.sequence_number,
                status="FAILED",
                message="Fake fail",
            )
        return StepResultDTO(
            sequence_number=config.sequence_number,
            status="COMPLETED",
            message=f"Done {self._name}",
        )


@pytest.fixture
def registry() -> StepRegistry:
    """Фикстура реестра с зарегистрированными FakeStep."""
    reg = StepRegistry()
    reg.register(0, FakeStep("flatten"))
    reg.register(1, FakeStep("prepare"))
    reg.register(2, FakeStep("dedup"))
    reg.register(4, FakeStep("ai_crop"))
    reg.register(5, FakeStep("vectorize"))
    reg.register(6, FakeStep("pose"))
    reg.register(7, FakeStep("colors"))
    reg.register(8, FakeStep("nsfw"))
    return reg


class TestResumeAfterCrash:
    """Тесты восстановления пайплайна после аварийного завершения."""

    def test_resume_resets_running_steps(
        self, tmp_path: Path, registry: StepRegistry, repository: JsonPipelineRepository
    ) -> None:
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry, repository)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        s0 = PipelineStep(sequence_number=0, name="flatten")
        pipeline.add_step(s0)
        pipeline.start_step(0)
        pipeline.complete_step(0)

        s1 = PipelineStep(sequence_number=1, name="prepare")
        pipeline.add_step(s1)
        pipeline.start_step(1)

        s2 = PipelineStep(sequence_number=2, name="dedup")
        pipeline.add_step(s2)

        repository.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[0, 1, 2],
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert len(results) == 3
        assert results[0].status == "SKIPPED"
        assert results[1].status == "COMPLETED"
        assert results[2].status == "COMPLETED"

        saved = repository.find_by_id(pipeline.id)
        assert saved is not None
        assert saved.status == PipelineStatus.COMPLETED
        assert saved.steps[0].status == StepStatus.COMPLETED
        assert saved.steps[1].status == StepStatus.COMPLETED
        assert saved.steps[2].status == StepStatus.COMPLETED
        assert saved.steps[1].error is None

    def test_resume_skips_already_completed(
        self, tmp_path: Path, registry: StepRegistry, repository: JsonPipelineRepository
    ) -> None:
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry, repository)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        for num in [0, 1]:
            s = PipelineStep(sequence_number=num, name=f"step_{num}")
            pipeline.add_step(s)
            pipeline.start_step(num)
            pipeline.complete_step(num)

        repository.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[0, 1],
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert all(r.status == "SKIPPED" for r in results)


class TestParallelExecution:
    """Тесты параллельного выполнения групп шагов."""

    def test_parallel_group_runs_all_steps(
        self, tmp_path: Path, repository: JsonPipelineRepository, registry: StepRegistry
    ) -> None:
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry, repository)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        for num in [4, 5, 6, 7, 8]:
            pipeline.add_step(PipelineStep(sequence_number=num, name=f"step_{num}"))
        repository.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[4, 5, 6, 7, 8],
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        statuses = {r.sequence_number: r.status for r in results}
        for num in (4, 5, 6, 7, 8):
            assert statuses[num] == "COMPLETED"

        saved = repository.find_by_id(pipeline.id)
        assert saved is not None
        for num in (5, 6, 7, 8):
            step = next(s for s in saved.steps if s.sequence_number == num)
            assert step.status == StepStatus.COMPLETED

    def test_parallel_group_stop_on_error(
        self, tmp_path: Path, repository: JsonPipelineRepository
    ) -> None:
        reg = StepRegistry()
        reg.register(5, FakeStep("v", fail=True))
        reg.register(6, FakeStep("p"))
        reg.register(7, FakeStep("c"))
        reg.register(8, FakeStep("n"))

        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(reg, repository)
        pipeline = orch.create_pipeline(source, tmp_path / "out")
        for num in [5, 6, 7, 8]:
            pipeline.add_step(PipelineStep(sequence_number=num, name=f"step_{num}"))
        repository.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[5, 6, 7, 8],
            stop_on_error=True,
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert any(r.status == "FAILED" for r in results)
        saved = repository.find_by_id(pipeline.id)
        assert saved is not None
        assert saved.status == PipelineStatus.FAILED