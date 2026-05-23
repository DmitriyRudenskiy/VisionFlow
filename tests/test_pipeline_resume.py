# tests/test_pipeline_resume.py
import pytest
from pathlib import Path

from src.application.pipeline.orchestrator import PipelineOrchestrator
from src.application.pipeline.step_registry import StepRegistry
from src.application.pipeline.dto import PipelineConfigDTO, StepResultDTO
from src.infrastructure.persistence.pipeline_repository import JsonPipelineRepository
from src.domain.pipeline.entities import PipelineAggregate, PipelineStep
from src.domain.pipeline.value_objects import StepStatus, PipelineStatus
from src.application.pipeline.steps.base_step import BaseStep


class FakeStep(BaseStep):
    def __init__(self, name: str, fail: bool = False):
        self._name = name
        self._fail = fail

    def execute(self, config) -> StepResultDTO:
        if self._fail:
            return StepResultDTO(step_number=config.step_number, status="FAILED", message="Fake fail")
        return StepResultDTO(step_number=config.step_number, status="COMPLETED", message=f"Done {self._name}")


@pytest.fixture
def tmp_repo(tmp_path):
    return JsonPipelineRepository(tmp_path / "pipelines")


@pytest.fixture
def registry():
    reg = StepRegistry()
    reg.register(0, FakeStep("flatten"))
    reg.register(1, FakeStep("prepare"))
    reg.register(2, FakeStep("dedup"))
    reg.register(5, FakeStep("vectorize"))
    reg.register(6, FakeStep("pose"))
    reg.register(7, FakeStep("colors"))
    reg.register(8, FakeStep("nsfw"))
    return reg


class TestResumeAfterCrash:
    def test_resume_resets_running_steps(self, tmp_path, registry, tmp_repo):
        """Имитирует kill -9: шаг 0 выполнен, шаг 1 в RUNNING, шаг 2 PENDING."""
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry, tmp_repo)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        # Шаг 0 — completed
        s0 = PipelineStep(step_number=0, step_name="flatten")
        pipeline.add_step(s0)
        pipeline.start_step(0)
        pipeline.complete_step(0)

        # Шаг 1 — running (процесс убит)
        s1 = PipelineStep(step_number=1, step_name="prepare")
        pipeline.add_step(s1)
        pipeline.start_step(1)

        # Шаг 2 — pending
        s2 = PipelineStep(step_number=2, step_name="dedup")
        pipeline.add_step(s2)

        tmp_repo.save(pipeline)

        # Resume
        config = PipelineConfigDTO(source_path=source, output_path=tmp_path / "out")
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert len(results) == 3
        assert results[0].status == "SKIPPED"   # 0 уже был готов
        assert results[1].status == "COMPLETED" # 1 перезапущен после сброса RUNNING→FAILED
        assert results[2].status == "COMPLETED" # 2 выполнен

        saved = tmp_repo.find_by_id(pipeline.id)
        assert saved.status == PipelineStatus.COMPLETED
        assert saved.steps[0].status == StepStatus.COMPLETED
        assert saved.steps[1].status == StepStatus.COMPLETED
        assert saved.steps[2].status == StepStatus.COMPLETED
        assert saved.steps[1].error is None  # ошибка прерывания очищена при старте

    def test_resume_skips_already_completed(self, tmp_path, registry, tmp_repo):
        """Повторный запуск не должен пере-выполнять завершённые шаги."""
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry, tmp_repo)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        for num in [0, 1]:
            s = PipelineStep(step_number=num, step_name=f"step_{num}")
            pipeline.add_step(s)
            pipeline.start_step(num)
            pipeline.complete_step(num)

        tmp_repo.save(pipeline)

        config = PipelineConfigDTO(source_path=source, output_path=tmp_path / "out")
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert all(r.status == "SKIPPED" for r in results)


class TestParallelExecution:
    def test_parallel_group_runs_all_steps(self, tmp_path, tmp_repo):
        """Шаги 5-8 выполняются параллельно и все завершаются успешно."""
        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(registry(), tmp_repo)
        pipeline = orch.create_pipeline(source, tmp_path / "out")

        for num in [4, 5, 6, 7, 8]:
            pipeline.add_step(PipelineStep(step_number=num, step_name=f"step_{num}"))
        tmp_repo.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[4, 5, 6, 7, 8]
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        statuses = {r.step_number: r.status for r in results}
        assert statuses[4] == "COMPLETED"
        assert statuses[5] == "COMPLETED"
        assert statuses[6] == "COMPLETED"
        assert statuses[7] == "COMPLETED"
        assert statuses[8] == "COMPLETED"

        saved = tmp_repo.find_by_id(pipeline.id)
        for num in [5, 6, 7, 8]:
            step = next(s for s in saved.steps if s.step_number == num)
            assert step.status == StepStatus.COMPLETED

    def test_parallel_group_stop_on_error(self, tmp_path, tmp_repo):
        """При stop_on_error=True падение одного шага из параллельной группы прерывает пайплайн."""
        reg = StepRegistry()
        reg.register(5, FakeStep("v", fail=True))
        reg.register(6, FakeStep("p"))
        reg.register(7, FakeStep("c"))
        reg.register(8, FakeStep("n"))

        source = tmp_path / "src"
        source.mkdir()

        orch = PipelineOrchestrator(reg, tmp_repo)
        pipeline = orch.create_pipeline(source, tmp_path / "out")
        for num in [5, 6, 7, 8]:
            pipeline.add_step(PipelineStep(step_number=num, step_name=f"step_{num}"))
        tmp_repo.save(pipeline)

        config = PipelineConfigDTO(
            source_path=source,
            output_path=tmp_path / "out",
            steps_to_run=[5, 6, 7, 8],
            stop_on_error=True
        )
        results = list(orch.execute(config, pipeline_id=pipeline.id))

        assert any(r.status == "FAILED" for r in results)
        saved = tmp_repo.find_by_id(pipeline.id)
        assert saved.status == PipelineStatus.FAILED