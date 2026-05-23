## CHANGELOG

## [K2.6 Thinking]

### Added
- Реальные реализации AI-клиентов вместо заглушек:
  - `DWPoseClient` — интеграция с `DwposeDetector`, паддинг изображений до 1024×1024
  - `QwenVLClient` — загрузка `Qwen/Qwen3-VL-Embedding-2B` через `transformers`
  - `SAM3Client` — сегментация `SAM3SemanticPredictor` с fallback-пrompts, поддержка режимов `square`/`mask`/`transparent`
  - `VisionTransformerClient` — perceptual hash + pixel-based similarity
  - `ColorExtractorClient` — извлечение палитры через `sklearn.cluster.KMeans` с ужиманием до 500k пикселей
  - `NSFWClient` — классификация через HF `nsfw_image_detection` pipeline
- Ленивая инициализация AI-клиентов через `BaseStep.prepare()` — модели загружаются только при выполнении соответствующего шага
- Изолированный интеграционный тест `test_color_palette_extraction.py` с 5 эталонными изображениями
- Фильтрация `scan_directory()` по `SUPPORTED_EXTENSIONS` — игнорирование `.DS_Store` и не-изображений
- Graceful fallback в `JsonPipelineSerializer` при отсутствии ключа `config`
- Реальные импорты в `scripts/verify_imports.py` вместо `pass`

### Changed
- `domain/pipeline/entities.py` — `PipelineStep.complete()` сбрасывает `error = None` для корректного `resume()`
- `application/pipeline/orchestrator.py` — `Lock` заменён на `RLock`, атомарное обновление статуса + сохранение агрегата
- `infrastructure/file_system.py` — `scan_directory()` возвращает `sorted()` список для детерминированности
- `color_palette_extraction_step.py` — дефолт `num_colors` изменён с 5 на 20, суффикс выходного файла `.json` вместо `_colors.json`
- `batch_file_step.py` — разделены счётчики `processed`/`skipped`/`error`, добавлена детальная расшифровка ошибок в `message`
- `prepare_images_step.py` — независимый счётчик `rename_counter` для генерации имён файлов
- Все step-файлы и `cli.py` — добавлен `from __future__ import annotations`

### Fixed
- Race condition при параллельном обновлении статуса пайплайна в `_process_step_result()`
- Двойное управление жизненным циклом `ThreadPoolExecutor` (конфликт `with` и `finally`)
- Недетерминированность порядка обхода файлов в `ExactDeduplicationStep` и `VisualDeduplicationStep`
- Сброс ошибки после `resume()` — тест `test_resume_resets_running_steps` проходит корректно
- 12+ ошибок mypy: дубли методов, `import-untyped`, `var-annotated`, `call-overload`, deleted variables

### Infrastructure
- Удалены дублирующиеся классы тестов в `test_pipeline_steps.py`
- Добавлены `# type: ignore` для опциональных ML-зависимостей (`dwpose`, `ultralytics`, `sklearn`)
- Глобальные переменные `_dwpose_import_error`, `_sam3_import_error` для корректной работы `mypy`

## [GLM-5] - Refactoring & Bugfixes

### Fixed (Исправлено)
*   **scripts/verify_imports.py**: Реализована фактическая проверка импортов модулей (Domain, Application, Infrastructure, AI Clients). Ранее блоки `try-except` были пустыми, что делало скрипт бесполезным.
*   **src/application/pipeline/steps/batch_file_step.py**: Добавлена валидация результата `process_file` (проверка на `None`) для предотвращения ошибок сериализации JSON.

### Changed (Изменено)
*   **src/application/pipeline/orchestrator.py**:
    *   Проведен рефакторинг с целью устранения дублирования кода (DRY).
    *   Логика создания конфигурации вынесена в метод `_build_step_config`.
    *   Логика выполнения шага изолирована в методе `_execute_step_logic`.
    *   Оптимизирована обработка ошибок и управление потоками в `_execute_parallel_group`.
*   **src/cli.py**: Улучшена структура файла, импорты сгруппированы по архитектурным слоям, повышена читаемость конфигурации.

### Technical (Технические изменения)
*   Улучшена отказоустойчивость оркестратора при параллельном выполнении шагов.

## [DEEPSEEK] — 2026-05-23

#### Fixed
- **DTO и конфигурация пайплайна**  
  - В `PipelineConfigDTO` поле `halt_on_failure` переименовано в `stop_on_error` для соответствия ожиданиям оркестратора.
- **Реестр шагов**  
  - В `StepRegistry` метод `get_registered_sequence_numbers()` переименован в `get_step_numbers()`.
- **Оркестратор пайплайна**  
  - Исправлены обращения к полям агрегата: `step_number` → `sequence_number`, `source_path` → `source_directory`, `output_path` → `output_directory`.
  - Устранены ошибки вызовов методов реестра и передачи параметров в DTO.
- **CLI**  
  - Класс хранилища заменён с несуществующего `FileSystemService` на `FileSystemStorage`.
  - Добавлена передача `JsonPipelineSerializer` при создании `JsonPipelineRepository`.
  - Удалён невалидный аргумент `default_mode` при создании `SmartCropStep`.
  - Приведены имена классов шагов к актуальным в импортах и регистрации.
- **Тесты**  
  - Устранена дублирующая фикстура `tmp_repo`, использована единая `repository` из `conftest`.
  - Обновлены обращения к полям сущностей в тестах согласно новым именам.
- **Скрипт верификации импортов**  
  - Пустые блоки заменены на реальные проверки импорта всех ключевых модулей (domain, application, infrastructure, AI-клиенты).

#### Changed
- Проведена унификация интерфейсов и именования во всех слоях приложения.
- Код приведён к состоянию, проходящему статический анализ `mypy` без ошибок.

## [K2.6 Thinking] — 2026-05-23

### Рефакторинг: DDD, SOLID, семантика имён

#### Изменено (Breaking Changes)
- **Доменные сущности**:
  - `PipelineStep.step_number` → `sequence_number`
  - `PipelineStep.step_name` → `name`
  - `PipelineAggregate.source_path` / `output_path` → `source_directory` / `output_directory`
  - `PipelineStep.touch()` → `update_modified_timestamp()`
- **DTO**:
  - `StepConfigDTO.step_number` → `sequence_number`
  - `StepResultDTO.step_number` → `sequence_number`
  - `PipelineConfigDTO.stop_on_error` → `halt_on_failure`
- **Порты**:
  - `FileSystemServicePort` → `StoragePort`
  - `AISegmenterPort` → `ImageSegmentationPort`
  - `VectorizationPort` → `ImageEmbeddingExtractorPort`
  - `PoseExtractorPort` → `PoseExtractionPort`
  - `ColorExtractorPort` → `ColorPaletteExtractorPort`
  - `NsfwClassifierPort` → `ContentSafetyClassifierPort`
  - `VisualDupDetectorPort` → `VisualDuplicateDetectorPort`
  - Методы `write_text` / `read_text` / `write_json` / `read_json` → `persist_text` / `load_text` / `persist_json` / `load_json`
- **Шаги пайплайна** — имена без порядковых номеров:
  - `Step0Flatten` → `FlattenDirectoriesStep`
  - `Step1Prepare` → `PrepareImagesStep`
  - `Step2Deduplicate` → `ExactDeduplicationStep`
  - `Step3VisualDups` → `VisualDeduplicationStep`
  - `Step4AICrop` → `SmartCropStep`
  - `Step5Vectorize` → `EmbeddingExtractionStep`
  - `Step6DWPose` → `PoseExtractionStep`
  - `Step7Colors` → `ColorPaletteExtractionStep`
  - `Step8NsfwScore` → `ContentSafetyClassificationStep`
- **Репозиторий** — конструктор теперь требует `PipelineSerializer`; сериализация вынесена из репозитория

#### Добавлено
- `BatchFileProcessingStep` — базовый класс для шагов 5–8, устраняющий дублирование шаблона «scan → process → write_json»
- `PipelineSerializer` / `JsonPipelineSerializer` — абстракция сериализации (OCP)
- `StoragePort.path_exists()` — явная проверка существования пути
- Атомарная запись репозитория через временный файл (`.tmp` → `replace`)
- Корректная отмена `Future` в параллельной группе при `halt_on_failure=True`
- `PipelineAggregate.find_step()` — безопасный поиск шага по номеру (возвращает `Optional`)

#### Исправлено
- 26 ошибок `mypy` — несоответствие имён портов, полей DTO, импортов в тестах и CLI
- Дублирование логики обработки результата шага в оркестраторе — вынесено в `_commit_step_result()`
- Дублирование логики старта шага — вынесено в `_attempt_step_activation()`
- Прямой вызов `open()` в шагах 3, 5–8 — заменён на методы `StoragePort`

#### Тесты
- Обновлены все тесты под новые имена полей, портов, конструкторов
- `tests/conftest.py` — централизованные фикстуры `file_storage` и `repository`
- `tests/test_pipeline_resume.py` — адаптирован под `sequence_number`, `halt_on_failure`, `JsonPipelineRepository` + serializer

#### Инфраструктура
- `scripts/verify_imports.py` — проверка разрешимости всех импортов после рефакторинга
- `scripts/run_tests.py` — кроссплатформенный запуск pytest + verify_imports

---

### Миграция

```python
# Старый код
from src.application.ports import FileSystemServicePort
from src.application.pipeline.steps.step_0_flatten import Step0Flatten

# Новый код
from src.application.ports import StoragePort
from src.application.pipeline.steps.flatten_directories_step import FlattenDirectoriesStep
```

```python
# Старый DTO
StepResultDTO(step_number=0, status="COMPLETED")

# Новый DTO
StepResultDTO(sequence_number=0, status="COMPLETED")
```

```python
# Старый репозиторий
repo = JsonPipelineRepository(Path("./data"))

# Новый репозиторий
repo = JsonPipelineRepository(
    storage_dir=Path("./data"),
    serializer=JsonPipelineSerializer(),
)
```

---

### Контрольный список проверки

- [ ] `python3 -m mypy src tests` — 0 ошибок
- [ ] `python3 -m pytest tests/ -v` — все тесты зелёные
- [ ] `python3 scripts/verify_imports.py` — успешно
- [ ] `python3 -m src.cli --help` — CLI запускается

---

## [GLM-5]

### Fixed (Исправлено)
*   **Typing (Mypy)**: Исправлено предупреждение `annotation-unchecked` в `StepRegistry`. Для метода `__init__` явно указан тип возвращаемого значения `-> None`.
*   **Tests**: Исправлена типизация в классе `FakeStep` (`tests/test_pipeline_resume.py`). Метод `execute` теперь принимает корректный `StepConfigDTO` вместо `PipelineConfigDTO`.
*   **Concurrency**: Исправлена потенциальная_race condition (состояние гонки) при сохранении состояния пайплайна. В метод `_save_pipeline` добавлен `threading.Lock`.
*   **Domain Logic**: Исправлена ошибка дублирования шагов в `PipelineAggregate`. Метод `add_step` теперь обновляет существующий шаг, вместо создания дубликата.

### Changed (Изменено)
*   **Orchestrator**: Улучшена логика режима Resume в `PipelineOrchestrator`. Теперь при восстановлении пайплайна автоматически добавляются шаги, переданные в конфигурации, но отсутствующие в сохраненном состоянии.

---

## [DEEPSEEK] – 2026-05-23

### Added
- Новый модуль `src/application/pipeline/steps/step_7_colors.py` с реализацией шага извлечения цветовой палитры изображений.
- Тест `TestStep7Colors.test_creates_colors_json` для проверки функциональности `Step7Colors`.
- Отсутствовавший шаг `4` (`ai_crop`) в фикстуру `registry` тестов (`test_pipeline_resume.py`).

### Fixed
- **Возобновление пайплайна**: исправлена логика `PipelineOrchestrator.execute()` – при передаче `pipeline_id` больше не добавляются новые шаги, используются только уже существующие в сохранённом агрегате.
- **Тесты возобновления**: в `test_resume_resets_running_steps` и `test_resume_skips_already_completed` явно заданы `steps_to_run`, чтобы избежать выполнения лишних шагов.
- **mypy**: устранена ошибка `Duplicate module named "entities"` путём добавления `__init__.py` во все каталоги пакетов (`src/domain`, `src/domain/image`, `src/domain/deduplication` и т.д.).
- **mypy**: убрано предупреждение `[annotation-unchecked]` за счёт полного аннотирования всех функций в тестовых файлах (`test_pipeline_resume.py`, `test_pipeline_steps.py`).

### Changed
- **Аннотации типов в тестах**: все фикстуры, вспомогательные функции и тестовые методы снабжены аннотациями возвращаемых значений и параметров, что повышает надёжность и устраняет предупреждения статического анализа.
- **Рефакторинг оркестратора**: метод `execute()` теперь чётко разделяет создание нового пайплайна (с добавлением шагов) и возобновление существующего (без добавления новых шагов), улучшая читаемость и предсказуемость поведения.

### Technical Debt
- Добавлены `__init__.py` во все директории Python-пакетов для корректной работы инструментов статического анализа (mypy, pylint, pytest).
- Рекомендовано использовать mypy версии 1.8+ для полной совместимости с Python 3.12.

## [K2.6 Thinking]

### Исправлено (Fixed)

- **Orchestrator** — `config.steps_to_run or ...` при пустом списке `[]` запускал **все** шаги вместо выбранных. Теперь используется явная проверка `is None`.
- **Orchestrator** — при `resume()` зависшие шаги в статусе `RUNNING` (например, после `kill -9`) не сбрасывались, что приводило к исключению `InvalidStepStateTransition` при попытке их перезапуска.
- **Orchestrator** — добавлена защита от ошибки в `start_step()` при возобновлении пайплайна: ошибка старта теперь корректно обрабатывается и yield'ится как `FAILED`, не прерывая цикл `execute()`.
- **Step 2 (Deduplicate)** — убран избыточный («мёртвый») код: проверки `is_dir()` и `parent == duplicates_path` при нерекурсивном сканировании файловой системы.
- **Step 3 (Visual Dups)** — убран избыточный фильтр `if f.is_file()` (порт `scan_directory` гарантирует возврат только файлов).
- **Step 4 (AI Crop)** — `cropped_count` **не инкрементировался**, если сегментер работал in-place (`cropped_image_path == file_path`).
- **Step 4 (AI Crop)** — незавершённая логика обработки коллизий имён файлов (`pass` вместо генерации уникального имени).
- **Steps 6/7/8 (DWPose, Colors, NSFW)** — при существующем JSON-файле шаг пропускал файл, но **не увеличивал** `processed_count`, что приводило к некорректной статистике (в отличие от Step 5).
- **FileSystemService** — `shutil.move` на Windows падал с ошибкой, если destination уже существовал. Добавлено предварительное удаление существующего destination.
- **PipelineRepository** — порядок именованных аргументов в `from_dict()` расходился с порядком полей `dataclass`, что усложняло рефакторинг.
- **Tests** — в `TestStep1Prepare` присутствовали противоречивые ассерты (`assert len == 1` и `assert len == 3` одновременно).
- **Tests** — в `TestStep2Deduplicate` проверка оставшегося в корне дубликата была нестабильной (не гарантировала, какой именно файл останется).

### Добавлено (Added)

- **Orchestrator** — **параллельное выполнение** шагов 5–8 (Vectorize, DWPose, Colors, NSFW) через `ThreadPoolExecutor`. State-transition (`start/complete/fail`) выполняется в главном потоке, тяжёлый `execute` — в пуле тредов.
- **Orchestrator** — `threading.Lock` (`_save_lock`) для thread-safe сохранения агрегата пайплайна в `JsonPipelineRepository`.
- **Orchestrator** — `yield` статуса `SKIPPED` для уже выполненных (`COMPLETED`) шагов при повторном запуске.
- **Step 0 (Flatten)** — защита от попытки переместить файл самого на себя (уже находящийся в корне). Добавлена обработка `OSError` при move.
- **Step 1 (Prepare)** — явный `skipped_count` и обработка ошибок при backup/copy и rename/move.
- **Step 3 (Visual Dups)** — добавлен счётчик `skipped_count` для файлов, которые не удалось обработать.
- **Step 5 (Vectorize)** — защита от пустого (`None` / пустой список) embedding, возвращаемого векторизатором.
- **Step 5 (Vectorize)** — `ensure_ascii=False` при записи JSON для корректной сериализации Unicode.
- **Domain (metadata/value_objects)** — строгая валидация `ColorEntry`: RGB-диапазон `0..255`, `percentage` `0..100`, корректный hex-формат (`#RRGGBB`).
- **Domain (metadata/value_objects)** — валидация `NsfwScore`: сумма `nsfw_value + safe_value` должна быть приблизительно равна `1.0` (допуск `±0.01`).
- **Domain (image/value_objects)** — `SUPPORTED_EXTENSIONS` переведён в `frozenset[str]` для иммутабельности и O(1) lookup.
- **Shared (base)** — добавлен `__repr__` для `BaseEntity` и `BaseValueObject` для удобной отладки в логах и консоли.
- **Application (ports)** — аннотации типов обновлены до современного синтаксиса (`list[Path]`, `tuple[float, float]`, `|` вместо `Optional`).
- **Application (dto)** — тип статуса `StepResultDTO.status` изменён на `Literal["COMPLETED", "FAILED", "SKIPPED"]` вместо сырой строки.
- **CLI** — добавлена обработка `KeyboardInterrupt` (код выхода `130`).
- **CLI** — логика summary теперь корректно отображает `SKIPPED`-шаги (⏭️) и считает их отдельно.
- **Tests** — новый модуль `tests/test_pipeline_resume.py` с интеграционными тестами:
  - Восстановление после аварийного прерывания (`kill -9`) и сброс зависших `RUNNING`-шагов.
  - Повторный запуск с пропуском уже выполненных (`COMPLETED`) шагов.
  - Параллельное выполнение группы шагов 5–8 с проверкой финальных статусов.
  - Обработка `stop_on_error=True` внутри параллельной группы.

### Изменено (Changed)

- **Orchestrator** — монолитный метод `execute()` декомпозирован на `_run_single_step()` и `_execute_parallel_group()` для разделения ответственности.
- **Step 4 (AI Crop)** — при коллизии имён при перемещении temp-файла в source теперь генерируется уникальное имя (`{stem}_{counter}{suffix}`), вместо незавершённой логики.
- **FileSystemService** — `move_file()` теперь предварительно удаляет существующий destination (файл или директорию) перед вызовом `shutil.move()`.
- **PipelineRepository** — порядок аргументов в `PipelineMapper.from_dict()` приведён к порядку объявления полей в `PipelineAggregate`.

### Улучшено (Improved)

- **Step 0 (Flatten)** — добавлены явные счётчики `processed_count` и `skipped_count` в `StepResultDTO`.
- **Step 1 (Prepare)** — корректная работа при повторном запуске (idempotent backup/rename).
- **Step 5 (Vectorize)** — убрана избыточная проверка `if not file_path.is_file()` (порт гарантирует файлы).
- **Steps 6/7/8** — поведение `processed_count` унифицировано с Step 5: пропуск существующего JSON засчитывается как обработанный.
- **Tests** — в `TestStep4AICrop` добавлена проверка `processed_count` для in-place и temp-file сценариев.
- **Step 3 (Visual Dups)** — HTML-отчёт теперь содержит `&lt;meta charset='utf-8'&gt;` и счётчик найденных групп.

## [GLM-5] - 2024-05-21

### Fixed
- **Step1Prepare**: Исправлена логика обработки файлов. Теперь шаг корректно игнорирует файлы с неподдерживаемыми расширениями (не-изображения), предотвращая ошибки при переименовании и бэкапе служебных файлов (например, `.txt`).
- **Step4AICrop**: Исправлена ошибка "разрыва" пайплайна. Обработанные изображения больше не перемещаются в подпапку `_ai_cropped`, а остаются в корневой директории источника (или заменяют оригинал), обеспечивая доступность файлов для последующих шагов.
- **VisionTransformerClient**: Исправлено поведение заглушки. Метод `calculate_phash` теперь возвращает уникальные значения вместо статического, что позволяет корректно тестировать сценарии без ложных срабатываний дедупликации.

### Changed
- **Step4AICrop**: Обновлена логика работы с файлами. Если AI-сегментер возвращает путь к временному файлу, он перемещается в рабочую директорию. Если возвращается исходный путь — файл остается на месте.

### Refactored
- **Tests (test_pipeline_steps.py)**: Проведен полный рефакторинг тестов. Реальные зависимости (AI-клиенты) заменены на `unittest.mock.MagicMock`. Тесты теперь изолированы, детерминированы и не зависят от реализации инфраструктурных заглушек.
- **Tests**: Улучшена читаемость тестов и добавлены проверки граничных случаев (например, обработка коллизий имен в `Step0Flatten`).


### [DEEPSEEK] — 2026-05-23

#### Добавлено
- Реализация `FileSystemService` для работы с файловой системой (копирование, перемещение, хеширование, сканирование директорий).
- Заглушки для AI-клиентов:  
  `VisionTransformerClient`, `SAM3Client`, `QwenVLClient`, `DWPoseClient`, `NSFWClient`, `ColorExtractorClient`.
- Модульные тесты для всех шагов пайплайна (`step_0` … `step_8`) с использованием `pytest` и временных директорий (`tmp_path`).

#### Исправлено
- **Порядок полей в датаклассах**  
  В сущностях `PipelineStep`, `PipelineAggregate`, `ImageFile`, `HashEntry`, `DuplicateGroup`, `ColorPalette`, `PoseData` поля со значениями по умолчанию теперь следуют после обязательных полей, что устраняет `TypeError` при создании экземпляров.
- **Отсутствующий импорт**  
  Добавлен `from typing import List` в `step_6_dwpose.py`, исправляющий `NameError`.
- **Обработка файлов без расширений**  
  В шаге `Step3VisualDups` добавлен перехват `InvalidImageFormat` и `ValueError` для пропуска файлов, не проходящих валидацию `FilePath`, что предотвращает аварийное завершение.
- **Дублирование кода**  
  Из `domain/deduplication/value_objects.py` удалены продублированные определения `HashEntry` и `DuplicateGroup` (оставлены только в `entities.py`).

#### Изменено
- **Рефакторинг наследования**  
  Все доменные сущности больше не наследуются от `BaseEntity`; поля `id`, `created_at`, `updated_at` и методы `touch()`, `__eq__`, `__hash__` внесены непосредственно в классы с соблюдением корректного порядка полей.
- **Логирование**  
  В шагах `Step4AICrop`, `Step5Vectorize`, `Step6DWPose`, `Step7Colors`, `Step8NsfwScore` заменены выводы `print()` на вызовы `logger.warning()` / `logger.info()`.
- **Упрощение логики перемещения**  
  В `Step0Flatten` удалён избыточный предварительный сбор имён файлов корневой директории; разрешение конфликтов теперь выполняется непосредственно в цикле перемещения.