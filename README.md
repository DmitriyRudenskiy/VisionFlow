# VisionFlow 🌊👁️

**VisionFlow** — это модульный пайплайн для интеллектуальной обработки изображений, построенный на принципах **Clean Architecture** и **DDD**. Проект объединяет классические алгоритмы компьютерного зрения и современные ML-модели (SAM, Qwen-VL, DWPose) в расширяемый рабочий процесс.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-Clean%20%7C%20DDD-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Ключевые возможности

Пайплайн состоит из последовательности шагов, которые можно включать или отключать:

1.  **Flatten & Prepare**: Рекурсивный поиск файлов, выравнивание структуры директорий, переименование по timestamp и создание резервных копий.
2.  **Deduplication**:
    *   Точное дублирование: Сравнение по MD5/SHA256 хешам.
    *   Визуальное дублирование: Использование pHash и эмбеддингов ViT для поиска похожих изображений.
3.  **AI Cropping**: Автоматическая обрезка (кроп) с использованием **SAM (Segment Anything Model)**.
4.  **Metadata Extraction**:
    *   **Векторизация**: Генерация эмбеддингов (Qwen-VL) для семантического поиска.
    *   **Позы**: Детекция ключевых точек скелета (DWPose).
    *   **Цвета**: Извлечение доминирующей цветовой палитры.
    *   **NSFW**: Классификация контента.
5.  **Resume Capability**: Автоматическое сохранение состояния пайплайна (JSON) с возможностью восстановления после сбоя.

---

## 🏗️ Архитектура

Проект разделен на 4 слоя согласно методологии Clean Architecture:

*   **Domain Layer**: Чистые бизнес-сущности (`Entity`), Объекты-значения (`Value Object`), Доменные исключения. Не зависят от фреймворков.
*   **Application Layer**: Сценарии использования (Steps), Порты (Interfaces), Оркестратор, DTO.
*   **Infrastructure Layer**: Реализация портов (Файловая система, AI-клиенты, Репозитории).
*   **Presentation Layer**: CLI-интерфейс.

```
src/
├── domain/         # Бизнес-логика (Pipeline, Image, Metadata)
├── application/    # Оркестрация и шаги обработки
├── infrastructure/ # Работа с OS, GPU, внешними API
└── cli.py          # Точка входа
```

---

## ⚙️ Установка

1.  **Клонирование репозитория**:
    ```bash
    git clone https://github.com/your-username/visionflow.git
    cd visionflow
    ```

2.  **Создание виртуального окружения**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate   # Windows
    ```

3.  **Установка зависимостей**:
    *(См. раздел "Зависимости" ниже)*
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Использование

Запуск полного цикла обработки:

```bash
python -m src.cli /path/to/input/images /path/to/output
```

**Опции CLI**:

*   `--steps 0 2 5`: Запуск только указанных шагов (например: Flatten, Deduplicate, Vectorize).
*   `--pipeline-id <UUID>`: Возобновление прерванного пайплайна по его ID.
*   `--no-stop-on-error`: Продолжить выполнение даже если один из шагов упал.

**Пример запуска конкретных шагов:**

```bash
python -m src.cli ./data/photos ./data/processed --steps 0 1 2
```

---

## 📂 Структура проекта

*   `src/domain/`: Ядро бизнес-логики.
    *   `pipeline/`: Сущности пайплайна (Aggregate Root).
    *   `image/`: Сущности файлов и валидация.
    *   `deduplication/`: Логика группировки дубликатов.
*   `src/application/`: Сценарии.
    *   `ports.py`: Интерфейсы (ABC) для инфраструктуры.
    *   `steps/`: Реализация конкретных шагов (Use Cases).
*   `src/infrastructure/`: Адаптеры.
    *   `ai/`: Клиенты к ML-моделям (PyTorch, Transformers).
    *   `persistence/`: Сохранение состояния в JSON.

---

## 🧪 Тестирование

Проект покрыт юнит- и интеграционными тестами (pytest).

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=src tests/
```

---

## 📦 Зависимости

Основные библиотеки:
*   `pydantic` / `dataclasses`: Валидация данных.
*   `Pillow`: Работа с изображениями.
*   `imagehash`: pHash.
*   `torch`, `transformers`: ML-инференс.
*   `pytest`: Тестирование.

---