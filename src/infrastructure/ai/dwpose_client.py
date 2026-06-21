import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoImageProcessor


def process_image(image_path: str):
    # Определяем устройство (GPU, если доступно, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Формируем пути к файлам
    input_path = Path(image_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    # Создаем имя выходного файла с постфиксом: original_name_result.json
    output_path = input_path.with_name(f"{input_path.stem}_result.json")

    print("Загрузка модели (это может занять некоторое время)...")
    config = AutoConfig.from_pretrained("akore/rtmw-x-384x288", trust_remote_code=True)
    model = AutoModel.from_pretrained("akore/rtmw-x-384x288", trust_remote_code=True)
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained("akore/rtmw-x-384x288")

    print(f"Обработка изображения: {input_path.name}")
    image = Image.open(input_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # Перемещаем входные данные на то же устройство, что и модель
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # coordinate_mode="model" -> координаты в пиксельном пространстве модели
        outputs = model(**inputs, coordinate_mode="model")

    # Извлекаем данные и конвертируем тензоры в обычные списки для JSON
    # Убираем batch-размерность (1), так как мы обрабатывали одно фото
    keypoints = outputs.keypoints[0].cpu().tolist()  # (133, 2)
    scores = outputs.scores[0].cpu().tolist()  # (133,)

    result_data = {
        "image": input_path.name,
        "keypoints": keypoints,
        "scores": scores
    }

    print(f"Сохранение результата в: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print("Готово!")


def main():
    parser = argparse.ArgumentParser(
        description="Утилита для оценки позы человека на изображении с использованием модели rtmw-x."
    )
    parser.add_argument(
        "image",
        type=str,
        help="Путь к входному изображению (например, person_crop.jpg)"
    )

    args = parser.parse_args()
    process_image(args.image)


if __name__ == "__main__":
    main()