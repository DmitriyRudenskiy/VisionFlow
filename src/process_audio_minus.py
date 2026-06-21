import argparse
import os
import torch
import torchaudio
import numpy as np
import librosa
from demucs.pretrained import get_model
from demucs.apply import apply_model


def extract_instrumental_ensemble(input_path, output_path):
    """
    Извлечение минусовки (инструментала) с использованием ансамбля (Ensemble) моделей Demucs.
    Усредняет результаты двух моделей, суммирует все стемы кроме вокала.
    """
    print(f"[*] Загрузка аудио и запуск Ensemble-разделения для {input_path}...")
    model_names = ['htdemucs', 'htdemucs_ft']
    instrumentals_list = []

    # Определение устройства (GPU если доступно, иначе CPU)
    # Добавлена поддержка Apple Silicon (M1/M2/M3) через MPS
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"[*] Используемое устройство: {device}")

    for name in model_names:
        print(f"  -> Запуск модели: {name}")
        model = get_model(name)
        model.to(device)

        # Загрузка аудио
        wav, sr = torchaudio.load(input_path)
        # Если аудио моно, делаем его стерео для Demucs
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        # Подготовка тензора [batch, channels, length]
        wav = wav.unsqueeze(0).to(device)

        # Нормализация
        ref = wav.mean(2)
        wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

        # Разделение на стемы
        with torch.no_grad():
            sources = apply_model(model, wav_norm, split=True, overlap=0.25, progress=False)[0]

        # Денормализация
        sources = sources * (ref.std() + 1e-8) + ref.mean()

        # В Demucs источники идут в порядке: 0=drums, 1=bass, 2=other, 3=vocals
        # Минусовка = барабаны + бас + другие инструменты (убираем индекс 3 - вокал)
        instrumental = sources[0] + sources[1] + sources[2]
        instrumentals_list.append(instrumental.cpu())

    # Ensemble: усреднение результатов двух моделей
    print("[*] Усреднение результатов (Ensemble)...")
    final_instrumental = torch.stack(instrumentals_list).mean(dim=0)

    # Сохранение результата
    torchaudio.save(output_path, final_instrumental, sr)
    print(f"[*] Минусовка сохранена в: {output_path}")


def calculate_bpm_librosa(input_path):
    """
    Определение BPM с помощью librosa (использование стабильного метода beat_track).
    """
    print(f"[*] Расчет BPM с помощью librosa...")
    y, sr = librosa.load(input_path, sr=22050)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # В разных версиях librosa tempo может возвращаться как массив с одним элементом или как число
    if isinstance(tempo, np.ndarray):
        bpm = int(round(tempo[0]))
    else:
        bpm = int(round(tempo))

    return bpm


def main():
    parser = argparse.ArgumentParser(
        description="Скрипт для извлечения минусовки (Ensemble) и определения BPM."
    )
    parser.add_argument("input", help="Путь к входному аудиофайлу (mp3, wav, flac и т.д.)")
    parser.add_argument("-o", "--output", default=None,
                        help="Путь для сохранения минусовки (по умолчанию: <имя_файла>_minus.wav)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: Файл '{args.input}' не найден.")
        return

    # Автоматическая генерация имени выходного файла, если не указан флаг -o
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"{base_name}_minus.wav"
    else:
        output_path = args.output

    # 1. Извлечение минусовки
    extract_instrumental_ensemble(args.input, output_path)

    # 2. Определение BPM (по исходному файлу)
    bpm = calculate_bpm_librosa(args.input)

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Минусовка:         {output_path}")
    print(f"Определенный BPM:  {bpm}")
    print("=" * 40)


if __name__ == "__main__":
    main()
