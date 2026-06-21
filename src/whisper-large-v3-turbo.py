import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# Функция для перевода секунд в формат субтитров (HH:MM:SS,mmm)
def format_time(seconds):
    if seconds is None:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    dtype=torch_dtype,
    low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    dtype=torch_dtype,
    device=device,
    return_timestamps=True  # <-- Включаем получение временных меток
)

audio_file = "/Users/user/Music/Новая папка с объектами/P12.mp3"
result = pipe(audio_file)

# Формируем текст в формате SRT
srt_text = ""
for i, chunk in enumerate(result["chunks"], start=1):
    start_time, end_time = chunk["timestamp"]
    text = chunk["text"].strip()

    srt_text += f"{i}\n"
    srt_text += f"{format_time(start_time)} --> {format_time(end_time)}\n"
    srt_text += f"{text}\n\n"

# Выводим в консоль
print(srt_text)

# Сохраняем в файл субтитров
with open("test.srt", "w", encoding="utf-8") as f:
    f.write(srt_text)

print("Субтитры успешно сохранены в файл test.srt")
