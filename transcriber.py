from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
import os
import torch
import shutil
import warnings
from tqdm import tqdm
from pydub import AudioSegment

# Определение переменных
models_dir = "models"
temp_dir = "temp_chunks"
input_dir = "my_files"
output_dir = "my_files"
os.environ["TORCH_HOME"] = models_dir


# Функция для создания директории
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Убедитесь, что директории для моделей, временных файлов и файлов ввода/вывода существуют
ensure_dir_exists(models_dir)
ensure_dir_exists(temp_dir)
ensure_dir_exists(input_dir)
ensure_dir_exists(output_dir)


# Определение функции для сохранения SRT
def save_srt(segments, srt_filename):
    with open(srt_filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(tqdm(segments, desc="Сохранение сегментов")):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            f.write(f"{i + 1}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")


# Определение функции для форматирования времени в SRT
def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


# Определение функции для удаления временных файлов
def clear_temp_files():
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            print(f"Удаление временного файла {filename}...")
            os.remove(file_path)
    print("Временные файлы удалены.")
    if os.listdir(temp_dir) == []:  # Удаляем папку, если она пуста
        os.rmdir(temp_dir)


# Определение функции для удаления кэша
def clear_cache():
    if os.path.exists(models_dir):
        print(f"Удаление кэша из папки {models_dir}...")
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Кэш успешно удален.")
    else:
        print("Папка кэша не найдена.")


# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")

# Определение имени модели и пути к кэшированной модели
model_name = "large-v3"
model_cache_path = os.path.join(models_dir, f"{model_name}.pt")


# Функция для загрузки модели
def load_model():
    print("Загрузка модели...")
    model = whisper.load_model(model_name).to(device)
    torch.save(model.state_dict(), model_cache_path)
    return model


# Игнорируем предупреждение FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    # Проверяем наличие кэшированной модели
    if os.path.exists(model_cache_path):
        print("Загрузка модели из файла...")
        model = whisper.load_model(model_name).to(device)
        model.load_state_dict(torch.load(model_cache_path, map_location=device))
    else:
        model = load_model()


def split_audio(filename, chunk_duration_ms=60000, overlap_ms=5000):
    audio = AudioSegment.from_file(filename)
    chunks = []
    start = 0
    chunk_duration = chunk_duration_ms - overlap_ms

    while start < len(audio):
        end = start + chunk_duration_ms
        chunk = audio[start:end]
        chunk_path = os.path.join(temp_dir, f"chunk_{start // chunk_duration_ms}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        start += chunk_duration  # Move start position with overlap
    return chunks


def transcribe_chunk(model, chunk_path):
    print(f"Транскрибирование {chunk_path}...")
    result = model.transcribe(chunk_path, language="ru", verbose=True)
    return result["segments"]


def transcribe_file(filename):
    # Получаем длительность файла
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video = VideoFileClip(filename)
        duration_sec = video.duration
    elif filename.lower().endswith(('.mp3', '.wav', '.flac')):
        audio = AudioFileClip(filename)
        duration_sec = audio.duration
    else:
        raise ValueError("Неподдерживаемый формат файла.")

    # Информация о длительности
    print(f"Длительность файла: {duration_sec:.2f} секунд")

    if duration_sec < 60:
        # Если длительность файла меньше 1 минуты, транскрибируем без разбиения
        audio_clip = filename  # Используем исходный файл без разбиения
        segments = transcribe_chunk(model, audio_clip)
        return segments

    # Иначе разбиваем файл на фрагменты
    chunks = split_audio(filename)
    all_segments = []

    total_chunks = len(chunks)
    with tqdm(total=total_chunks, desc="Транскрибирование", unit='chunk') as pbar:
        for chunk_path in chunks:
            segments = transcribe_chunk(model, chunk_path)
            all_segments.extend(segments)
            pbar.update(1)
            os.remove(chunk_path)  # Удаление временного файла
    return all_segments


while True:
    filename = input(f"Введите имя файла (с расширением) из папки {input_dir} или 'exit' для выхода: ")

    if filename.lower() == 'exit':
        clear_temp_files()
        clear_cache()
        print("Выход из программы.")
        break

    file_path = os.path.join(input_dir, filename)
    if not os.path.isfile(file_path):
        print(f"Файл {filename} не найден в папке {input_dir}.")
        continue

    print(f"Файл {filename} найден. Начинаем транскрибирование...")

    try:
        # Транскрибирование файла
        all_segments = transcribe_file(file_path)

        print("Транскрибирование завершено.")
    except Exception as e:
        print(f"Произошла ошибка при транскрибировании: {e}")
        continue

    if not all_segments:
        print("Результат транскрибирования не содержит сегменты.")
        continue

    print("Сохранение в файл SRT...")
    srt_filename = os.path.splitext(filename)[0] + ".srt"
    srt_path = os.path.join(output_dir, srt_filename)
    try:
        save_srt(all_segments, srt_path)
        print(f"Результат транскрибирования сохранен в файл {srt_path}.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении SRT файла: {e}")

    clear_temp_files()
    clear_cache()
