import os
import warnings

import torch
import whisper
from moviepy.editor import VideoFileClip, AudioFileClip

import file_manager
import model_loader
import settings
import srt_saver
import transcriber

models_dir = "models"
os.environ["TORCH_HOME"] = models_dir
model_name = input("Введите название модели Whisper: ")
file_manager.ensure_dir_exists(models_dir)
model_cache_path = os.path.join(models_dir, f"{model_name}.pt")
device = settings.get_device()

# Игнорируем предупреждение FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

    # Проверяем наличие кэшированной модели
    if os.path.exists(model_cache_path):
        print("Загрузка модели из файла...")
        model = whisper.load_model(model_name).to(device)
        model.load_state_dict(torch.load(model_cache_path, map_location=device))
    else:
        model = model_loader.load_model(model_name, model_cache_path, device)


while True:
    filename = input("Введите имя файла (с расширением): ")

    if filename.lower() == 'exit':
        file_manager.clear_temp_files()
        file_manager.clear_cache()
        print("Выход из программы.")
        break

    if not os.path.isfile(filename):
        print(f"Файл {filename} не найден в папке {'my_files'}.")
        continue
    print(f"Файл {filename} найден. Начинаем транскрибирование...")

    try:
        # Получение длительности файла с помощью moviepy
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video = VideoFileClip(filename)
        elif filename.lower().endswith(('.mp3', '.wav', '.flac')):
            audio = AudioFileClip(filename)
        else:
            raise ValueError("Неподдерживаемый формат файла.")

        # Транскрибирование файла
        all_segments = transcriber.transcribe_file(filename)

        print("Транскрибирование завершено.")
    except Exception as e:
        print(f"Произошла ошибка при транскрибировании: {e}")
        continue

    if not all_segments:
        print("Результат транскрибирования не содержит сегменты.")
        continue

    print("Сохранение в файл SRT...")
    srt_filename = os.path.splitext(filename)[0] + ".srt"
    try:
        srt_saver.save_srt(all_segments, srt_filename)
        print(f"Результат транскрибирования сохранен в файл {srt_filename}.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении SRT файла: {e}")

    file_manager.clear_temp_files()
    file_manager.clear_cache()
