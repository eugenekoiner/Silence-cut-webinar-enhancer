import os
import warnings

import file_manager
import model_loader
import settings
import srt_saver
import transcriber

my_files_dir = "my_files"
os.environ["TORCH_HOME"] = file_manager.models_dir
model_name = input("Введите название модели Whisper: ")
file_manager.ensure_dir_exists(file_manager.models_dir)
model_cache_path = os.path.join(file_manager.models_dir, f"{model_name}.pt")
device = settings.get_device()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)

model = model_loader.load_model(model_name, model_cache_path, device)

while True:
    filename = input("Введите имя файла (с расширением): ")
    if filename.lower() == 'exit':
        file_manager.clear_temp_files()
        file_manager.clear_cache(model_cache_path)
        print("Выход из программы.")
        break

    file_manager.ensure_dir_exists(my_files_dir)

    if not os.path.isfile(os.path.join(my_files_dir, filename)):
        print(f"Файл {filename} не найден в папке {'my_files'}.")
        continue

    print(f"Файл {filename} найден. Начинаем транскрибирование...")
    try:
        all_segments = transcriber.transcribe_file(filename, model)
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
        srt_saver.save_srt(all_segments, os.path.join(my_files_dir, srt_filename))
        print(f"Результат транскрибирования сохранен в файл {srt_filename}.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении SRT файла: {e}")

    file_manager.clear_temp_files()
    file_manager.clear_cache(file_manager.temp_audio_dir)
