import os
import shutil

my_files_dir = 'my_files'
temp_audio_dir = 'temp_audio'
models_dir = 'models'
def clear_temp_files():
    temp_files = [f for f in os.listdir() if f.endswith(".temp")]
    for filename in temp_files:
        print(f"Удаление временного файла {filename}...")
        os.remove(filename)
    print("Временные файлы удалены.")

def clear_cache(path):
    if os.path.exists(path):
        print(f"Удаление кэша из папки {path}...")
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Кэш успешно удален.")
    else:
        print("Папка кэша не найдена.")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
