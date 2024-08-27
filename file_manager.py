import os
import shutil


def clear_temp_files():
    temp_files = [f for f in os.listdir() if f.endswith(".temp")]
    for filename in temp_files:
        print(f"Удаление временного файла {filename}...")
        os.remove(filename)
    print("Временные файлы удалены.")


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


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)