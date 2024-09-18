import os
import time
import threading
import whisper
import subprocess
import torch
import re
import warnings

model_name = "medium"
speed_factor = 1.25
offset_dB = -3
silence_gap = 0.5

# Определим пути для всех файлов относительно папки скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '.models')
SOURCE_DIR = os.path.join(SCRIPT_DIR, '.source')
TEMP_DIR = os.path.join(SCRIPT_DIR, '.temp')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '.output')

# Создаем директории, если они не существуют
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

# Функция для очистки временных файлов
def cleanup(temp_files):
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except PermissionError:
                print(f"Не удалось удалить {file}. Попробую снова...")
                time.sleep(1)


def clear_cache():
    temp_files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if
                  f.endswith(('.tmp', '.wav', '.srt', '.mp4'))]
    cleanup(temp_files)


# Пересчет таймкодов субтитров с учетом будущего изменения скорости
def adjust_subtitles(segments, speed_factor):
    adjusted_segments = []
    for segment in segments:
        start = segment['start'] / speed_factor  # Пересчитываем, учитывая будущую коррекцию скорости
        end = segment['end'] / speed_factor
        adjusted_segments.append((start, end, segment['text']))
    return adjusted_segments


# Функция для транскрибирования
def transcribe_audio(model, audio_path):
    result = model.transcribe(audio_path, language="ru")
    return result['segments']


# Функция для таймера
def start_timer(stop_event):
    start_time = time.time()  # Начальное время
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"{formatted_time}", end="\r")
        time.sleep(1)


# Загрузка модели Whisper
def load_model():
    # Определяем устройство: CPU или GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Подготовка к транскрибации.")
    print(f"Транскрибация будет выполняться на {device}.")
    print(f"Инициализация Whisper. Модель \"{model_name}\".")
    model = whisper.load_model(model_name, download_root=MODEL_DIR, device=device)
    return model


# Функция для получения порога тишины
def get_silence_threshold(input_path):
    command = [
        'ffmpeg', '-i', input_path,
        '-af', 'ebur128', '-f', 'null', '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, universal_newlines=True)
    loudness_log = result.stderr

    # Регулярное выражение для извлечения порога тишины
    match = re.search(r'Integrated loudness:[\s\S]*?Threshold:\s*(-?\d+\.\d+)\s*LUFS', loudness_log)
    print('Порог тишины в видео: ', float(match.group(1))+offset_dB)
    if match:
        return float(match.group(1))
    else:
        print("Не удалось найти 'Threshold' в логах.")
        return None


# Функция для вычисления порога тишины
def calculate_silence_threshold(loudness):
    return loudness + offset_dB


# Функция для анализа аудио и извлечения данных о тишине
def analyze_audio(input_path):
    loudness = get_silence_threshold(input_path)
    if loudness is None:
        raise RuntimeError("Не удалось определить порог тишины файла.")

    silence_threshold = calculate_silence_threshold(loudness)
    silence_threshold_str = f"{silence_threshold}dB"

    command = [
        'ffmpeg', '-i', input_path,
        '-af', f'silencedetect=n={silence_threshold_str}:d={silence_gap}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, universal_newlines=True)
    silence_log = result.stderr

    silence_intervals = []
    start_time = None

    for line in silence_log.split('\n'):
        start_match = re.search(r'silence_start:\s*([\d.]+)', line)
        if start_match:
            start_time = float(start_match.group(1))
        end_match = re.search(r'silence_end:\s*([\d.]+)', line)
        if end_match:
            end_time = float(end_match.group(1))
            if start_time is not None:
                if end_time >= start_time:
                    silence_intervals.append((start_time, end_time))
                else:
                    print(f"Invalid interval detected: start_time={start_time}, end_time={end_time}")
                start_time = None

    return silence_intervals



# Функция для удаления тишины, основываясь на данных о тишине
def remove_silence_using_metadata(input_path, silence_intervals, output_path):
    if not silence_intervals:
        print("Нет участков тишины для удаления.")
        return

    # Создаем фильтр для удаления тишины
    filter_complex = ''.join([f"between(t,{start},{end})+" for start, end in silence_intervals])[:-1]

    # Проверяем, что фильтр не пустой
    if not filter_complex:
        raise ValueError("Фильтр для удаления тишины пустой.")

    command = [
        'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
        '-vf', f"select='not({filter_complex})',setpts=N/FRAME_RATE/TB",
        '-af', f"aselect='not({filter_complex})',asetpts=N/SR/TB",
        '-b:v', '5000k',
        output_path
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении ffmpeg: {e}")
        raise


# Функция для изменения скорости видео и аудио
def speed_up_video(input_path, output_path, speed_factor):
    command = [
        'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
        '-filter:v', f'setpts={1 / speed_factor}*PTS',
        '-filter:a', f'atempo={speed_factor}',
        output_path
    ]
    subprocess.run(command, check=True)


# Основной скрипт
def main():
    try:
        video_file_name = input("Введите название видеофайла (с расширением): ")
        video_path = os.path.join(SOURCE_DIR, video_file_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError("Файл не найден. Проверьте путь и имя файла в папке .source.")

        # Промежуточные файлы
        temp_no_silence_video = os.path.join(TEMP_DIR, "temp_no_silence.mp4")
        final_video_path = os.path.join(OUTPUT_DIR, video_file_name.split('.')[0] + "_output.mp4")
        temp_srt = os.path.join(OUTPUT_DIR, video_file_name.split('.')[0] + "_output.srt")

        # Анализируем аудио для получения данных о тишине
        print("Подготовка к удалению тишины...")
        silence_intervals = analyze_audio(video_path)
        print('Интервалс: ', silence_intervals)

        # Удаляем тишину (с учетом видео и аудио)
        print("Удаление тишины из видео...")
        remove_silence_using_metadata(video_path, silence_intervals, temp_no_silence_video)

        # Загружаем модель для транскрибации
        model = load_model()

        # Запуск таймера в отдельном потоке
        try:
            stop_event = threading.Event()
            print("Транскрибация...")
            timer_thread = threading.Thread(target=start_timer, args=(stop_event,))
            timer_thread.start()
            segments = transcribe_audio(model, temp_no_silence_video)
        finally:
            stop_event.set()
            timer_thread.join()

        print("\nТранскрибация завершена.")
        # Планируем изменение скорости

        # Корректируем таймкоды субтитров с учетом будущего ускорения
        adjusted_segments = adjust_subtitles(segments, speed_factor)

        # Создаем файл субтитров
        with open(temp_srt, 'w') as srt_file:
            for i, (start, end, text) in enumerate(adjusted_segments, start=1):
                start_time = time.strftime("%H:%M:%S", time.gmtime(start))
                end_time = time.strftime("%H:%M:%S", time.gmtime(end))
                srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

        # Ускоряем видео с учетом фактора скорости
        print(f"Ускорение видео в {speed_factor}...")
        speed_up_video(temp_no_silence_video, final_video_path, speed_factor)

        print(f"Видео готово и сохранено по пути: {final_video_path}")
        print(f"Субтитры сохранены по пути: {temp_srt}")
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        clear_cache()
        print("Все временные файлы удалены.")


if __name__ == "__main__":
    main()
