import os
import json
import time
import threading
import whisper
import subprocess
import torch
import re
import warnings
from tqdm import tqdm

# Определяем глобальные переменные
model_name = None
speed_factor = None
offset_dB = None
silence_gap = None

# Определим пути для всех файлов относительно папки скрипта
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '.models')
SOURCE_DIR = os.path.join(SCRIPT_DIR, '.source')
TEMP_DIR = os.path.join(SCRIPT_DIR, '.temp')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '.output')
CONFIG_DIR = os.path.join(SCRIPT_DIR, '.config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
temp_output_path = os.path.join(TEMP_DIR, "final_voice_only.mp4")

# Создаем директории, если они не существуют
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)


def initialize_params():
    global model_name, speed_factor, offset_dB, silence_gap
    # Дефолтные параметры
    DEFAULT_MODEL_NAME = "medium"
    DEFAULT_SPEED_FACTOR = 1.25
    DEFAULT_OFFSET_DB = -3
    DEFAULT_SILENCE_GAP = 0.5

    if os.path.exists(CONFIG_FILE):
        print(f"Загрузка конфигурации из файла {CONFIG_FILE}...")
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        model_name = config.get('model_name', DEFAULT_MODEL_NAME)
        speed_factor = config.get('speed_factor', DEFAULT_SPEED_FACTOR)
        offset_dB = config.get('offset_dB', DEFAULT_OFFSET_DB)
        silence_gap = config.get('silence_gap', DEFAULT_SILENCE_GAP)
    else:
        print("Конфигурационный файл не найден. Введите параметры вручную.")
        model_name = input(f"Введите название модели (по умолчанию '{DEFAULT_MODEL_NAME}'): ") or DEFAULT_MODEL_NAME
        speed_factor = input(f"Во сколько вы хотите ускорить видео (1 если ускорение не нужно, по умолчанию {DEFAULT_SPEED_FACTOR}): ")
        speed_factor = float(speed_factor) if speed_factor else DEFAULT_SPEED_FACTOR
        offset_dB = input(f"Настройки чувствительности тишины ({DEFAULT_OFFSET_DB} по умолчанию): ")
        offset_dB = float(offset_dB) if offset_dB else DEFAULT_OFFSET_DB
        silence_gap = input(f"Настройки ожидания тишины ({DEFAULT_SILENCE_GAP} по умолчанию): ")
        silence_gap = float(silence_gap) if silence_gap else DEFAULT_SILENCE_GAP

        config = {
            'model_name': model_name,
            'speed_factor': speed_factor,
            'offset_dB': offset_dB,
            'silence_gap': silence_gap
        }
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)

    return model_name, speed_factor, offset_dB, silence_gap


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
                  f.endswith(('.tmp', '.txt', '.srt', '.mp4'))]
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

# Функция для чтения stderr в отдельном потоке
def read_stderr(process, log_container):
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_container.append(output)

# Функция для получения порога тишины
def get_silence_threshold(input_path):
    command = [
        'ffmpeg', '-i', input_path,
        '-af', 'ebur128', '-f', 'null', '-',
        '-progress', 'pipe:1'
    ]
    message = "Определение порога тишины"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Лог для stderr
    loudness_log = []

    # Запускаем поток для чтения stderr
    stderr_thread = threading.Thread(target=read_stderr, args=(process, loudness_log))
    stderr_thread.start()

    # Отображаем прогресс
    ffmpeg_progress_bar(process, get_video_duration_in_seconds(input_path), message)

    # Дожидаемся завершения потока stderr
    stderr_thread.join()

    # Объединяем все строки из stderr в один лог
    loudness_log_str = ''.join(loudness_log)

    # Регулярное выражение для извлечения порога тишины
    match = re.search(r'Integrated loudness:[\s\S]*?Threshold:\s*(-?\d+\.\d+)\s*LUFS', loudness_log_str)

    if match:
        threshold = float(match.group(1))
        print('Порог тишины в видео: ', threshold)
        return threshold
    else:
        print("Не удалось найти 'Threshold' в логах.")
        return None

# Функция для анализа аудио и извлечения данных о тишине
def analyze_audio(input_path):
    loudness = get_silence_threshold(input_path) #-39.7
    if loudness is None:
        raise RuntimeError("Не удалось определить порог тишины файла.")

    silence_threshold = loudness  + offset_dB
    silence_threshold_str = f"{silence_threshold}dB"

    command = [
        'ffmpeg', '-i', input_path,
        '-af', f'silencedetect=n={silence_threshold_str}:d={silence_gap}',
        '-f', 'null', '-',
        '-progress', 'pipe:1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # Лог для stderr
    silence_log = []

    # Запускаем поток для чтения stderr
    stderr_thread = threading.Thread(target=read_stderr, args=(process, silence_log))
    stderr_thread.start()

    # Отображаем прогресс
    ffmpeg_progress_bar(process, get_video_duration_in_seconds(input_path), "Поиск отрезков тишины")

    # Дожидаемся завершения потока stderr
    stderr_thread.join()

    # Объединяем все строки из stderr в один лог
    silence_log_str = ''.join(silence_log)

    silence_intervals = []
    start_time = None

    for line in silence_log_str.split('\n'):
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

def ffmpeg_progress_bar(process, total_duration_seconds, message):
    try:
        pbar = tqdm(total=100, desc=message, unit="%")

        # Временные переменные для точности прогресса
        last_percent = 0

        for line in process.stdout:
            if 'out_time_ms=' in line:
                try:
                    time_seconds = int(line.split('=')[1].strip()) / 1000 / 1000
                    percentage = (time_seconds / total_duration_seconds) * 100
                    percentage = min(100, round(percentage))  # Ограничиваем до 100%

                    # Обновляем прогресс-бар только если процент изменился
                    if percentage > last_percent:
                        pbar.update(percentage - last_percent)
                        last_percent = percentage

                except ValueError:
                    continue  # Игнорируем строки с ошибками

        process.wait()
        pbar.update(100 - last_percent)  # Завершаем прогресс-бар


    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении ffmpeg: {e}")
        raise


def get_video_duration_in_seconds(input_path):
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ]
    result = float(subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip())  # в секундах
    return result


# Функция для разбиения на куски с учетом тишины
def split_video_on_silence(input_path, silence_intervals, chunk_duration=15 * 60, search_window=5 * 60):
    video_duration = get_video_duration_in_seconds(input_path)
    split_points = []
    current_position = 0

    while video_duration - current_position > 5*60:
        target_time = current_position + chunk_duration
        search_start = max(0, target_time - search_window)
        search_end = min(video_duration, target_time + search_window)

        # Ищем самый длинный интервал тишины в окне поиска
        longest_silence = None
        for start, end in silence_intervals:
            if search_start <= start <= search_end:
                silence_duration = end - start
                if longest_silence is None or silence_duration > (longest_silence[1] - longest_silence[0]):
                    longest_silence = (start, end)

        if longest_silence:
            split_points.append(longest_silence[0])
            current_position = longest_silence[0]
        else:
            split_points.append(target_time)
            current_position = target_time

    if video_duration > current_position:
        split_points.remove(current_position)

    return split_points


# Разрезаем видео по найденным точкам
def split_video_by_points(input_path, split_points):
    temp_files = []
    points = [0] + split_points  # Добавляем начало видео как первую точку

    for i, start in enumerate(points, 1):  # Нумерация начинается с 1
        end = split_points[i - 1] if i <= len(split_points) else None
        output_chunk = os.path.join(TEMP_DIR, f"chunk_{i}.mp4")

        command = [
            'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
            '-ss', str(start)
        ] + (['-to', str(end)] if end else []) + ['-c', 'copy', '-b:v', '5000k', output_chunk]

        subprocess.run(command, check=True)
        temp_files.append(output_chunk)

    return temp_files


# Пересчитываем интервалы тишины для каждого фрагмента
def adjust_silence_intervals_for_chunks(silence_intervals, split_points):
    chunk_silence_intervals = []

    for i, start in enumerate(split_points):
        end = split_points[i + 1] if i + 1 < len(split_points) else None
        chunk_intervals = [(s - start, e - start) for s, e in silence_intervals if start <= s < (end or float('inf'))]
        chunk_silence_intervals.append(chunk_intervals)

    return chunk_silence_intervals


def remove_silence_using_metadata(input_path, silence_intervals, output_path):
    if not silence_intervals:
        print("Тишина не найдена. Используется исходное видео.")
        command = ['ffmpeg', '-loglevel', 'quiet', '-i', input_path, '-c', 'copy', output_path]
        subprocess.run(command, check=True)
        return

    # Создаем фильтр для удаления тишины
    filter_complex = ''.join([f"between(t,{start},{end})+" for start, end in silence_intervals])[:-1]

    # Проверяем, что фильтр не пустой
    if not filter_complex:
        raise ValueError("Фильтр для удаления тишины пустой.")

    command = [
        'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
        '-vf', f"select='({filter_complex})',setpts=N/FRAME_RATE/TB",
        '-af', f"aselect='({filter_complex})',asetpts=N/SR/TB",
        '-b:v', '5000k',
        '-progress', 'pipe:1',
        output_path
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    message = "Удаление тишины"
    ffmpeg_progress_bar(process, get_video_duration_in_seconds(input_path), message)




# Удаление тишины по кусочкам
def remove_silence_from_chunks(chunks, silence_intervals_list):
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        output_chunk = os.path.join(TEMP_DIR, f"chunk_no_silence_{i + 1}.mp4")
        silence_intervals = silence_intervals_list[i]

        remove_silence_using_metadata(chunk, silence_intervals, output_chunk)
        processed_chunks.append(output_chunk)

    return processed_chunks


# Соединение всех фрагментов
def concat_chunks(chunks, output_path):
    concat_file = os.path.join(TEMP_DIR, "concat_list.txt")
    with open(concat_file, 'w') as f:
        for chunk in chunks:
            f.write(f"file '{chunk}'\n")

    command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-loglevel', 'quiet', '-i', concat_file, '-c', 'copy', output_path
    ]
    subprocess.run(command, check=True)


# Функция для изменения скорости видео и аудио
def speed_up_video(input_path, output_path, speed_factor):
    command = [
        'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
        '-filter:v', f'setpts={1 / speed_factor}*PTS',
        '-filter:a', f'atempo={speed_factor}',
        '-progress', 'pipe:1',
        output_path
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) # subprocess.run(command, check=True)
    message = "Ускорение видео"
    ffmpeg_progress_bar(process, get_video_duration_in_seconds(input_path), message)

# Основной скрипт
def main():
    initialize_params()
    try:
        video_file_name = input("Введите название видеофайла (с расширением): ")
        input_path = os.path.join(SOURCE_DIR, video_file_name)
        if not os.path.exists(input_path):
            raise FileNotFoundError("Файл не найден. Проверьте путь и имя файла в папке .source.")

        # Промежуточные файлы
        temp_srt = os.path.join(OUTPUT_DIR, video_file_name.split('.')[0] + "_output.srt")

        # Разбиваем видео на куски по тишине
        print('ДЛИНА ВИДЕО!!!!!!!!!!!!!!', get_video_duration_in_seconds(input_path)/60, ' минут' )
        if get_video_duration_in_seconds(input_path) > 20 * 60:
            aprox_silence_intervals = analyze_audio(input_path)
            split_points = split_video_on_silence(input_path, aprox_silence_intervals)
            video_chunks = split_video_by_points(input_path, split_points)
            # Пересчитываем интервалы тишины для каждого куска
            chunk_silence_intervals = adjust_silence_intervals_for_chunks(aprox_silence_intervals, split_points)
            # Удаляем тишину из каждого куска
            no_silence_chunks = remove_silence_from_chunks(video_chunks, chunk_silence_intervals)
            # Соединяем куски без тишины в один файл
            temp_no_silence_video = os.path.join(TEMP_DIR, "final_no_silence.mp4")
            concat_chunks(no_silence_chunks, temp_no_silence_video)
        else:
            whisper_voice_analyze(input_path)

        # Загружаем модель для транскрибации
        model = load_model()
        voice_only_video = temp_output_path

        # Запуск таймера в отдельном потоке
        try:
            stop_event = threading.Event()
            print("Транскрибация...")
            timer_thread = threading.Thread(target=start_timer, args=(stop_event,))
            timer_thread.start()
            segments = transcribe_audio(model, voice_only_video)
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
        if speed_factor != 1:
            final_input_path = os.path.join(OUTPUT_DIR, video_file_name.split('.')[0] + "_output.mp4")
            speed_up_video(voice_only_video, final_input_path, speed_factor)
        else:
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file_name)[0]}_output.mp4")
            os.rename(voice_only_video, output_path)

        print("Готово!")
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    finally:
        clear_cache()
        print("Все временные файлы удалены.")


def whisper_voice_analyze(input_path, output_path):
    print('Определение участков с голосом')
    segments = transcribe_audio(whisper.load_model("tiny"), input_path)
    silence_intervals = [(seg['start'], seg['end']) for seg in segments]
    print('Вырезаем все лишнее')
    remove_silence_using_metadata(input_path, silence_intervals, output_path)
    return


if __name__ == "__main__":
    main()
