import os
import json
import time
import threading
import subprocess
import torch
import whisper
import re
import warnings
from tqdm import tqdm
import datetime
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import tempfile

model_name = None
speed_factor = None
offset_dB = None
silence_gap = None
video_path = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, '.models')
SOURCE_DIR = os.path.join(SCRIPT_DIR, '.source')
TEMP_DIR = os.path.join(SCRIPT_DIR, '.temp')
TEMP_VIDEO_DIR = None
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '.output')
CONFIG_DIR = os.path.join(SCRIPT_DIR, '.config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=FutureWarning)
temp_no_silence_video = None
final_video_path = None
final_srt_path = None
video_file_name = None
continue_counter = 0


def initialize_params():
    global model_name, speed_factor, offset_dB, silence_gap, need_transcription
    DEFAULT_MODEL_NAME = "base"
    DEFAULT_SPEED_FACTOR = 1.25
    DEFAULT_OFFSET_DB = 1
    DEFAULT_SILENCE_GAP = 0.5
    DEFAULT_NEED_TRANSCRIPTION = 'no'

    if os.path.exists(CONFIG_FILE):
        print(f"Загрузка конфигурации из файла {os.path.basename(CONFIG_FILE)}...")
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        model_name = config.get('model_name', DEFAULT_MODEL_NAME)
        speed_factor = config.get('speed_factor', DEFAULT_SPEED_FACTOR)
        offset_dB = config.get('offset_dB', DEFAULT_OFFSET_DB)
        silence_gap = config.get('silence_gap', DEFAULT_SILENCE_GAP)
        need_transcription = config.get('transcription', DEFAULT_NEED_TRANSCRIPTION)
    else:
        print("Конфигурационный файл не найден. Введите параметры вручную.")
        model_name = input(f"Введите название модели (по умолчанию '{DEFAULT_MODEL_NAME}'): ") or DEFAULT_MODEL_NAME
        speed_factor = input(f"Во сколько вы хотите ускорить видео (1 если ускорение не нужно, по умолчанию {DEFAULT_SPEED_FACTOR}): ")
        speed_factor = float(speed_factor) if speed_factor else DEFAULT_SPEED_FACTOR
        offset_dB = input(f"Настройки чувствительности тишины ({DEFAULT_OFFSET_DB} по умолчанию): ")
        offset_dB = float(offset_dB) if offset_dB else DEFAULT_OFFSET_DB
        silence_gap = input(f"Настройки ожидания тишины ({DEFAULT_SILENCE_GAP} по умолчанию): ")
        silence_gap = float(silence_gap) if silence_gap else DEFAULT_SILENCE_GAP
        need_transcription = input(f"Нужны ли субтитры ({DEFAULT_NEED_TRANSCRIPTION} по умолчанию): " or DEFAULT_NEED_TRANSCRIPTION)
        config = {
            'model_name': model_name,
            'speed_factor': speed_factor,
            'offset_dB': offset_dB,
            'silence_gap': silence_gap,
            'need_transcription': need_transcription
        }
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    return model_name, speed_factor, offset_dB, silence_gap, need_transcription

def clear_cache():
    if not os.path.exists(TEMP_VIDEO_DIR):
        for file in os.listdir(TEMP_DIR):
            path = os.path.join(TEMP_DIR, file)
            try:
                shutil.rmtree(path) if os.path.isdir(path) else os.unlink(path)
            except Exception as e:
                print(f'Ошибка при удалении {path}: {e}')

def onetime_print():
    global continue_counter
    if continue_counter == 0:
        print('Продолжаем с места последней остановки.')
        continue_counter += 1

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Инициализация Whisper. Модель \"{model_name}\". Используется {device}.")
    model = whisper.load_model(model_name, device=device, download_root=MODEL_DIR)
    return model

def read_stderr(process, log_container):
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_container.append(output)

def timer_thread(start_time, stop_event, progress_bar):
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        progress_bar.set_postfix_str(f"Прошло времени: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
        time.sleep(1)

def adjust_subtitles(segments, speed_factor):
    adjusted_segments = []
    for segment in segments:
        start = round(segment['start'] / speed_factor, 2)
        end = round(segment['end'] / speed_factor, 2)
        adjusted_segments.append((start, end, segment['text']))
    return adjusted_segments

def get_silence_threshold(TEMP_VIDEO_DIR, input_path):
    try:
        threshold_file_path = os.path.join(TEMP_VIDEO_DIR, f"{os.path.splitext(os.path.basename(input_path))[0]}_threshold.txt")
        if os.path.exists(threshold_file_path):
            with open(threshold_file_path, 'r', encoding='utf-8') as f:
                threshold = float(f.read().strip())
                return threshold
        command = [
            'ffmpeg', '-i', input_path,
            '-af', 'ebur128', '-f', 'null', '-',
            '-progress', 'pipe:1'
        ]
        message = f"Анализ {os.path.basename(input_path)}"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        loudness_log = []
        stderr_thread = threading.Thread(target=read_stderr, args=(process, loudness_log))
        stderr_thread.start()
        ffmpeg_progress(process, get_video_duration_in_seconds(input_path), message)
        stderr_thread.join()
        loudness_log_str = ''.join(loudness_log)
        match = re.search(r'Integrated loudness:[\s\S]*?Threshold:\s*(-?\d+\.\d+)\s*LUFS', loudness_log_str)
        if match:
            threshold = float(match.group(1))
            with open(threshold_file_path, 'w', encoding='utf-8') as f:
                f.write(str(threshold))
            return threshold
        else:
            print("Не удалось найти 'Threshold' в логах.")
            return None
    except (KeyboardInterrupt, subprocess.CalledProcessError, Exception):
        raise


def analyze_audio(TEMP_VIDEO_DIR, offset_dB, input_path, save_name=None, get_non_silence=True):
    if save_name is None:
        save_name = os.path.splitext(os.path.basename(input_path))[0]
    video_duration = get_video_duration_in_seconds(input_path)
    interval_type = 'non_silence' if get_non_silence else 'silence'
    intervals_file_path = os.path.join(TEMP_VIDEO_DIR, f'{save_name}_{interval_type}_intervals.txt')
    if os.path.exists(intervals_file_path):
        with open(intervals_file_path, 'r', encoding='utf-8') as f:
            intervals = json.load(f)
        return intervals
    loudness = get_silence_threshold(TEMP_VIDEO_DIR, input_path)
    if loudness is None:
        raise RuntimeError("Не удалось определить порог тишины файла.")
    silence_threshold = loudness + offset_dB
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log', dir=TEMP_VIDEO_DIR) as temp_file:
        log_file_path = temp_file.name
    command = [
        'ffmpeg', '-i', input_path,
        '-af', f'silencedetect=n={silence_threshold}dB:d=0.5',
        '-f', 'null', '-',
        '-progress', 'pipe:1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    threading.Thread(target=ffmpeg_progress, args=(process, video_duration, f"Поиск отрезков тишины {save_name}")).start()
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        while True:
            line = process.stderr.readline()
            if not line:
                break
            log_file.write(line)
    process.wait()
    with open(log_file_path, 'r', encoding='utf-8') as log_file:
        silence_log_str = log_file.read()
    silence_intervals, non_silence_intervals = [], []
    start_time = 0.0
    for line in silence_log_str.split('\n'):
        start_match = re.search(r'silence_start:\s*([\d.]+)', line)
        if start_match:
            end_time = float(start_match.group(1))
            non_silence_intervals.append((start_time, end_time))
            silence_intervals.append((end_time, None))
        end_match = re.search(r'silence_end:\s*([\d.]+)', line)
        if end_match:
            start_time = float(end_match.group(1))
            if silence_intervals and silence_intervals[-1][1] is None:
                silence_intervals[-1] = (silence_intervals[-1][0], start_time)
    if start_time is not None and video_duration - start_time > 5:
        non_silence_intervals.append((start_time, video_duration))
    intervals = non_silence_intervals if get_non_silence else silence_intervals
    with open(intervals_file_path, 'w', encoding='utf-8') as f:
        json.dump(intervals, f)
    os.remove(log_file_path)
    return intervals


def calculate_remaining_duration(non_silence_intervals):
    remaining_duration = 0
    for start, end in non_silence_intervals:
        remaining_duration += (end - start)
    return round(float(remaining_duration), 2)

def ffmpeg_progress(process, total_duration_seconds, message):
    try:
        total_duration_seconds = round(total_duration_seconds, 2)
        pbar = tqdm(total=100, desc=message, unit="%")
        last_percent = 0
        for line in process.stdout:
            if 'out_time_ms=' in line:
                try:
                    time_seconds = round(int(line.split('=')[1].strip())/1000/1000, 2)
                    percentage = (time_seconds / total_duration_seconds) * 100
                    percentage = min(100, round(percentage))
                    if percentage > last_percent:
                        pbar.update(percentage - last_percent)
                        last_percent = percentage
                except ValueError:
                    continue
        process.wait()
        pbar.update(100 - last_percent)
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        raise


def get_video_duration_in_seconds(input_path):
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ]
    result = float(subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout.strip())  # в секундах
    return result

def split_video_on_silence(input_path, silence_intervals, chunk_duration=10 * 60, search_window=5 * 60):
    split_points = []
    split_points_file_path = os.path.join(TEMP_VIDEO_DIR, f'{os.path.splitext(os.path.basename(input_path))[0]}_split_points.txt')
    if os.path.exists(split_points_file_path):
        with open(split_points_file_path, 'r', encoding='utf-8') as f:
            split_points = json.load(f)
        return split_points
    video_duration = get_video_duration_in_seconds(input_path)
    current_position = 0
    while video_duration - current_position > 5*60:
        target_time = current_position + chunk_duration
        if target_time >= video_duration:
            break
        search_start = max(0, target_time - search_window)
        search_end = min(video_duration, target_time + search_window)
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
    if video_duration - current_position < 5*60:
        split_points.remove(current_position)
    with open(split_points_file_path, 'w', encoding='utf-8') as f:
        json.dump(split_points, f)
    return split_points

def split_video_by_points(input_path, split_points):
    points = [0] + split_points
    total_chunks = len(points)
    temp_files = []
    current_chunk = None
    try:
        if not len(get_chunks(silence=False)) == total_chunks:
            with tqdm(total=total_chunks, desc="Splitting video", unit="chunks") as pbar:
                for i, start in enumerate(points, 1):
                    end = split_points[i - 1] if i <= len(split_points) else None
                    current_chunk = os.path.join(TEMP_VIDEO_DIR, f"{os.path.splitext(video_file_name)[0]}_chunk_{i}.mp4")
                    if os.path.exists(current_chunk):
                        temp_files.append(current_chunk)
                        pbar.update(1)
                        continue
                    command = [
                                  'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
                                  '-ss', str(start)
                              ] + (['-to', str(end)] if end else []) + ['-c', 'copy', '-b:v', '2000k', current_chunk]
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                    process.wait()
                    if process.returncode != 0:
                        raise subprocess.CalledProcessError(process.returncode, command)
                    temp_files.append(current_chunk)
                    pbar.update(1)
            return temp_files
        else:
            return get_chunks(silence=False)
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        if current_chunk and os.path.exists(current_chunk):
            os.remove(current_chunk)
        raise

def get_chunks_non_silence_intervals():
    chunks = get_chunks(silence=False)
    pattern = re.compile(rf"{re.escape(os.path.splitext(video_file_name)[0])}_chunk_(\d+)_non_silence_intervals\.txt")
    intervals = get_temp_txts(pattern)
    if len(chunks) != len(intervals):
        intervals = [None] * len(chunks)
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    analyze_audio,
                    TEMP_VIDEO_DIR,
                    offset_dB,
                    chunk
                ): i for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                intervals[idx] = future.result()
    return intervals

def concatenate_chunks():
    output_file = os.path.join(TEMP_VIDEO_DIR, f"{os.path.splitext(video_file_name)[0]}_final_no_silence.mp4")
    if os.path.exists(output_file):
        return output_file
    video_chunks = get_chunks()
    if not video_chunks:
        raise FileNotFoundError(
            f"Видео файлы по шаблону {os.path.splitext(video_file_name)[0]}_chunk_*_no_silence.mp4 не найдены в {TEMP_VIDEO_DIR}.")
    concat_file_path = os.path.join(TEMP_VIDEO_DIR, f'{os.path.splitext(video_file_name)[0]}_concat_list.txt')
    if not os.path.exists(concat_file_path):
        with open(concat_file_path, 'w', encoding='cp1251') as concat_file:
            for chunk in video_chunks:
                concat_file.write(f"file '{chunk}'\n")
    command = [
        'ffmpeg', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
        '-i', concat_file_path, '-c', 'copy', '-progress', 'pipe:1',  output_file
    ]
    total_duration_seconds = sum(get_video_duration_in_seconds(chunk) for chunk in video_chunks)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    message = f"Соединение видеофрагментов для {video_file_name}"
    try:
        ffmpeg_progress(process, total_duration_seconds, message)
    except (KeyboardInterrupt, subprocess.CalledProcessError) as e:
        if not KeyboardInterrupt:
            print(f"Ошибка при соединении фрагментов: {e.stderr}")
        if (process.returncode is not None and process.returncode != 0) or process.poll() is None:
            if os.path.exists(output_file):
                os.remove(output_file)
                print(f"Файл {os.path.basename(output_file)} удален из-за ошибки.")
        raise
    return output_file



def get_chunks(silence=True):
    if get_video_duration_in_seconds(video_path) > 20*60:
        if silence:
            chunk_pattern = re.compile(rf"{re.escape(os.path.splitext(video_file_name)[0])}_chunk_(\d+)_no_silence\.mp4")
        else:
            chunk_pattern = re.compile(rf"{re.escape(os.path.splitext(video_file_name)[0])}_chunk_(\d+)\.mp4")
        video_chunks = sorted([
            os.path.join(TEMP_VIDEO_DIR, f) for f in os.listdir(TEMP_VIDEO_DIR)
            if chunk_pattern.match(f)
        ], key=lambda x: int(chunk_pattern.search(x).group(1)))
        return video_chunks
    else:
        if silence:
            return [temp_no_silence_video]
        else:
            return [video_path]

def get_temp_txts(pattern):
    temp_txts = sorted([
        os.path.join(TEMP_VIDEO_DIR, f) for f in os.listdir(TEMP_VIDEO_DIR)
        if pattern.match(f)
    ], key=lambda x: int(pattern.search(x).group(1)))
    return temp_txts

def remove_silence_using_metadata(input_path, output_path, TEMP_VIDEO_DIR):
    if os.path.exists(output_path):
        return
    intervals_file_path = os.path.join(TEMP_VIDEO_DIR, f'{os.path.splitext(os.path.basename(input_path))[0]}_non_silence_intervals.txt')
    if os.path.exists(intervals_file_path):
        with open(intervals_file_path, 'r', encoding='utf-8') as f:
            non_silence_intervals = json.load(f)
    if not non_silence_intervals:
        print("Тишина не найдена. Используется исходное видео.")
        command = ['ffmpeg', '-loglevel', 'quiet', '-i', input_path, '-c', 'copy', output_path]
        subprocess.run(command, check=True)
        return
    filter_complex = ''.join([
        f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{idx}];"
        f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{idx}];"
        for idx, (start, end) in enumerate(non_silence_intervals)
    ])
    if not filter_complex:
        raise ValueError("Фильтр для удаления тишины пустой.")
    concat_inputs = ''.join([f"[v{idx}][a{idx}]" for idx in range(len(non_silence_intervals))])
    filter_complex += f"{concat_inputs}concat=n={len(non_silence_intervals)}:v=1:a=1[v][a]"
    filter_file_path = os.path.join(TEMP_VIDEO_DIR, f'{os.path.splitext(os.path.basename(input_path))[0]}_silence_filter.txt')
    if not os.path.exists(filter_file_path):
        with open(filter_file_path, 'w', encoding='utf-8') as f:
            f.write(filter_complex)
    command = [
        'ffmpeg', '-loglevel', 'error', '-i', input_path,
        '-filter_complex_script', filter_file_path,
        '-map', '[v]', '-map', '[a]',
        '-b:v', '2m',
        '-progress', 'pipe:1',
        output_path
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    message = f"Удаление тишины в {os.path.basename(input_path)}"
    try:
        ffmpeg_progress(process, calculate_remaining_duration(non_silence_intervals), message)
    except (KeyboardInterrupt, subprocess.CalledProcessError) as e:
        print(f"Ошибка при выполнении ffmpeg: {e.stderr}")
        if (process.returncode is not None and process.returncode != 0) or process.poll() is None:
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Файл {os.path.basename(output_path)} удален из-за ошибки.")
        raise

def get_silence_threshold_for_chunks():
    chunks = get_chunks(silence=False)
    pattern = re.compile(rf"{re.escape(os.path.splitext(video_file_name)[0])}_chunk_(\d+)_threshold\.txt")
    thresholds = get_temp_txts(pattern)
    if len(chunks) != len(thresholds):
        thresholds = [None] * len(chunks)
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    get_silence_threshold,
                    TEMP_VIDEO_DIR,
                    chunk
                ): i for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                thresholds[idx] = future.result()
    return thresholds

def remove_silence_from_chunks(chunks):
    processed_chunks = [None] * len(chunks)
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                remove_silence_using_metadata,
                chunk,
                os.path.join(TEMP_VIDEO_DIR,
                             f"{os.path.splitext(os.path.basename(chunk))[0]}_no_silence.mp4"), TEMP_VIDEO_DIR
            ): i for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            processed_chunks[idx] = future.result()
    return processed_chunks

def transcribe_chunk(chunk_path, model):
    torch.cuda.set_per_process_memory_fraction
    temp_srt_path = os.path.join(TEMP_VIDEO_DIR, f"{os.path.splitext(os.path.basename(chunk_path))[0]}_temp_srt.txt")
    if os.path.exists(temp_srt_path):
        return
    segments = model.transcribe(chunk_path, language="ru")
    adjusted_segments = adjust_subtitles(segments['segments'], speed_factor)
    with open(temp_srt_path, 'w', encoding='utf-8') as srt_file:
        for i, (start, end, text) in enumerate(adjusted_segments, start=1):
            start_time = time.strftime("%H:%M:%S", time.gmtime(start))
            end_time = time.strftime("%H:%M:%S", time.gmtime(end))
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")
    torch.cuda.empty_cache()

def transcribe_all_chunks():
    chunks = get_chunks()
    pattern = re.compile(rf"{re.escape(os.path.splitext(video_file_name)[0])}_chunk_(\d+)_no_silence_temp_srt\.txt")
    if not len(get_temp_txts(pattern)) == len(chunks):
        model = load_model()
        print('Поскольку whisper не отдает прогресса до момента полного выполнения, может казаться что ничего не происходит.')
        print('Прогресс бар может обновляться раз в несколько минут.')
        start_time = time.time()
        stop_event = threading.Event()
        with tqdm(total=len(chunks), desc="Транскрибация", unit="chunk", leave=True, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]') as pbar:
            timer = threading.Thread(target=timer_thread, args=(start_time, stop_event, pbar))
            timer.start()
            for chunk in chunks:
                transcribe_chunk(chunk, model)
                pbar.update(1)
            stop_event.set()
            timer.join()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nОбщее время транскрибации: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

def concatenate_srt_files():
    video_chunks = get_chunks()
    try:
        with open(final_srt_path, 'w', encoding='utf-8') as final_srt:
            subtitle_counter = 1
            accumulated_time = 0.0
            for chunk in video_chunks:
                temp_srt_path = os.path.join(TEMP_VIDEO_DIR, f"{os.path.splitext(os.path.basename(chunk))[0]}_temp_srt.txt")
                chunk_duration = get_video_duration_in_seconds(chunk) / speed_factor
                if os.path.exists(temp_srt_path):
                    with open(temp_srt_path, 'r', encoding='utf-8', errors='replace') as temp_srt:
                        for line in temp_srt:
                            if '-->' in line:
                                times = line.strip().split(' --> ')
                                start_time = add_time_to_timestamp(times[0], accumulated_time)
                                end_time = add_time_to_timestamp(times[1], accumulated_time)
                                final_srt.write(f"{subtitle_counter}\n{start_time} --> {end_time}\n")
                                subtitle_counter += 1
                            else:
                                final_srt.write(line)
                else:
                    print(f"SRT файл для {os.path.basename(chunk)} не найден.")
                accumulated_time += chunk_duration
            print(f"Финальный файл субтитров сохранен как {os.path.basename(final_srt_path)}.")
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        traceback.print_exc()
        if os.path.exists(final_srt_path):
            os.remove(final_srt_path)

def add_time_to_timestamp(timestamp, accumulated_time):
    h, m, s = map(float, timestamp.split(":"))
    total_seconds = h * 3600 + m * 60 + s + accumulated_time
    return time.strftime("%H:%M:%S", time.gmtime(total_seconds))

def speed_up_video(input_path, output_path, speed_factor):
    if os.path.exists(output_path):
        print(f"Финальный файл {os.path.basename(output_path)} уже существует.")
        return
    command = [
        'ffmpeg', '-loglevel', 'quiet', '-i', input_path,
        '-filter:v', f'setpts={1 / speed_factor}*PTS',
        '-filter:a', f'atempo={speed_factor}',
        '-c:v', 'h264_nvenc',
        '-b:v', '500k',
        '-progress', 'pipe:1',
        output_path
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        message = "Ускорение видео"
        ffmpeg_progress(process, get_video_duration_in_seconds(temp_no_silence_video)/speed_factor, message)
    except (KeyboardInterrupt, subprocess.CalledProcessError) as e:
        if not KeyboardInterrupt:
            print(f"Ошибка при выполнении ffmpeg: {e.stderr}")
        if (process.returncode is not None and process.returncode != 0) or process.poll() is None:
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Файл {os.path.basename(output_path)} удален из-за ошибки.")
        raise



def main():
    initialize_params()
    global video_path
    global TEMP_VIDEO_DIR
    global temp_no_silence_video
    global final_video_path
    global final_srt_path
    global video_file_name
    try:
        video_file_name = input("Введите название видеофайла (с расширением): ")
        start_time = time.time()
        final_video_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_file_name)[0] + "_output.mp4")
        final_srt_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_file_name)[0] + "_output.srt")
        if os.path.exists(final_video_path) and os.path.exists(final_srt_path):
            print(f"Финальный файл {os.path.basename(final_video_path)} уже существует.")
            return
        TEMP_VIDEO_DIR = os.path.join(TEMP_DIR, os.path.splitext(video_file_name)[0])
        clear_cache()
        os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
        video_path = os.path.join(SOURCE_DIR, video_file_name)

        temp_no_silence_video = os.path.join(TEMP_VIDEO_DIR, f'{os.path.splitext(video_file_name)[0]}_final_no_silence.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError("Файл не найден. Проверьте путь и имя файла в папке .source.")
        print('Длительность видео', datetime.timedelta(seconds=int(get_video_duration_in_seconds(video_path))))
        if get_video_duration_in_seconds(video_path) > 20*60:
            video_chunks = get_chunks(silence=False)
            if not video_chunks:
                aprox_silence_intervals = analyze_audio(TEMP_VIDEO_DIR, offset_dB, video_path, get_non_silence=False)
                split_points = split_video_on_silence(video_path, aprox_silence_intervals)
                video_chunks = split_video_by_points(video_path, split_points)
            get_silence_threshold_for_chunks()
            get_chunks_non_silence_intervals()
            remove_silence_from_chunks(video_chunks)
            if not os.path.exists(os.path.join(OUTPUT_DIR, final_srt_path)) and need_transcription == 'yes':
                transcribe_all_chunks()
                concatenate_srt_files()
            concatenate_chunks()
        else:
            analyze_audio(TEMP_VIDEO_DIR, offset_dB, video_path)
            remove_silence_using_metadata(video_path, temp_no_silence_video, TEMP_VIDEO_DIR)
            if not os.path.exists(os.path.join(OUTPUT_DIR, final_srt_path)) and need_transcription == 'yes':
                transcribe_all_chunks()
                concatenate_srt_files()
        if speed_factor != 1:
            speed_up_video(temp_no_silence_video, final_video_path, speed_factor)
        else:
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_file_name)[0]}_processed.mp4")
            os.rename(temp_no_silence_video, output_path)
        print("Готово!")
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"Затраченное время на обработку видеофайла: {formatted_time}")
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
