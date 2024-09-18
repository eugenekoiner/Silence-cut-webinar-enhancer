import os

from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm

import file_manager


def transcribe_file(file_name, model):
    file_path = os.path.join(file_manager.my_files_dir, file_name)
    file_duration_sec = get_file_duration(file_name, file_path)
    print(f"Длительность файла: {file_duration_sec:.2f} секунд.")

    if file_duration_sec > 60:
        chunks = split_audio(file_path)
        all_segments = []
        total_chunks = len(chunks)
        with tqdm(total=total_chunks, desc="Транскрибирование", unit='chunk') as pbar:
            for chunk_path in chunks:
                segments = transcribe_chunk(chunk_path, model)
                all_segments.extend(segments)
                pbar.update(1)
                os.remove(chunk_path)
    else:
        all_segments = transcribe_chunk(file_path, model)

    return all_segments

def split_audio(file_path, chunk_duration_ms=60000, overlap_ms=5000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    start = 0
    chunk_duration = chunk_duration_ms - overlap_ms
    while start < len(audio):
        end = start + chunk_duration_ms
        chunk = audio[start:end]
        chunk_path = os.path.join(file_manager.temp_audio_dir, f"chunk_{start // chunk_duration_ms}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
        start += chunk_duration
    return chunks

def transcribe_chunk(chunk_path, model):
    print(f"Транскрибирование {chunk_path}...")
    result = model.transcribe(chunk_path, language="ru", verbose=True)
    return result["segments"]

def get_file_duration(file_name,file_path):
    if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
        video = VideoFileClip(file_path)
        return video.duration
    elif file_name.lower().endswith(('.mp3', '.wav', '.flac')):
        audio = AudioFileClip(file_path)
        return audio.duration
    else:
        raise ValueError("Неподдерживаемый формат файла.")
