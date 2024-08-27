from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import os
import numpy as np
from tqdm import tqdm


class Transcriber:
    def __init__(self, model, file_manager):
        self.model = model
        self.file_manager = file_manager

    def split_audio(self, filename, chunk_duration_ms=60000, overlap_ms=5000):
        audio = AudioSegment.from_file(filename)
        chunks = []
        start = 0
        chunk_duration = chunk_duration_ms - overlap_ms

        while start < len(audio):
            end = start + chunk_duration_ms
            chunk = audio[start:end]
            chunk_path = os.path.join(self.file_manager.temp_dir, f"chunk_{start // chunk_duration_ms}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            start += chunk_duration  # Move start position with overlap
        return chunks

    def transcribe_chunk(self, chunk_path):
        print(f"Транскрибирование {chunk_path}...")
        result = self.model.transcribe(chunk_path, language="ru", verbose=True)
        return result["segments"]

    def transcribe_file(self, filename):
        file_duration_sec = self.get_file_duration(filename)
        print(f"Длительность файла: {file_duration_sec:.2f} секунд.")

        if file_duration_sec > 60:
            chunks = self.split_audio(filename)
            all_segments = []

            total_chunks = len(chunks)
            with tqdm(total=total_chunks, desc="Транскрибирование", unit='chunk') as pbar:
                for chunk_path in chunks:
                    segments = self.transcribe_chunk(chunk_path)
                    all_segments.extend(segments)
                    pbar.update(1)
                    os.remove(chunk_path)  # Удаление временного файла
        else:
            print("Файл меньше минуты. Выполняется транскрибирование без разбиения на фрагменты.")
            all_segments = self.transcribe_chunk(filename)

        return all_segments

    def get_file_duration(self, filename):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video = VideoFileClip(filename)
            return video.duration
        elif filename.lower().endswith(('.mp3', '.wav', '.flac')):
            audio = AudioFileClip(filename)
            return audio.duration
        else:
            raise ValueError("Неподдерживаемый формат файла.")
