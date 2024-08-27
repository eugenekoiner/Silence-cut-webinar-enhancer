from tqdm import tqdm


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
