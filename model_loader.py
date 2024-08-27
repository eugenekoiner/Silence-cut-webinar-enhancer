# Функция для загрузки модели
import torch
import whisper


def load_model(model_name, model_cache_path, device):
    print("Загрузка модели...")
    model = whisper.load_model(model_name).to(device)
    torch.save(model.state_dict(), model_cache_path)
    return model