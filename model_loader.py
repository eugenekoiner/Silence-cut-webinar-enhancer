import torch
import whisper


def load_model(model_name, model_cache_path, device):
    while model_name not in whisper.available_models():
        model_name = input(f"Введите корректное название модели: ")
    print("Загрузка модели...")
    model = whisper.load_model(model_name).to(device)
    model.load_state_dict(torch.load(model_cache_path, map_location=device))
    return model
