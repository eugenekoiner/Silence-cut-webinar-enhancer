import torch


# Проверка доступности GPU
def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используемое устройство: {device}")
    return device
