"""
Утилиты для работы с моделью GigaAM
"""

import torch
import os

def setup_gigaam_model(device):
    """Инициализация модели GigaAM-CTC"""
    try:
        import gigaam
        # Загрузка модели GigaAM-CTC
        model = gigaam.load_model("ctc")
        model.to(device)

        # Получаем процессор из модели
        processor = model.processor

        print("✓ GigaAM-CTC model loaded successfully")
        return model, processor

    except ImportError as e:
        raise ImportError(
            "GigaAM not installed. Please run setup_environment.py first"
        ) from e

def save_gigaam_model(model, processor, output_dir):
    """Сохранение модели GigaAM"""
    # Создаем директорию если не существует
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем модель и процессор
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"✓ Model saved to {output_dir}")

def load_gigaam_model(model_path, device):
    """Загрузка дообученной модели GigaAM"""
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(model_path)
        model = Wav2Vec2ForCTC.from_pretrained(model_path)
        model.to(device)

        return model, processor
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {e}")
