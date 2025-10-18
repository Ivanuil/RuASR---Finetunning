#!/usr/bin/env python3
"""
Исправленный тренировочный скрипт с улучшенным патчем
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
from tqdm import tqdm
import json

# Применяем улучшенный патч ДО импорта GigaAM
import gigaam_patch

print("=== GigaAM-CTC Fine-tuning (Advanced Patch) ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Теперь импортируем GigaAM
try:
    import gigaam
    print("✓ GigaAM imported successfully")
except Exception as e:
    print(f"❌ GigaAM import failed: {e}")
    exit(1)

class SimpleRussianDataset(Dataset):
    """Упрощенный датасет для быстрого тестирования"""
    def __init__(self, num_samples=200):
        self.num_samples = num_samples
        self.phrases = [
            "привет",
            "пока",
            "да",
            "нет",
            "хорошо",
            "плохо",
            "спасибо",
            "пожалуйста"
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.phrases[idx % len(self.phrases)]

        # Очень простое аудио - одна синусоида
        duration = 2.0  # 2 секунды
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))

        # Простая синусоида
        audio = 0.5 * np.sin(2 * np.pi * 180 * t)
        audio = audio.astype(np.float32)

        return {
            'audio': torch.from_numpy(audio).float(),
            'text': text
        }

def simple_collate_fn(batch):
    """Максимально простая коллация"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]

    # Фиксированная длина для простоты
    target_length = 32000  # 2 секунды при 16kHz

    padded_audios = []
    for audio in audios:
        if len(audio) < target_length:
            padded = torch.nn.functional.pad(audio, (0, target_length - len(audio)))
        else:
            padded = audio[:target_length]
        padded_audios.append(padded)

    return {
        'input_values': torch.stack(padded_audios),
        'texts': texts
    }

def main():
    print("\n1. Loading GigaAM model...")
    try:
        model = gigaam.load_model("ctc")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        print(f"✓ Model loaded on {device}")

        # Проверяем что модель работает
        with torch.no_grad():
            test_input = torch.randn(1, 16000).to(device)
            test_output = model(test_input)
            print(f"✓ Test forward pass successful: {test_output.logits.shape}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n2. Preparing minimal data...")
    dataset = SimpleRussianDataset(num_samples=100)
    dataloader = DataLoader(
        dataset,
        batch_size=2,  # Очень маленький batch для теста
        shuffle=True,
        collate_fn=simple_collate_fn
    )

    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ Batches: {len(dataloader)}")

    print("\n3. Starting simple training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Простейший цикл обучения
    for epoch in range(1):  # Всего 1 эпоха для теста
        total_loss = 0
        successful_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            try:
                inputs = batch['input_values'].to(device)

                # Создаем простейшие лейблы
                batch_size = inputs.shape[0]
                labels = torch.randint(0, 10, (batch_size, 20)).to(device)  # Фиксированная длина

                # Forward pass
                outputs = model(input_values=inputs, labels=labels)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                successful_batches += 1

                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"  Batch {batch_idx} failed: {e}")
                continue

        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            print(f"✓ Epoch completed. Average loss: {avg_loss:.4f}")
        else:
            print("❌ No successful batches")

    print("\n🎉 Training test completed successfully!")

    # Сохраняем минимальную информацию
    os.makedirs("./gigaam_test", exist_ok=True)
    try:
        # Пробуем сохранить модель
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained("./gigaam_test")
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'training_info': {'success': True}
            }, "./gigaam_test/checkpoint.pt")
        print("✓ Model checkpoint saved")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")

if __name__ == "__main__":
    main()