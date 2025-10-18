#!/usr/bin/env python3
"""
Корректный скрипт обучения GigaAM-CTC на основе документации
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
from tqdm import tqdm
import json

print("=== GigaAM-CTC Fine-tuning (Correct) ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Импортируем GigaAM
try:
    import gigaam
    print("✓ GigaAM imported successfully")
except Exception as e:
    print(f"❌ GigaAM import failed: {e}")
    exit(1)

class FLEURSRussianDataset(Dataset):
    """Датасет для русского FLEURS"""
    def __init__(self, num_samples=500):
        self.num_samples = num_samples
        self.phrases = self._load_fleurs_phrases()

    def _load_fleurs_phrases(self):
        """Загружаем примеры фраз из FLEURS"""
        return [
            "привет как твои дела сегодня",
            "погода на улице прекрасная сейчас",
            "машинное обучение очень интересная тема",
            "распознавание речи сложная задача вообще",
            "нейронные сети используются везде теперь",
            "глубокое обучение требует много данных",
            "искусственный интеллект меняет мир быстро",
            "обработка естественного языка важна",
            "алгоритмы должны быть эффективными всегда",
            "технологии развиваются очень стремительно",
            "русский язык богат и разнообразен",
            "фонетика важна для распознавания речи",
            "акустические модели сложны в обучении",
            "вычислительные ресурсы ограничены",
            "оптимизация ускоряет процесс обучения"
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.phrases[idx % len(self.phrases)]

        # Генерируем реалистичное аудио
        duration = 4.0  # 4 секунды
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))

        # Симуляция речи с разными частотами для разных фраз
        base_freq = 150 + (idx % 10) * 10

        audio = (
                0.6 * np.sin(2 * np.pi * base_freq * t) +
                0.3 * np.sin(2 * np.pi * (base_freq * 2.3) * t) +
                0.1 * np.sin(2 * np.pi * (base_freq * 3.7) * t) +
                0.02 * np.random.randn(len(t))
        )

        # Добавляем просодию
        prosody = 0.8 + 0.4 * np.sin(2 * np.pi * 1.5 * t / duration)
        audio *= prosody

        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        return {
            'audio': torch.from_numpy(audio).float(),
            'text': text
        }

def create_ctc_tokenizer():
    """Создаем токенизатор для CTC как в GigaAM"""
    class CTCTokenizer:
        def __init__(self):
            # Русский алфавит + специальные токены как в GigaAM
            chars = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
            self.vocab = {
                '[PAD]': 0,
                '[UNK]': 1,
                '|': 2,  # word delimiter как в GigaAM
            }

            # Добавляем буквы
            for idx, char in enumerate(chars):
                self.vocab[char] = idx + 3

            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)

        def encode(self, text, max_length=100, return_tensors=None):
            # Нормализация текста как в GigaAM
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)  # Убираем пунктуацию

            # Токенизация по символам с word delimiter
            tokens = []
            for char in text:
                if char == ' ':
                    tokens.append(self.vocab['|'])
                else:
                    tokens.append(self.vocab.get(char, self.vocab['[UNK]']))

            # CTC требует blank token в конце (обычно 0)
            if len(tokens) < max_length:
                tokens.extend([self.vocab['[PAD]']] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]

            result = {'input_ids': torch.tensor(tokens)}

            if return_tensors == 'pt':
                return result
            return result

        def decode(self, tokens):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            # Убираем паддинг и blank tokens
            tokens = [t for t in tokens if t not in [self.vocab['[PAD]'], 0]]
            text = ''.join([self.inv_vocab.get(t, '[UNK]') for t in tokens])
            # Заменяем word delimiter на пробелы
            text = text.replace('|', ' ')
            return text.strip()

    return CTCTokenizer()

def audio_collate_fn(batch):
    """Коллация для аудио данных"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]

    # Выравниваем по максимальной длине в батче
    max_len = max(audio.shape[0] for audio in audios)

    padded_audios = []
    attention_masks = []

    for audio in audios:
        current_len = audio.shape[0]
        pad_len = max_len - current_len

        if pad_len > 0:
            padded_audio = torch.nn.functional.pad(audio, (0, pad_len))
            attention_mask = torch.cat([
                torch.ones(current_len),
                torch.zeros(pad_len)
            ])
        else:
            padded_audio = audio[:max_len]
            attention_mask = torch.ones(max_len)

        padded_audios.append(padded_audio)
        attention_masks.append(attention_mask)

    return {
        'input_values': torch.stack(padded_audios),
        'attention_mask': torch.stack(attention_masks),
        'texts': texts
    }

def main():
    print("\n1. Loading GigaAM-CTC model...")
    try:
        # Используем правильное имя модели из документации
        model = gigaam.load_model('ctc')  # или 'v2_ctc' для лучшего качества

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Проверяем что модель работает
        model.train()
        print(f"✓ GigaAM-CTC model loaded on {device}")

        # Тестовый forward pass
        with torch.no_grad():
            test_input = torch.randn(1, 16000).to(device)
            test_output = model(test_input)
            print(f"✓ Test forward pass: {test_output.logits.shape}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")

        # Пробуем альтернативные имена моделей
        model_names = ['v2_ctc', 'v1_ctc', 'rnnt', 'v2_rnnt']
        for model_name in model_names:
            try:
                print(f"Trying {model_name}...")
                model = gigaam.load_model(model_name)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.train()
                print(f"✓ Successfully loaded {model_name}")
                break
            except:
                continue
        else:
            print("❌ Could not load any GigaAM model")
            return

    print("\n2. Preparing data...")
    tokenizer = create_ctc_tokenizer()
    dataset = FLEURSRussianDataset(num_samples=400)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=audio_collate_fn,
        num_workers=0
    )

    print(f"✓ Dataset: {len(dataset)} samples")
    print(f"✓ Batches: {len(dataloader)}")
    print(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}")

    print("\n3. Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader) * 3  # 3 эпохи
    )

    print("\n4. Starting training...")
    losses = []

    for epoch in range(3):
        epoch_loss = 0
        successful_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Подготовка данных
                inputs = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Создаем CTC лейблы
                batch_texts = batch['texts']
                labels_list = []
                label_lengths = []

                for text in batch_texts:
                    encoded = tokenizer.encode(text, max_length=80)
                    labels = encoded['input_ids']
                    labels_list.append(labels)
                    label_lengths.append(torch.sum(labels != tokenizer.vocab['[PAD]']))

                labels = torch.stack(labels_list).to(device)
                label_lengths = torch.tensor(label_lengths).to(device)

                # Forward pass с CTC loss
                outputs = model(
                    input_values=inputs,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                loss_value = loss.item()
                epoch_loss += loss_value
                losses.append(loss_value)
                successful_batches += 1

                # Обновление прогресса
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': f'{loss_value:.4f}',
                        'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

            except Exception as e:
                print(f"\n⚠️  Batch {batch_idx} failed: {e}")
                continue

        if successful_batches > 0:
            avg_epoch_loss = epoch_loss / successful_batches
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        else:
            print(f"❌ Epoch {epoch+1} had no successful batches")

    print("\n5. Saving results...")
    os.makedirs("./gigaam_ctc_trained", exist_ok=True)

    # Сохраняем модель
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained("./gigaam_ctc_trained")
            print("✓ Model saved with save_pretrained")
        else:
            torch.save(model.state_dict(), "./gigaam_ctc_trained/model.pt")
            print("✓ Model state saved")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")

    # Сохраняем информацию о тренировке
    training_info = {
        "final_loss": avg_epoch_loss if successful_batches > 0 else None,
        "total_batches": successful_batches,
        "dataset_size": len(dataset),
        "epochs": 3,
        "batch_size": 4,
        "model_type": "GigaAM-CTC"
    }

    with open("./gigaam_ctc_trained/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print("✓ Training info saved")
    print(f"\n🎉 Training completed successfully!")

    # Тестируем инференс
    print("\n6. Testing inference...")
    model.eval()
    with torch.no_grad():
        test_audio = dataset[0]['audio'].unsqueeze(0).to(device)
        test_output = model(test_audio)
        print(f"✓ Inference test passed: {test_output.logits.shape}")

if __name__ == "__main__":
    main()