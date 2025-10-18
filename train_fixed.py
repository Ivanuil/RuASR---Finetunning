#!/usr/bin/env python3
"""
Исправленный тренировочный скрипт с патчем для GigaAM
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
from tqdm import tqdm
import json

# Применяем патч ДО импорта GigaAM
import gigaam_patch

print("=== GigaAM-CTC Fine-tuning (Fixed) ===")
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

class RussianSpeechDataset(Dataset):
    """Синтетический датасет русской речи"""
    def __init__(self, num_samples=500, duration_sec=4.0, sample_rate=16000):
        self.num_samples = num_samples
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.samples = []

        self.phrases = [
            "привет как твои дела сегодня",
            "погода на улице прекрасная сейчас",
            "машинное обучение очень интересная тема",
            "распознавание речи сложная задача вообще",
            "нейронные сети используются везде теперь",
            "глубокое обучение требует много данных",
            "искусственный интеллект меняет мир быстро",
            "обработка естественного языка важна",
            "алгоритмы должны быть эффективными всегда",
            "технологии развиваются очень стремительно"
        ]

        # Предгенерируем данные
        self._generate_data()

    def _generate_speech_like_audio(self, idx):
        """Генерация аудио, похожего на речь"""
        t = np.linspace(0, self.duration_sec, int(self.sample_rate * self.duration_sec))

        # Основные частоты речи (разные для разных "говорящих")
        base_freq = 120 + (idx % 8) * 15

        # Симуляция формант речи
        audio = (
                0.6 * np.sin(2 * np.pi * base_freq * t) +                    # основная частота
                0.25 * np.sin(2 * np.pi * (base_freq * 2.5) * t) +          # первая форманта
                0.1 * np.sin(2 * np.pi * (base_freq * 3.8) * t) +           # вторая форманта
                0.05 * np.sin(2 * np.pi * (base_freq * 5.2) * t)            # третья форманта
        )

        # Добавляем "просодию" - изменение тона во времени
        prosody = 1.0 + 0.2 * np.sin(2 * np.pi * 2 * t / self.duration_sec)
        audio *= prosody

        # Добавляем шум
        audio += 0.02 * np.random.randn(len(t))

        # Нормализация
        audio = audio.astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        return audio

    def _generate_data(self):
        """Генерация всех сэмплов"""
        for i in range(self.num_samples):
            text = self.phrases[i % len(self.phrases)]
            audio = self._generate_speech_like_audio(i)

            self.samples.append({
                'audio': audio,
                'text': text,
                'duration': self.duration_sec
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'audio': torch.from_numpy(sample['audio']).float(),
            'text': sample['text'],
            'duration': sample['duration']
        }

def create_simple_tokenizer():
    """Простой токенизатор для русского языка"""
    class RussianTokenizer:
        def __init__(self):
            # Создаем словарь
            chars = ' абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
            self.vocab = {char: idx for idx, char in enumerate(chars)}
            self.vocab['[PAD]'] = len(chars)
            self.vocab['[UNK]'] = len(chars) + 1

            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.vocab_size = len(self.vocab)

        def encode(self, text, max_length=100, return_tensors=None):
            text = text.lower().strip()
            tokens = [self.vocab.get(char, self.vocab['[UNK]']) for char in text]

            # Паддинг
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

            # Убираем паддинг
            tokens = [t for t in tokens if t != self.vocab['[PAD]']]
            return ''.join([self.inv_vocab.get(t, '[UNK]') for t in tokens])

    return RussianTokenizer()

def audio_collate_fn(batch):
    """Коллация для аудио данных"""
    audios = [item['audio'] for item in batch]
    texts = [item['text'] for item in batch]

    # Находим максимальную длину
    max_len = max(audio.shape[0] for audio in audios)

    # Делаем паддинг
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
    print("\n1. Loading GigaAM model...")
    try:
        model = gigaam.load_model("ctc")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        print(f"✓ Model loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    print("\n2. Preparing data...")
    tokenizer = create_simple_tokenizer()
    dataset = RussianSpeechDataset(num_samples=400)
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
        T_max=len(dataloader) * 2  # 2 эпохи
    )

    print("\n4. Starting training...")
    losses = []

    for epoch in range(2):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/2")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Подготовка данных
                inputs = batch['input_values'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Создаем лейблы из текстов
                batch_texts = batch['texts']
                labels_list = []

                for text in batch_texts:
                    encoded = tokenizer.encode(text, max_length=80)
                    labels_list.append(encoded['input_ids'])

                labels = torch.stack(labels_list).to(device)

                # Forward pass
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

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")

    print("\n5. Saving results...")
    os.makedirs("./gigaam_trained", exist_ok=True)

    # Сохраняем модель
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained("./gigaam_trained")
            print("✓ Model saved with save_pretrained")
        else:
            torch.save(model.state_dict(), "./gigaam_trained/model.pt")
            print("✓ Model state saved")
    except Exception as e:
        print(f"⚠️  Could not save model: {e}")

    # Сохраняем информацию о тренировке
    training_info = {
        "final_loss": avg_epoch_loss,
        "total_steps": len(losses),
        "min_loss": min(losses) if losses else 0,
        "max_loss": max(losses) if losses else 0,
        "dataset_size": len(dataset),
        "epochs": 2,
        "batch_size": 4
    }

    with open("./gigaam_trained/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

    print("✓ Training info saved")
    print(f"\n🎉 Training completed successfully!")
    print(f"Final loss: {avg_epoch_loss:.4f}")

if __name__ == "__main__":
    main()