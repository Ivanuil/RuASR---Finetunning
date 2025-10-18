"""
Обработка и подготовка данных для GigaAM
"""

import torch
import torchaudio
import re
from transformers import Wav2Vec2Processor

class DataProcessor:
    def __init__(self, processor, max_duration=10.0):
        self.processor = processor
        self.max_duration = max_duration

    @staticmethod
    def normalize_text(text: str) -> str:
        """Нормализация текста согласно требованиям"""
        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление знаков препинания, кроме апострофа для латиницы
        text = re.sub(r'[^\w\s\']', '', text)

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def prepare_dataset(self, batch):
        """Подготовка батча данных для GigaAM"""
        audio = batch["audio"]

        # Аудио уже должно быть 16kHz из-за cast_column
        waveform = torch.tensor(audio["array"], dtype=torch.float32)
        sample_rate = audio["sampling_rate"]

        # Дополнительная проверка sample rate
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Извлечение features с помощью процессора GigaAM
        inputs = self.processor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values[0]

        # Нормализация текста
        normalized_text = self.normalize_text(batch["transcription"])

        # Токенизация текста
        with self.processor.as_target_processor():
            labels = self.processor(
                normalized_text,
                return_tensors="pt"
            ).input_ids[0]

        return {
            "input_values": input_values,
            "labels": labels,
            "attention_mask": torch.ones_like(input_values)
        }

    @property
    def data_collator(self):
        """Data collator для CTC"""
        return DataCollatorCTCWithPadding(processor=self.processor)

class DataCollatorCTCWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

        # Замена паддинга в лейблах на -100
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
