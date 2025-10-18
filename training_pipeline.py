"""
Пайплайн обучения модели GigaAM-CTC
"""

import torch
from datasets import load_dataset, Audio
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from data_processor import DataProcessor
from metrics_calculator import MetricsCalculator
from model_utils import setup_gigaam_model, save_gigaam_model
import os

class TrainingPipeline:
    def __init__(self, batch_size, learning_rate, num_epochs, max_duration, output_dir):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_duration = max_duration
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # Инициализация компонентов
        self.processor = None
        self.model = None
        self.data_processor = None
        self.trainer = None

    def setup_components(self):
        """Инициализация всех компонентов GigaAM"""
        print("Setting up GigaAM components...")

        # Загрузка модели и процессора GigaAM
        self.model, self.processor = setup_gigaam_model(self.device)

        # Инициализация обработчика данных
        self.data_processor = DataProcessor(
            processor=self.processor,
            max_duration=self.max_duration
        )

    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        print("Loading and preparing FLEURS dataset...")

        # Загрузка датасета
        dataset = load_dataset("google/fleurs", "ru")

        # Конвертация аудио в правильный формат
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        # Фильтрация по длительности
        def filter_by_duration(example):
            return example["audio"]["duration"] <= self.max_duration

        dataset = dataset.filter(filter_by_duration)

        print(f"Train samples: {len(dataset['train'])}")
        print(f"Validation samples: {len(dataset['validation'])}")
        print(f"Test samples: {len(dataset['test'])}")

        # Подготовка данных
        train_dataset = dataset["train"].map(
            self.data_processor.prepare_dataset,
            remove_columns=dataset["train"].column_names
        )

        eval_dataset = dataset["validation"].map(
            self.data_processor.prepare_dataset,
            remove_columns=dataset["validation"].column_names
        )

        return train_dataset, eval_dataset

    def setup_training(self, train_dataset, eval_dataset):
        """Настройка обучения"""
        print("Setting up training...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            group_by_length=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            num_train_epochs=self.num_epochs,
            fp16=torch.cuda.is_available(),
            learning_rate=self.learning_rate,
            warmup_steps=300,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            logging_steps=100,
            push_to_hub=False,
            report_to=None,
            dataloader_num_workers=0 if os.name == 'nt' else 2,
            remove_unused_columns=False,
        )

        self.trainer = Trainer(
            model=self.model,
            data_collator=self.data_processor.data_collator,
            args=training_args,
            compute_metrics=lambda pred: MetricsCalculator.compute_metrics(pred, self.processor),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

    def run_training(self, resume_from_checkpoint=None):
        """Запуск обучения"""
        print("Starting GigaAM training pipeline...")

        # Настройка компонентов
        self.setup_components()

        # Подготовка данных
        train_dataset, eval_dataset = self.load_and_prepare_data()

        # Настройка обучения
        self.setup_training(train_dataset, eval_dataset)

        # Запуск обучения
        print("Starting training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Сохранение модели
        save_gigaam_model(self.model, self.processor, self.output_dir)

        # Сохранение метрик обучения
        self.trainer.save_metrics("train", train_result.metrics)

        # Финальная оценка
        print("Final evaluation...")
        final_metrics = self.trainer.evaluate(eval_dataset)

        print("\n=== Training Completed ===")
        print(f"Final WER: {final_metrics['eval_wer']:.4f}")
        print(f"Final CER: {final_metrics['eval_cer']:.4f}")

        # Проверка достижения цели
        if final_metrics['eval_wer'] < 0.08:
            print("🎉 Target achieved! WER < 8%")
        else:
            print("⚠️  Target not reached. WER >= 8%")

        return final_metrics
