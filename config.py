"""
Конфигурационные параметры
"""

class Config:
    # Модель
    MODEL_NAME = "ai-forever/gigam-ctc"

    # Данные
    DATASET_NAME = "google/fleurs"
    LANGUAGE = "ru"
    MAX_DURATION = 10.0

    # Обучение
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    WARMUP_STEPS = 500

    # Пути
    OUTPUT_DIR = "./gigam-ctc-fleurs-ru"
    LOGGING_DIR = "./logs"

    # Ранняя остановка
    EARLY_STOPPING_PATIENCE = 3
