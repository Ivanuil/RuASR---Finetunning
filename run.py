"""
Основной скрипт для запуска всего пайплайна
"""

import sys
import os

def main():
    print("=== GigaAM-CTC FLEURS-Ru Fine-tuning ===")

    # Проверяем установлен ли GigaAM
    try:
        import gigaam
        print("✓ GigaAM is installed")
    except ImportError:
        print("❌ GigaAM not found. Setting up environment...")
        from setup_environment import setup_gigaam, test_installation
        if not setup_gigaam():
            sys.exit(1)
        if not test_installation():
            sys.exit(1)

    # Запускаем обучение
    from train_fixed import main as train_main
    train_main()

if __name__ == "__main__":
    main()
