"""
Скрипт для установки и настройки окружения GigaAM
"""

import subprocess
import sys
import os

def setup_gigaam():
    """Установка GigaAM из репозитория"""
    print("Setting up GigaAM environment...")

    # Проверяем, установлен ли ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("✓ ffmpeg is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg first:")
        print("Windows: https://ffmpeg.org/download.html#build-windows")
        print("Linux: sudo apt install ffmpeg")
        print("Mac: brew install ffmpeg")
        return False

    # Клонируем и устанавливаем GigaAM
    if not os.path.exists("GigaAM"):
        print("Cloning GigaAM repository...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/salute-developers/GigaAM.git"
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone GigaAM: {e}")
            return False

    # Устанавливаем пакет
    print("Installing GigaAM package...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "GigaAM"
        ], check=True)
        print("✓ GigaAM installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install GigaAM: {e}")
        return False

def test_installation():
    """Тестирование установки"""
    try:
        import gigaam
        model = gigaam.load_model("ctc")
        print("✓ GigaAM imported successfully")
        print(f"✓ Model loaded: {model}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GigaAM: {e}")
        return False

if __name__ == "__main__":
    if setup_gigaam():
        test_installation()
