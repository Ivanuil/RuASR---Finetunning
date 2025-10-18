# `check_ffmpeg.py` - скрипт для проверки и настройки FFmpeg
import os
import subprocess
import sys

def check_ffmpeg():
    """Проверка доступности FFmpeg"""
    try:
        # Проверяем разные возможные команды
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg found via 'ffmpeg' command")
            return True
    except FileNotFoundError:
        pass

    # Проверяем возможные пути установки Windows
    possible_paths = [
        r"C:\Program Files\FFmpeg\bin",
        r"C:\ffmpeg\bin",
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages\FFmpeg.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"),
        r"C:\Users\{}\AppData\Local\Microsoft\WinGet\Packages\FFmpeg.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe".format(os.getenv('USERNAME'))
    ]

    for path in possible_paths:
        if os.path.exists(path):
            ffmpeg_path = os.path.join(path, "ffmpeg.exe")
            if os.path.exists(ffmpeg_path):
                print(f"✓ FFmpeg found at: {ffmpeg_path}")
                # Добавляем в PATH для текущей сессии
                os.environ["PATH"] += os.pathsep + path
                return True

    return False

def install_ffmpeg_windows():
    """Установка FFmpeg для Windows"""
    print("Installing FFmpeg...")

    methods = [
        # Метод 1: Через chocolatey
        ["choco", "install", "ffmpeg", "-y"],
        # Метод 2: Скачивание и распаковка
    ]

    for method in methods:
        try:
            if method[0] == "choco":
                # Проверяем установлен ли chocolatey
                subprocess.run(["choco", "--version"], capture_output=True, check=True)
                print("Installing FFmpeg via Chocolatey...")
                subprocess.run(method, check=True)
                return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    # Метод 3: Ручная установка
    print("\n📥 Please install FFmpeg manually:")
    print("1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases")
    print("2. Choose 'ffmpeg-master-latest-win64-gpl.zip'")
    print("3. Extract to C:\\ffmpeg")
    print("4. Add C:\\ffmpeg\\bin to your system PATH")
    return False

if __name__ == "__main__":
    if check_ffmpeg():
        print("FFmpeg is ready!")
    else:
        print("FFmpeg not found in PATH")
        if os.name == 'nt':  # Windows
            install_ffmpeg_windows()
