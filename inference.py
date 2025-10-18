"""
Скрипт для инференса на новых аудиофайлах
"""

import argparse
import torch
import torchaudio
from model_utils import load_model

def inference_audio(audio_path, model_path):
    """Инференс на одном аудиофайле"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model, processor = load_model(model_path, device)
    model.eval()

    # Загрузка аудио
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ресемплинг
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Предобработка
    inputs = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Инференс
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

def main():
    parser = argparse.ArgumentParser(description='Run inference on audio files')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')

    args = parser.parse_args()

    transcription = inference_audio(args.audio_path, args.model_path)
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    main()
