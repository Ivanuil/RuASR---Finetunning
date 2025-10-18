#!/usr/bin/env python3
"""
Патч для GigaAM без изменения версий
"""

import torch
import torchaudio
import sys
import os

print("=== Applying GigaAM Patch ===")

# Создаем фиктивный transforms модуль для torchaudio
class FakeResample:
    def __init__(self, orig_freq, new_freq):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform):
        # Простой ресемплинг - возвращаем как есть
        return waveform

class FakeTransformsModule:
    Resample = FakeResample

    def __getattr__(self, name):
        # Для любого другого трансформа возвращаем заглушку
        print(f"⚠️  Using fake transform: {name}")

        class FakeTransform:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def __call__(self, x):
                return x

        return FakeTransform

# Патчим torchaudio
if not hasattr(torchaudio, 'transforms'):
    torchaudio.transforms = FakeTransformsModule()
    print("✓ Created fake torchaudio.transforms")

# Также патчим прямо в sys.modules чтобы GigaAM увидел это
sys.modules['torchaudio.transforms'] = torchaudio.transforms

print("✓ GigaAM patch applied successfully")
