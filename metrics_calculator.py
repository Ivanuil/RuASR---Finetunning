"""
Вычисление метрик WER и CER
"""

import numpy as np
from jiwer import wer

class MetricsCalculator:
    @staticmethod
    def compute_metrics(pred):
        """Вычисление WER и CER"""
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        # Декодирование предсказаний и меток
        pred_str = pred.processor.batch_decode(pred_ids)
        label_str = pred.processor.batch_decode(pred.label_ids, group_tokens=False)

        # WER
        wer_score = wer(label_str, pred_str)

        # CER
        cer_score = MetricsCalculator.calculate_cer(label_str, pred_str)

        return {"wer": wer_score, "cer": cer_score}

    @staticmethod
    def calculate_cer(refs, hyps):
        """Вычисление Character Error Rate"""
        cers = []
        for ref, hyp in zip(refs, hyps):
            ref_chars = list(ref.replace(" ", ""))
            hyp_chars = list(hyp.replace(" ", ""))

            # Расчет расстояния Левенштейна
            distances = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

            for i in range(len(ref_chars) + 1):
                distances[i][0] = i
            for j in range(len(hyp_chars) + 1):
                distances[0][j] = j

            for i in range(1, len(ref_chars) + 1):
                for j in range(1, len(hyp_chars) + 1):
                    cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
                    distances[i][j] = min(
                        distances[i-1][j] + 1,
                        distances[i][j-1] + 1,
                        distances[i-1][j-1] + cost
                    )

            cer = distances[-1][-1] / max(len(ref_chars), 1)
            cers.append(cer)

        return np.mean(cers)
