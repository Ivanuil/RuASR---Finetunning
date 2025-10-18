"""
Оценка модели на тестовом наборе
"""

import argparse
from datasets import load_dataset
from training_pipeline import TrainingPipeline
from data_processor import DataProcessor
from model_utils import load_model

def evaluate_model(model_path, output_file="evaluation_results.txt"):
    """Полная оценка модели"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    model, processor = load_model(model_path, device)

    # Загрузка данных
    dataset = load_dataset("google/fleurs", "ru")
    data_processor = DataProcessor(processor=processor)

    test_dataset = dataset["test"].map(
        data_processor.prepare_dataset,
        remove_columns=dataset["test"].column_names
    )

    # Настройка trainer для оценки
    training_args = TrainingArguments(
        output_dir="./eval_temp",
        per_device_eval_batch_size=8,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_processor.data_collator,
        compute_metrics=MetricsCalculator.compute_metrics,
    )

    # Оценка
    results = trainer.evaluate(test_dataset)

    # Сохранение результатов
    with open(output_file, "w") as f:
        f.write("=== Evaluation Results ===\n")
        f.write(f"WER: {results['eval_wer']:.4f}\n")
        f.write(f"CER: {results['eval_cer']:.4f}\n")
        f.write(f"Loss: {results['eval_loss']:.4f}\n")

    print(f"Evaluation completed. Results saved to {output_file}")
    print(f"WER: {results['eval_wer']:.4f}")
    print(f"CER: {results['eval_cer']:.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_file', type=str, default="evaluation_results.txt",
                        help='Output file for results')

    args = parser.parse_args()

    evaluate_model(args.model_path, args.output_file)

if __name__ == "__main__":
    main()
