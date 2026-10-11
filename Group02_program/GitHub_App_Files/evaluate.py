"""Compare the fine-tuned classifier and pretrained embedding retriever."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import pandas as pd

from banking77_core import (
    DEFAULT_MODEL_DIR,
    HuggingFaceEmbeddingRetriever,
    HuggingFaceIntentClassifier,
    load_banking77,
    make_reference_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/banking77_comparison.csv"))
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--references-per-intent", type=int, default=25)
    return parser.parse_args()


def metrics_for(predictions: list[int], labels: list[int]) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def classifier_label_id(label: str, class_names: list[str]) -> int:
    normalized = label.removeprefix("LABEL_")
    if normalized in class_names:
        return class_names.index(normalized)
    try:
        return int(normalized)
    except ValueError as error:
        raise ValueError(f"Classifier returned an unknown label: {label}") from error


def evaluate_classifier(
    classifier: HuggingFaceIntentClassifier,
    texts: list[str],
    labels: list[int],
    class_names: list[str],
) -> tuple[dict[str, Any], list[int]]:
    start = time.perf_counter()
    predictions = classifier.predict_many(texts, top_k=1)
    inference_seconds = time.perf_counter() - start
    predicted_labels = [
        classifier_label_id(row[0]["intent"], class_names)
        for row in predictions
    ]
    accuracy, macro_f1 = metrics_for(predicted_labels, labels)
    return (
        {
            "pipeline": "fine_tuned_distilbert",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "setup_seconds": "",
            "inference_seconds": inference_seconds,
            "seconds_per_sample": inference_seconds / max(1, len(texts)),
            "n_samples": len(texts),
        },
        predicted_labels,
    )


def evaluate_retriever(
    train_split: Any,
    texts: list[str],
    labels: list[int],
    class_names: list[str],
    references_per_intent: int,
) -> tuple[dict[str, Any], list[int]]:
    reference_texts, reference_labels = make_reference_examples(
        train_split,
        examples_per_intent=references_per_intent,
    )
    setup_start = time.perf_counter()
    retriever = HuggingFaceEmbeddingRetriever(
        reference_texts,
        reference_labels,
        class_names,
    )
    setup_seconds = time.perf_counter() - setup_start

    inference_start = time.perf_counter()
    predictions = retriever.predict_many(texts, top_k=1)
    inference_seconds = time.perf_counter() - inference_start
    predicted_labels = [int(row[0]["label_id"]) for row in predictions]
    accuracy, macro_f1 = metrics_for(predicted_labels, labels)
    return (
        {
            "pipeline": "pretrained_embedding_retrieval",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "setup_seconds": setup_seconds,
            "inference_seconds": inference_seconds,
            "seconds_per_sample": inference_seconds / max(1, len(texts)),
            "n_samples": len(texts),
        },
        predicted_labels,
    )


def main() -> None:
    args = parse_args()
    if args.max_test_samples < 0:
        raise ValueError("--max-test-samples cannot be negative")
    if args.references_per_intent < 1:
        raise ValueError("--references-per-intent must be positive")
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at {args.model_dir}. Run `python train.py` first."
        )

    dataset, class_names = load_banking77()
    test_split = dataset["test"]
    if args.max_test_samples:
        test_split = test_split.select(range(min(args.max_test_samples, len(test_split))))
    texts = [str(row["text"]) for row in test_split]
    labels = [int(row["label"]) for row in test_split]

    classifier = HuggingFaceIntentClassifier(args.model_dir, batch_size=args.batch_size)
    classifier_result, classifier_predictions = evaluate_classifier(
        classifier,
        texts,
        labels,
        class_names,
    )
    retrieval_result, retrieval_predictions = evaluate_retriever(
        dataset["train"],
        texts,
        labels,
        class_names,
        args.references_per_intent,
    )
    rows = [classifier_result, retrieval_result]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prediction_table = pd.DataFrame(
        {
            "text": texts,
            "actual_intent": [class_names[label] for label in labels],
            "fine_tuned_distilbert": [
                class_names[label] for label in classifier_predictions
            ],
            "pretrained_embedding_retrieval": [
                class_names[label] for label in retrieval_predictions
            ],
        }
    )
    workbook_path = args.output.with_suffix(".xlsx")
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as workbook:
        pd.DataFrame(rows).to_excel(workbook, sheet_name="Model Comparison", index=False)
        prediction_table.to_excel(workbook, sheet_name="Test Predictions", index=False)
    print(f"Wrote comparison results to {args.output}")
    print(f"Wrote Excel workbook to {workbook_path}")
    for row in rows:
        print(
            f"{row['pipeline']}: accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"seconds_per_sample={row['seconds_per_sample']:.6f}"
        )


if __name__ == "__main__":
    main()
