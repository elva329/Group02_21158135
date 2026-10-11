"""Fine-tune DistilBERT on the official BANKING77 training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from banking77_core import DEFAULT_MODEL_DIR, MAX_LENGTH, load_banking77

BASE_MODEL_ID = "distilbert/distilbert-base-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=BASE_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    dataset, class_names = load_banking77()
    split = dataset["train"].train_test_split(
        test_size=0.1,
        seed=args.seed,
        stratify_by_column="label",
    )

    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokens = tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)
        tokens["labels"] = batch["label"]
        return tokens

    remove_columns = dataset["train"].column_names
    train_data = split["train"].map(tokenize_batch, batched=True, remove_columns=remove_columns)
    validation_data = split["test"].map(
        tokenize_batch,
        batched=True,
        remove_columns=remove_columns,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    validation_metrics = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / "label_names.json").open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)
    print(f"Saved fine-tuned model to {args.output_dir}")
    print(f"Validation metrics: {validation_metrics}")


if __name__ == "__main__":
    main()
