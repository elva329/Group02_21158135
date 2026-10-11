"""Shared data loading and inference code for the BANKING77 project."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

DATASET_ID = "PolyAI/banking77"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 128


def model_weights_exist(model_dir: str | Path) -> bool:
    """Return whether a Hugging Face model directory contains weight files."""
    model_path = Path(model_dir)
    weight_files = (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    return (
        any((model_path / filename).is_file() for filename in weight_files)
        or any(model_path.glob("model-*.safetensors"))
        or any(model_path.glob("pytorch_model-*.bin"))
    )


def resolve_default_model_dir() -> Path:
    """Find the default fine-tuned model directory across common project layouts."""
    candidates = [
        Path(__file__).resolve().parent / "Fine-tuned_Model_files" / "banking77-distilbert",
        Path(__file__).resolve().parents[2] / "Group02_Dataset_files" / "Fine-tuned_Model_files" / "banking77-distilbert",
        Path(__file__).resolve().parents[1] / "Python_notebooks" / "Fine-tuned_Model_files" / "banking77-distilbert",
        Path.cwd() / "Group02_Dataset_files" / "Fine-tuned_Model_files" / "banking77-distilbert",
        Path.cwd() / "Fine-tuned_Model_files" / "banking77-distilbert",
    ]
    for candidate in candidates:
        if model_weights_exist(candidate):
            return candidate
    return candidates[0]


DEFAULT_MODEL_DIR = resolve_default_model_dir()
REFERENCE_EXAMPLES_PER_INTENT = 25


def _load_local_banking77() -> tuple[Any, list[str]] | None:
    """Attempt to load BANKING77 dataset from local CSV files in the workspace."""
    try:
        import pandas as pd
        from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

        candidates = [
            Path(__file__).resolve().parents[2] / "Group02_Dataset_files",
            Path(__file__).resolve().parents[1] / "Dataset_files",
            Path.cwd() / "Group02_Dataset_files",
            Path.cwd() / "Dataset_files",
        ]
        data_dir = next(
            (c for c in candidates if (c / "train.csv").is_file() and (c / "test.csv").is_file()),
            None,
        )
        if data_dir is None:
            return None

        train_df = pd.read_csv(data_dir / "train.csv")
        if (data_dir / "validation.csv").is_file():
            val_df = pd.read_csv(data_dir / "validation.csv")
            full_train_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            full_train_df = train_df
        test_df = pd.read_csv(data_dir / "test.csv")

        intent_map = dict(zip(full_train_df["label"], full_train_df["intent"]))
        class_names = [intent_map[i] for i in range(len(intent_map))]
        features = Features({"text": Value("string"), "label": ClassLabel(names=class_names)})

        dataset = DatasetDict({
            "train": Dataset.from_pandas(full_train_df[["text", "label"]], features=features),
            "test": Dataset.from_pandas(test_df[["text", "label"]], features=features),
        })
        return dataset, class_names
    except Exception:
        return None


def load_banking77() -> tuple[Any, list[str]]:
    """Load the official Hugging Face dataset and its intent names with local CSV fallback."""
    from datasets import ClassLabel, DatasetDict, load_dataset

    try:
        loaded_dataset = load_dataset(DATASET_ID, trust_remote_code=True)
        if isinstance(loaded_dataset, DatasetDict):
            label_feature = loaded_dataset["train"].features.get("label")
            if isinstance(label_feature, ClassLabel):
                class_names = list(label_feature.names)
                return loaded_dataset, class_names
    except Exception:
        local_result = _load_local_banking77()
        if local_result is not None:
            return local_result
        raise

    local_result = _load_local_banking77()
    if local_result is not None:
        return local_result
    raise TypeError(f"{DATASET_ID} did not load as a map-style DatasetDict")


def make_reference_examples(
    train_split: Any,
    examples_per_intent: int = REFERENCE_EXAMPLES_PER_INTENT,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Select a reproducible, balanced subset of training examples."""
    if examples_per_intent < 1:
        raise ValueError("examples_per_intent must be positive")

    by_label: dict[int, list[str]] = defaultdict(list)
    for row in train_split:
        by_label[int(row["label"])].append(str(row["text"]))

    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    for label in sorted(by_label):
        examples = by_label[label]
        rng.shuffle(examples)
        selected = examples[:examples_per_intent]
        texts.extend(selected)
        labels.extend([label] * len(selected))

    if not texts:
        raise ValueError("The training split contained no examples")
    return texts, labels


def normalize_classifier_output(outputs: Any, batch_size: int) -> list[list[dict[str, Any]]]:
    """Normalize Transformers outputs into app predictions across API shapes."""
    if batch_size == 1:
        if isinstance(outputs, dict):
            grouped_outputs = [[outputs]]
        elif outputs and isinstance(outputs[0], dict):
            grouped_outputs = [list(outputs)]
        elif outputs and isinstance(outputs[0], list):
            grouped_outputs = [list(outputs[0])]
        else:
            grouped_outputs = [[]]
    elif outputs and isinstance(outputs[0], dict):
        grouped_outputs = [[item] for item in outputs]
    else:
        grouped_outputs = [list(item) for item in outputs]

    return [
        [{"intent": str(item["label"]), "score": float(item["score"])} for item in row]
        for row in grouped_outputs
    ]


def rank_neighbor_intents(
    neighbors: Sequence[tuple[int, float]],
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Rank intents by top-neighbor votes, breaking ties by mean cosine similarity."""
    by_label: dict[int, list[float]] = defaultdict(list)
    for label_id, similarity in neighbors:
        if label_id < 0 or label_id >= len(class_names):
            raise ValueError(f"Unknown intent label id: {label_id}")
        by_label[label_id].append(float(similarity))

    ranked = sorted(
        by_label.items(),
        key=lambda item: (-len(item[1]), -sum(item[1]) / len(item[1]), class_names[item[0]]),
    )
    return [
        {
            "label_id": label_id,
            "intent": class_names[label_id],
            "score": sum(similarities) / len(similarities),
            "votes": len(similarities),
        }
        for label_id, similarities in ranked
    ]


class HuggingFaceIntentClassifier:
    """Adapter for a fine-tuned Hugging Face text-classification model."""

    def __init__(self, model_ref: str | Path, batch_size: int = 16) -> None:
        from transformers import pipeline

        self.batch_size = batch_size
        self.classifier = pipeline(
            "text-classification",
            model=str(model_ref),
            tokenizer=str(model_ref),
            device=_pipeline_device(),
        )

    def predict_many(self, texts: Sequence[str], top_k: int = 3) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        raw_outputs = self.classifier(
            list(texts),
            truncation=True,
            max_length=MAX_LENGTH,
            batch_size=self.batch_size,
            top_k=top_k,
        )
        return normalize_classifier_output(raw_outputs, len(texts))


class HuggingFaceEmbeddingRetriever:
    """Intent retrieval using a pretrained Hugging Face feature-extraction model."""

    def __init__(
        self,
        reference_texts: Sequence[str],
        reference_labels: Sequence[int],
        class_names: Sequence[str],
        model_id: str = EMBEDDING_MODEL_ID,
        batch_size: int = 32,
        neighbor_count: int = 5,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if len(reference_texts) != len(reference_labels):
            raise ValueError("Reference texts and labels must have the same length")
        if not reference_texts:
            raise ValueError("At least one reference example is required")
        if neighbor_count < 1:
            raise ValueError("neighbor_count must be positive")

        self.torch = torch
        self.class_names = list(class_names)
        self.batch_size = batch_size
        self.neighbor_count = neighbor_count
        self.device = _torch_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id).to(self.device)
        self.encoder.eval()
        from transformers import pipeline

        self.feature_extraction_pipeline = pipeline(
            "feature-extraction",
            model=self.encoder,
            tokenizer=self.tokenizer,
        )
        self.reference_labels = torch.tensor(reference_labels, dtype=torch.long)
        self.reference_vectors = self._encode(reference_texts)

    def _encode(self, texts: Sequence[str]) -> Any:
        torch = self.torch
        encoded_batches = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            with torch.inference_mode():
                hidden_states = self.encoder(**tokens).last_hidden_state
            attention_mask = tokens["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            encoded_batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu())
        return torch.cat(encoded_batches, dim=0)

    def predict_many(self, texts: Sequence[str], top_k: int = 3) -> list[list[dict[str, Any]]]:
        if not texts:
            return []
        torch = self.torch
        query_vectors = self._encode(texts)
        similarities = query_vectors @ self.reference_vectors.T
        neighbor_count = min(self.neighbor_count, self.reference_vectors.shape[0])
        scores, indices = torch.topk(similarities, k=neighbor_count, dim=1)

        predictions: list[list[dict[str, Any]]] = []
        for row_scores, row_indices in zip(scores, indices):
            neighbors = [
                (int(self.reference_labels[index]), float(score))
                for score, index in zip(row_scores.tolist(), row_indices.tolist())
            ]
            predictions.append(rank_neighbor_intents(neighbors, self.class_names)[:top_k])
        return predictions


def _torch_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pipeline_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return -1
