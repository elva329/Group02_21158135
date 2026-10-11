import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from banking77_core import (
    DEFAULT_MODEL_DIR,
    make_reference_examples,
    model_weights_exist,
    normalize_classifier_output,
    rank_neighbor_intents,
    resolve_default_model_dir,
)


class ClassifierOutputTests(unittest.TestCase):
    def test_normalizes_single_item_top_k_output(self) -> None:
        output = [
            {"label": "card_arrival", "score": 0.8},
            {"label": "card_delivery_estimate", "score": 0.2},
        ]
        self.assertEqual(
            normalize_classifier_output(output, 1),
            [[
                {"intent": "card_arrival", "score": 0.8},
                {"intent": "card_delivery_estimate", "score": 0.2},
            ]],
        )

    def test_normalizes_batch_single_prediction_output(self) -> None:
        output = [
            {"label": "card_arrival", "score": 0.8},
            {"label": "cash_withdrawal", "score": 0.7},
        ]
        self.assertEqual(
            normalize_classifier_output(output, 2),
            [
                [{"intent": "card_arrival", "score": 0.8}],
                [{"intent": "cash_withdrawal", "score": 0.7}],
            ],
        )


class RetrievalRankingTests(unittest.TestCase):
    def test_neighbor_vote_count_ranks_before_similarity(self) -> None:
        names = ["card_arrival", "card_payment"]
        ranked = rank_neighbor_intents(
            [(0, 0.55), (1, 0.99), (0, 0.60)],
            names,
        )
        self.assertEqual(ranked[0]["intent"], "card_arrival")
        self.assertEqual(ranked[0]["votes"], 2)

    def test_unknown_intent_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_neighbor_intents([(2, 0.5)], ["card_arrival"])


class ModelResolutionTests(unittest.TestCase):
    def test_default_model_dir_resolves_valid_path(self) -> None:
        resolved = resolve_default_model_dir()
        self.assertIsInstance(resolved, Path)
        self.assertTrue(resolved.is_dir(), f"Resolved path does not exist: {resolved}")

    def test_reference_examples_validation(self) -> None:
        with self.assertRaises(ValueError):
            make_reference_examples([], examples_per_intent=0)

    def test_config_without_weights_is_not_a_model_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "config.json").touch()
            self.assertFalse(model_weights_exist(model_dir))

    def test_model_weights_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "model.safetensors").touch()
            self.assertTrue(model_weights_exist(model_dir))


if __name__ == "__main__":
    unittest.main()
