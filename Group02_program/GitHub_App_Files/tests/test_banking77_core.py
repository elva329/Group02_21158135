import unittest

from banking77_core import normalize_classifier_output, rank_neighbor_intents


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


if __name__ == "__main__":
    unittest.main()
