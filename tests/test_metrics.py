import math
import unittest

from movie_rec.metrics import (
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics,
    recall_at_k,
)


class MetricTests(unittest.TestCase):
    def test_precision_recall_hit_on_known_list(self):
        ranked = [1, 2, 3, 4, 5]
        relevant = {2, 5, 9}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 3), 1 / 3)
        self.assertEqual(hit_at_k(ranked, relevant, 3), 1.0)
        self.assertEqual(hit_at_k(ranked, relevant, 1), 0.0)
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 1), 0.0)

    def test_precision_uses_k_even_when_the_list_is_shorter(self):
        self.assertAlmostEqual(precision_at_k([7], {7}, 10), 0.1)

    def test_binary_ndcg_matches_hand_computed_dcg(self):
        ranked = [1, 2, 3]
        relevant = {1, 3}
        dcg = 1.0 / math.log2(2) + 0.0 / math.log2(3)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        self.assertAlmostEqual(dcg_at_k([1.0, 0.0], 2), dcg)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 2), dcg / idcg)

    def test_ndcg_is_one_for_perfect_prefix(self):
        self.assertEqual(ndcg_at_k([4, 5, 6], {4, 5}, 2), 1.0)

    def test_ndcg_is_zero_when_nothing_is_relevant(self):
        self.assertEqual(ndcg_at_k([1, 2, 3], set(), 3), 0.0)
        self.assertEqual(recall_at_k([1, 2, 3], set(), 3), 0.0)

    def test_idcg_truncates_to_k_when_more_items_are_relevant(self):
        ranked = [9, 8, 7]
        relevant = {9, 1, 2, 3}
        dcg = 1.0 / math.log2(2)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevant, 2), dcg / idcg)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            precision_at_k([1], {1}, 0)
        with self.assertRaises(ValueError):
            ranking_metrics([1], {1}, -1)

    def test_ranking_metrics_bundle(self):
        result = ranking_metrics([10, 20, 30], {20}, 2)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertEqual(result["hit_rate"], 1.0)
        self.assertGreater(result["ndcg"], 0.0)


if __name__ == "__main__":
    unittest.main()
