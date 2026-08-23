import unittest

from ibm_recommender.metrics import hit_rate_at_k, precision_at_k, recall_at_k


class MetricTests(unittest.TestCase):
    def test_ranking_metrics(self):
        recommended = [10, 20, 30, 40]
        relevant = {20, 40, 50}
        self.assertEqual(precision_at_k(recommended, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(recommended, relevant, 4), 2 / 3)
        self.assertEqual(hit_rate_at_k(recommended, relevant, 1), 0.0)
        self.assertEqual(hit_rate_at_k(recommended, relevant, 2), 1.0)

    def test_invalid_k_and_duplicate_recommendations(self):
        with self.assertRaises(ValueError):
            precision_at_k([1], {1}, 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            precision_at_k([1, 1], {1}, 2)

    def test_recall_requires_relevant_items(self):
        with self.assertRaises(ValueError):
            recall_at_k([1], set(), 1)


if __name__ == "__main__":
    unittest.main()
