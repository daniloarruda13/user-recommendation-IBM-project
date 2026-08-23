import importlib.util
import unittest


HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))


@unittest.skipUnless(HAS_DEPS, "requires numpy and pandas")
class DataAndRankingTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd

        self.raw = pd.DataFrame(
            {
                "article_id": ["2.0", "1", 2.0, "3.0", "1.0", "4.0"],
                "title": ["Two", "One", "Two", "Three", "One", "Four"],
                "email": [
                    "B@EXAMPLE.COM",
                    "a@example.com",
                    "b@example.com",
                    None,
                    "a@example.com",
                    "",
                ],
            }
        )

    def test_canonical_article_ids(self):
        from ibm_recommender.data import canonical_article_id

        self.assertEqual(canonical_article_id("1429.0"), 1429)
        self.assertEqual(canonical_article_id(7), 7)
        with self.assertRaises(ValueError):
            canonical_article_id("7.5")

    def test_stable_user_encoding_and_anonymous_policy(self):
        from ibm_recommender.data import encode_users

        dropped = encode_users(self.raw, anonymous="drop")
        self.assertEqual(dropped["user_id"].tolist(), [2, 1, 2, 1])
        single = encode_users(self.raw, anonymous="single")
        self.assertEqual(single.loc[single["email"].isna(), "user_id"].tolist(), [0, 0])

    def test_invalid_anonymous_policy_is_rejected(self):
        from ibm_recommender.data import encode_users

        with self.assertRaises(ValueError):
            encode_users(self.raw, anonymous="guess")

    def test_popularity_ranking_is_deterministic(self):
        from ibm_recommender.ranking import article_names, top_articles

        ranking = top_articles(self.raw, 4)
        self.assertEqual(ranking["article_id"].tolist(), [1, 2, 3, 4])
        self.assertEqual(ranking["views"].tolist(), [2, 2, 1, 1])
        self.assertEqual(article_names([2, 1], self.raw), ["Two", "One"])

    def test_popularity_exclusion_and_n_validation(self):
        from ibm_recommender.ranking import top_articles

        self.assertEqual(top_articles(self.raw, 2, exclude={1})["article_id"].tolist(), [2, 3])
        with self.assertRaises(ValueError):
            top_articles(self.raw, 0)


if __name__ == "__main__":
    unittest.main()
