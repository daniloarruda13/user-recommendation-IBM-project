import importlib.util
import unittest


HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))


@unittest.skipUnless(HAS_DEPS, "requires numpy and pandas")
class CollaborativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd

        rows = []
        titles = {article: f"Article {article}" for article in range(1, 8)}
        user_articles = {
            1: [1, 2],
            2: [1, 2, 3],
            3: [1, 2, 4],
            4: [1, 5, 6, 7],
        }
        for user_id, articles in user_articles.items():
            for article_id in articles:
                rows.append(
                    {
                        "user_id": user_id,
                        "article_id": article_id,
                        "title": titles[article_id],
                        "email": f"{user_id}@example.com",
                    }
                )
        rows.append(rows[0].copy())  # repeated event must remain binary in the matrix
        cls.interactions = pd.DataFrame(rows)

    def test_matrix_collapses_duplicate_events(self):
        from ibm_recommender.matrix import create_user_item_matrix

        matrix = create_user_item_matrix(self.interactions)
        self.assertEqual(matrix.shape, (4, 7))
        self.assertEqual(int(matrix.loc[1, 1]), 1)
        self.assertEqual(int(matrix.loc[1].sum()), 2)

    def test_neighbors_sort_similarity_before_activity(self):
        from ibm_recommender.matrix import create_user_item_matrix, similar_users

        matrix = create_user_item_matrix(self.interactions)
        neighbors = similar_users(1, matrix)
        self.assertEqual(neighbors["neighbor_id"].tolist(), [2, 3, 4])
        self.assertEqual(neighbors["similarity"].tolist(), [2, 2, 1])
        self.assertEqual(neighbors["num_interactions"].tolist(), [3, 3, 4])

    def test_similarity_sum_does_not_overflow_compact_matrix(self):
        import pandas as pd

        from ibm_recommender.matrix import similar_users

        matrix = pd.DataFrame(
            [[1] * 130, [1] * 130],
            index=[1, 2],
            columns=range(130),
            dtype="int8",
        )
        neighbors = similar_users(1, matrix)
        self.assertEqual(int(neighbors.loc[0, "similarity"]), 130)

    def test_known_user_recommendations_exclude_seen_and_duplicates(self):
        from ibm_recommender.recommend import recommend_article_ids

        recommendations = recommend_article_ids(1, self.interactions, 4)
        self.assertEqual(recommendations[:2], [3, 4])
        self.assertFalse({1, 2}.intersection(recommendations))
        self.assertEqual(len(recommendations), len(set(recommendations)))

    def test_unknown_user_receives_popularity_fallback(self):
        from ibm_recommender.recommend import recommend_article_ids
        from ibm_recommender.ranking import top_articles

        expected = top_articles(self.interactions, 3)["article_id"].tolist()
        self.assertEqual(recommend_article_ids(999, self.interactions, 3), expected)

    def test_unknown_neighbor_user_is_reported(self):
        from ibm_recommender.matrix import create_user_item_matrix, similar_users

        with self.assertRaises(KeyError):
            similar_users(999, create_user_item_matrix(self.interactions))


if __name__ == "__main__":
    unittest.main()
