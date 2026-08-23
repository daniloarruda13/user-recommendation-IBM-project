import importlib.util
import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
HAS_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))


@unittest.skipUnless(HAS_DEPS, "requires numpy and pandas")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ibm_recommender.data import encode_users, load_articles, load_interactions

        cls.interactions = load_interactions(ROOT / "data" / "user-item-interactions.csv")
        cls.encoded = encode_users(cls.interactions)
        cls.articles = load_articles(ROOT / "data" / "articles_community.csv")

    def test_committed_dataset_dimensions(self):
        self.assertEqual(len(self.interactions), 45993)
        self.assertEqual(self.interactions["article_id"].nunique(), 714)
        self.assertEqual(self.interactions["email"].isna().sum(), 17)
        self.assertEqual(self.encoded["user_id"].nunique(), 5148)
        self.assertEqual(len(self.encoded), 45976)
        self.assertEqual(len(self.articles), 1051)

    def test_real_popularity_baseline(self):
        from ibm_recommender.ranking import top_articles

        ranking = top_articles(self.interactions, 2)
        self.assertEqual(ranking["article_id"].tolist(), [1429, 1330])
        self.assertEqual(ranking["views"].tolist(), [937, 927])

    def test_real_user_item_matrix(self):
        from ibm_recommender.matrix import create_user_item_matrix

        matrix = create_user_item_matrix(self.encoded)
        self.assertEqual(matrix.shape, (5148, 714))
        self.assertTrue(matrix.isin([0, 1]).all().all())


class RepositoryTests(unittest.TestCase):
    def test_notebook_and_project_metadata(self):
        notebook_path = ROOT / "Recommendations_with_IBM.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], "ibm-article-recommender")

    def test_readme_documents_historical_artifacts(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").casefold()
        self.assertIn("historical", readme)
        self.assertIn("pickle", readme)
        self.assertIn("implicit", readme)


if __name__ == "__main__":
    unittest.main()
