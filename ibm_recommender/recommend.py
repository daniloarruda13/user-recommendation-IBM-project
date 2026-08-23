"""Collaborative recommendations with deterministic cold-start fallback."""

from __future__ import annotations

from .matrix import create_user_item_matrix, similar_users
from .ranking import top_articles


def recommend_article_ids(user_id: int, interactions, n: int = 10, *, user_item=None) -> list[int]:
    """Recommend unseen articles using weighted neighbors and popularity ties."""
    import pandas as pd

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    matrix = create_user_item_matrix(interactions) if user_item is None else user_item
    if user_id not in matrix.index:
        return top_articles(interactions, n)["article_id"].astype(int).tolist()

    seen = set(matrix.columns[matrix.loc[user_id].gt(0)].astype(int))
    neighbors = similar_users(user_id, matrix)
    neighbors = neighbors.loc[neighbors["similarity"].gt(0)]
    if neighbors.empty:
        return top_articles(interactions, n, exclude=seen)["article_id"].astype(int).tolist()

    neighbor_matrix = matrix.loc[neighbors["neighbor_id"]]
    weighted_scores = neighbor_matrix.T.dot(neighbors.set_index("neighbor_id")["similarity"])
    candidates = pd.DataFrame(
        {"article_id": weighted_scores.index.astype(int), "score": weighted_scores.to_numpy()}
    )
    candidates = candidates.loc[candidates["score"].gt(0) & ~candidates["article_id"].isin(seen)]
    popularity = interactions.groupby("article_id").size().rename("views")
    candidates = candidates.join(popularity, on="article_id").fillna({"views": 0})
    candidates = candidates.sort_values(
        ["score", "views", "article_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    recommendations = candidates["article_id"].astype(int).head(n).tolist()
    if len(recommendations) < n:
        excluded = seen | set(recommendations)
        fallback = top_articles(interactions, n - len(recommendations), exclude=excluded)
        recommendations.extend(fallback["article_id"].astype(int).tolist())
    return recommendations
