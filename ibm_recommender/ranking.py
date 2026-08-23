"""Popularity ranking and article-name lookup."""

from __future__ import annotations


def top_articles(interactions, n: int = 10, *, exclude=()):
    """Return a deterministic popularity table sorted by views then ID."""
    import pandas as pd

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    required = {"article_id", "title"}
    missing = sorted(required.difference(interactions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    from .data import canonical_article_id

    excluded = {canonical_article_id(value) for value in exclude}
    working = interactions.loc[:, ["article_id", "title"]].copy()
    working["article_id"] = working["article_id"].map(canonical_article_id)
    counts = working.groupby("article_id", sort=False).size().rename("views")
    title_counts = (
        working.groupby(["article_id", "title"], sort=False)
        .size()
        .rename("title_views")
        .reset_index()
        .sort_values(
            ["article_id", "title_views", "title"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("article_id", keep="first")
        .set_index("article_id")["title"]
    )
    ranking = pd.concat([counts, title_counts], axis=1).reset_index()
    ranking = ranking.loc[~ranking["article_id"].isin(excluded)]
    ranking = ranking.sort_values(
        ["views", "article_id"], ascending=[False, True], kind="stable"
    )
    return ranking.head(n).reset_index(drop=True)


def article_names(article_ids, interactions) -> list[str]:
    """Return titles in requested ID order, rejecting missing IDs."""
    ranking = top_articles(interactions, interactions["article_id"].nunique())
    lookup = ranking.set_index("article_id")["title"].to_dict()
    missing = [article_id for article_id in article_ids if int(article_id) not in lookup]
    if missing:
        raise KeyError(f"Unknown article IDs: {missing}")
    return [str(lookup[int(article_id)]) for article_id in article_ids]
