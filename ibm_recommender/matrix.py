"""Binary implicit-feedback matrices and deterministic user neighbors."""

from __future__ import annotations


def create_user_item_matrix(interactions):
    """Create an int8 user-by-article matrix with duplicate events collapsed."""
    import pandas as pd

    required = {"user_id", "article_id"}
    missing = sorted(required.difference(interactions.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if interactions[list(required)].isna().any().any():
        raise ValueError("user_id and article_id cannot be missing")
    matrix = pd.crosstab(interactions["user_id"], interactions["article_id"])
    matrix = matrix.gt(0).astype("int8")
    return matrix.sort_index().sort_index(axis=1)


def similar_users(user_id: int, user_item):
    """Rank neighbors by dot-product similarity, activity, then user ID."""
    import pandas as pd

    if user_id not in user_item.index:
        raise KeyError(f"Unknown user_id {user_id}")
    # int8 is compact for storage, but can overflow while summing hundreds of
    # shared articles. Promote before matrix multiplication.
    target = user_item.loc[user_id].to_numpy(dtype="int32")
    similarities = user_item.to_numpy(dtype="int32") @ target
    result = pd.DataFrame(
        {
            "neighbor_id": user_item.index.to_numpy(),
            "similarity": similarities,
            "num_interactions": user_item.sum(axis=1).to_numpy(),
        }
    )
    result = result.loc[result["neighbor_id"] != user_id]
    return result.sort_values(
        ["similarity", "num_interactions", "neighbor_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
