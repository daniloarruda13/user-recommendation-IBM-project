"""Loading and canonicalization for IBM article interactions."""

from __future__ import annotations

from pathlib import Path


def canonical_article_id(value) -> int:
    """Normalize integer-like IDs such as ``1429.0`` to Python integers."""
    import pandas as pd

    if pd.isna(value):
        raise ValueError("Article IDs cannot be missing")
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid article ID {value!r}") from exc
    if not numeric.is_integer():
        raise ValueError(f"Article ID {value!r} is not integer-like")
    return int(numeric)


def load_interactions(path: str | Path):
    """Load interactions with canonical IDs and normalized optional emails."""
    import pandas as pd

    frame = pd.read_csv(path)
    _require_columns(frame, {"article_id", "title", "email"})
    result = frame.loc[:, ["article_id", "title", "email"]].copy()
    result["article_id"] = result["article_id"].map(canonical_article_id)
    result["title"] = result["title"].astype("string").str.strip()
    if result["title"].isna().any() or result["title"].eq("").any():
        raise ValueError("Interaction titles cannot be missing or blank")
    result["email"] = result["email"].astype("string").str.strip().str.casefold()
    result.loc[result["email"].eq(""), "email"] = pd.NA
    return result


def load_articles(path: str | Path):
    """Load article metadata and deterministically remove duplicate IDs."""
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"article_id", "doc_full_name", "doc_description", "doc_body"}
    _require_columns(frame, required)
    result = frame.loc[:, sorted(required)].copy()
    result["article_id"] = result["article_id"].map(canonical_article_id)
    result = result.sort_index().drop_duplicates("article_id", keep="first")
    return result.reset_index(drop=True)


def encode_users(interactions, *, anonymous: str = "drop"):
    """Add stable user IDs, with explicit handling for anonymous interactions.

    ``anonymous='drop'`` excludes anonymous rows from personalization while
    leaving the original interaction frame available for popularity ranking.
    ``anonymous='single'`` assigns all anonymous rows user ID 0.
    """
    import pandas as pd

    _require_columns(interactions, {"article_id", "title", "email"})
    if anonymous not in {"drop", "single"}:
        raise ValueError("anonymous must be 'drop' or 'single'")
    result = interactions.copy()
    result["article_id"] = result["article_id"].map(canonical_article_id)
    result["email"] = result["email"].astype("string").str.strip().str.casefold()
    result.loc[result["email"].eq(""), "email"] = pd.NA
    known_emails = sorted(result.loc[result["email"].notna(), "email"].unique())
    mapping = {email: index for index, email in enumerate(known_emails, start=1)}
    result["user_id"] = result["email"].map(mapping).astype("Int64")
    if anonymous == "drop":
        result = result.loc[result["user_id"].notna()].copy()
    else:
        result["user_id"] = result["user_id"].fillna(0)
    result["user_id"] = result["user_id"].astype(int)
    return result.reset_index(drop=True)


def _require_columns(frame, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
