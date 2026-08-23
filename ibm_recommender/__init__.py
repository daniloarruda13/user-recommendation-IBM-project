"""Deterministic recommendation helpers for the IBM article dataset."""

from .data import encode_users, load_articles, load_interactions
from .matrix import create_user_item_matrix, similar_users
from .metrics import hit_rate_at_k, precision_at_k, recall_at_k
from .recommend import recommend_article_ids
from .ranking import article_names, top_articles

__all__ = [
    "article_names",
    "create_user_item_matrix",
    "encode_users",
    "hit_rate_at_k",
    "load_articles",
    "load_interactions",
    "precision_at_k",
    "recall_at_k",
    "recommend_article_ids",
    "similar_users",
    "top_articles",
]
