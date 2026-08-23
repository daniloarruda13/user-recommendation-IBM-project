"""Ranking metrics for implicit-feedback recommendation evaluation."""

from __future__ import annotations


def precision_at_k(recommended, relevant, k: int) -> float:
    recommendations, relevant_items = _prepare(recommended, relevant, k)
    return len(set(recommendations).intersection(relevant_items)) / k


def recall_at_k(recommended, relevant, k: int) -> float:
    recommendations, relevant_items = _prepare(recommended, relevant, k)
    if not relevant_items:
        raise ValueError("relevant must contain at least one item for recall")
    return len(set(recommendations).intersection(relevant_items)) / len(relevant_items)


def hit_rate_at_k(recommended, relevant, k: int) -> float:
    recommendations, relevant_items = _prepare(recommended, relevant, k)
    return float(bool(set(recommendations).intersection(relevant_items)))


def _prepare(recommended, relevant, k: int):
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    recommendations = list(recommended)
    if len(recommendations) != len(set(recommendations)):
        raise ValueError("recommended items must be unique")
    return recommendations[:k], set(relevant)
