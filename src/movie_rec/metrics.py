"""Ranking metrics for the shared evaluation protocol.

Formulas match the project evaluation plan:

* Precision@K = (# relevant in top K) / K
* Recall@K = (# relevant in top K) / (# relevant in the test set)
* NDCG@K = DCG@K / IDCG@K with binary gains and log2(rank + 1)
* Hit Rate@K = 1 if the top K contains at least one relevant item, else 0

Queries are macro-averaged by the evaluation runner. The seed movie is never a
relevant item: recommenders always exclude it, matching Member 1.
"""

from __future__ import annotations

import math


def _validate_k(k):
    if k < 1:
        raise ValueError("k must be positive.")
    return k


def _hits(ranked_ids, relevant, k):
    relevant = set(relevant)
    return sum(1 for movie_id in ranked_ids[:k] if movie_id in relevant), relevant


def precision_at_k(ranked_ids, relevant, k):
    k = _validate_k(k)
    hits, _ = _hits(ranked_ids, relevant, k)
    return hits / k


def recall_at_k(ranked_ids, relevant, k):
    k = _validate_k(k)
    hits, relevant = _hits(ranked_ids, relevant, k)
    if not relevant:
        return 0.0
    return hits / len(relevant)


def dcg_at_k(gains, k):
    k = _validate_k(k)
    total = 0.0
    for rank, gain in enumerate(gains[:k], start=1):
        total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids, relevant, k):
    k = _validate_k(k)
    relevant = set(relevant)
    gains = [1.0 if movie_id in relevant else 0.0 for movie_id in ranked_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg_at_k(ideal, k) if ideal else 0.0
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_at_k(ranked_ids, relevant, k):
    k = _validate_k(k)
    hits, _ = _hits(ranked_ids, relevant, k)
    return 1.0 if hits else 0.0


def ranking_metrics(ranked_ids, relevant, k):
    """Return the four metrics for one evaluation query."""
    return {
        "precision": precision_at_k(ranked_ids, relevant, k),
        "recall": recall_at_k(ranked_ids, relevant, k),
        "ndcg": ndcg_at_k(ranked_ids, relevant, k),
        "hit_rate": hit_at_k(ranked_ids, relevant, k),
    }
