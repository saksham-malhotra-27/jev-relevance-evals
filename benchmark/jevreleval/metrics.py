"""Information-retrieval metrics for the benchmark.

Every function takes a ranked list of document ids and the set of gold
(relevant) ids and returns the metric value, so the exact same code scores the
BM25 baseline, the Jev-filtered ranking, and the LLM-judge output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jevreleval.constants import BENCHMARK_CANDIDATE_K


def recall_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Fraction of gold documents present among the top ``k`` ranked ids."""
    if not gold_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & gold_ids) / len(gold_ids)


def precision_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Fraction of the top ``k`` ranked ids that are relevant."""
    if k < 1 or not ranked_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & gold_ids) / min(k, len(ranked_ids))


def hit_rate_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """1.0 when at least one gold document appears in the top ``k``."""
    return 1.0 if any(doc_id in gold_ids for doc_id in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: set[str]) -> float:
    """1/rank of the first gold document; 0.0 when nothing relevant is ranked."""
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain (binary relevance) at ``k``."""
    if not gold_ids:
        return 0.0
    relevant_at = {
        position: 1.0 for position, doc_id in enumerate(ranked_ids[:k], start=1)
        if doc_id in gold_ids
    }
    dcg = sum(gain / math.log2(position + 1) for position, gain in relevant_at.items())
    ideal_count = min(k, len(gold_ids))
    ideal_dcg = sum(
        1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1)
    )
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def rank_metrics(
    ranked_ids: Sequence[str], gold_ids: set[str], *, k: int = BENCHMARK_CANDIDATE_K
) -> dict[str, float]:
    """Compute the standard metric family for one ranked list.

    Args:
        ranked_ids: Document ids in retrieval order.
        gold_ids: Ids of the documents that genuinely answer the query.
        k: Cutoff for the @k metrics (Recall@k, Precision@k, Hit-rate@k, nDCG@k).

    Returns:
        A dict with keys ``recall_at_k``, ``precision_at_k``, ``hit_rate_at_k``,
        ``mrr``, and ``ndcg_at_k``.
    """
    return {
        "recall_at_k": recall_at_k(ranked_ids, gold_ids, k),
        "precision_at_k": precision_at_k(ranked_ids, gold_ids, k),
        "hit_rate_at_k": hit_rate_at_k(ranked_ids, gold_ids, k),
        "mrr": reciprocal_rank(ranked_ids, gold_ids),
        "ndcg_at_k": ndcg_at_k(ranked_ids, gold_ids, k),
    }


def summarize(
    per_query: list[dict[str, float]]
) -> dict[str, float]:
    """Aggregate a list of per-query metric dicts into means and percentiles.

    Returns a dict of ``<metric>_mean`` and ``<metric>_p50``/``<metric>_p95``
    entries (p50/p95 only for keys present in at least one row).
    """
    if not per_query:
        return {}
    keys = sorted({key for row in per_query for key in row})
    summary: dict[str, float] = {}
    for key in keys:
        values = sorted(row[key] for row in per_query if key in row)
        if not values:
            continue
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_p50"] = _percentile(values, 0.50)
        summary[f"{key}_p95"] = _percentile(values, 0.95)
    return summary


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    index = math.ceil(fraction * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


__all__ = [
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "rank_metrics",
    "summarize",
]