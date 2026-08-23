"""Retrieval and latency metrics."""
import statistics


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Fraction of gold documents present in the top-k retrieved.
    
    Args:
        - retrieved_ids: List of document IDs retrieved by the system, ordered by relevance.
        - gold_ids: List of document IDs that are considered relevant (ground truth).
        - k: The number of top retrieved documents to consider.
    Returns:
        - Recall at k: The fraction of gold documents that are present in the top-k retrieved documents.
    """
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / len(set(gold_ids))


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    1/rank of the first gold hit, 0.0 if none retrieved.
    
    Args:
        - retrieved_ids: List of document IDs retrieved by the system, ordered by relevance.
        - gold_ids: List of document IDs that are considered relevant (ground truth).
    Returns:
        - Reciprocal rank: The reciprocal of the rank of the first relevant document in the retrieved list.
    """
    gold = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def mrr(rr_scores: list[float]) -> float:
    """
    Mean Reciprocal Rank (MRR) is the average of the reciprocal ranks of results for a set of queries.

    Args:
        - rr_scores: List of reciprocal rank scores for individual queries.
    Returns:
        - MRR: The mean of the reciprocal rank scores.
    """
    return statistics.fmean(rr_scores) if rr_scores else 0.0


def latency_stats(seconds: list[float]) -> dict[str, float]:
    """
    Compute latency statistics: p50, p95, mean.

    Args:
        - seconds: List of latency measurements in seconds.
    Returns:
        - Dictionary containing p50, p95, and mean latency.
    """
    s = sorted(seconds)
    return {"p50": statistics.median(s),
            "p95": s[min(int(0.95 * len(s)), len(s) - 1)],
            "mean": statistics.fmean(s)}
