"""IR metrics and bootstrap statistics. Pure Python + numpy, no network."""
from __future__ import annotations

import math
from typing import Callable, Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, int]]  # query_id -> {doc_id: graded relevance}
Run = Mapping[str, Sequence[str]]  # query_id -> ranked doc ids


def recall_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    relevant = {d for d, g in rel.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def hit_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    return 1.0 if any(rel.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def mrr_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Mapping[str, int], k: int) -> float:
    """Standard (trec_eval-style) nDCG with linear gain, as used by BEIR."""
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


METRICS: dict[str, Callable[[Sequence[str], Mapping[str, int], int], float]] = {
    "recall": recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def per_query(run: Run, qrels: Qrels, metric: str, k: int) -> dict[str, float]:
    fn = METRICS[metric]
    return {qid: fn(run.get(qid, []), rel, k) for qid, rel in qrels.items()}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": round(float(arr.mean()), 4),
        "lo": round(float(np.quantile(means, alpha / 2)), 4),
        "hi": round(float(np.quantile(means, 1 - alpha / 2)), 4),
        "n": int(arr.size),
    }


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 13) -> dict[str, float]:
    """Mean difference (b - a) with 95% CI and a two-sided bootstrap p-value."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert xa.shape == xb.shape
    d = xb - xa
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "delta": round(float(d.mean()), 4),
        "lo": round(float(np.quantile(means, 0.025)), 4),
        "hi": round(float(np.quantile(means, 0.975)), 4),
        "p_value": round(float(min(1.0, p)), 4),
        "n": int(d.size),
    }


def evaluate(run: Run, qrels: Qrels, ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in ("recall", "mrr", "ndcg"):
        for k in ks:
            out[f"{metric}@{k}"] = bootstrap_ci(list(per_query(run, qrels, metric, k).values()))
    return out
