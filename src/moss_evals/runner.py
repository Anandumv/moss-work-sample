"""Run retrieval against a Moss cloud index (loaded locally) or a SessionIndex built from a corpus."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from moss import DocumentInfo, MossClient, QueryOptions

from . import metrics

QueryTransform = Callable[[str], Awaitable[str]]


@dataclass
class RetrievalConfig:
    alpha: float = 0.6
    top_k: int = 10
    candidate_depth: Optional[int] = None
    min_score: Optional[float] = None
    label: str = "default"


@dataclass
class RetrievalRun:
    config: RetrievalConfig
    ranked: dict[str, list[str]] = field(default_factory=dict)
    texts: dict[str, list[str]] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    transformed_queries: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "ranked": self.ranked,
            "scores": self.scores,
            "latency_ms": self.latency_ms,
            "transformed_queries": self.transformed_queries,
        }


class Target:
    """Queryable Moss index or session. Build with `from_index` or `from_corpus`."""

    def __init__(self, query_fn: Callable[[str, QueryOptions], Awaitable[Any]], closer: Callable[[], Awaitable[None]], info: dict[str, Any]):
        self._query = query_fn
        self._close = closer
        self.info = info

    @classmethod
    async def from_index(cls, client: MossClient, index_name: str) -> "Target":
        t0 = time.perf_counter()
        await client.load_index(index_name)
        load_ms = (time.perf_counter() - t0) * 1000

        async def q(text: str, opts: QueryOptions):
            return await client.query(index_name, text, opts)

        async def close():
            await client.unload_index(index_name)

        return cls(q, close, {"kind": "cloud_index", "index": index_name, "load_ms": round(load_ms, 2)})

    @classmethod
    async def from_corpus(
        cls,
        client: MossClient,
        name: str,
        docs: Sequence[DocumentInfo],
        model_id: str = "moss-minilm",
        batch: int = 256,
        query_embedder: Optional[Callable[[str], list[float]]] = None,
    ) -> "Target":
        t0 = time.perf_counter()
        sess = await client.session(name, model_id=model_id, auto_load_on_init=False)
        for i in range(0, len(docs), batch):
            await sess.add_docs(list(docs[i : i + batch]))
        build_ms = (time.perf_counter() - t0) * 1000

        async def q(text: str, opts: QueryOptions):
            if query_embedder is not None:
                opts.embedding = query_embedder(text)
            return await sess.query(text, opts)

        async def close():
            await sess.close()

        return cls(q, close, {"kind": "session", "model_id": model_id, "docs": len(docs), "build_ms": round(build_ms, 2)})

    async def close(self) -> None:
        await self._close()

    async def run(
        self,
        queries: dict[str, str],
        cfg: RetrievalConfig,
        transform: Optional[QueryTransform] = None,
    ) -> RetrievalRun:
        out = RetrievalRun(config=cfg)
        for qid, text in queries.items():
            if transform is not None:
                text2 = await transform(text)
                out.transformed_queries[qid] = text2
                text = text2
            opts = QueryOptions(top_k=cfg.top_k, alpha=cfg.alpha, candidate_depth=cfg.candidate_depth, min_score=cfg.min_score)
            t0 = time.perf_counter()
            res = await self._query(text, opts)
            out.latency_ms[qid] = round((time.perf_counter() - t0) * 1000, 3)
            out.ranked[qid] = [d.id for d in res.docs]
            out.texts[qid] = [d.text for d in res.docs]
            out.scores[qid] = [round(float(d.score), 4) for d in res.docs]
        return out


def score_run(run: RetrievalRun, qrels: metrics.Qrels, ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, Any]:
    from moss_eval.common import summarize_latencies  # reuse latency summariser

    lat = list(run.latency_ms.values())
    return {
        "label": run.config.label,
        "metrics": metrics.evaluate(run.ranked, qrels, ks),
        "latency_ms": {**summarize_latencies(lat), "p99": round(float(sorted(lat)[int(0.99 * (len(lat) - 1))]), 3) if lat else 0.0},
    }
