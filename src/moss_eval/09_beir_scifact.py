"""
Suite 9: BEIR SciFact (5,183 docs, 300 test queries, official qrels) on Moss.

Compares moss-minilm, moss-mediumlm (in-process SessionIndex) and an external embedding
(BAAI/bge-small-en-v1.5 via fastembed, plugged in as Moss custom vectors) across an
alpha sweep (1.0 = dense ... 0.0 = Moss keyword/BM25), reporting nDCG@10, Recall@10/100 and
MRR@10 with 2,000-sample bootstrap 95% CIs plus paired-bootstrap deltas.
Sessions are saved to data/sessions for reuse by later suites (no re-embedding).
"""
import asyncio
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
from moss import DocumentInfo, MossClient, QueryOptions

from moss_eval.common import ROOT, load_credentials, save_log, summarize_latencies, percentile
from moss_evals import metrics

DATA = ROOT / "data" / "scifact"
SESS_DIR = ROOT / "data" / "sessions"
RUNS_DIR = ROOT / "runs"
ALPHAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


def load_scifact():
    corpus = pq.read_table(DATA / "corpus.parquet").to_pylist()
    queries = {str(q["_id"]): q["text"] for q in pq.read_table(DATA / "queries.parquet").to_pylist()}
    qrels: dict[str, dict[str, int]] = {}
    for line in (DATA / "test.tsv").read_text().splitlines()[1:]:
        qid, did, score = line.split("\t")
        qrels.setdefault(qid, {})[did] = int(score)
    test_queries = {qid: queries[qid] for qid in qrels}
    docs = [(str(d["_id"]), (d["title"] + ". " + d["text"]).strip()) for d in corpus]
    return docs, test_queries, qrels


async def eval_target(name, query_fn, queries, qrels, alphas):
    out = {}
    rankings = {}
    for alpha in alphas:
        ranked, lat = {}, []
        for qid, text in queries.items():
            t0 = time.perf_counter()
            res = await query_fn(text, alpha)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked[qid] = [d.id for d in res.docs]
        rankings[str(alpha)] = ranked
        per_q = {m: metrics.per_query(ranked, qrels, m, k) for m, k in [("ndcg", 10), ("recall", 10), ("recall", 100), ("mrr", 10)]}
        out[str(alpha)] = {
            "ndcg@10": metrics.bootstrap_ci(list(metrics.per_query(ranked, qrels, "ndcg", 10).values())),
            "recall@10": metrics.bootstrap_ci(list(metrics.per_query(ranked, qrels, "recall", 10).values())),
            "recall@100": metrics.bootstrap_ci(list(metrics.per_query(ranked, qrels, "recall", 100).values())),
            "mrr@10": metrics.bootstrap_ci(list(metrics.per_query(ranked, qrels, "mrr", 10).values())),
            "latency_ms": {**summarize_latencies(lat), "p99": round(percentile(lat, 99), 2)},
        }
        print(f"  {name:22s} alpha={alpha:<4} nDCG@10={out[str(alpha)]['ndcg@10']['mean']:.4f} "
              f"[{out[str(alpha)]['ndcg@10']['lo']:.4f},{out[str(alpha)]['ndcg@10']['hi']:.4f}] "
              f"R@100={out[str(alpha)]['recall@100']['mean']:.4f} p50={out[str(alpha)]['latency_ms']['p50']}ms", flush=True)
    return out, rankings


async def main():
    pid, pkey = load_credentials()
    docs, queries, qrels = load_scifact()
    print(f"SciFact: {len(docs)} docs, {len(queries)} test queries, {sum(len(v) for v in qrels.values())} qrels", flush=True)
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    client = MossClient(pid, pkey)
    report = {"dataset": {"name": "BEIR/scifact", "docs": len(docs), "queries": len(queries)}, "targets": {}}
    all_rankings = {}

    for model_id in ["moss-minilm", "moss-mediumlm"]:
        name = f"scifact-{model_id}"
        t0 = time.perf_counter()
        sess = await client.session(name, model_id=model_id, auto_load_on_init=False)
        for i in range(0, len(docs), 256):
            await sess.add_docs([DocumentInfo(id=d, text=t) for d, t in docs[i:i + 256]])
        build_ms = (time.perf_counter() - t0) * 1000
        print(f"{model_id}: built {len(docs)} docs in {build_ms/1000:.1f}s", flush=True)

        async def qf(text, alpha, s=sess):
            return await s.query(text, QueryOptions(top_k=100, alpha=alpha))

        res, rk = await eval_target(model_id, qf, queries, qrels, ALPHAS)
        t_s = time.perf_counter()
        await sess.save_to_disk(str(SESS_DIR))
        save_ms = (time.perf_counter() - t_s) * 1000
        await sess.close()
        report["targets"][model_id] = {"build_ms": round(build_ms, 1), "docs_per_s": round(len(docs) / (build_ms / 1000), 1),
                                       "save_to_disk_ms": round(save_ms, 1), "alphas": res}
        all_rankings[model_id] = rk

    # External embedding plugged into Moss as custom vectors
    from fastembed import TextEmbedding
    ext_name = "BAAI/bge-small-en-v1.5"
    emb = TextEmbedding(model_name=ext_name)
    t0 = time.perf_counter()
    doc_vecs = [v.tolist() for v in emb.embed([t for _, t in docs], batch_size=64)]
    embed_ms = (time.perf_counter() - t0) * 1000
    qvecs = {qid: v.tolist() for qid, v in zip(queries, emb.query_embed(list(queries.values())))}
    t1 = time.perf_counter()
    sess = await client.session("scifact-bge-small-custom", model_id="custom", auto_load_on_init=False)
    for i in range(0, len(docs), 512):
        await sess.add_docs([DocumentInfo(id=d, text=t, embedding=doc_vecs[i + j]) for j, (d, t) in enumerate(docs[i:i + 512])])
    ingest_ms = (time.perf_counter() - t1) * 1000
    text_to_qid = {v: k for k, v in queries.items()}

    async def qf_custom(text, alpha, s=sess):
        return await s.query(text, QueryOptions(top_k=100, alpha=alpha, embedding=qvecs[text_to_qid[text]]))

    res, rk = await eval_target("bge-small (custom)", qf_custom, queries, qrels, ALPHAS)
    await sess.save_to_disk(str(SESS_DIR))
    await sess.close()
    report["targets"]["custom:bge-small-en-v1.5"] = {"external_embed_ms": round(embed_ms, 1), "moss_ingest_ms": round(ingest_ms, 1), "alphas": res}
    all_rankings["custom:bge-small-en-v1.5"] = rk
    await client.close()

    # Paired comparisons on nDCG@10
    def pq_ndcg(model, alpha):
        return [metrics.ndcg_at_k(all_rankings[model][str(alpha)].get(q, []), qrels[q], 10) for q in qrels]

    comps = {}
    best = {m: max(ALPHAS, key=lambda a: report["targets"][m]["alphas"][str(a)]["ndcg@10"]["mean"]) for m in all_rankings}
    comps["mediumlm_vs_minilm_dense"] = metrics.paired_bootstrap(pq_ndcg("moss-minilm", 1.0), pq_ndcg("moss-mediumlm", 1.0))
    comps["minilm_best_hybrid_vs_dense"] = metrics.paired_bootstrap(pq_ndcg("moss-minilm", 1.0), pq_ndcg("moss-minilm", best["moss-minilm"]))
    comps["minilm_dense_vs_bm25"] = metrics.paired_bootstrap(pq_ndcg("moss-minilm", 0.0), pq_ndcg("moss-minilm", 1.0))
    comps["mediumlm_best_hybrid_vs_dense"] = metrics.paired_bootstrap(pq_ndcg("moss-mediumlm", 1.0), pq_ndcg("moss-mediumlm", best["moss-mediumlm"]))
    comps["bge_custom_dense_vs_minilm_dense"] = metrics.paired_bootstrap(pq_ndcg("moss-minilm", 1.0), pq_ndcg("custom:bge-small-en-v1.5", 1.0))
    comps["bge_custom_best_vs_minilm_best"] = metrics.paired_bootstrap(pq_ndcg("moss-minilm", best["moss-minilm"]), pq_ndcg("custom:bge-small-en-v1.5", best["custom:bge-small-en-v1.5"]))
    report["best_alpha"] = best
    report["paired_ndcg10"] = comps
    (RUNS_DIR / "09_scifact_rankings.json").write_text(json.dumps(all_rankings))
    print("best alphas:", best)
    for k, v in comps.items():
        print(f"  {k}: delta={v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p={v['p_value']}")
    print("Saved:", save_log("09_beir_scifact.json", report, [pkey]))


if __name__ == "__main__":
    asyncio.run(main())
