"""
Suite 10: Multilingual retrieval gap and two fixes, measured.

Dataset: first 100 SciFact test queries translated to Hindi, Spanish and Malayalam by
anthropic/claude-haiku-5.5 (cached in data/multilingual/), retrieved against the English corpus.
Conditions (nDCG@10 with bootstrap CI, paired deltas vs native-query baseline):
  en_upper          : original English query, moss-minilm dense
  native_minilm     : translated query, moss-minilm dense           (the gap)
  native_hybrid     : translated query, moss-minilm alpha=0.6
  fix_translate     : moss/fastlane-1 translates query -> English, then moss-minilm dense
  fix_multilingual  : paraphrase-multilingual-MiniLM-L12-v2 vectors via Moss custom embeddings
Also re-runs the 15 multilingual queries of the 120-doc bench (Suite 6) with fix_translate.
"""
import asyncio
import importlib
import json
import time
from pathlib import Path

import httpx
from moss import DocumentInfo, MossClient, QueryOptions

from moss_eval.common import ROOT, estimate_inference_cost, load_credentials, percentile, save_log
from moss_evals import metrics

_m09 = importlib.import_module("moss_eval.09_beir_scifact")
_m06 = importlib.import_module("moss_eval.06_deep_retrieval_eval")
SESS_DIR = ROOT / "data" / "sessions"
ML_DIR = ROOT / "data" / "multilingual"
LANGS = {"hi": "Hindi", "es": "Spanish", "ml": "Malayalam"}
BASE = "https://inference.moss.dev/v1"


class LLM:
    def __init__(self, auth):
        self.auth = auth
        self.h = httpx.AsyncClient(timeout=60)
        self.cost = 0.0
        self.lat = []

    async def call(self, model, instructions, text, max_tok=120):
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                r = await self.h.post(f"{BASE}/responses", headers={"Authorization": self.auth, "Content-Type": "application/json", "X-Moss-Cache": "off"},
                                  json={"model": model, "instructions": instructions, "input": text, "max_output_tokens": max_tok})
            except (httpx.TransportError, httpx.TimeoutException):
                self.transport_retries = getattr(self, "transport_retries", 0) + 1
                await asyncio.sleep(2 ** attempt)
                continue
            ms = (time.perf_counter() - t0) * 1000
            if r.status_code in (429, 502, 503):
                await asyncio.sleep(float(r.headers.get("retry-after", 2)))
                continue
            r.raise_for_status()
            d = r.json()
            u = d.get("usage") or {}
            self.cost += estimate_inference_cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0))
            out = "".join(c.get("text", "") for it in d.get("output", []) if it.get("type") == "message" for c in it.get("content", []) if c.get("type") == "output_text").strip()
            return out, ms
        raise RuntimeError("rate limited")


async def build_dataset(llm, queries):
    ML_DIR.mkdir(parents=True, exist_ok=True)
    path = ML_DIR / "scifact_test100_hi_es_ml.jsonl"
    done = {json.loads(l)["qid"]: json.loads(l) for l in path.read_text().splitlines()} if path.exists() else {}
    with path.open("a") as f:  # append + resume: a dropped connection must not lose finished rows
        for qid, q in list(queries.items())[:100]:
            if qid in done:
                continue
            row = {"qid": qid, "en": q}
            for code, name in LANGS.items():
                row[code], _ = await llm.call("anthropic/claude-haiku-5.5", f"Translate the text into natural {name} as a native speaker would write a search query. Keep scientific terms accurate; output only the translation.", q)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            done[qid] = row
    return [done[q] for q in list(queries)[:100]]


async def main():
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    llm = LLM(auth)
    docs, queries, qrels = _m09.load_scifact()
    data = await build_dataset(llm, queries)
    gen_cost = llm.cost
    sub_qrels = {r["qid"]: qrels[r["qid"]] for r in data}
    print(f"dataset: {len(data)} queries x {len(LANGS)} langs; generation cost ${gen_cost:.4f}", flush=True)

    client = MossClient(pid, pkey)
    sess = await client.session("scifact-moss-minilm", auto_load_on_init=False)
    t0 = time.perf_counter()
    await sess.load_from_disk(str(SESS_DIR))
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"loaded minilm snapshot in {load_ms:.0f} ms ({sess.doc_count} docs)", flush=True)

    async def retrieve(qtexts, alpha=1.0, s=sess, emb=None):
        run = {}
        for qid, t in qtexts.items():
            opts = QueryOptions(top_k=10, alpha=alpha)
            if emb is not None:
                opts.embedding = emb[qid]
            run[qid] = [d.id for d in (await s.query(t, opts)).docs]
        return run

    def ndcg_list(run):
        return [metrics.ndcg_at_k(run.get(q, []), sub_qrels[q], 10) for q in sub_qrels]

    results, per_q = {}, {}
    en = {r["qid"]: r["en"] for r in data}
    per_q["en_upper"] = ndcg_list(await retrieve(en))
    translations, tr_lat = {}, []
    for code in LANGS:
        native = {r["qid"]: r[code] for r in data}
        per_q[f"{code}|native_minilm"] = ndcg_list(await retrieve(native))
        per_q[f"{code}|native_hybrid"] = ndcg_list(await retrieve(native, alpha=0.6))
        tr = {}
        for qid, t in native.items():
            tr[qid], ms = await llm.call("moss/fastlane-1", "Translate the user's search query into concise English. Keep scientific terms, identifiers and numbers unchanged. Output only the translation.", t, 80)
            tr_lat.append(ms)
        translations[code] = tr
        per_q[f"{code}|fix_translate"] = ndcg_list(await retrieve(tr))
        print(f"  {code}: native={sum(per_q[f'{code}|native_minilm'])/100:.4f} hybrid={sum(per_q[f'{code}|native_hybrid'])/100:.4f} translate={sum(per_q[f'{code}|fix_translate'])/100:.4f}", flush=True)
    await sess.close()

    # Fix 2: multilingual embedding via Moss custom vectors
    from fastembed import TextEmbedding
    mname = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    emb = TextEmbedding(model_name=mname)
    t0 = time.perf_counter()
    dvecs = [v.tolist() for v in emb.embed([t for _, t in docs], batch_size=64)]
    embed_ms = (time.perf_counter() - t0) * 1000
    csess = await client.session("scifact-multilingual-custom", model_id="custom", auto_load_on_init=False)
    for i in range(0, len(docs), 512):
        await csess.add_docs([DocumentInfo(id=d, text=t, embedding=dvecs[i + j]) for j, (d, t) in enumerate(docs[i:i + 512])])
    qenc_lat = []
    for code in ["en"] + list(LANGS):
        texts = {r["qid"]: r[code] for r in data}
        t1 = time.perf_counter()
        qv = {qid: v.tolist() for qid, v in zip(texts, emb.embed(list(texts.values())))}
        qenc_lat.append((time.perf_counter() - t1) * 1000 / len(texts))
        per_q[f"{code}|fix_multilingual"] = ndcg_list(await retrieve(texts, alpha=1.0, s=csess, emb=qv))
        print(f"  {code}: multilingual-custom={sum(per_q[f'{code}|fix_multilingual'])/100:.4f}", flush=True)
    await csess.close()

    for k, v in per_q.items():
        results[k] = metrics.bootstrap_ci(v)
    deltas = {}
    for code in LANGS:
        base = per_q[f"{code}|native_minilm"]
        for fix in ["native_hybrid", "fix_translate", "fix_multilingual"]:
            deltas[f"{code}|{fix}_vs_native"] = metrics.paired_bootstrap(base, per_q[f"{code}|{fix}"])
        deltas[f"{code}|gap_vs_en"] = metrics.paired_bootstrap(per_q["en_upper"], base)
    deltas["en|multilingual_vs_minilm"] = metrics.paired_bootstrap(per_q["en_upper"], per_q["en|fix_multilingual"])

    # 120-doc bench (Suite 6) multilingual stratum with translate-then-retrieve
    await client.load_index("eval-bench-minilm")
    ml_cases = [q for q in _m06.EVAL_60 if q["stratum"] == "multilingual"]
    bench = {"before_r1": 0, "after_r1": 0, "rows": []}
    for q in ml_cases:
        before = [d.id for d in (await client.query("eval-bench-minilm", q["query"], QueryOptions(top_k=5, alpha=1.0))).docs]
        tq, ms = await llm.call("moss/fastlane-1", "Translate the user's search query into concise English. Keep identifiers and numbers unchanged. Output only the translation.", q["query"], 80)
        after = [d.id for d in (await client.query("eval-bench-minilm", tq, QueryOptions(top_k=5, alpha=1.0))).docs]
        bench["before_r1"] += int(bool(before) and before[0] == q["target"])
        bench["after_r1"] += int(bool(after) and after[0] == q["target"])
        bench["rows"].append({"id": q["id"], "translated": tq, "before_top1": before[:1], "after_top1": after[:1], "target": q["target"], "translate_ms": round(ms, 1)})
    await client.close()
    await llm.h.aclose()
    print(f"  120-doc bench multilingual R@1: {bench['before_r1']}/15 -> {bench['after_r1']}/15", flush=True)

    report = {
        "dataset": {"source": "BEIR scifact test (first 100 queries)", "langs": LANGS, "generator": "anthropic/claude-haiku-5.5",
                    "file": "data/multilingual/scifact_test100_hi_es_ml.jsonl"},
        "ndcg@10": results, "paired_deltas": deltas,
        "translate_latency_ms": {"p50": round(percentile(tr_lat, 50), 1), "p95": round(percentile(tr_lat, 95), 1), "p99": round(percentile(tr_lat, 99), 1), "n": len(tr_lat)},
        "multilingual_embed": {"model": mname, "doc_embed_ms": round(embed_ms, 1), "query_encode_ms_per_query": [round(x, 2) for x in qenc_lat], "note": "max_seq_length 128 tokens: abstracts are truncated"},
        "bench120_multilingual": bench, "translations_sample": {c: dict(list(t.items())[:5]) for c, t in translations.items()},
        "est_cost_usd": round(llm.cost, 5), "dataset_generation_cost_usd": round(gen_cost, 5), "snapshot_load_ms": round(load_ms, 1),
    }
    print(json.dumps({k: v for k, v in deltas.items()}, indent=1))
    print("Saved:", save_log("10_multilingual_fix.json", report, [pkey]), "cost", round(llm.cost, 4))


if __name__ == "__main__":
    asyncio.run(main())
