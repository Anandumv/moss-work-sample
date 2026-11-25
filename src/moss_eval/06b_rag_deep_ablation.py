"""
Suite 6b: min_score Threshold Calibration (Local SessionIndex, $0) +
RAG Tool-Call Query-Rewrite Trace & Un-truncated Reasoning Budget Ablation.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from moss import MossClient, QueryOptions
from moss_eval.common import estimate_inference_cost, load_credentials, save_log, summarize_latencies
import importlib
_m06 = importlib.import_module("moss_eval.06_deep_retrieval_eval")
build_corpus, EVAL_60, UNANSWERABLE_TRAPS = _m06.build_corpus, _m06.EVAL_60, _m06.UNANSWERABLE_TRAPS

BASE_URL = "https://inference.moss.dev/v1"
IDX = "eval-bench-minilm"


def norm_match(ans: str, facts: list[str], expect_refusal: bool = False) -> bool:
    low = ans.replace("\u2019", "'").replace("\u2018", "'").lower()
    if expect_refusal:
        markers = [
            "don't know", "do not know", "not mentioned", "no information",
            "cannot find", "not provided", "not in", "don't have", "do not have",
            "not available", "does not contain", "doesn't contain", "unable to find",
            "don't have information", "no mention", "not specified",
        ]
        return any(m in low for m in markers)
    return all(any(alt.lower() in low for alt in f.split("|")) for f in facts)


async def score_calibration(pid: str, pkey: str) -> dict[str, Any]:
    corpus = build_corpus()
    client = MossClient(pid, pkey)
    sess = await client.session("cal-minilm", model_id="moss-minilm", auto_load_on_init=False)
    await sess.add_docs(corpus)

    pos_top1, neg_top1 = [], []
    trap_top1 = []
    for q in EVAL_60:
        res = await sess.query(q["query"], QueryOptions(top_k=3, alpha=1.0))
        if not res.docs:
            continue
        top = res.docs[0]
        if top.id == q["target"]:
            pos_top1.append(top.score)
        else:
            neg_top1.append({"id": q["id"], "stratum": q["stratum"], "got": top.id, "score": round(top.score, 4)})
    for u in UNANSWERABLE_TRAPS:
        res = await sess.query(u["query"], QueryOptions(top_k=3, alpha=1.0))
        if res.docs:
            trap_top1.append({"id": u["id"], "got": res.docs[0].id, "score": round(res.docs[0].score, 4)})
    await sess.close()
    await client.close()

    # Sweep min_score thresholds
    sweep = []
    for thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
        tp = sum(1 for s in pos_top1 if s >= thr)
        fp_wrong_doc = sum(1 for x in neg_top1 if x["score"] >= thr)
        fp_trap = sum(1 for x in trap_top1 if x["score"] >= thr)
        sweep.append({
            "min_score": thr,
            "true_pos_kept": f"{tp}/{len(pos_top1)}",
            "wrong_doc_kept": f"{fp_wrong_doc}/{len(neg_top1)}",
            "unanswerable_trap_triggered": f"{fp_trap}/{len(trap_top1)}",
        })

    return {
        "correct_top1_scores": summarize_latencies(pos_top1),
        "wrong_top1_scores": [x["score"] for x in neg_top1],
        "trap_top1_details": trap_top1,
        "threshold_sweep": sweep,
    }


async def rag_ablation(pid: str, pkey: str) -> dict[str, Any]:
    auth = f"Bearer {pid}:{pkey}"
    sys_prompt = "You are a precise policy assistant. Answer in 1-2 short sentences using ONLY the retrieved context. If the answer is not in the context, explicitly say you do not have that information."
    subset = (
        [q for q in EVAL_60 if q["stratum"] == "paraphrase"][:5]
        + [q for q in EVAL_60 if q["stratum"] == "adversarial"][:10]
        + [q for q in EVAL_60 if q["stratum"] == "multilingual"][:5]
        + UNANSWERABLE_TRAPS
    )

    configs = [
        ("gpt6_luna_indexes_450tok", "openai/gpt-6-luna", "indexes", 450),
        ("gpt6_luna_moss_search_450tok", "openai/gpt-6-luna", "moss_search", 450),
        ("haiku55_moss_search_450tok", "anthropic/claude-haiku-5.5", "moss_search", 450),
    ]

    out: dict[str, Any] = {}
    total_cost = 0.0
    async with httpx.AsyncClient(timeout=90.0) as h:
        for label, model_id, mode, max_tok in configs:
            ans_pass, trap_pass = 0, 0
            ttfts = []
            cfg_cost = 0.0
            failures = []
            rewrites = []
            for c in subset:
                hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off"}
                body: dict[str, Any] = {
                    "model": model_id,
                    "stream": True,
                    "instructions": sys_prompt,
                    "input": c["query"],
                    "max_output_tokens": max_tok,
                }
                if mode == "indexes":
                    hdrs["X-Moss-Indexes"] = IDX
                else:
                    body["tools"] = [{"type": "moss_search", "indexes": [IDX], "top_k": 5}]

                t0 = time.perf_counter()
                ttft = None
                parts = []
                usage = {}
                mcp_calls = 0
                mcp_args = []
                async with h.stream("POST", f"{BASE_URL}/responses", headers=hdrs, json=body) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        ev = json.loads(raw)
                        et = ev.get("type", "")
                        if et == "response.output_text.delta":
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            parts.append(ev.get("delta") or "")
                        elif et == "response.mcp_call.completed":
                            mcp_calls += 1
                        elif et == "response.output_item.done":
                            item = ev.get("item") or {}
                            if item.get("type") == "mcp_call":
                                mcp_args.append(item.get("arguments"))
                        elif et in ("response.completed", "response.incomplete"):
                            usage = (ev.get("response") or {}).get("usage") or {}

                ans = "".join(parts)
                if ttft is not None:
                    ttfts.append(ttft)
                cost = estimate_inference_cost(
                    model_id,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    mcp_calls if mode == "moss_search" else 1,
                )
                cfg_cost += cost
                total_cost += cost
                if c.get("id", "").startswith("m") and mcp_args:
                    rewrites.append({"id": c["id"], "args": mcp_args})
                is_trap = bool(c.get("expect_refusal"))
                ok = norm_match(ans, c.get("fact", []), expect_refusal=is_trap)
                if is_trap:
                    if ok:
                        trap_pass += 1
                else:
                    if ok:
                        ans_pass += 1
                if not ok:
                    failures.append({"id": c["id"], "answer": ans[:200]})
                await asyncio.sleep(0.12)

            out[label] = {
                "answerable": f"{ans_pass}/20",
                "abstention": f"{trap_pass}/4",
                "overall": round((ans_pass + trap_pass) / 24, 4),
                "ttft_ms": summarize_latencies(ttfts),
                "est_cost_usd": round(cfg_cost, 6),
                "multilingual_rewrites": rewrites,
                "failures": failures,
            }
    out["total_cost_usd"] = round(total_cost, 6)
    return out


async def main() -> None:
    pid, pkey = load_credentials()
    cal = await score_calibration(pid, pkey)
    rag = await rag_ablation(pid, pkey)
    path = save_log("06b_rag_deep_ablation.json", {"score_calibration": cal, "rag_ablation": rag}, [pkey])
    print("Saved:", path)
    print("Correct top1 scores:", cal["correct_top1_scores"])
    print("Wrong top1 scores:", cal["wrong_top1_scores"])
    print("Trap top1 details:", cal["trap_top1_details"])
    print("Threshold sweep:", cal["threshold_sweep"])
    for k, v in rag.items():
        if isinstance(v, dict):
            print(f"{k:30s} | ans={v['answerable']} trap={v['abstention']} overall={v['overall']:.3f} | TTFT p50={v['ttft_ms']['p50']}ms | cost=${v['est_cost_usd']}")
            if v["multilingual_rewrites"]:
                print("  rewrites:", v["multilingual_rewrites"][:3])
    print("Suite 6b est cost USD:", rag["total_cost_usd"])


if __name__ == "__main__":
    asyncio.run(main())
