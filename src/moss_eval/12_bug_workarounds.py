"""
Suite 12: Workarounds for the top Inference bugs, measured before/after.

A) ISS-01 context loss in stateless function-tool loops with X-Moss-Indexes.
   10 cases on eval-bench-minilm; each turn 0 forces a function call, turn 1 sends the
   function_call_output back. Conditions:
     baseline      : [user, function_call, function_call_output] + X-Moss-Indexes
     W1_requery    : baseline + the user's question re-appended as a trailing user message
     W2_moss_search: moss_search tool instead of X-Moss-Indexes
     W3_client_rag : retrieve client-side with MossClient (local, ~5 ms), pass passages in
                     instructions on both turns, X-Moss-Retrieve: off
B) ISS-04 reasoning budget starvation: qwen3.8-flash / glm-5.3-flash with
   max_output_tokens=450 at default vs reasoning.effort=low vs none.
"""
import asyncio
import importlib
import json
import time
from typing import Any

import httpx
from moss import MossClient, QueryOptions

from moss_eval.common import estimate_inference_cost, load_credentials, save_log, summarize_latencies

BASE = "https://inference.moss.dev/v1"
IDX = "eval-bench-minilm"
_m06 = importlib.import_module("moss_eval.06_deep_retrieval_eval")
EVAL = {q["id"]: q for q in _m06.EVAL_60}
CASE_IDS = ["d01", "d03", "d06", "d08", "d10", "a01", "a03", "a05", "a08", "a12"]
TOOL = {
    "type": "function",
    "name": "get_order_status",
    "description": "Look up the live delivery status of a customer order by order id.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
}
INSTR = "You are a support agent. Use get_order_status for order status, and use the retrieved knowledge for policy questions. Answer both parts in at most 3 sentences."


def fact_ok(text: str, facts: list[str]) -> bool:
    low = text.lower()
    return all(any(alt.lower() in low for alt in f.split("|")) for f in facts)


async def post(h, auth, body, extra_headers=None):
    hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off", **(extra_headers or {})}
    for attempt in range(5):
        t0 = time.perf_counter()
        r = await h.post(f"{BASE}/responses", headers=hdrs, json=body)
        ms = (time.perf_counter() - t0) * 1000
        if r.status_code == 429 and attempt < 4:
            await asyncio.sleep(float(r.headers.get("retry-after", 2)))
            continue
        return r, ms
    return r, ms


def text_of(data):
    return "".join(c.get("text", "") for it in data.get("output", []) if it.get("type") == "message" for c in it.get("content", []) if c.get("type") == "output_text")


async def run_case(h, auth, client, cond, case, model):
    question = f"My order 4471 hasn't arrived - what's its status? Also: {case['query']}"
    user = {"role": "user", "content": question}
    tools = [TOOL]
    headers0, instr = {}, INSTR
    if cond in ("baseline", "W1_requery"):
        headers0 = {"X-Moss-Indexes": IDX}
    elif cond == "W2_moss_search":
        tools = [TOOL, {"type": "moss_search", "indexes": [IDX], "top_k": 5}]
    elif cond == "W3_client_rag":
        t0 = time.perf_counter()
        res = await client.query(IDX, case["query"], QueryOptions(top_k=5, alpha=1.0))
        local_ms = (time.perf_counter() - t0) * 1000
        instr = INSTR + "\n\nRETRIEVED KNOWLEDGE:\n" + "\n".join(f"- {d.text}" for d in res.docs)
        headers0 = {"X-Moss-Indexes": IDX, "X-Moss-Retrieve": "off"}
    body0 = {"model": model, "instructions": instr, "input": [user], "tools": tools, "max_output_tokens": 200}
    if cond != "W2_moss_search":
        body0["tool_choice"] = "required"
    r0, ms0 = await post(h, auth, body0, headers0)
    d0 = r0.json() if r0.status_code == 200 else {}
    out0 = d0.get("output", [])
    fc = next((x for x in out0 if x.get("type") == "function_call"), None)
    row = {"case": case["id"], "cond": cond, "model": model, "turn0_status": r0.status_code,
           "turn0_in_tok": (d0.get("usage") or {}).get("input_tokens"), "turn0_retrieve_ms": r0.headers.get("x-moss-retrieve-ms"),
           "turn0_ms": round(ms0, 1), "got_function_call": fc is not None}
    if cond == "W3_client_rag":
        row["local_retrieve_ms"] = round(local_ms, 2)
    cost = estimate_inference_cost(model, row["turn0_in_tok"] or 0, (d0.get("usage") or {}).get("output_tokens", 0), 0, 1 if cond in ("baseline", "W1_requery") else 0)
    if fc is None:
        final = text_of(d0)
        row.update({"final": final[:300], "correct": fact_ok(final, case["fact"]), "note": "no function call on turn 0", "cost": cost})
        return row
    items = [user] + [x for x in out0 if x.get("type") in ("function_call", "mcp_call", "mcp_list_tools")] + [
        {"type": "function_call_output", "call_id": fc["call_id"], "output": json.dumps({"order_id": "4471", "status": "out for delivery", "eta_minutes": 12})}]
    if cond == "W1_requery":
        items.append({"role": "user", "content": question})
    body1 = {"model": model, "instructions": instr, "input": items, "tools": tools, "max_output_tokens": 220}
    r1, ms1 = await post(h, auth, body1, headers0)
    d1 = r1.json() if r1.status_code == 200 else {"error": r1.text[:300]}
    final = text_of(d1)
    u1 = d1.get("usage") or {}
    cost += estimate_inference_cost(model, u1.get("input_tokens", 0), u1.get("output_tokens", 0), 0, 1 if cond in ("baseline", "W1_requery") else 0)
    row.update({"turn1_status": r1.status_code, "turn1_in_tok": u1.get("input_tokens"), "turn1_retrieve_ms": r1.headers.get("x-moss-retrieve-ms"),
                "turn1_ms": round(ms1, 1), "final": final[:300], "correct": fact_ok(final, case["fact"]),
                "mentions_status": "out for delivery" in final.lower() or "12" in final, "cost": round(cost, 6),
                "error": d1.get("error") if r1.status_code != 200 else None})
    return row


async def part_a(pid, pkey):
    auth = f"Bearer {pid}:{pkey}"
    client = MossClient(pid, pkey)
    await client.load_index(IDX)
    rows = []
    async with httpx.AsyncClient(timeout=120) as h:
        for model in ["moss/fastlane-1", "openai/gpt-6-luna"]:
            for cond in ["baseline", "W1_requery", "W2_moss_search", "W3_client_rag"]:
                for cid in CASE_IDS:
                    row = await run_case(h, auth, client, cond, EVAL[cid], model)
                    rows.append(row)
                    await asyncio.sleep(0.6)
                sub = [r for r in rows if r["cond"] == cond and r["model"] == model]
                print(f"[A] {model:18s} {cond:15s} correct={sum(r['correct'] for r in sub)}/{len(sub)} "
                      f"status_mentioned={sum(bool(r.get('mentions_status')) for r in sub)}/{len(sub)}", flush=True)
    await client.close()
    summary = {}
    for model in ["moss/fastlane-1", "openai/gpt-6-luna"]:
        for cond in ["baseline", "W1_requery", "W2_moss_search", "W3_client_rag"]:
            sub = [r for r in rows if r["cond"] == cond and r["model"] == model]
            tot = [r.get("turn0_ms", 0) + r.get("turn1_ms", 0) for r in sub]
            summary[f"{model}|{cond}"] = {
                "correct": f"{sum(r['correct'] for r in sub)}/{len(sub)}",
                "status_mentioned": f"{sum(bool(r.get('mentions_status')) for r in sub)}/{len(sub)}",
                "turn1_retrieve_ms_values": sorted({str(r.get('turn1_retrieve_ms')) for r in sub}),
                "turn1_in_tok_mean": round(sum((r.get('turn1_in_tok') or 0) for r in sub) / len(sub), 1),
                "two_turn_ms": summarize_latencies(tot),
                "cost_usd": round(sum(r.get("cost", 0) for r in sub), 6),
            }
    return {"rows": rows, "summary": summary}


async def part_b(pid, pkey):
    auth = f"Bearer {pid}:{pkey}"
    out = {}
    async with httpx.AsyncClient(timeout=120) as h:
        for model in ["qwen/qwen3.8-flash", "z-ai/glm-5.3-flash"]:
            for label, extra in [("default", {}), ("effort_low", {"reasoning": {"effort": "low"}}), ("effort_none", {"reasoning": {"effort": "none"}})]:
                runs = []
                for i in range(5):
                    r, ms = await post(h, auth, {"model": model, "instructions": "Answer in one short sentence using ONLY the retrieved context.",
                                                  "input": EVAL["d01"]["query"], "max_output_tokens": 450, **extra}, {"X-Moss-Indexes": IDX})
                    d = r.json() if r.status_code == 200 else {}
                    u = d.get("usage") or {}
                    txt = text_of(d)
                    runs.append({"status": r.status_code, "resp_status": d.get("status"), "ms": round(ms, 1), "visible": bool(txt.strip()),
                                 "correct": fact_ok(txt, EVAL["d01"]["fact"]), "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
                                 "output_tokens": u.get("output_tokens"), "cost": estimate_inference_cost(model, u.get("input_tokens", 0), u.get("output_tokens", 0), 0, 1),
                                 "error": r.text[:200] if r.status_code != 200 else None})
                    await asyncio.sleep(0.3)
                ok_ms = [x["ms"] for x in runs if x["status"] == 200]
                out[f"{model}|{label}"] = {"visible": f"{sum(x['visible'] for x in runs)}/5", "correct": f"{sum(x['correct'] for x in runs)}/5",
                                           "http": sorted({x['status'] for x in runs}), "latency_ms": summarize_latencies(ok_ms),
                                           "mean_reasoning_tokens": round(sum((x['reasoning_tokens'] or 0) for x in runs) / 5, 1),
                                           "cost_usd": round(sum(x["cost"] for x in runs), 6), "runs": runs}
                v = out[f"{model}|{label}"]
                print(f"[B] {model:20s} {label:12s} visible={v['visible']} correct={v['correct']} http={v['http']} p50={v['latency_ms']['p50']}ms r_tok={v['mean_reasoning_tokens']}", flush=True)
    return out


async def main():
    pid, pkey = load_credentials()
    a = await part_a(pid, pkey)
    b = await part_b(pid, pkey)
    total = sum(r.get("cost", 0) for r in a["rows"]) + sum(v["cost_usd"] for v in b.values())
    print("Saved:", save_log("12_bug_workarounds.json", {"context_loss": a, "reasoning_budget": b, "est_cost_usd": round(total, 6)}, [pkey]))
    print(json.dumps(a["summary"], indent=1))
    print("est cost", round(total, 4))


if __name__ == "__main__":
    asyncio.run(main())
