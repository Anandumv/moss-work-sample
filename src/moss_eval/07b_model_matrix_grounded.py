"""
Suite 7b: 13-Model Grounded Matrix on eval-bench-minilm with max_output_tokens=450
so thinking/reasoning models emit full visible answers and true TTFT/reasoning-token overhead.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from moss_eval.common import estimate_inference_cost, load_credentials, save_log, summarize_latencies

BASE_URL = "https://inference.moss.dev/v1"
IDX = "eval-bench-minilm"


async def stream_one(client: httpx.AsyncClient, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    ttft = None
    first_any_event_ms = None
    parts: list[str] = []
    final_resp: dict[str, Any] = {}
    async with client.stream("POST", f"{BASE_URL}/responses", headers=headers, json=payload) as r:
        status = r.status_code
        rh = {k.lower(): v for k, v in r.headers.items()}
        if status != 200:
            return {"status": status, "error": (await r.aread()).decode("utf-8", "replace")[:300]}
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            if first_any_event_ms is None:
                first_any_event_ms = (time.perf_counter() - t0) * 1000
            ev = json.loads(raw)
            et = ev.get("type", "")
            if et == "response.output_text.delta":
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(ev.get("delta") or "")
            elif et in ("response.completed", "response.incomplete", "response.failed"):
                final_resp = ev.get("response") or {}
    return {
        "status": status,
        "response_status": final_resp.get("status"),
        "ttft_ms": round(ttft, 2) if ttft is not None else None,
        "first_event_ms": round(first_any_event_ms, 2) if first_any_event_ms is not None else None,
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
        "retrieve_ms": float(rh["x-moss-retrieve-ms"]) if "x-moss-retrieve-ms" in rh else None,
        "text": "".join(parts),
        "usage": final_resp.get("usage") or {},
    }


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    configs = [
        ("moss/fastlane-1", "moss/fastlane-1", {}),
        ("openai/gpt-6-luna", "openai/gpt-6-luna", {}),
        ("openai/gpt-6.1-terra", "openai/gpt-6.1-terra", {}),
        ("anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5", {}),
        ("anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5.5", {}),
        ("google/gemini-3.5-flash-lite", "google/gemini-3.5-flash-lite", {}),
        ("google/gemini-3.8-flash (default)", "google/gemini-3.8-flash", {}),
        ("google/gemini-3.8-flash (effort=none)", "google/gemini-3.8-flash", {"reasoning": {"effort": "none"}}),
        ("deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", {}),
        ("qwen/qwen3.8-flash", "qwen/qwen3.8-flash", {}),
        ("z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", {}),
        ("moonshotai/kimi-k3", "moonshotai/kimi-k3", {}),
        ("minimax/minimax-m3.1-flash", "minimax/minimax-m3.1-flash", {}),
    ]

    results: dict[str, Any] = {}
    total_cost = 0.0
    async with httpx.AsyncClient(timeout=120.0) as client:
        for label, model_id, extra in configs:
            ttfts, totals, reason_toks = [], [], []
            runs = []
            cost_sum = 0.0
            for i in range(3):
                r = await stream_one(
                    client,
                    {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Indexes": IDX, "X-Moss-Cache": "off"},
                    {
                        "model": model_id,
                        "stream": True,
                        "instructions": "Answer in one short sentence using ONLY the retrieved context.",
                        "input": "What is the KoruMart Auckland CBD express courier SLA and minimum basket?",
                        "max_output_tokens": 450,
                        **extra,
                    },
                )
                u = r.get("usage") or {}
                c = estimate_inference_cost(
                    model_id,
                    u.get("input_tokens", 0),
                    u.get("output_tokens", 0),
                    (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    1,
                )
                cost_sum += c
                total_cost += c
                if r.get("ttft_ms") is not None:
                    ttfts.append(r["ttft_ms"])
                if r.get("total_ms") is not None:
                    totals.append(r["total_ms"])
                rt = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                reason_toks.append(rt)
                txt = (r.get("text") or "").lower()
                ok = ("18" in txt) and ("35" in txt)
                runs.append({
                    "run": i,
                    "response_status": r.get("response_status"),
                    "ttft_ms": r.get("ttft_ms"),
                    "total_ms": r.get("total_ms"),
                    "retrieve_ms": r.get("retrieve_ms"),
                    "reasoning_tokens": rt,
                    "output_tokens": u.get("output_tokens", 0),
                    "fact_ok": ok,
                    "text_preview": (r.get("text") or r.get("error") or "")[:160],
                })
                await asyncio.sleep(0.15)
            results[label] = {
                "model": model_id,
                "accuracy": f"{sum(1 for x in runs if x['fact_ok'])}/{len(runs)}",
                "ttft_ms": summarize_latencies(ttfts) if ttfts else None,
                "total_ms": summarize_latencies(totals) if totals else None,
                "avg_reasoning_tokens": round(sum(reason_toks) / len(reason_toks), 1),
                "est_cost_usd": round(cost_sum, 6),
                "runs": runs,
            }

    results["total_cost_usd"] = round(total_cost, 6)
    path = save_log("07b_model_matrix_grounded.json", results, [pkey])
    print("Saved:", path)
    for label, d in results.items():
        if isinstance(d, dict) and "accuracy" in d:
            p50 = d["ttft_ms"]["p50"] if d["ttft_ms"] else None
            tot = d["total_ms"]["p50"] if d["total_ms"] else None
            print(f"{label:38s} | acc={d['accuracy']} | ttft_p50={p50}ms | tot_p50={tot}ms | r_tok={d['avg_reasoning_tokens']} | cost=${d['est_cost_usd']}")
    print("Total 07b cost USD:", total_cost)


if __name__ == "__main__":
    asyncio.run(main())
