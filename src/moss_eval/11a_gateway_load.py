"""
Suite 11a: Gateway load & tail latency on /v1/responses (moss/fastlane-1 + X-Moss-Indexes).

Open-loop (constant arrival rate) ramp so slow responses do not throttle the load generator
(avoids coordinated omission). Documented limits for fastlane-1: 120 RPM sustained / 10 burst;
vendor models 300 / 50. Records status mix, client latency p50/p95/p99, X-Moss-Retrieve-Ms,
Retry-After and x-ratelimit-* headers; then measures a Retry-After-honouring client and bursts.
"""
import asyncio
import time
from collections import Counter
from typing import Any

import httpx

from moss_eval.common import load_credentials, percentile, save_log

BASE = "https://inference.moss.dev/v1"
IDX = "eval-bench-minilm"
QUERIES = ["What is the KoruPass annual price?", "How long are gift vouchers valid?", "What is escalation code ESC-917 for?",
           "What is the webhook partner API rate limit?", "When does gpu-l40s-prod scale up?", "How fast is Patroni failover?"]


def stats(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(percentile(xs, 50), 1), "p95": round(percentile(xs, 95), 1), "p99": round(percentile(xs, 99), 1), "max": round(max(xs), 1)}


async def one(h, auth, i, model="moss/fastlane-1", retry=False, max_retries=3):
    body = {"model": model, "input": QUERIES[i % len(QUERIES)], "max_output_tokens": 8}
    hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Indexes": IDX, "X-Moss-Cache": "off"}
    t0 = time.perf_counter()
    attempts = 0
    waited = 0.0
    while True:
        attempts += 1
        try:
            r = await h.post(f"{BASE}/responses", headers=hdrs, json=body)
        except httpx.HTTPError as e:
            return {"status": "conn_error", "err": str(e)[:100], "ms": (time.perf_counter() - t0) * 1000, "attempts": attempts}
        if r.status_code == 429 and retry and attempts <= max_retries:
            ra = float(r.headers.get("retry-after", "1"))
            waited += ra
            await asyncio.sleep(ra)
            continue
        break
    usage = (r.json().get("usage") or {}) if r.status_code == 200 else {}
    rl = {k: v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit") or k.lower() == "retry-after"}
    code = None
    if r.status_code != 200:
        try:
            code = r.json().get("error", {}).get("code")
        except Exception:
            code = r.text[:80]
    return {"status": r.status_code, "code": code, "ms": (time.perf_counter() - t0) * 1000, "retrieve_ms": r.headers.get("x-moss-retrieve-ms"),
            "headers": rl, "attempts": attempts, "retry_wait_s": waited, "in_tok": usage.get("input_tokens", 0), "out_tok": usage.get("output_tokens", 0)}


async def open_loop(h, auth, rps, duration_s, model="moss/fastlane-1", retry=False):
    tasks = []
    start = time.perf_counter()
    n = int(rps * duration_s)
    for i in range(n):
        target = start + i / rps
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(h, auth, i, model, retry)))
    res = await asyncio.gather(*tasks)
    wall = time.perf_counter() - start
    ok = [r for r in res if r["status"] == 200]
    sc = Counter(str(r["status"]) for r in res)
    ra = Counter(r["headers"].get("retry-after") or r["headers"].get("Retry-After") for r in res if r["status"] == 429)
    return {
        "target_rps": rps, "duration_s": duration_s, "sent": n, "wall_s": round(wall, 1), "status_counts": dict(sc),
        "error_codes": dict(Counter(r.get("code") for r in res if r["status"] != 200)),
        "success_rate": round(len(ok) / n, 3), "goodput_rps": round(len(ok) / wall, 2),
        "latency_ok_ms": stats([r["ms"] for r in ok]),
        "retrieve_ms": stats([float(r["retrieve_ms"]) for r in ok if r.get("retrieve_ms")]),
        "retry_after_values": {str(k): v for k, v in ra.items()},
        "sample_429_headers": next((r["headers"] for r in res if r["status"] == 429), None),
        "sample_200_headers": next((r["headers"] for r in ok), None),
        "mean_attempts": round(sum(r["attempts"] for r in res) / n, 2),
        "tokens": {"in": sum(r["in_tok"] for r in ok), "out": sum(r["out_tok"] for r in ok)},
    }


async def burst(h, auth, n, model):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[one(h, auth, i, model) for i in range(n)])
    return {"model": model, "n": n, "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "status_counts": dict(Counter(str(r["status"]) for r in res)),
            "latency_ok_ms": stats([r["ms"] for r in res if r["status"] == 200]),
            "sample_429": next(({"headers": r["headers"], "code": r["code"]} for r in res if r["status"] == 429), None),
            "tokens": {"in": sum(r["in_tok"] for r in res), "out": sum(r["out_tok"] for r in res)}}


async def main():
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    report: dict[str, Any] = {}
    limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=60, limits=limits) as h:
        # warm the index on the gateway
        warm = [await one(h, auth, i) for i in range(3)]
        report["warmup_retrieve_ms"] = [w.get("retrieve_ms") for w in warm]
        await asyncio.sleep(61)  # let the per-minute window reset
        ramp = []
        for rps in [0.5, 1, 2, 3, 4]:
            res = await open_loop(h, auth, rps, 40)
            ramp.append(res)
            print(f"[ramp] {rps} rps: {res['status_counts']} goodput={res['goodput_rps']} p50={res['latency_ok_ms'].get('p50')} "
                  f"p95={res['latency_ok_ms'].get('p95')} p99={res['latency_ok_ms'].get('p99')} retry-after={res['retry_after_values']}", flush=True)
            await asyncio.sleep(61)
        report["open_loop_ramp"] = ramp
        res = await open_loop(h, auth, 4, 40, retry=True)
        report["retry_after_client_4rps"] = res
        print(f"[retry] 4 rps with Retry-After honouring: {res['status_counts']} success={res['success_rate']} p50={res['latency_ok_ms'].get('p50')} p99={res['latency_ok_ms'].get('p99')} attempts={res['mean_attempts']}", flush=True)
        await asyncio.sleep(61)
        report["burst_fastlane_20"] = await burst(h, auth, 20, "moss/fastlane-1")
        print("[burst] fastlane 20:", report["burst_fastlane_20"]["status_counts"], flush=True)
        await asyncio.sleep(61)
        report["burst_haiku_60"] = await burst(h, auth, 60, "anthropic/claude-haiku-5.5")
        print("[burst] haiku 60:", report["burst_haiku_60"]["status_counts"], flush=True)

    # cost estimate: fastlane tokens + retrievals on 200s; haiku burst at its prices
    from moss_eval.common import estimate_inference_cost
    cost = 0.0
    for blk in report["open_loop_ramp"] + [report["retry_after_client_4rps"]]:
        ok = blk["status_counts"].get("200", 0)
        cost += estimate_inference_cost("moss/fastlane-1", blk["tokens"]["in"], blk["tokens"]["out"], 0, ok)
    b = report["burst_fastlane_20"]
    cost += estimate_inference_cost("moss/fastlane-1", b["tokens"]["in"], b["tokens"]["out"], 0, b["status_counts"].get("200", 0))
    b = report["burst_haiku_60"]
    cost += estimate_inference_cost("anthropic/claude-haiku-5.5", b["tokens"]["in"], b["tokens"]["out"], 0, b["status_counts"].get("200", 0))
    report["est_cost_usd"] = round(cost + 3 * 0.0007, 4)
    print("Saved:", save_log("11a_gateway_load.json", report, [pkey]), "est cost", report["est_cost_usd"])


if __name__ == "__main__":
    asyncio.run(main())
