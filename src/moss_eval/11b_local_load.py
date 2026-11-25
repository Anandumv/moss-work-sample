"""
Suite 11b: local (in-process) retrieval under load + cold start.

- Open-loop QPS ramp (latency measured from the *scheduled* send time, so queueing is counted,
  avoiding coordinated omission) on eval-bench-minilm (120 docs, cloud index loaded locally)
  and the SciFact 5,183-doc moss-minilm session snapshot.
- Closed-loop max throughput at concurrency 1/4/16.
- Cold start in fresh subprocesses (x3): import, client init, load_index / load_from_disk,
  first query, then warm median.
Caveat: the box is shared; another tenant process used ~5 CPUs during the run (load avg recorded).
"""
import asyncio
import json
import os
import subprocess
import sys
import time

from moss import MossClient, QueryOptions

from moss_eval.common import ROOT, load_credentials, percentile, save_log

QUERIES = None


def summarize(lat):
    lat = sorted(lat)
    return {"n": len(lat), "p50": round(percentile(lat, 50), 3), "p95": round(percentile(lat, 95), 3),
            "p99": round(percentile(lat, 99), 3), "max": round(lat[-1], 3)} if lat else {"n": 0}


async def open_loop(qfn, texts, rate, seconds):
    n = int(rate * seconds)
    start = time.perf_counter() + 0.05
    lat, errs = [], 0

    async def one(i):
        nonlocal errs
        sched = start + i / rate
        delay = sched - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        try:
            await qfn(texts[i % len(texts)])
            lat.append((time.perf_counter() - sched) * 1000)
        except Exception:
            errs += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    return {"target_qps": rate, "achieved_qps": round(len(lat) / wall, 1), "errors": errs, **summarize(lat)}


async def closed_loop(qfn, texts, conc, seconds=8):
    stop = time.perf_counter() + seconds
    lat = []

    async def worker(w):
        i = w
        while time.perf_counter() < stop:
            t0 = time.perf_counter()
            await qfn(texts[i % len(texts)])
            lat.append((time.perf_counter() - t0) * 1000)
            i += conc

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(conc)))
    return {"concurrency": conc, "throughput_qps": round(len(lat) / (time.perf_counter() - t0), 1), **summarize(lat)}


COLD_SNIPPET = r'''
import asyncio, json, time, sys
t0 = time.perf_counter()
from moss import MossClient, QueryOptions
from moss_eval.common import load_credentials
t_import = time.perf_counter()
async def main(kind):
    pid, pkey = load_credentials()
    c = MossClient(pid, pkey)
    t_client = time.perf_counter()
    if kind == "cloud_index":
        await c.load_index("eval-bench-minilm")
        t_load = time.perf_counter()
        q = lambda s: c.query("eval-bench-minilm", s, QueryOptions(top_k=5))
    else:
        s = await c.session(kind, auto_load_on_init=False)
        await s.load_from_disk(sys.argv[2])
        t_load = time.perf_counter()
        q = lambda x: s.query(x, QueryOptions(top_k=10))
    a = time.perf_counter(); await q("how does the refund policy work"); first = (time.perf_counter() - a) * 1000
    warm = []
    for i in range(20):
        a = time.perf_counter(); await q(f"protein folding mechanism {i}"); warm.append((time.perf_counter() - a) * 1000)
    warm.sort()
    print(json.dumps({"import_ms": (t_import - t0) * 1000, "client_ms": (t_client - t_import) * 1000,
                      "load_ms": (t_load - t_client) * 1000, "first_query_ms": first, "warm_p50_ms": warm[10]}))
asyncio.run(main(sys.argv[1]))
'''


def cold_start(kind, reps=3):
    out = []
    for _ in range(reps):
        r = subprocess.run([sys.executable, "-c", COLD_SNIPPET, kind, str(ROOT / "data" / "sessions")], capture_output=True, text=True,
                           cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=600)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        out.append({k: round(v, 1) for k, v in json.loads(line[-1]).items()} if line else {"error": r.stderr[-300:]})
    return out


async def main():
    pid, pkey = load_credentials()
    from moss_eval import common  # noqa
    import importlib
    m09 = importlib.import_module("moss_eval.09_beir_scifact")
    _, sq, _ = m09.load_scifact()
    sci_texts = list(sq.values())
    bench_texts = ["refund policy for damaged items", "how do I reset my password", "shipping times to Auckland",
                   "cancel subscription before renewal", "API rate limit headers", "order status lookup",
                   "data retention policy", "change delivery address after ordering"] * 4
    client = MossClient(pid, pkey)
    await client.load_index("eval-bench-minilm")
    sess = await client.session("scifact-moss-minilm", auto_load_on_init=False)
    await sess.load_from_disk(str(ROOT / "data" / "sessions"))
    targets = {
        "eval-bench-minilm (120 docs, top_k=5)": lambda s: client.query("eval-bench-minilm", s, QueryOptions(top_k=5)),
        "scifact-minilm session (5,183 docs, top_k=10)": lambda s: sess.query(s, QueryOptions(top_k=10)),
        "scifact-minilm session hybrid a=0.6": lambda s: sess.query(s, QueryOptions(top_k=10, alpha=0.6)),
    }
    report = {"load_avg_start": os.getloadavg(), "cpus": os.cpu_count(), "targets": {}}
    for name, fn in targets.items():
        texts = bench_texts if "bench" in name else sci_texts
        for t in texts[:20]:
            await fn(t)
        res = {"closed_loop": [], "open_loop": []}
        for c in [1, 4, 16]:
            r = await closed_loop(fn, texts, c)
            res["closed_loop"].append(r)
            print(f"[closed] {name:45s} c={c:<3} {r['throughput_qps']:>7} qps p50={r['p50']}ms p99={r['p99']}ms", flush=True)
        cap = res["closed_loop"][0]["throughput_qps"]
        for frac in [0.25, 0.5, 0.75, 0.9, 1.1, 1.5]:
            rate = max(5, round(cap * frac))
            r = await open_loop(fn, texts, rate, 8)
            r["fraction_of_c1_capacity"] = frac
            res["open_loop"].append(r)
            print(f"[open]   {name:45s} {rate:>6} qps -> {r['achieved_qps']} p50={r['p50']} p95={r['p95']} p99={r['p99']}ms", flush=True)
        report["targets"][name] = res
    await sess.close()
    await client.close()
    report["load_avg_after_ramp"] = os.getloadavg()
    report["cold_start"] = {"cloud_index eval-bench-minilm": cold_start("cloud_index"),
                            "session snapshot scifact-moss-minilm": cold_start("scifact-moss-minilm")}
    if (ROOT / "data" / "sessions").exists() and any("mediumlm" in p.name for p in (ROOT / "data" / "sessions").iterdir()):
        report["cold_start"]["session snapshot scifact-moss-mediumlm"] = cold_start("scifact-moss-mediumlm")
    for k, v in report["cold_start"].items():
        print("[cold]", k, v, flush=True)
    report["load_avg_end"] = os.getloadavg()
    print("Saved:", save_log("11b_local_load.json", report, [pkey]))


if __name__ == "__main__":
    asyncio.run(main())
