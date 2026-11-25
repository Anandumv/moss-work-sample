"""Write charts/*.png from logs/*.json. Skips a chart if its log is missing."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moss_eval.common import ROOT

LOGS, OUT = ROOT / "logs", ROOT / "charts"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name):
    p = LOGS / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


def scifact():
    d = load("09_beir_scifact.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, t in d["targets"].items():
        al = sorted(t["alphas"], key=float)
        x = [float(a) for a in al]
        m = [t["alphas"][a]["ndcg@10"]["mean"] for a in al]
        lo = [t["alphas"][a]["ndcg@10"]["lo"] for a in al]
        hi = [t["alphas"][a]["ndcg@10"]["hi"] for a in al]
        ax.plot(x, m, marker="o", label=name)
        ax.fill_between(x, lo, hi, alpha=0.12)
    ax.set_xlabel("alpha (0 = keyword/BM25 only, 1 = dense only)")
    ax.set_ylabel("nDCG@10 (95% bootstrap CI)")
    ax.set_title("BEIR SciFact (5,183 docs, 300 test queries): hybrid alpha sweep")
    ax.legend()
    save(fig, "scifact_alpha_sweep.png")


def multilingual():
    d = load("10_multilingual_fix.json")
    if not d:
        return
    r = d["ndcg@10"]
    langs = list(d["dataset"]["langs"])
    conds = [("native_minilm", "native query, minilm"), ("native_hybrid", "native, hybrid a=0.6"),
             ("fix_translate", "translate->minilm"), ("fix_multilingual", "multilingual custom vectors")]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    w = 0.2
    for i, (c, lab) in enumerate(conds):
        xs = [j + (i - 1.5) * w for j in range(len(langs))]
        ms = [r[f"{l}|{c}"]["mean"] for l in langs]
        err = [[r[f"{l}|{c}"]["mean"] - r[f"{l}|{c}"]["lo"] for l in langs], [r[f"{l}|{c}"]["hi"] - r[f"{l}|{c}"]["mean"] for l in langs]]
        ax.bar(xs, ms, w, yerr=err, capsize=2, label=lab)
    ax.axhline(r["en_upper"]["mean"], color="k", ls="--", lw=1, label=f"English query upper bound ({r['en_upper']['mean']:.3f})")
    ax.set_xticks(range(len(langs)), [d["dataset"]["langs"][l] for l in langs])
    ax.set_ylabel("nDCG@10")
    ax.set_title("SciFact test-100 with translated queries: gap and fixes")
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "multilingual_fix.png")


def gateway():
    d = load("11a_gateway_load.json")
    if not d:
        return
    ramp = d["open_loop_ramp"]
    x = [r["target_rps"] for r in ramp]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a1.plot(x, [r["goodput_rps"] for r in ramp], marker="o", label="goodput (HTTP 200/s)")
    a1.plot(x, x, ls=":", c="grey", label="offered")
    a1.axhline(2.0, c="r", ls="--", lw=1, label="documented 120 RPM")
    a1b = a1.twinx()
    a1b.bar(x, [r["status_counts"].get("429", 0) / r["sent"] for r in ramp], width=0.25, alpha=0.25, color="r")
    a1b.set_ylabel("429 fraction", color="r")
    a1b.set_ylim(0, 1)
    a1.set_xlabel("offered load (req/s, open loop)")
    a1.set_ylabel("req/s")
    a1.legend(fontsize=7, loc="upper left")
    a1.set_title("Gateway /responses (fastlane-1 + retrieval)")
    for p in ["p50", "p95", "p99"]:
        a2.plot(x, [r["latency_ok_ms"][p] for r in ramp], marker="o", label=f"end-to-end {p}")
    a2.plot(x, [r["retrieve_ms"]["p99"] for r in ramp], marker="s", label="X-Moss-Retrieve-Ms p99")
    a2.set_yscale("log")
    a2.set_xlabel("offered load (req/s)")
    a2.set_ylabel("ms (successful requests)")
    a2.legend(fontsize=7)
    a2.set_title("Latency vs load")
    save(fig, "gateway_load.png")


def local():
    d = load("11b_local_load.json")
    if not d:
        return
    fig, axes = plt.subplots(1, len(d["targets"]), figsize=(4 * len(d["targets"]), 3.4), squeeze=False)
    for ax, (name, t) in zip(axes[0], d["targets"].items()):
        ol = t["open_loop"]
        x = [r["target_qps"] for r in ol]
        for p in ["p50", "p95", "p99"]:
            ax.plot(x, [r[p] for r in ol], marker="o", label=p)
        ax.set_yscale("log")
        ax.set_xlabel("offered QPS (open loop)")
        ax.set_ylabel("ms incl. queueing")
        ax.set_title(name, fontsize=8)
        ax.legend(fontsize=7)
    save(fig, "local_load.png")


def workaround():
    d = load("12_bug_workarounds.json")
    if not d:
        return
    s = d["context_loss"]["summary"]
    conds = ["baseline", "W1_requery", "W2_moss_search", "W3_client_rag"]
    models = ["moss/fastlane-1", "openai/gpt-6-luna"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    for i, m in enumerate(models):
        vals = [int(s[f"{m}|{c}"]["correct"].split("/")[0]) for c in conds]
        ax.bar([j + (i - 0.5) * w for j in range(len(conds))], vals, w, label=m)
    ax.set_xticks(range(len(conds)), ["baseline\n(X-Moss-Indexes)", "W1 re-append\nuser turn", "W2 moss_search\ntool", "W3 client-side\nMoss retrieval"])
    ax.set_ylabel("cases correct after tool call (of 10)")
    ax.set_ylim(0, 10.5)
    ax.set_title("ISS-01 context loss in function-tool loops: before / after")
    ax.legend(fontsize=7)
    save(fig, "iss01_workarounds.png")


def tasks_budget():
    d = load("13_tasks_adversarial.json")
    if not d:
        return
    rows = d["budget_across_tiers"]
    names = [r["model"] for r in rows]
    ratio = [(r.get("total_usd") or 0) / r["cap_usd"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(range(len(rows)), ratio, color=["tab:red" if x > 1 else "tab:blue" for x in ratio])
    ax.axhline(1.0, c="k", ls="--", lw=1, label="max_cost_usd cap ($0.02)")
    ax.set_xticks(range(len(rows)), [n.replace("/", "/\n") for n in names], fontsize=7)
    ax.set_ylabel("billed / cap")
    ax.set_title("Tasks API: same task, $0.02 cap, by model tier")
    ax.legend(fontsize=7)
    save(fig, "tasks_budget_overshoot.png")


def evals_demo():
    base = ROOT / "examples" / "demo_out"
    runs = [("baseline", "run_baseline.json"), ("translate non-Latin", "run_translate_nonlatin.json"), ("translate all", "run_translate_all.json")]
    if not all((base / f).exists() for _, f in runs):
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, (lab, f) in enumerate(runs):
        m = json.loads((base / f).read_text())["summary"]["metrics"]
        keys = ["recall@5", "mrr@10", "ndcg@10"]
        xs = [j + (i - 1) * 0.27 for j in range(len(keys))]
        ax.bar(xs, [m[k]["mean"] for k in keys], 0.27, yerr=[[m[k]["mean"] - m[k]["lo"] for k in keys], [m[k]["hi"] - m[k]["mean"] for k in keys]], capsize=2, label=lab)
    ax.set_xticks(range(3), ["Recall@5", "MRR@10", "nDCG@10"])
    ax.set_ylim(0.5, 1.02)
    ax.set_title("moss-evals regression set (60 cases from Suite 6 failures + guards)")
    ax.legend(fontsize=7)
    save(fig, "moss_evals_regression.png")


if __name__ == "__main__":
    for f in (scifact, multilingual, gateway, local, workaround, tasks_budget, evals_demo):
        f()
