"""Compare a run to thresholds and an optional baseline. Writes Markdown and JUnit, sets the exit code."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from xml.sax.saxutils import escape

from . import metrics


def per_case_table(run: dict[str, Any], qrels: dict[str, dict[str, int]], k: int) -> dict[str, dict[str, float]]:
    ranked = run["ranked"]
    return {
        qid: {
            f"hit@{k}": metrics.hit_at_k(ranked.get(qid, []), rel, k),
            f"mrr@{k}": metrics.mrr_at_k(ranked.get(qid, []), rel, k),
            f"ndcg@{k}": metrics.ndcg_at_k(ranked.get(qid, []), rel, k),
            f"recall@{k}": metrics.recall_at_k(ranked.get(qid, []), rel, k),
        }
        for qid, rel in qrels.items() if rel
    }


def gate(
    current: dict[str, dict[str, float]],
    thresholds: dict[str, dict[str, float]],
    baseline: Optional[dict[str, dict[str, float]]] = None,
) -> list[dict[str, Any]]:
    results = []
    for metric, rule in thresholds.items():
        vals = [c[metric] for c in current.values() if metric in c and c[metric] is not None]
        if not vals:
            results.append({"gate": metric, "passed": False, "detail": "metric missing from run"})
            continue
        ci = metrics.bootstrap_ci(vals)
        passed = True
        details = [f"mean={ci['mean']:.4f} [95% CI {ci['lo']:.4f}, {ci['hi']:.4f}] n={ci['n']}"]
        if "min" in rule and ci["mean"] < rule["min"]:
            passed = False
            details.append(f"below min {rule['min']}")
        if baseline is not None and "max_drop" in rule:
            shared = [q for q in current if q in baseline and metric in baseline[q] and metric in current[q]]
            if shared:
                pb = metrics.paired_bootstrap([baseline[q][metric] for q in shared], [current[q][metric] for q in shared])
                details.append(f"vs baseline delta={pb['delta']:+.4f} [{pb['lo']:+.4f}, {pb['hi']:+.4f}] p={pb['p_value']}")
                if pb["delta"] < -rule["max_drop"] and pb["p_value"] < 0.05:
                    passed = False
                    details.append(f"significant drop > {rule['max_drop']}")
        results.append({"gate": metric, "passed": passed, "detail": "; ".join(details), "ci": ci})
    return results


def newly_failing(current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]], key: str) -> list[str]:
    return sorted(q for q in current if q in baseline and baseline[q].get(key, 0) > 0 and current[q].get(key, 0) == 0)


def to_markdown(title: str, gates: list[dict[str, Any]], regressions: list[str], strata: dict[str, Any], extra: dict[str, Any]) -> str:
    ok = all(g["passed"] for g in gates)
    lines = [f"# {title}", "", f"**Result: {'PASS' if ok else 'FAIL'}**", "", "| Gate | Status | Detail |", "| --- | --- | --- |"]
    for g in gates:
        lines.append(f"| `{g['gate']}` | {'pass' if g['passed'] else '**FAIL**'} | {g['detail']} |")
    if strata:
        lines += ["", "## By stratum", "", "| Stratum | n | " + " | ".join(next(iter(strata.values()))["metrics"].keys()) + " |",
                  "| --- | ---: | " + " | ".join("---:" for _ in next(iter(strata.values()))["metrics"]) + " |"]
        for s, v in strata.items():
            lines.append(f"| {s} | {v['n']} | " + " | ".join(f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]" for m in v["metrics"].values()) + " |")
    if regressions:
        lines += ["", f"## Newly failing cases ({len(regressions)})", ""] + [f"- `{r}`" for r in regressions[:50]]
    for k, v in extra.items():
        lines += ["", f"## {k}", "", "```json", json.dumps(v, indent=2, ensure_ascii=False)[:4000], "```"]
    return "\n".join(lines) + "\n"


def to_junit(suite: str, gates: list[dict[str, Any]], regressions: list[str]) -> str:
    cases = []
    for g in gates:
        body = "" if g["passed"] else f'<failure message="{escape(g["detail"], {chr(34): "&quot;"})}"/>'
        cases.append(f'  <testcase classname="{suite}.gates" name="{escape(g["gate"])}">{body}</testcase>')
    for r in regressions:
        cases.append(f'  <testcase classname="{suite}.regressions" name="{escape(r)}"><failure message="hit dropped from 1 to 0 vs baseline"/></testcase>')
    fails = sum(1 for g in gates if not g["passed"]) + len(regressions)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<testsuite name="{suite}" tests="{len(cases)}" failures="{fails}">\n' + "\n".join(cases) + "\n</testsuite>\n"


def write(path: Optional[str], text: str) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
