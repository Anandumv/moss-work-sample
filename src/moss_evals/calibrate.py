"""Judge calibration: compare the LLM judge's `correct` verdict with a deterministic check
(all reference facts present in the answer, case/space-insensitive). Reports agreement,
Cohen's kappa and the disagreement cases so a human can adjudicate them."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from . import dataset as ds


def _norm(s: str) -> str:
    s = s.lower().replace("°", "")
    return re.sub(r"\s+", " ", s).strip()


def facts_present(answer: str, facts: list[str]) -> bool:
    """Every fact must appear; a fact may list alternatives separated by '|' (e.g. "1,500|1500")."""
    a = _norm(answer)
    return all(any(_norm(alt) in a for alt in f.split("|")) for f in facts)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def calibrate(cases: dict[str, dict[str, Any]], judged: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for qid, v in judged.items():
        facts = cases.get(qid, {}).get("facts")
        if not facts or v.get("correct") is None:
            continue
        rows.append({"id": qid, "judge": bool(v["correct"]), "deterministic": facts_present(v.get("answer", ""), facts),
                     "faithfulness": v.get("faithfulness"), "answer": v.get("answer", "")[:240]})
    j = [r["judge"] for r in rows]
    d = [r["deterministic"] for r in rows]
    tp = sum(x and y for x, y in zip(j, d))
    return {
        "n": len(rows),
        "agreement": round(sum(x == y for x, y in zip(j, d)) / len(rows), 4) if rows else None,
        "cohen_kappa": round(cohen_kappa(j, d), 4) if rows else None,
        "judge_true_det_false": [r for r in rows if r["judge"] and not r["deterministic"]],
        "judge_false_det_true": [r for r in rows if not r["judge"] and r["deterministic"]],
        "confusion": {"both_true": tp, "judge_only": sum(x and not y for x, y in zip(j, d)),
                      "det_only": sum(y and not x for x, y in zip(j, d)), "both_false": sum(not x and not y for x, y in zip(j, d))},
    }


def cmd_calibrate(a) -> int:
    cases = {c["id"]: c for c in ds.read_jsonl(a.dataset)}
    judged = json.loads(Path(a.judged).read_text())["cases"]
    out = calibrate(cases, judged)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"judge vs deterministic: n={out['n']} agreement={out['agreement']} kappa={out['cohen_kappa']} confusion={out['confusion']}")
    return 0
