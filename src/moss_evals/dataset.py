"""Turn production failure logs into a deduplicated, stratified regression dataset.

Input (JSONL, one production event per line), e.g. from a voice agent's trace store:
  {"query": "...", "retrieved_ids": ["a","b"], "expected_ids": ["c"], "answer": "...",
   "reference_answer": "...", "feedback": "thumbs_down" | "escalated" | "wrong_answer",
   "source": "prod-2026-11-24", "lang": optional}

Output (JSONL): one regression case per unique normalized query with qrels and tags.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

_ID_PATTERN = re.compile(r"\b([A-Z]{2,}-\d{2,}|\d{4}|v\d+(\.\d+)*|\d+\.\d+\.\d+\.\d+)\b")
_SCRIPT_BUCKETS = {
    "MALAYALAM": "ml", "DEVANAGARI": "hi", "HIRAGANA": "ja", "KATAKANA": "ja", "CJK": "zh",
    "ARABIC": "ar", "TAMIL": "ta", "CYRILLIC": "ru",
}


def normalize_query(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).lower().strip()
    q = re.sub(r"\s+", " ", q)
    # strip trailing punctuation *then* whitespace: French "valides ?" and "VALIDES " must collide
    return re.sub(r"[\s?.!,;:]+$", "", q)


def detect_script(q: str) -> str:
    counts: dict[str, int] = {}
    for ch in q:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            bucket = next((v for k, v in _SCRIPT_BUCKETS.items() if name.startswith(k)), "latin")
            counts[bucket] = counts.get(bucket, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


def tag_case(q: str) -> list[str]:
    tags = []
    script = detect_script(q)
    if script != "latin":
        tags.append(f"lang:{script}")
        tags.append("stratum:multilingual")
    if _ID_PATTERN.search(q) or re.search(r"\bnot\b|\binstead of\b|\bvs\.?\b", q, re.I):
        tags.append("stratum:adversarial")
    if not any(t.startswith("stratum:") for t in tags):
        tags.append("stratum:general")
    return tags


def case_id(norm_q: str) -> str:
    return "reg-" + hashlib.sha1(norm_q.encode()).hexdigest()[:10]


def build_regression_dataset(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for ev in events:
        q = (ev.get("query") or "").strip()
        expected = ev.get("expected_ids") or []
        if not q or not expected:
            # Without a labelled target we cannot score retrieval; keep for judge-only evals.
            if q and ev.get("reference_answer"):
                expected = []
            else:
                continue
        nq = normalize_query(q)
        cid = case_id(nq)
        c = cases.setdefault(cid, {
            "id": cid,
            "query": q,
            "qrels": {},
            "reference_answer": ev.get("reference_answer"),
            "facts": ev.get("facts"),
            "tags": tag_case(q),
            "occurrences": 0,
            "sources": [],
            "first_failure": {"retrieved_ids": ev.get("retrieved_ids"), "answer": ev.get("answer"), "feedback": ev.get("feedback")},
        })
        for d in expected:
            c["qrels"][d] = max(c["qrels"].get(d, 0), 1)
        c["occurrences"] += 1
        if ev.get("source") and ev["source"] not in c["sources"]:
            c["sources"].append(ev["source"])
        if not c.get("reference_answer") and ev.get("reference_answer"):
            c["reference_answer"] = ev["reference_answer"]
    # Most frequent production failures first: they are what the gate should protect.
    return sorted(cases.values(), key=lambda c: (-c["occurrences"], c["id"]))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
