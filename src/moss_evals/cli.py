"""moss-evals CLI.

  moss-evals build-dataset --failures prod_failures.jsonl --out regression.jsonl
  moss-evals retrieve --dataset regression.jsonl --index my-index --alpha 0.6 --out run.json
  moss-evals retrieve --dataset regression.jsonl --corpus corpus.jsonl --model moss-minilm --out run.json [--translate]
  moss-evals judge --dataset regression.jsonl --run run.json --out judged.json
  moss-evals report --dataset regression.jsonl --run run.json [--judged judged.json] [--baseline base.json] \
                    --thresholds thresholds.json --md report.md --junit junit.xml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import dataset as ds
from . import metrics, report
from .runner import RetrievalConfig, Target, score_run


def _creds() -> tuple[str, str]:
    from moss_eval.common import load_credentials
    return load_credentials()


def _qrels(cases: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {c["id"]: c["qrels"] for c in cases if c.get("qrels")}


async def _translate_factory(llm, model: str, scope: str = "non-latin"):
    cache: dict[str, str] = {}

    async def transform(q: str) -> str:
        if scope == "non-latin" and ds.detect_script(q) == "latin":
            return q
        if q not in cache:
            resp = await llm.respond({
                "model": model,
                "instructions": "Translate the user's search query into concise English. Keep identifiers, numbers and product codes unchanged. Output only the translation.",
                "input": q, "max_output_tokens": 80,
            }, headers={"X-Moss-Cache": "off"})
            cache[q] = llm.text_of(resp).strip() or q
        return cache[q]
    return transform


async def cmd_retrieve(a: argparse.Namespace) -> int:
    from moss import DocumentInfo, MossClient
    from .judge import MossLLM

    pid, pkey = _creds()
    cases = ds.read_jsonl(a.dataset)
    queries = {c["id"]: c["query"] for c in cases if c.get("qrels")}
    client = MossClient(pid, pkey)
    if a.index:
        target = await Target.from_index(client, a.index)
    else:
        docs = [DocumentInfo(id=d["id"], text=d["text"], metadata=d.get("metadata")) for d in ds.read_jsonl(a.corpus)]
        target = await Target.from_corpus(client, a.session_name, docs, model_id=a.model)
    llm = MossLLM(pid, pkey) if a.translate else None
    transform = await _translate_factory(llm, a.translate_model, a.translate_scope) if llm else None
    cfg = RetrievalConfig(alpha=a.alpha, top_k=a.top_k, label=a.label or f"alpha={a.alpha}")
    run = await target.run(queries, cfg, transform=transform)
    await target.close()
    await client.close()
    if llm:
        await llm.close()
    scored = score_run(run, _qrels(cases))
    out = {"target": target.info, **run.to_json(), "texts": run.texts, "summary": scored}
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    m = scored["metrics"]
    print(f"[{cfg.label}] hit-free summary: recall@5={m['recall@5']['mean']:.3f} mrr@10={m['mrr@10']['mean']:.3f} "
          f"ndcg@10={m['ndcg@10']['mean']:.3f} p50={scored['latency_ms']['p50']}ms -> {a.out}")
    return 0


async def cmd_judge(a: argparse.Namespace) -> int:
    from .judge import MossLLM

    pid, pkey = _creds()
    cases = {c["id"]: c for c in ds.read_jsonl(a.dataset)}
    run = json.loads(Path(a.run).read_text())
    llm = MossLLM(pid, pkey)
    out = {}
    ids = [q for q in run["ranked"] if q in cases][: a.limit or None]
    for qid in ids:
        passages = run["texts"][qid][: a.context_k]
        answer = await llm.generate(cases[qid]["query"], passages, model=a.gen_model)
        verdict = await llm.judge(cases[qid]["query"], passages, answer, cases[qid].get("reference_answer"), model=a.judge_model)
        out[qid] = {"answer": answer, **verdict}
        print(f"  {qid}: faith={verdict.get('faithfulness')} correct={verdict.get('correct')} abstained={verdict.get('abstained')}")
    await llm.close()
    Path(a.out).write_text(json.dumps({"gen_model": a.gen_model, "judge_model": a.judge_model, "cases": out, "usage": llm.usage}, indent=2, ensure_ascii=False))
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    cases = ds.build_regression_dataset(ds.read_jsonl(a.failures))
    ds.write_jsonl(a.out, cases)
    tags: dict[str, int] = {}
    for c in cases:
        for t in c["tags"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"{len(cases)} regression cases -> {a.out}; tags={tags}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    cases = ds.read_jsonl(a.dataset)
    qrels = _qrels(cases)
    run = json.loads(Path(a.run).read_text())
    current = report.per_case_table(run, qrels, a.k)
    if a.judged:
        judged = json.loads(Path(a.judged).read_text())["cases"]
        for qid, v in judged.items():
            row = current.setdefault(qid, {})
            row["faithfulness"] = v.get("faithfulness")
            if v.get("correct") is not None:
                row["answer_correct"] = 1.0 if v["correct"] else 0.0
    baseline = None
    if a.baseline:
        b = json.loads(Path(a.baseline).read_text())
        baseline = report.per_case_table(b, qrels, a.k)
    thresholds = json.loads(Path(a.thresholds).read_text())
    gates = report.gate(current, thresholds, baseline)
    regressions = report.newly_failing(current, baseline, f"hit@{a.k}") if baseline else []
    strata: dict[str, Any] = {}
    for c in cases:
        if c["id"] not in current:
            continue
        for t in c["tags"]:
            if t.startswith("stratum:"):
                strata.setdefault(t, []).append(c["id"])
    strata_out = {s: {"n": len(ids), "metrics": {m: metrics.bootstrap_ci([current[i][m] for i in ids]) for m in (f"hit@{a.k}", f"ndcg@{a.k}")}} for s, ids in strata.items()}
    md = report.to_markdown(a.title, gates, regressions, strata_out, {})
    report.write(a.md, md)
    report.write(a.junit, report.to_junit("moss_evals", gates, regressions))
    print(md)
    return 0 if all(g["passed"] for g in gates) and not (regressions and a.fail_on_regression) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="moss-evals", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-dataset")
    b.add_argument("--failures", required=True)
    b.add_argument("--out", required=True)

    r = sub.add_parser("retrieve")
    r.add_argument("--dataset", required=True)
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--index")
    src.add_argument("--corpus")
    r.add_argument("--model", default="moss-minilm")
    r.add_argument("--session-name", default="moss-evals-session")
    r.add_argument("--alpha", type=float, default=0.6)
    r.add_argument("--top-k", type=int, default=10)
    r.add_argument("--label")
    r.add_argument("--translate", action="store_true", help="Translate non-Latin-script queries to English before retrieval")
    r.add_argument("--translate-model", default="moss/fastlane-1")
    r.add_argument("--translate-scope", choices=["non-latin", "all"], default="non-latin",
                   help="non-latin: only translate non-Latin-script queries (cheap); all: every query (catches es/fr/de)")
    r.add_argument("--out", required=True)

    j = sub.add_parser("judge")
    j.add_argument("--dataset", required=True)
    j.add_argument("--run", required=True)
    j.add_argument("--gen-model", default="moss/fastlane-1")
    j.add_argument("--judge-model", default="openai/gpt-6-luna")
    j.add_argument("--context-k", type=int, default=3)
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--out", required=True)

    rp = sub.add_parser("report")
    rp.add_argument("--dataset", required=True)
    rp.add_argument("--run", required=True)
    rp.add_argument("--judged")
    rp.add_argument("--baseline")
    rp.add_argument("--thresholds", required=True)
    rp.add_argument("--k", type=int, default=5)
    rp.add_argument("--title", default="moss-evals report")
    rp.add_argument("--md")
    rp.add_argument("--junit")
    rp.add_argument("--fail-on-regression", action="store_true")

    c = sub.add_parser("calibrate", help="Agreement between the LLM judge and a deterministic reference-fact check")
    c.add_argument("--dataset", required=True)
    c.add_argument("--judged", required=True)
    c.add_argument("--out")

    a = p.parse_args(argv)
    if a.cmd == "calibrate":
        from .calibrate import cmd_calibrate
        return cmd_calibrate(a)
    if a.cmd == "build-dataset":
        return cmd_build(a)
    if a.cmd == "report":
        return cmd_report(a)
    return asyncio.run({"retrieve": cmd_retrieve, "judge": cmd_judge}[a.cmd](a))


if __name__ == "__main__":
    sys.exit(main())
