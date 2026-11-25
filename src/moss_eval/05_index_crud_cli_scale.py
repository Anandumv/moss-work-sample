"""
Suite 5: Control Plane CRUD, Metadata Filters, Custom Embeddings, Sessions, Multi-Index,
2,000-Doc Scale Benchmark (local + gateway), and moss-cli Checks.
"""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any
import httpx
from moss import (
    DocumentInfo,
    GetDocumentsOptions,
    InvalidArgumentError,
    MossClient,
    MossError,
    MutationOptions,
    ParentGrouping,
    QueryOptions,
)
from moss_eval.common import load_credentials, save_log, summarize_latencies

BASE_URL = "https://inference.moss.dev/v1"


def run_cli(args: list[str], env_extra: dict[str, str]) -> dict[str, Any]:
    env = {**os.environ, **env_extra}
    t0 = time.perf_counter()
    p = subprocess.run(["moss", *args], capture_output=True, text=True, env=env, timeout=60)
    dt = round((time.perf_counter() - t0) * 1000, 2)
    parsed = None
    if p.stdout.strip():
        try:
            parsed = json.loads(p.stdout)
        except Exception:
            parsed = p.stdout.strip()[:500]
    return {"cmd": ["moss", *args], "returncode": p.returncode, "elapsed_ms": dt, "stdout": parsed, "stderr": p.stderr.strip()[:300]}


def make_custom_vec(seed: int, dim: int = 8) -> list[float]:
    vec = [((seed * 7 + i * 13) % 100) / 100.0 + 0.01 for i in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [round(x / norm, 6) for x in vec]


async def main() -> None:
    pid, pkey = load_credentials()
    env_creds = {"MOSS_PROJECT_ID": pid, "MOSS_PROJECT_KEY": pkey}
    report: dict[str, Any] = {}

    # 1. moss-cli checks
    cli_checks = {}
    cli_checks["version"] = run_cli(["version"], env_creds)
    cli_checks["index_list"] = run_cli(["--json", "index", "list"], env_creds)
    cli_checks["index_get_vac"] = run_cli(["--json", "index", "get", "voice-agent-context"], env_creds)
    cli_checks["query_vac"] = run_cli(["--json", "query", "voice-agent-context", "refund damaged item", "--top-k", "3"], env_creds)
    cli_checks["query_filtered"] = run_cli(
        ["--json", "query", "eval-grounding-v1", "express delivery", "--filter", '{"field":"region","condition":{"$eq":"auckland"}}'],
        env_creds,
    )
    cli_checks["whoami_no_token"] = run_cli(["whoami"], env_creds)
    report["cli_checks"] = cli_checks

    client = MossClient(pid, pkey)

    # Clean up leftover temp indexes if any
    existing = {idx.name for idx in await client.list_indexes()}
    for old in ["temp-crud-idx", "temp-custom-idx", "eval-scale-2k"]:
        if old in existing:
            await client.delete_index(old)

    # 2. Index CRUD + JobHandle + Upsert False vs True + Delete Docs + Delete Index
    crud_docs = [
        DocumentInfo(id="c1", text="Auckland hub cold-chain SKU 101", metadata={"city": "auckland", "temp": "4", "active": "true", "loc": "-36.8485,174.7633"}),
        DocumentInfo(id="c2", text="Wellington hub dry-goods SKU 202", metadata={"city": "wellington", "temp": "18", "active": "true", "loc": "-41.2865,174.7762"}),
        DocumentInfo(id="c3", text="Christchurch depot frozen SKU 303", metadata={"city": "christchurch", "temp": "-18", "active": "false", "loc": "-43.5321,172.6362"}),
        DocumentInfo(id="c4", text="Hamilton depot ambient SKU 404", metadata={"city": "hamilton", "active": "true"}),
    ]
    handle = await client.create_index("temp-crud-idx", crud_docs, "moss-minilm", wait=False)
    job_initial = await handle.status()
    job_done = await handle.wait()
    idx_info = await client.get_index("temp-crud-idx")

    # add_docs upsert=False with duplicate c1 + new c5
    await client.add_docs(
        "temp-crud-idx",
        [
            DocumentInfo(id="c1", text="OVERWRITTEN?", metadata={"city": "auckland", "temp": "99"}),
            DocumentInfo(id="c5", text="Dunedin micro-hub SKU 505", metadata={"city": "dunedin", "temp": "12", "active": "true"}),
        ],
        MutationOptions(upsert=False),
    )
    c1_after_no_upsert = await client.get_docs("temp-crud-idx", GetDocumentsOptions(doc_ids=["c1", "c5"]))

    # Now upsert=True
    await client.add_docs(
        "temp-crud-idx",
        [DocumentInfo(id="c1", text="Auckland hub cold-chain SKU 101 updated", metadata={"city": "auckland", "temp": "4", "active": "true", "loc": "-36.8485,174.7633"})],
        MutationOptions(upsert=True),
    )
    c1_after_upsert = await client.get_docs("temp-crud-idx", GetDocumentsOptions(doc_ids=["c1"]))

    # Verify cloud get_docs rejects filter
    cloud_get_docs_filter_err = None
    try:
        await client.get_docs("temp-crud-idx", GetDocumentsOptions(filter={"field": "city", "condition": {"$eq": "auckland"}}))
    except Exception as e:
        cloud_get_docs_filter_err = f"{type(e).__name__}: {e}"

    # Load and test metadata filters, min_score, top_k bounds
    await client.load_index("temp-crud-idx")
    q_eq = await client.query("temp-crud-idx", "depot SKU", QueryOptions(top_k=5, filter={"field": "city", "condition": {"$eq": "wellington"}}))
    q_ne = await client.query("temp-crud-idx", "depot SKU", QueryOptions(top_k=5, filter={"field": "city", "condition": {"$ne": "auckland"}}))
    q_range = await client.query("temp-crud-idx", "depot SKU", QueryOptions(top_k=5, filter={"$and": [{"field": "temp", "condition": {"$gte": -20}}, {"field": "temp", "condition": {"$lte": 10}}]}))
    q_near = await client.query("temp-crud-idx", "hub", QueryOptions(top_k=5, filter={"field": "loc", "condition": {"$near": "-36.85,174.76,50000"}}))
    q_missing_field = await client.query("temp-crud-idx", "SKU", QueryOptions(top_k=5, filter={"field": "nonexistent_field", "condition": {"$ne": "x"}}))
    q_min_score = await client.query("temp-crud-idx", "Auckland cold-chain", QueryOptions(top_k=5, min_score=0.5))
    q_top_k_0 = await client.query("temp-crud-idx", "SKU", QueryOptions(top_k=0))

    min_score_alpha0_err = None
    try:
        await client.query("temp-crud-idx", "SKU", QueryOptions(alpha=0.0, min_score=0.2))
    except Exception as e:
        min_score_alpha0_err = f"{type(e).__name__}: {e}"

    top_k_neg_err = None
    try:
        await client.query("temp-crud-idx", "SKU", QueryOptions(top_k=-1))
    except Exception as e:
        top_k_neg_err = f"{type(e).__name__}: {e}"

    # delete_docs + delete_index
    await client.delete_docs("temp-crud-idx", ["c4", "c5"])
    docs_after_del = await client.get_docs("temp-crud-idx")
    await client.unload_index("temp-crud-idx")
    deleted = await client.delete_index("temp-crud-idx")

    report["crud_and_filters"] = {
        "job_id": handle.job_id,
        "job_initial_status": str(job_initial.status),
        "job_done_status": str(job_done.status),
        "index_info": {"name": idx_info.name, "doc_count": idx_info.doc_count, "model_id": idx_info.model.id},
        "upsert_false_c1_text": [d.text for d in c1_after_no_upsert if d.id == "c1"],
        "upsert_false_added_c5": any(d.id == "c5" for d in c1_after_no_upsert),
        "upsert_true_c1_text": [d.text for d in c1_after_upsert],
        "cloud_get_docs_filter_error": cloud_get_docs_filter_err,
        "filter_eq_ids": [d.id for d in q_eq.docs],
        "filter_ne_ids": [d.id for d in q_ne.docs],
        "filter_range_ids": [d.id for d in q_range.docs],
        "filter_near_auckland_ids": [d.id for d in q_near.docs],
        "filter_missing_field_count": len(q_missing_field.docs),
        "min_score_0_5_scores": [(d.id, round(d.score, 4)) for d in q_min_score.docs],
        "top_k_0_count": len(q_top_k_0.docs),
        "min_score_alpha0_error": min_score_alpha0_err,
        "top_k_neg_error": top_k_neg_err,
        "docs_after_delete_docs_ids": [d.id for d in docs_after_del],
        "delete_index_result": deleted,
    }

    # 3. Custom embeddings + multi-index query + SessionIndex disk persistence & lock
    custom_docs = [
        DocumentInfo(id="v1", text="vector doc 1", embedding=make_custom_vec(1)),
        DocumentInfo(id="v2", text="vector doc 2", embedding=make_custom_vec(2)),
    ]
    await client.create_index("temp-custom-idx", custom_docs, model_id="custom")
    await client.load_index("temp-custom-idx")
    custom_res = await client.query("temp-custom-idx", "ignored", QueryOptions(top_k=2, embedding=make_custom_vec(1)))
    custom_no_vec_err = None
    try:
        await client.query("temp-custom-idx", "hello")
    except Exception as e:
        custom_no_vec_err = f"{type(e).__name__}: {e}"
    await client.unload_index("temp-custom-idx")
    await client.delete_index("temp-custom-idx")

    # Multi-index query
    loaded_multi = await client.load_indexes(["voice-agent-context", "eval-grounding-v1", "nonexistent-idx-check"])
    multi_res = await client.query_multi_index(loaded_multi.loaded, "refund damaged cold chain", QueryOptions(top_k=6, alpha=0.7))
    await client.unload_indexes(loaded_multi.loaded)

    # SessionIndex disk save/load, session.lock conflict, sort_by, group_by
    tmpdir = tempfile.mkdtemp(prefix="moss_sess_")
    try:
        s1 = await client.session("sess-persist-test", auto_load_on_init=False)
        await s1.add_docs([
            DocumentInfo(id="p1#0", text="Part 1 of order SLA", metadata={"parent": "sla", "ord": "1", "ts": "100"}),
            DocumentInfo(id="p1#1", text="Part 2 of order SLA", metadata={"parent": "sla", "ord": "2", "ts": "200"}),
            DocumentInfo(id="p2#0", text="Returns section", metadata={"parent": "ret", "ord": "1", "ts": "50"}),
        ])
        await s1.save_to_disk(tmpdir)
        # Try concurrent load_from_disk while s1 is alive -> expect MossError INTERNAL lock error
        s2 = await client.session("sess-persist-test", auto_load_on_init=False)
        lock_err = None
        try:
            await s2.load_from_disk(tmpdir)
        except Exception as e:
            lock_err = f"{type(e).__name__}: {e}"
        await s2.close()

        sess_grouped = await s1.get_docs(GetDocumentsOptions(sort_by="ts", ascending=True, group_by=ParentGrouping("parent", "ord")))
        await s1.close()

        s3 = await client.session("sess-persist-test", auto_load_on_init=False)
        restored_count = await s3.load_from_disk(tmpdir)
        await s3.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    report["custom_multi_session"] = {
        "custom_top_id": custom_res.docs[0].id if custom_res.docs else None,
        "custom_missing_vec_err": custom_no_vec_err,
        "multi_loaded": loaded_multi.loaded,
        "multi_failed": loaded_multi.failed_codes,
        "multi_hits": [(d.index_name, d.id, round(d.score, 4)) for d in multi_res.docs],
        "session_lock_err_while_open": lock_err,
        "session_grouped_ids": [d.id for d in sess_grouped],
        "session_restored_count": restored_count,
    }

    # 4. 2,000-Document Scale Benchmark (Local + Gateway)
    domains = ["grocery", "logistics", "billing", "coldchain", "returns", "api", "security", "support", "loyalty", "inventory"]
    scale_docs = [
        DocumentInfo(
            id=f"scale-{i:04d}",
            text=f"Policy record #{i} for {domains[i % 10]} operations in zone Z{i % 25}: threshold value is {100 + i} units and SLA code OP-{i:04d}.",
            metadata={"domain": domains[i % 10], "zone": f"Z{i % 25}", "seq": str(i)},
        )
        for i in range(2000)
    ]
    t_ing0 = time.perf_counter()
    await client.create_index("eval-scale-2k", scale_docs, "moss-minilm", wait=True)
    ingest_2k_ms = round((time.perf_counter() - t_ing0) * 1000, 2)

    t_ld0 = time.perf_counter()
    await client.load_index("eval-scale-2k")
    load_2k_ms = round((time.perf_counter() - t_ld0) * 1000, 2)

    local_2k_lat = []
    local_2k_filtered_lat = []
    for i in range(100):
        q = f"threshold value for {domains[i % 10]} zone Z{i % 25} OP-{i * 17:04d}"
        tq = time.perf_counter()
        await client.query("eval-scale-2k", q, QueryOptions(top_k=5, alpha=0.8))
        local_2k_lat.append((time.perf_counter() - tq) * 1000)

        tf = time.perf_counter()
        await client.query("eval-scale-2k", q, QueryOptions(top_k=5, filter={"field": "domain", "condition": {"$eq": domains[i % 10]}}))
        local_2k_filtered_lat.append((time.perf_counter() - tf) * 1000)

    # Gateway retrieve latency on 2,000-doc index (10 calls)
    gw_retrieve_2k = []
    gw_ttft_2k = []
    async with httpx.AsyncClient(timeout=60.0) as h:
        hdrs = {
            "Authorization": f"Bearer {pid}:{pkey}",
            "Content-Type": "application/json",
            "X-Moss-Indexes": "eval-scale-2k",
            "X-Moss-Cache": "off",
        }
        for i in range(10):
            t0 = time.perf_counter()
            ttft = None
            async with h.stream(
                "POST",
                f"{BASE_URL}/responses",
                headers=hdrs,
                json={"model": "moss/fastlane-1", "stream": True, "input": f"What is the threshold value for OP-{i * 100:04d}?", "max_output_tokens": 25},
            ) as resp:
                r_ms = resp.headers.get("x-moss-retrieve-ms")
                if r_ms is not None:
                    gw_retrieve_2k.append(float(r_ms))
                async for line in resp.aiter_lines():
                    if line.startswith("data:") and '"response.output_text.delta"' in line and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
            if ttft is not None:
                gw_ttft_2k.append(ttft)

    await client.unload_index("eval-scale-2k")
    await client.delete_index("eval-scale-2k")
    await client.close()

    report["scale_2k"] = {
        "doc_count": 2000,
        "ingest_build_ms": ingest_2k_ms,
        "cold_load_ms": load_2k_ms,
        "local_query_100x_ms": summarize_latencies(local_2k_lat),
        "local_filtered_100x_ms": summarize_latencies(local_2k_filtered_lat),
        "gateway_cold_retrieve_ms": gw_retrieve_2k[0] if gw_retrieve_2k else None,
        "gateway_warm_retrieve_9x_ms": summarize_latencies(gw_retrieve_2k[1:]),
        "gateway_warm_ttft_9x_ms": summarize_latencies(gw_ttft_2k[1:]),
    }

    path = save_log("05_index_crud_cli_scale.json", report, [pkey])
    print("Saved:", path)
    print("CLI index_list rc:", cli_checks["index_list"]["returncode"], "| query_vac rc:", cli_checks["query_vac"]["returncode"], "| whoami rc:", cli_checks["whoami_no_token"]["returncode"])
    print("CRUD & filters:", json.dumps(report["crud_and_filters"], indent=2))
    print("Custom/Multi/Session:", json.dumps(report["custom_multi_session"], indent=2))
    print("Scale 2K:", json.dumps(report["scale_2k"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
