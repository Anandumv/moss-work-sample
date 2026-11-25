"""
Suite 4B: Live Browser Tasks API Tests
Tests happy path (moss-fast, live_view, output_schema), Idempotency-Key replay & 422 conflict,
files.list/content & 409 while running, trace, MossInferenceSync, verification failure behavior,
cancellation (running + already-completed), forced budget_exceeded, and Malayalam/unreachable URL.
"""
import asyncio
import time
import uuid
from typing import Any
import httpx
from moss import MossInference, MossInferenceSync
from moss.inference import InferenceError
from moss_eval.common import load_credentials, save_log

BASE_URL = "https://inference.moss.dev/v1"


def summarize_task(t: Any) -> dict[str, Any]:
    raw = t.raw if hasattr(t, "raw") else t
    return {
        "id": raw.get("id"),
        "status": raw.get("status"),
        "model": raw.get("model"),
        "effort": raw.get("effort"),
        "live_url": raw.get("live_url"),
        "trace_available": raw.get("trace_available"),
        "video_available": raw.get("video_available"),
        "video_state": raw.get("video_state"),
        "cancel_requested": raw.get("cancel_requested"),
        "result": raw.get("result"),
        "verification": raw.get("verification"),
        "usage": raw.get("usage"),
        "error": raw.get("error"),
        "metadata": raw.get("metadata"),
        "created_at": raw.get("created_at"),
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "files_expire_at": raw.get("files_expire_at"),
    }


async def run_async_tests(pid: str, pkey: str) -> dict[str, Any]:
    auth = f"Bearer {pid}:{pkey}"
    hdrs = {"Authorization": auth, "Content-Type": "application/json"}
    out: dict[str, Any] = {}

    # 1. Idempotency-Key test + Happy Path Task (example.com, moss-fast, live_view, output_schema)
    idem_key = f"ws-idem-{uuid.uuid4().hex[:12]}"
    payload_happy = {
        "goal": "Read the main heading and the href URL of the 'More information...' link on this page.",
        "environment": {"type": "browser", "start_url": "https://example.com", "live_view": True},
        "model": "moss-fast",
        "effort": "low",
        "metadata": {"suite": "work_sample", "case": "happy_example"},
        "limits": {"max_runtime_seconds": 90, "max_cost_usd": 0.15},
        "output_schema": {
            "type": "object",
            "properties": {
                "heading": {"type": "string"},
                "link_url": {"type": "string"},
            },
            "required": ["heading", "link_url"],
            "additionalProperties": False,
        },
    }
    async with httpx.AsyncClient(timeout=30.0) as h:
        r1 = await h.post(f"{BASE_URL}/tasks", headers={**hdrs, "Idempotency-Key": idem_key}, json=payload_happy)
        r2 = await h.post(f"{BASE_URL}/tasks", headers={**hdrs, "Idempotency-Key": idem_key}, json=payload_happy)
        r3 = await h.post(
            f"{BASE_URL}/tasks",
            headers={**hdrs, "Idempotency-Key": idem_key},
            json={**payload_happy, "goal": "Different goal with same idempotency key"},
        )
        t1_json = r1.json()
        t1_id = t1_json["id"]
        out["idempotency"] = {
            "first_status": r1.status_code,
            "first_id": t1_id,
            "replay_status": r2.status_code,
            "replay_id": r2.json().get("id"),
            "replay_header": r2.headers.get("idempotent-replayed"),
            "same_id": t1_id == r2.json().get("id"),
            "collision_status": r3.status_code,
            "collision_body": r3.json(),
        }

        # While Task 1 is running, probe /files (expect 409) and /trace
        r_files_early = await h.get(f"{BASE_URL}/tasks/{t1_id}/files", headers=hdrs)
        r_trace_early = await h.get(f"{BASE_URL}/tasks/{t1_id}/trace", headers=hdrs)
        out["task1_mid_run_probes"] = {
            "files_early_status": r_files_early.status_code,
            "files_early_body": r_files_early.json(),
            "trace_early_status": r_trace_early.status_code,
            "trace_early_body": r_trace_early.json() if "json" in r_trace_early.headers.get("content-type", "") else r_trace_early.text[:200],
        }

    async with MossInference(pid, pkey) as client:
        t1_done = await client.tasks.wait(t1_id, timeout=110)
        files_list = await client.tasks.files.list(t1_id)
        downloaded_files = []
        for f in files_list.data[:5]:
            fc = await client.tasks.files.content(t1_id, f.path)
            downloaded_files.append({
                "path": f.path,
                "listed_bytes": f.bytes,
                "actual_bytes": len(fc.data),
                "content_type": fc.content_type,
            })

        trace_res: Any = None
        try:
            tr = await client.tasks.trace(t1_id)
            trace_res = {"ok": True, "type": type(tr).__name__, "len_or_keys": len(tr) if isinstance(tr, list) else list(tr.keys()) if isinstance(tr, dict) else str(tr)[:200]}
        except InferenceError as e:
            trace_res = {"ok": False, "status": e.status, "code": e.code, "message": e.message}

        # Cancel an already-completed task -> docs say "unchanged"
        cancel_done = await client.tasks.cancel(t1_id)

        out["task1_happy"] = {
            "task": summarize_task(t1_done),
            "files_count": len(files_list.data),
            "files_sample": downloaded_files,
            "trace": trace_res,
            "cancel_after_complete_status": cancel_done.status,
            "cancel_after_complete_requested": cancel_done.cancel_requested,
        }

        # 2. Concurrent submit (2 tasks) + cancellation (one immediately, one after 4s)
        t_c1, t_c2 = await asyncio.gather(
            client.tasks.create(
                goal="Browse Wikipedia and summarize 5 articles.",
                environment={"type": "browser", "start_url": "https://en.wikipedia.org/wiki/Main_Page"},
                model="moss-fast",
                limits={"max_runtime_seconds": 120, "max_cost_usd": 0.15},
            ),
            client.tasks.create(
                goal="Browse Wikipedia and list 10 featured articles with dates.",
                environment={"type": "browser", "start_url": "https://en.wikipedia.org/wiki/Main_Page"},
                model="moss-fast",
                limits={"max_runtime_seconds": 120, "max_cost_usd": 0.15},
            ),
        )
        c1_ack = await client.tasks.cancel(t_c1.id)
        await asyncio.sleep(4.0)
        c2_ack = await client.tasks.cancel(t_c2.id)
        c1_final = await client.tasks.wait(t_c1.id, timeout=45)
        c2_final = await client.tasks.wait(t_c2.id, timeout=45)
        out["cancellation_and_concurrency"] = {
            "immediate_cancel_ack": {"status": c1_ack.status, "cancel_requested": c1_ack.cancel_requested},
            "immediate_cancel_final": summarize_task(c1_final),
            "delayed_4s_cancel_ack": {"status": c2_ack.status, "cancel_requested": c2_ack.cancel_requested},
            "delayed_4s_cancel_final": summarize_task(c2_final),
        }

        # 3. Forced budget_exceeded test (max_cost_usd=0.01 on multi-step Hacker News task)
        t_budget = await client.tasks.run(
            goal="Go to Hacker News, open each of the top 5 stories, read the top 5 comments on each story, and compare their themes in detail.",
            environment={"type": "browser", "start_url": "https://news.ycombinator.com"},
            model="moss-balanced",
            effort="medium",
            limits={"max_runtime_seconds": 120, "max_cost_usd": 0.01},
            timeout=140,
        )
        out["forced_budget_cap_0_01"] = summarize_task(t_budget)

        # 4. Unreachable domain + Malayalam goal test
        t_unreach = await client.tasks.run(
            goal="ഈ പേജിന്റെ തലക്കെട്ട് വായിക്കുക. If the site cannot be reached, report the exact failure.",
            environment={"type": "browser", "start_url": "https://nonexistent-domain-moss-eval-999.invalid"},
            model="moss-fast",
            effort="low",
            limits={"max_runtime_seconds": 60, "max_cost_usd": 0.05},
            timeout=80,
        )
        out["unreachable_and_malayalam"] = summarize_task(t_unreach)

    return out


def run_sync_verification_test(pid: str, pkey: str) -> dict[str, Any]:
    # 5. MossInferenceSync + output_schema mismatch / impossible numbers test
    with MossInferenceSync(pid, pkey) as sync_client:
        t0 = time.perf_counter()
        done = sync_client.tasks.run(
            goal="Look at https://example.com only (do not leave this page). Report the exact Bitcoin USD price shown on example.com, or if no Bitcoin price exists on the page, set btc_price_usd to null or state clearly what the page contains.",
            environment={"type": "browser", "start_url": "https://example.com"},
            model="moss-fast",
            effort="low",
            limits={"max_runtime_seconds": 90, "max_cost_usd": 0.15},
            output_schema={
                "type": "object",
                "properties": {
                    "btc_price_usd": {"type": "number"},
                    "treasury_yield_pct": {"type": "number"},
                },
                "required": ["btc_price_usd", "treasury_yield_pct"],
                "additionalProperties": False,
            },
            timeout=110,
        )
        dt = round(time.perf_counter() - t0, 2)
        return {"elapsed_s": dt, "task": summarize_task(done)}


async def main() -> None:
    pid, pkey = load_credentials()
    results = await run_async_tests(pid, pkey)
    results["sync_schema_mismatch_probe"] = run_sync_verification_test(pid, pkey)

    # Sum observed task costs
    task_costs = []
    for k in ["task1_happy", "forced_budget_cap_0_01", "unreachable_and_malayalam", "sync_schema_mismatch_probe"]:
        t = results[k].get("task") if "task" in results[k] else results[k]
        c = ((t.get("usage") or {}).get("cost") or {}).get("total_usd")
        if c is not None:
            task_costs.append((k, c))
    for ck in ["immediate_cancel_final", "delayed_4s_cancel_final"]:
        c = ((results["cancellation_and_concurrency"][ck].get("usage") or {}).get("cost") or {}).get("total_usd")
        if c is not None:
            task_costs.append((ck, c))

    results["observed_task_costs_usd"] = task_costs
    results["total_observed_task_cost_usd"] = round(sum(c for _, c in task_costs), 6)

    path = save_log("04b_tasks_live_runs.json", results, [pkey])
    print("Saved:", path)
    print("Task costs:", task_costs, "| Total:", results["total_observed_task_cost_usd"])
    print("Idempotency:", results["idempotency"])
    print("Task1 mid-run probes:", results["task1_mid_run_probes"])
    print("Task1 happy:", results["task1_happy"]["task"]["status"], results["task1_happy"]["task"]["result"], results["task1_happy"]["task"]["verification"], "live_url=", results["task1_happy"]["task"]["live_url"], "files=", results["task1_happy"]["files_count"], "trace=", results["task1_happy"]["trace"])
    print("Cancel immediate:", results["cancellation_and_concurrency"]["immediate_cancel_final"]["status"], "cost=", results["cancellation_and_concurrency"]["immediate_cancel_final"]["usage"])
    print("Cancel 4s:", results["cancellation_and_concurrency"]["delayed_4s_cancel_final"]["status"], "cost=", results["cancellation_and_concurrency"]["delayed_4s_cancel_final"]["usage"])
    print("Budget cap 0.01:", results["forced_budget_cap_0_01"]["status"], results["forced_budget_cap_0_01"]["error"], results["forced_budget_cap_0_01"]["usage"])
    print("Unreachable/ML:", results["unreachable_and_malayalam"]["status"], results["unreachable_and_malayalam"]["error"], results["unreachable_and_malayalam"]["result"])
    print("Sync schema mismatch:", results["sync_schema_mismatch_probe"]["task"]["status"], results["sync_schema_mismatch_probe"]["task"]["result"], results["sync_schema_mismatch_probe"]["task"]["verification"])


if __name__ == "__main__":
    asyncio.run(main())
