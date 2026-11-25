"""
Suite 4A: Tasks API Validation, Boundary, Security (SSRF/Path Traversal), and Error Probes.
Zero-cost synchronous checks; any probe that unexpectedly returns 202 is immediately cancelled.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from moss_eval.common import load_credentials, save_log

BASE_URL = "https://inference.moss.dev/v1"


async def probe(
    h: httpx.AsyncClient,
    method: str,
    path: str,
    headers: dict[str, str],
    body: Any = None,
    auto_cancel: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if method == "GET":
        r = await h.get(f"{BASE_URL}{path}", headers=headers)
    else:
        r = await h.post(f"{BASE_URL}{path}", headers=headers, json=body)
    dt = round((time.perf_counter() - t0) * 1000, 2)
    try:
        parsed = r.json()
    except Exception:
        parsed = r.text[:400]

    cancelled = None
    if auto_cancel and r.status_code in (200, 202) and isinstance(parsed, dict) and str(parsed.get("id", "")).startswith("task_"):
        tid = parsed["id"]
        rc = await h.post(f"{BASE_URL}/tasks/{tid}/cancel", headers=headers)
        cancelled = {"cancel_status": rc.status_code, "cancel_body": rc.json() if rc.headers.get("content-type", "").startswith("application/json") else rc.text[:200]}

    return {
        "status": r.status_code,
        "elapsed_ms": dt,
        "body": parsed,
        "auto_cancelled": cancelled,
    }


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    hdrs = {"Authorization": auth, "Content-Type": "application/json"}
    results: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=30.0) as h:
        # Auth checks
        results["bad_auth_create"] = await probe(h, "POST", "/tasks", {"Authorization": f"Bearer {pid}:wrong", "Content-Type": "application/json"}, {"goal": "x", "environment": {"type": "browser"}})
        results["no_auth_get"] = await probe(h, "GET", "/tasks/task_00000000000000000000000000000000", {})

        # Missing / invalid required fields
        results["empty_goal"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "", "environment": {"type": "browser"}})
        results["whitespace_goal"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "   \n ", "environment": {"type": "browser"}})
        results["over_20k_goal"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "a" * 20001, "environment": {"type": "browser"}})
        results["missing_env"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello"})
        results["bad_env_type"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "shell"}})

        # start_url validation & SSRF probes
        for label, url in [
            ("start_url_javascript", "javascript:alert(1)"),
            ("start_url_file", "file:///etc/passwd"),
            ("start_url_ftp", "ftp://example.com"),
            ("start_url_not_url", "not-a-url"),
            ("start_url_loopback_ssrf", "http://127.0.0.1:8080/"),
            ("start_url_metadata_ssrf", "http://169.254.169.254/latest/meta-data/"),
            ("start_url_localhost", "http://localhost:3000/"),
        ]:
            results[label] = await probe(
                h, "POST", "/tasks", hdrs,
                {"goal": "Read page", "environment": {"type": "browser", "start_url": url}, "limits": {"max_cost_usd": 0.01, "max_runtime_seconds": 60}},
            )

        # Model, effort, limits, unknown fields, payload size, schema
        results["bad_model"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "model": "nonexistent-model"})
        results["bad_effort"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "effort": "ultra"})
        results["runtime_too_low"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "limits": {"max_runtime_seconds": 10}})
        results["runtime_too_high"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "limits": {"max_runtime_seconds": 99999}})
        results["cost_too_low"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "limits": {"max_cost_usd": 0.001}})
        results["cost_negative"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "limits": {"max_cost_usd": -1}})
        results["cost_too_high"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "limits": {"max_cost_usd": 100}})
        results["unknown_top_field"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "unknown_xyz": True})
        results["oversized_metadata"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "metadata": {"blob": "x" * 4500}})
        results["schema_non_object_root"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "output_schema": {"type": "string"}})
        results["schema_malformed"] = await probe(h, "POST", "/tasks", hdrs, {"goal": "hello", "environment": {"type": "browser"}, "output_schema": {"type": "object", "properties": "bad"}})

        # >256 KiB body -> 413
        results["over_256kib_body"] = await probe(
            h, "POST", "/tasks", hdrs,
            {"goal": "hello", "environment": {"type": "browser"}, "metadata": {"big": "x" * 270_000}},
        )

        # Nonexistent & malformed task IDs
        results["nonexistent_task_id"] = await probe(h, "GET", "/tasks/task_00000000000000000000000000000000", hdrs)
        results["malformed_task_id"] = await probe(h, "GET", "/tasks/not-a-valid-id", hdrs)
        results["cancel_nonexistent"] = await probe(h, "POST", "/tasks/task_00000000000000000000000000000000/cancel", hdrs, {})
        results["files_traversal_probe"] = await probe(h, "GET", "/tasks/task_00000000000000000000000000000000/files/../../etc/passwd", hdrs)

    path = save_log("04a_tasks_validation_probes.json", results, [pkey])
    print("Saved:", path)
    for k, v in results.items():
        b = v["body"]
        msg = b.get("error", b) if isinstance(b, dict) else b
        ac = " [AUTO-CANCELLED!]" if v["auto_cancelled"] else ""
        print(f"[{k}]: HTTP {v['status']}{ac} -> {str(msg)[:140]}")


if __name__ == "__main__":
    asyncio.run(main())
