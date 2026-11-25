"""
Suite 8: Deep Browser Tasks Scenarios (inference.moss.dev/v1/tasks)
Tests multi-page navigation, interactive form submission, dynamic JS waiting,
login wall vs demo auth, nested/enum output_schema, model tier comparison
(moss-fast, moss-balanced, moss-advanced, deepseek/deepseek-v4.1-flash),
cost accounting, and 5-task concurrency + cancel race.
"""
import asyncio
import time
from typing import Any
import httpx
from moss_eval.common import load_credentials, save_log

BASE = "https://inference.moss.dev/v1"


def make_payload(
    goal: str,
    start_url: str,
    model: str = "moss-balanced",
    max_cost_usd: float = 0.25,
    max_runtime_seconds: int = 120,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p: dict[str, Any] = {
        "goal": goal,
        "environment": {"type": "browser", "start_url": start_url},
        "model": model,
        "limits": {"max_runtime_seconds": max_runtime_seconds, "max_cost_usd": max_cost_usd},
    }
    if output_schema is not None:
        p["output_schema"] = output_schema
    return p


async def run_task(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    name: str,
    payload: dict[str, Any],
    timeout_s: float = 150.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = await client.post(f"{BASE}/tasks", headers=headers, json=payload)
    submit_ms = round((time.perf_counter() - t0) * 1000, 2)
    if r.status_code != 202:
        return {
            "name": name,
            "submit_status": r.status_code,
            "submit_ms": submit_ms,
            "submit_body": r.text[:400],
        }
    task = r.json()
    tid = task["id"]
    statuses = [task.get("status")]
    while task.get("status") in ("queued", "running"):
        if time.perf_counter() - t0 > timeout_s:
            await client.post(f"{BASE}/tasks/{tid}/cancel", headers=headers)
            return {
                "name": name,
                "id": tid,
                "submit_status": 202,
                "timed_out": True,
                "statuses_seen": statuses,
                "elapsed_s": round(time.perf_counter() - t0, 2),
            }
        await asyncio.sleep(2.0)
        gr = await client.get(f"{BASE}/tasks/{tid}", headers=headers)
        task = gr.json()
        if task.get("status") != statuses[-1]:
            statuses.append(task.get("status"))

    elapsed_s = round(time.perf_counter() - t0, 2)
    files_res = await client.get(f"{BASE}/tasks/{tid}/files", headers=headers)
    files_j = files_res.json() if files_res.status_code == 200 else {"status": files_res.status_code}
    trace_res = await client.get(f"{BASE}/tasks/{tid}/trace", headers=headers)
    trace_j = trace_res.json() if trace_res.status_code == 200 else {"status": trace_res.status_code}

    return {
        "name": name,
        "id": tid,
        "submit_status": 202,
        "statuses_seen": statuses,
        "final_status": task.get("status"),
        "error": task.get("error"),
        "result": task.get("result"),
        "verification": task.get("verification"),
        "usage": task.get("usage"),
        "steps_count": len(trace_j.get("events") or []) if isinstance(trace_j, dict) else None,
        "elapsed_s": elapsed_s,
        "files_list": files_j.get("files") if isinstance(files_j, dict) else files_j,
        "trace_video": trace_j.get("video") if isinstance(trace_j, dict) else None,
    }


async def main() -> None:
    pid, pkey = load_credentials()
    headers = {"Authorization": f"Bearer {pid}:{pkey}", "Content-Type": "application/json"}

    quotes_schema = {
        "type": "object",
        "properties": {
            "quotes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"author": {"type": "string"}, "text": {"type": "string"}},
                    "required": ["author", "text"],
                },
            }
        },
        "required": ["quotes"],
    }

    scenarios: list[tuple[str, dict[str, Any]]] = [
        ("multipage_travel_books", make_payload(
            goal="Click the 'Travel' category in the left sidebar, then extract the first 3 books shown in Travel with title, numeric price in GBP, and whether it is in stock.",
            start_url="https://books.toscrape.com/",
            model="moss-balanced",
            max_cost_usd=0.25,
            output_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "currency": {"type": "string", "enum": ["GBP"]},
                    "books": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "price_gbp": {"type": "number"},
                                "in_stock": {"type": "boolean"},
                            },
                            "required": ["title", "price_gbp", "in_stock"],
                        },
                    },
                },
                "required": ["category", "currency", "books"],
            },
        )),
        ("interactive_form_httpbin", make_payload(
            goal="Fill the pizza order form: Customer name='Anandu MV', Telephone='021-555-0199', Email='eval@example.com', Size='Medium', Topping='Bacon', Delivery instructions='Leave at front door', then click Submit order and return the echoed form fields.",
            start_url="https://httpbin.org/forms/post",
            model="moss-balanced",
            max_cost_usd=0.25,
            output_schema={
                "type": "object",
                "properties": {
                    "custname": {"type": "string"},
                    "size": {"type": "string"},
                    "comments": {"type": "string"},
                },
                "required": ["custname", "size", "comments"],
            },
        )),
        ("dynamic_js_loading", make_payload(
            goal="Click the 'Start' button, wait for the loading spinner to finish, and extract the revealed text.",
            start_url="https://the-internet.herokuapp.com/dynamic_loading/1",
            model="moss-balanced",
            max_cost_usd=0.20,
            output_schema={
                "type": "object",
                "properties": {"revealed_text": {"type": "string"}},
                "required": ["revealed_text"],
            },
        )),
        ("login_wall_unauth_secure", make_payload(
            goal="Extract the secret area confirmation message from this page without logging in.",
            start_url="https://the-internet.herokuapp.com/secure",
            model="moss-balanced",
            max_cost_usd=0.15,
        )),
        ("login_demo_auth_flow", make_payload(
            goal="Log in using username 'tomsmith' and password 'SuperSecretPassword!', then extract the green confirmation banner text.",
            start_url="https://the-internet.herokuapp.com/login",
            model="moss-balanced",
            max_cost_usd=0.20,
            output_schema={
                "type": "object",
                "properties": {
                    "logged_in": {"type": "boolean"},
                    "banner_text": {"type": "string"},
                },
                "required": ["logged_in", "banner_text"],
            },
        )),
        ("js_quotes_fast", make_payload(
            goal="Extract the first 2 quotes and their authors from this JavaScript-rendered page.",
            start_url="https://quotes.toscrape.com/js/",
            model="moss-fast",
            max_cost_usd=0.15,
            output_schema=quotes_schema,
        )),
        ("js_quotes_balanced", make_payload(
            goal="Extract the first 2 quotes and their authors from this JavaScript-rendered page.",
            start_url="https://quotes.toscrape.com/js/",
            model="moss-balanced",
            max_cost_usd=0.20,
            output_schema=quotes_schema,
        )),
        ("js_quotes_advanced", make_payload(
            goal="Extract the first 2 quotes and their authors from this JavaScript-rendered page.",
            start_url="https://quotes.toscrape.com/js/",
            model="moss-advanced",
            max_cost_usd=0.45,
            output_schema=quotes_schema,
        )),
        ("js_quotes_deepseek_probe", make_payload(
            goal="Extract the first 2 quotes and their authors from this JavaScript-rendered page.",
            start_url="https://quotes.toscrape.com/js/",
            model="deepseek/deepseek-v4.1-flash",
            max_cost_usd=0.15,
            output_schema=quotes_schema,
        )),
    ]

    scenario_results = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for name, payload in scenarios:
            res = await run_task(client, headers, name, payload)
            scenario_results.append(res)
            print(
                f"[scenario] {name:28s} -> {res.get('final_status') or res.get('submit_status')} "
                f"| trace_events={res.get('steps_count')} | cost={(res.get('usage') or {}).get('cost_usd')} "
                f"| browser={(res.get('usage') or {}).get('browser_usd')} | verif={(res.get('verification') or {}).get('passed')}",
                flush=True,
            )

        # 5-task concurrency test
        t_c0 = time.perf_counter()
        conc_reqs = [
            client.post(
                f"{BASE}/tasks",
                headers=headers,
                json=make_payload("Return the H1 heading of this page.", "https://httpbin.org/html", "moss-fast", 0.05, 90),
            )
            for _ in range(5)
        ]
        conc_resps = await asyncio.gather(*conc_reqs)
        conc_ids = []
        conc_statuses = []
        conc_bodies = []
        for cr in conc_resps:
            conc_statuses.append(cr.status_code)
            if cr.status_code == 202:
                conc_ids.append(cr.json()["id"])
            else:
                conc_bodies.append(cr.text[:200])

        conc_final = []
        pending = set(conc_ids)
        while pending and (time.perf_counter() - t_c0 < 120):
            await asyncio.sleep(2.0)
            for tid in list(pending):
                gr = await client.get(f"{BASE}/tasks/{tid}", headers=headers)
                tj = gr.json()
                if tj.get("status") not in ("queued", "running"):
                    conc_final.append({
                        "id": tid,
                        "status": tj.get("status"),
                        "error": tj.get("error"),
                        "usage": tj.get("usage"),
                        "duration_s": (tj.get("usage") or {}).get("duration_s"),
                    })
                    pending.remove(tid)

        # Cancel race: t=0 and t=0.2s
        cancel_races = []
        for delay in (0.0, 0.2):
            r_cr = await client.post(
                f"{BASE}/tasks",
                headers=headers,
                json=make_payload("Return the page title.", "https://books.toscrape.com/", "moss-fast", 0.08, 90),
            )
            if r_cr.status_code == 202:
                tid = r_cr.json()["id"]
                if delay > 0:
                    await asyncio.sleep(delay)
                rc = await client.post(f"{BASE}/tasks/{tid}/cancel", headers=headers)
                await asyncio.sleep(2.5)
                gf = await client.get(f"{BASE}/tasks/{tid}", headers=headers)
                fj = gf.json()
                cancel_races.append({
                    "delay_s": delay,
                    "id": tid,
                    "cancel_http": rc.status_code,
                    "cancel_resp_status": rc.json().get("status") if rc.status_code == 200 else rc.text[:150],
                    "final_status": fj.get("status"),
                    "usage": fj.get("usage"),
                })

    total_tasks_usd = 0.0
    for item in scenario_results + conc_final + cancel_races:
        u = item.get("usage") or {}
        total_tasks_usd += (u.get("model_usd") or 0.0) + (u.get("browser_usd") or 0.0) + (u.get("traffic_usd") or 0.0)

    report = {
        "scenarios": scenario_results,
        "concurrency_5x": {
            "submit_statuses": conc_statuses,
            "reject_bodies": conc_bodies,
            "accepted_count": len(conc_ids),
            "wall_s": round(time.perf_counter() - t_c0, 2),
            "final": conc_final,
        },
        "cancel_races": cancel_races,
        "est_total_suite8_usd": round(total_tasks_usd, 6),
    }
    path = save_log("08_tasks_deep_scenarios.json", report, [pkey])
    print("Saved:", path)
    print("Concurrency 5x:", conc_statuses, "wall_s:", report["concurrency_5x"]["wall_s"])
    print("Cancel races:", cancel_races)
    print("Exact Suite 8 total USD (model+browser+traffic):", report["est_total_suite8_usd"])


if __name__ == "__main__":
    asyncio.run(main())
