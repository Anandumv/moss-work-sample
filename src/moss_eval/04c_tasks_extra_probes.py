"""
Suite 4C: Extra Tasks API probes:
- Save screenshot from task1_happy to verify what the cloud browser rendered
- Probe video endpoint / files path for session-1.mp4
- Run one moss-balanced extraction on books.toscrape.com while polling every 1s for live_url
"""
import asyncio
import time
import httpx
from moss import MossInference
from moss_eval.common import load_credentials, save_log

BASE_URL = "https://inference.moss.dev/v1"
TASK1_ID = "task_d1a69ea62c62402e83a66dadf0efba72"


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    hdrs = {"Authorization": auth, "Content-Type": "application/json"}
    out = {}

    async with MossInference(pid, pkey) as client:
        # 1. Download screenshot from task1_happy to inspect visually
        fc = await client.tasks.files.content(TASK1_ID, "screenshots/runner/00001-step-2.png")
        shot_path = "/tmp/moss_task1_step2.png"
        with open(shot_path, "wb") as f:
            f.write(fc.data)
        out["task1_screenshot_saved"] = shot_path

    # 2. Probe video / session-1.mp4 access on task1_happy
    async with httpx.AsyncClient(timeout=20.0) as h:
        video_probes = {}
        for path in [
            f"/tasks/{TASK1_ID}/files/session-1.mp4",
            f"/tasks/{TASK1_ID}/files/video/session-1.mp4",
            f"/tasks/{TASK1_ID}/video",
        ]:
            r = await h.get(f"{BASE_URL}{path}", headers=hdrs)
            video_probes[path] = {"status": r.status_code, "content_type": r.headers.get("content-type"), "len": len(r.content), "preview": r.text[:200] if "json" in r.headers.get("content-type", "") else None}
        out["video_probes"] = video_probes

        # 3. Live moss-balanced extraction on books.toscrape.com with 1s poll for live_url
        r_create = await h.post(
            f"{BASE_URL}/tasks",
            headers=hdrs,
            json={
                "goal": "Extract the title and GBP price of the very first book shown on the page.",
                "environment": {"type": "browser", "start_url": "https://books.toscrape.com/", "live_view": True},
                "model": "moss-balanced",
                "effort": "low",
                "limits": {"max_runtime_seconds": 90, "max_cost_usd": 0.15},
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "price_gbp": {"type": "number"},
                    },
                    "required": ["title", "price_gbp"],
                    "additionalProperties": False,
                },
            },
        )
        t_data = r_create.json()
        tid = t_data["id"]
        live_urls_seen = []
        statuses_seen = []
        t0 = time.perf_counter()
        while True:
            rg = await h.get(f"{BASE_URL}/tasks/{tid}", headers=hdrs)
            tj = rg.json()
            statuses_seen.append((round(time.perf_counter() - t0, 1), tj.get("status"), tj.get("live_url")))
            if tj.get("live_url"):
                live_urls_seen.append(tj.get("live_url"))
            if tj.get("status") not in ("queued", "running"):
                out["books_balanced_task"] = tj
                break
            await asyncio.sleep(1.0)
        out["books_poll_timeline"] = statuses_seen
        out["live_urls_seen"] = live_urls_seen

    path = save_log("04c_tasks_extra_probes.json", out, [pkey])
    print("Saved:", path)
    print("Video probes:", out["video_probes"])
    print("live_urls_seen:", live_urls_seen)
    bt = out["books_balanced_task"]
    print("Books task:", bt.get("status"), bt.get("result"), bt.get("verification"), bt.get("usage"))


if __name__ == "__main__":
    asyncio.run(main())
