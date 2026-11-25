"""Suite 13b: start_url admission probes (re-run of ISS-02 plus new schemes). Anything accepted is cancelled immediately."""
import asyncio
import time

import httpx

from moss_eval.common import load_credentials, save_log

BASE = "https://inference.moss.dev/v1"
URLS = {
    "loopback_127": "http://127.0.0.1:8080/",
    "localhost": "http://localhost:3000/",
    "metadata_169_254": "http://169.254.169.254/latest/meta-data/",
    "rfc1918_10": "http://10.0.0.1/",
    "rfc1918_192_168": "http://192.168.1.1/",
    "ipv6_loopback": "http://[::1]:8080/",
    "decimal_ip_loopback": "http://2130706433/",
    "hex_ip_loopback": "http://0x7f000001/",
    "nip_io_loopback_dns": "http://127.0.0.1.nip.io/",
    "metadata_gcp_host": "http://metadata.google.internal/computeMetadata/v1/",
    "data_url": "data:text/html,<h1>hi</h1>",
    "file_url": "file:///etc/passwd",
    "javascript_url": "javascript:alert(1)",
    "ftp_url": "ftp://example.com/",
    "control_public": "https://example.com/",
}


async def main():
    pid, pkey = load_credentials()
    hdrs = {"Authorization": f"Bearer {pid}:{pkey}", "Content-Type": "application/json"}
    out = {}
    async with httpx.AsyncClient(timeout=30) as h:
        for label, url in URLS.items():
            body = {"goal": "Read the page title.", "environment": {"type": "browser", "start_url": url}, "model": "moss-fast",
                    "limits": {"max_cost_usd": 0.01, "max_runtime_seconds": 60}}
            r = await h.post(f"{BASE}/tasks", headers=hdrs, json=body)
            e = {"url": url, "status": r.status_code, "body": r.text[:240]}
            if r.status_code == 202:
                tid = r.json()["id"]
                c = await h.post(f"{BASE}/tasks/{tid}/cancel", headers=hdrs)
                await asyncio.sleep(3)
                t = (await h.get(f"{BASE}/tasks/{tid}", headers=hdrs)).json()
                e.update({"task_id": tid, "cancel_http": c.status_code, "final_status": t.get("status"),
                          "total_usd": ((t.get("usage") or {}).get("cost") or {}).get("total_usd"), "error": t.get("error")})
            out[label] = e
            print(f"{label:22s} {r.status_code} {e.get('final_status','')} {e['body'][:110]}", flush=True)
    print("Saved:", save_log("13b_tasks_url_admission.json", {"probed_at_local": time.strftime("%Y-%m-%d %H:%M %Z"), "probes": out}, [pkey]))


if __name__ == "__main__":
    asyncio.run(main())
