"""Suite 12b: why does gpt-6-luna return empty text under W1 (re-appended user turn)? Inspect output item types,
then test W1 + tool_choice='none' on turn 1 (3 cases each)."""
import asyncio, importlib, json
import httpx
from moss_eval.common import load_credentials, save_log, estimate_inference_cost
m = importlib.import_module("moss_eval.12_bug_workarounds")

async def main():
    pid, pkey = load_credentials(); auth = f"Bearer {pid}:{pkey}"
    rows, cost = [], 0.0
    async with httpx.AsyncClient(timeout=120) as h:
        for variant in ["W1_auto", "W1_tool_choice_none"]:
            for cid in ["d01", "d03", "a01"]:
                case = m.EVAL[cid]
                question = f"{case['query']} Also, what's the status of order 4471?"
                user = {"role": "user", "content": question}
                hd = {"X-Moss-Indexes": m.IDX}
                r0, _ = await m.post(h, auth, {"model": "openai/gpt-6-luna", "instructions": m.INSTR, "input": [user], "tools": [m.TOOL], "tool_choice": "required", "max_output_tokens": 200}, hd)
                d0 = r0.json(); fc = next(x for x in d0["output"] if x.get("type") == "function_call")
                items = [user, fc, {"type": "function_call_output", "call_id": fc["call_id"], "output": json.dumps({"order_id": "4471", "status": "out for delivery", "eta_minutes": 12})}, {"role": "user", "content": question}]
                b1 = {"model": "openai/gpt-6-luna", "instructions": m.INSTR, "input": items, "tools": [m.TOOL], "max_output_tokens": 220}
                if variant == "W1_tool_choice_none":
                    b1["tool_choice"] = "none"
                r1, _ = await m.post(h, auth, b1, hd)
                d1 = r1.json() if r1.status_code == 200 else {"error": r1.text[:300]}
                for d in (d0, d1):
                    u = d.get("usage") or {}
                    cost += estimate_inference_cost("openai/gpt-6-luna", u.get("input_tokens", 0), u.get("output_tokens", 0), 0, 1)
                final = m.text_of(d1)
                rows.append({"variant": variant, "case": cid, "status": r1.status_code, "turn1_output_types": [x.get("type") for x in d1.get("output", [])],
                             "turn1_function_calls": [x.get("arguments") for x in d1.get("output", []) if x.get("type") == "function_call"],
                             "final": final[:300], "correct": m.fact_ok(final, case["fact"]), "mentions_status": "out for delivery" in final.lower(),
                             "error": d1.get("error")})
                print(rows[-1], flush=True)
    print("Saved:", save_log("12b_luna_w1_probe.json", {"rows": rows, "est_cost_usd": round(cost, 5)}, [pkey]), round(cost, 4))
asyncio.run(main())
