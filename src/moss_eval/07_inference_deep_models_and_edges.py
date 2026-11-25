"""
Suite 7: Deep Inference Model Matrix, Parameter Quirks, Long-Context Stress,
Streaming Incomplete / Truncation, Tool-Calling / MCP SSRF, and Error Envelope Audit.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from moss_eval.common import (
    estimate_inference_cost,
    load_credentials,
    save_log,
    summarize_latencies,
)

BASE_URL = "https://inference.moss.dev/v1"
IDX = "eval-grounding-v1"
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


async def stream_response(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    events: list[str] = []
    final_resp: dict[str, Any] = {}
    status = None
    resp_headers: dict[str, str] = {}
    raw_err = None

    async with client.stream("POST", f"{BASE_URL}/responses", headers=headers, json=payload) as r:
        status = r.status_code
        resp_headers = {k.lower(): v for k, v in r.headers.items()}
        if status != 200:
            raw_err = (await r.aread()).decode("utf-8", "replace")
            return {
                "status": status,
                "error": raw_err[:500],
                "headers": {k: v for k, v in resp_headers.items() if k.startswith("x-moss") or k == "retry-after"},
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            ev = json.loads(raw)
            et = ev.get("type", "")
            events.append(et)
            if et == "response.output_text.delta":
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                text_parts.append(ev.get("delta") or "")
            elif et in ("response.completed", "response.incomplete", "response.failed"):
                final_resp = ev.get("response") or {}

    total_ms = (time.perf_counter() - t0) * 1000
    usage = final_resp.get("usage") or {}
    return {
        "status": status,
        "response_status": final_resp.get("status"),
        "incomplete_details": final_resp.get("incomplete_details"),
        "ttft_ms": round(ttft, 2) if ttft is not None else None,
        "total_ms": round(total_ms, 2),
        "retrieve_ms": float(resp_headers["x-moss-retrieve-ms"]) if "x-moss-retrieve-ms" in resp_headers else None,
        "text": "".join(text_parts),
        "usage": usage,
        "event_counts": {k: events.count(k) for k in set(events)},
        "request_id": resp_headers.get("x-request-id"),
        "response_id": final_resp.get("id"),
    }


async def run_model_matrix(client: httpx.AsyncClient, auth: str) -> dict[str, Any]:
    configs = [
        ("moss/fastlane-1", "moss/fastlane-1", {}),
        ("openai/gpt-6-luna", "openai/gpt-6-luna", {}),
        ("openai/gpt-6.1-terra", "openai/gpt-6.1-terra", {}),
        ("anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5.5", {}),
        ("anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5.5", {}),
        ("google/gemini-3.5-flash-lite", "google/gemini-3.5-flash-lite", {}),
        ("google/gemini-3.8-flash (default)", "google/gemini-3.8-flash", {}),
        ("google/gemini-3.8-flash (effort=none)", "google/gemini-3.8-flash", {"reasoning": {"effort": "none"}}),
        ("deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash", {}),
        ("qwen/qwen3.8-flash", "qwen/qwen3.8-flash", {}),
        ("z-ai/glm-5.3-flash", "z-ai/glm-5.3-flash", {}),
        ("moonshotai/kimi-k3", "moonshotai/kimi-k3", {}),
        ("minimax/minimax-m3.1-flash", "minimax/minimax-m3.1-flash", {}),
    ]

    results: dict[str, Any] = {}
    total_cost = 0.0
    for label, model_id, extra in configs:
        ttfts, totals, retrieves = [], [], []
        runs = []
        cfg_cost = 0.0
        for i in range(3):
            hdrs = {
                "Authorization": auth,
                "Content-Type": "application/json",
                "X-Moss-Indexes": IDX,
                "X-Moss-Cache": "off",
            }
            payload = {
                "model": model_id,
                "stream": True,
                "instructions": "Answer concisely using ONLY the retrieved context.",
                "input": "What is the exact maximum refund for spoiled produce in Wellington?",
                "max_output_tokens": 120,
                **extra,
            }
            r = await stream_response(client, hdrs, payload)
            u = r.get("usage") or {}
            cost = estimate_inference_cost(
                model_id,
                u.get("input_tokens", 0),
                u.get("output_tokens", 0),
                (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                1 if r.get("status") == 200 else 0,
            )
            cfg_cost += cost
            total_cost += cost
            if r.get("ttft_ms") is not None:
                ttfts.append(r["ttft_ms"])
            if r.get("total_ms") is not None:
                totals.append(r["total_ms"])
            if r.get("retrieve_ms") is not None:
                retrieves.append(r["retrieve_ms"])
            ok = "45" in (r.get("text") or "")
            runs.append({
                "run": i,
                "status": r.get("status"),
                "response_status": r.get("response_status"),
                "ttft_ms": r.get("ttft_ms"),
                "total_ms": r.get("total_ms"),
                "retrieve_ms": r.get("retrieve_ms"),
                "usage": u,
                "fact_ok": ok,
                "text_preview": (r.get("text") or r.get("error") or "")[:160],
            })
            await asyncio.sleep(0.15)
        results[label] = {
            "model": model_id,
            "ttft_ms": summarize_latencies(ttfts) if ttfts else None,
            "total_ms": summarize_latencies(totals) if totals else None,
            "retrieve_ms": summarize_latencies(retrieves) if retrieves else None,
            "accuracy": f"{sum(1 for x in runs if x['fact_ok'])}/{len(runs)}",
            "est_cost_usd": round(cfg_cost, 6),
            "runs": runs,
        }
    results["total_matrix_cost_usd"] = round(total_cost, 6)
    return results


async def run_param_matrix(client: httpx.AsyncClient, auth: str) -> dict[str, Any]:
    cases = [
        ("anthropic_haiku_temperature", {"model": "anthropic/claude-haiku-5.5", "input": "Say ok.", "temperature": 0.2, "max_output_tokens": 16}),
        ("anthropic_haiku_top_p", {"model": "anthropic/claude-haiku-5.5", "input": "Say ok.", "top_p": 0.9, "max_output_tokens": 16}),
        ("anthropic_sonnet_tool_required", {
            "model": "anthropic/claude-sonnet-5.5",
            "input": "Check weather for Auckland.",
            "tools": [{"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}],
            "tool_choice": "required",
            "max_output_tokens": 64,
        }),
        ("anthropic_haiku_json_object", {"model": "anthropic/claude-haiku-5.5", "input": "Return JSON {\"ok\":true}", "text": {"format": {"type": "json_object"}}, "max_output_tokens": 32}),
        ("openai_luna_temperature", {"model": "openai/gpt-6-luna", "input": "Say ok.", "temperature": 0.5, "max_output_tokens": 16}),
        ("openai_luna_reasoning_low", {"model": "openai/gpt-6-luna", "input": "Say ok.", "reasoning": {"effort": "low"}, "max_output_tokens": 64}),
        ("openai_luna_seed", {"model": "openai/gpt-6-luna", "input": "Say ok.", "seed": 42, "max_output_tokens": 16}),
        ("gemini38_effort_invalid", {"model": "google/gemini-3.8-flash", "input": "Say ok.", "reasoning": {"effort": "ultra"}, "max_output_tokens": 16}),
        ("fastlane_temperature", {"model": "moss/fastlane-1", "input": "Say ok.", "temperature": 0.0, "max_output_tokens": 16}),
        ("fastlane_reasoning_low", {"model": "moss/fastlane-1", "input": "Say ok.", "reasoning": {"effort": "low"}, "max_output_tokens": 32}),
        ("fastlane_input_image", {
            "model": "moss/fastlane-1",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this image in 5 words."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{TINY_PNG_B64}"},
            ]}],
            "max_output_tokens": 32,
        }),
        ("gemini38_input_image", {
            "model": "google/gemini-3.8-flash",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this image in 5 words."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{TINY_PNG_B64}"},
            ]}],
            "max_output_tokens": 48,
        }),
    ]
    out: dict[str, Any] = {}
    hdrs = {"Authorization": auth, "Content-Type": "application/json"}
    for name, body in cases:
        r = await client.post(f"{BASE_URL}/responses", headers=hdrs, json=body)
        resp_j = None
        try:
            resp_j = r.json()
        except Exception:
            pass
        out[name] = {
            "status": r.status_code,
            "body_preview": (r.text or "")[:300],
            "error": resp_j.get("error") if isinstance(resp_j, dict) else None,
        }
        await asyncio.sleep(0.1)
    return out


async def run_long_context(client: httpx.AsyncClient, auth: str) -> dict[str, Any]:
    filler_unit = "Standard warehouse operational telemetry logs confirm nominal chiller operation across bay A1 through D9 during routine overnight checks. "
    targets = [
        ("10k_tokens", 420, "NEEDLE-ALPHA-7391"),
        ("50k_tokens", 2100, "NEEDLE-BETA-4820"),
        ("100k_tokens", 4200, "NEEDLE-GAMMA-9104"),
    ]
    out: dict[str, Any] = {}
    cost = 0.0
    for label, reps, needle in targets:
        half = reps // 2
        prompt = (
            (filler_unit * half)
            + f"\nCRITICAL OVERRIDE CODE FOR AUDIT: The vault audit passkey is {needle}.\n"
            + (filler_unit * half)
            + "\nQuestion: What is the exact vault audit passkey mentioned above? Reply with only the code."
        )
        hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off"}
        r = await stream_response(client, hdrs, {
            "model": "moss/fastlane-1",
            "stream": True,
            "input": prompt,
            "max_output_tokens": 40,
        })
        u = r.get("usage") or {}
        c = estimate_inference_cost("moss/fastlane-1", u.get("input_tokens", 0), u.get("output_tokens", 0), 0, 0)
        cost += c
        out[label] = {
            "status": r.get("status"),
            "ttft_ms": r.get("ttft_ms"),
            "total_ms": r.get("total_ms"),
            "usage": u,
            "needle_recalled": needle in (r.get("text") or ""),
            "text": r.get("text"),
            "est_cost_usd": round(c, 6),
        }
        await asyncio.sleep(0.2)
    out["total_cost_usd"] = round(cost, 6)
    return out


async def run_streaming_and_tool_edges(client: httpx.AsyncClient, auth: str) -> dict[str, Any]:
    hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off"}
    out: dict[str, Any] = {}

    # 1. Streaming incomplete (max_output_tokens=10)
    out["stream_incomplete"] = await stream_response(client, hdrs, {
        "model": "moss/fastlane-1",
        "stream": True,
        "input": "Write a 200-word essay on retrieval-augmented generation.",
        "max_output_tokens": 10,
    })

    # 2. Structured output truncated mid-JSON
    out["json_schema_truncated"] = await stream_response(client, hdrs, {
        "model": "moss/fastlane-1",
        "stream": True,
        "input": "Return a detailed 3-item summary of Auckland, Wellington, and Christchurch.",
        "max_output_tokens": 14,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cities",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "city1": {"type": "string"},
                        "city2": {"type": "string"},
                        "city3": {"type": "string"},
                    },
                    "required": ["city1", "city2", "city3"],
                    "additionalProperties": False,
                },
            }
        },
    })

    # 3. Chat Completions stream_options include_usage=false
    chunks = []
    async with client.stream("POST", f"{BASE_URL}/chat/completions", headers=hdrs, json={
        "model": "moss/fastlane-1",
        "stream": True,
        "stream_options": {"include_usage": False},
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 12,
    }) as r:
        async for line in r.aiter_lines():
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                chunks.append(json.loads(line[5:].strip()))
    out["chat_stream_include_usage_false"] = {
        "chunk_count": len(chunks),
        "any_chunk_has_usage": any(c.get("usage") is not None for c in chunks),
        "last_chunk": chunks[-1] if chunks else None,
    }

    # 4. tool_choice="none" vs "required" vs parallel_tool_calls=false
    fn_tools = [
        {"type": "function", "name": "fn_a", "description": "Fn A", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}},
        {"type": "function", "name": "fn_b", "description": "Fn B", "parameters": {"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]}},
    ]
    r_none = await client.post(f"{BASE_URL}/responses", headers=hdrs, json={
        "model": "moss/fastlane-1",
        "input": "Call fn_a with x=1 and fn_b with y=2.",
        "tools": fn_tools,
        "tool_choice": "none",
        "max_output_tokens": 40,
    })
    r_req = await client.post(f"{BASE_URL}/responses", headers=hdrs, json={
        "model": "moss/fastlane-1",
        "input": "What is 2+2? Do not explain.",
        "tools": fn_tools,
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_output_tokens": 40,
    })
    out["tool_choice_none"] = {
        "status": r_none.status_code,
        "output_types": [o.get("type") for o in (r_none.json().get("output") or [])] if r_none.status_code == 200 else r_none.text[:200],
    }
    out["tool_choice_required_seq"] = {
        "status": r_req.status_code,
        "output_types": [o.get("type") for o in (r_req.json().get("output") or [])] if r_req.status_code == 200 else r_req.text[:200],
    }

    # 5. moss_search with nonexistent index
    r_bad_idx = await client.post(f"{BASE_URL}/responses", headers=hdrs, json={
        "model": "moss/fastlane-1",
        "input": "Search policy.",
        "tools": [{"type": "moss_search", "indexes": ["nonexistent-eval-idx-999"], "top_k": 3}],
        "max_output_tokens": 60,
    })
    out["moss_search_nonexistent_index"] = {
        "status": r_bad_idx.status_code,
        "body_preview": r_bad_idx.text[:400],
    }

    # 6. MCP SSRF probe (server_url pointing to localhost & metadata IP)
    for tag, url in [("mcp_localhost", "http://127.0.0.1:9999/mcp"), ("mcp_metadata", "http://169.254.169.254/latest/meta-data/")]:
        rm = await client.post(f"{BASE_URL}/responses", headers=hdrs, json={
            "model": "moss/fastlane-1",
            "input": "Use the remote mcp tool to read status.",
            "tools": [{"type": "mcp", "server_label": "probe", "server_url": url}],
            "max_output_tokens": 60,
        })
        out[tag] = {"status": rm.status_code, "body_preview": rm.text[:350]}

    # 7. Error envelope consistency across endpoints
    err_checks = {}
    r_models_bad = await client.get(f"{BASE_URL}/models", headers={"Authorization": "Bearer bad:key"})
    r_models_no = await client.get(f"{BASE_URL}/models")
    r_get_resp = await client.get(f"{BASE_URL}/responses/resp_doesnotexist", headers=hdrs)
    r_bad_model = await client.post(f"{BASE_URL}/responses", headers=hdrs, json={"model": "vendor/not-a-model", "input": "hi"})
    r_empty_msgs = await client.post(f"{BASE_URL}/chat/completions", headers=hdrs, json={"model": "moss/fastlane-1", "messages": []})
    for k, resp in [
        ("get_models_bad_auth", r_models_bad),
        ("get_models_no_auth", r_models_no),
        ("get_response_unknown_id", r_get_resp),
        ("post_responses_invalid_model", r_bad_model),
        ("post_chat_empty_messages", r_empty_msgs),
    ]:
        err_checks[k] = {"status": resp.status_code, "body_preview": resp.text[:300]}
    out["error_envelopes"] = err_checks
    return out


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        matrix = await run_model_matrix(client, auth)
        params = await run_param_matrix(client, auth)
        long_ctx = await run_long_context(client, auth)
        edges = await run_streaming_and_tool_edges(client, auth)

    report = {
        "model_matrix": matrix,
        "param_matrix": params,
        "long_context": long_ctx,
        "streaming_and_tool_edges": edges,
        "total_suite7_est_cost_usd": round(matrix["total_matrix_cost_usd"] + long_ctx["total_cost_usd"], 6),
    }
    path = save_log("07_inference_deep_models_and_edges.json", report, [pkey])
    print("Saved:", path)
    print("\n--- MODEL MATRIX ---")
    for label, d in matrix.items():
        if isinstance(d, dict) and "accuracy" in d:
            p50 = d["ttft_ms"]["p50"] if d["ttft_ms"] else None
            tot = d["total_ms"]["p50"] if d["total_ms"] else None
            print(f"{label:38s} | acc={d['accuracy']} | ttft_p50={p50}ms | tot_p50={tot}ms | cost=${d['est_cost_usd']}")
    print("\n--- LONG CONTEXT ---")
    for k, v in long_ctx.items():
        if isinstance(v, dict):
            print(f"{k:12s} | status={v['status']} | in_tok={v['usage'].get('input_tokens')} | ttft={v['ttft_ms']}ms | needle={v['needle_recalled']} | cost=${v['est_cost_usd']}")
    print("\n--- KEY EDGE CHECKS ---")
    for k, v in params.items():
        print(f"param.{k:32s} -> {v['status']} {v['body_preview'][:100]}")
    print("Total Suite 7 est cost USD:", report["total_suite7_est_cost_usd"])


if __name__ == "__main__":
    asyncio.run(main())
