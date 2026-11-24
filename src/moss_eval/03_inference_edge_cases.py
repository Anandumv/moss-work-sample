"""
Suite 3: Moss Inference Edge Cases, RTT Baseline, Auth/Validation, Concurrency & Rate Limits.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from moss import MossInference
from moss.inference import InferenceError, ResponseFunctionToolCall
from moss_eval.common import (
    estimate_inference_cost,
    load_credentials,
    save_log,
    summarize_latencies,
)

BASE_URL = "https://inference.moss.dev/v1"
EVAL_INDEX = "eval-grounding-v1"


async def raw_post(
    h: httpx.AsyncClient, path: str, headers: dict[str, str], body: Any
) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = await h.post(f"{BASE_URL}{path}", headers=headers, json=body)
    dt = round((time.perf_counter() - t0) * 1000, 2)
    try:
        parsed = r.json()
    except Exception:
        parsed = r.text[:500]
    return {
        "status": r.status_code,
        "elapsed_ms": dt,
        "headers": {
            k.lower(): v
            for k, v in r.headers.items()
            if any(x in k.lower() for x in ("moss", "ratelimit", "retry", "should"))
        },
        "body": parsed,
    }


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    hdrs = {"Authorization": auth, "Content-Type": "application/json"}
    est_cost = 0.0
    report: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=60.0) as h:
        # 1. Baseline HTTP RTT (keep-alive GET /v1/models x 10)
        rtt_ms = []
        for i in range(10):
            t0 = time.perf_counter()
            r = await h.get(f"{BASE_URL}/models", headers={"Authorization": auth})
            rtt_ms.append((time.perf_counter() - t0) * 1000)
        report["baseline_rtt_models_ms"] = summarize_latencies(rtt_ms)
        report["baseline_rtt_warm_ms"] = summarize_latencies(rtt_ms[1:])

        # 2. Check raw Chat Completions moss_tool_steps
        r_cc = await raw_post(
            h,
            "/chat/completions",
            {**hdrs, "X-Moss-Cache": "off"},
            {
                "model": "moss/fastlane-1",
                "messages": [{"role": "user", "content": "What is escalation code ESC-917?"}],
                "tools": [{"type": "moss_search", "indexes": [EVAL_INDEX], "top_k": 3}],
                "max_tokens": 50,
            },
        )
        report["raw_chat_completions_moss_search"] = {
            "status": r_cc["status"],
            "has_moss_tool_steps": isinstance(r_cc["body"], dict) and "moss_tool_steps" in r_cc["body"],
            "moss_tool_steps": r_cc["body"].get("moss_tool_steps") if isinstance(r_cc["body"], dict) else None,
            "content": (r_cc["body"].get("choices") or [{}])[0].get("message", {}).get("content") if isinstance(r_cc["body"], dict) else None,
        }

        # 3. Diagnose function-tool + X-Moss-Indexes context retention across turns
        r_ft0 = await raw_post(
            h,
            "/responses",
            {**hdrs, "X-Moss-Indexes": EVAL_INDEX, "X-Moss-Cache": "off"},
            {
                "model": "moss/fastlane-1",
                "instructions": "Use lookup_sku_stock AND the retrieved storage policy.",
                "input": [{"role": "user", "content": "Check stock for SKU MNG-4402 and tell me its storage temperature policy."}],
                "tools": [{
                    "type": "function",
                    "name": "lookup_sku_stock",
                    "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
                }],
                "max_output_tokens": 80,
            },
        )
        step0_in_tok = ((r_ft0["body"].get("usage") or {}).get("input_tokens")) if isinstance(r_ft0["body"], dict) else None
        out0 = r_ft0["body"].get("output", []) if isinstance(r_ft0["body"], dict) else []
        fc = next((x for x in out0 if x.get("type") == "function_call"), None)
        step1_info = None
        if fc:
            r_ft1 = await raw_post(
                h,
                "/responses",
                {**hdrs, "X-Moss-Indexes": EVAL_INDEX, "X-Moss-Cache": "off"},
                {
                    "model": "moss/fastlane-1",
                    "instructions": "Use lookup_sku_stock AND the retrieved storage policy.",
                    "input": [
                        {"role": "user", "content": "Check stock for SKU MNG-4402 and tell me its storage temperature policy."},
                        fc,
                        {"type": "function_call_output", "call_id": fc["call_id"], "output": '{"cases_in_stock": 42}'},
                    ],
                    "tools": [{
                        "type": "function",
                        "name": "lookup_sku_stock",
                        "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
                    }],
                    "max_output_tokens": 80,
                },
            )
            step1_info = {
                "step1_input_tokens": (r_ft1["body"].get("usage") or {}).get("input_tokens"),
                "step1_retrieve_ms": r_ft1["headers"].get("x-moss-retrieve-ms"),
                "step1_output": r_ft1["body"].get("output"),
            }
        report["func_tool_retrieval_diag"] = {
            "step0_input_tokens": step0_in_tok,
            "step0_retrieve_ms": r_ft0["headers"].get("x-moss-retrieve-ms"),
            "step1": step1_info,
        }

        # 4. Edge cases (input shapes, refused params, auth, wrong indexes, oversized)
        edge_cases = {}
        edge_cases["empty_string_input"] = await raw_post(h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": "", "max_output_tokens": 20})
        edge_cases["whitespace_input"] = await raw_post(h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": "   \n\t  ", "max_output_tokens": 20})
        edge_cases["empty_list_input"] = await raw_post(h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": [], "max_output_tokens": 20})
        edge_cases["unicode_mixed_emoji_rtl"] = await raw_post(
            h, "/responses", {**hdrs, "X-Moss-Cache": "off"},
            {"model": "moss/fastlane-1", "input": "Reply with OK: മലയാളം 🚀 مرحبا بالعالم 日本語", "max_output_tokens": 20},
        )
        edge_cases["refused_previous_response_id"] = await raw_post(
            h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": "hi", "previous_response_id": "resp_123"}
        )
        edge_cases["refused_unknown_field"] = await raw_post(
            h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": "hi", "nonexistent_param": 123}
        )
        edge_cases["refused_anthropic_temperature"] = await raw_post(
            h, "/responses", hdrs, {"model": "anthropic/claude-haiku-5.5", "input": "hi", "temperature": 0.5}
        )
        edge_cases["model_not_found"] = await raw_post(
            h, "/responses", hdrs, {"model": "openai/does-not-exist-99", "input": "hi"}
        )
        edge_cases["missing_provider_key"] = await raw_post(
            h, "/responses", hdrs, {"model": "gpt-6-luna", "input": "hi"}
        )
        edge_cases["bad_auth_key"] = await raw_post(
            h, "/responses", {"Authorization": f"Bearer {pid}:moss_wrongkey_0000", "Content-Type": "application/json"},
            {"model": "moss/fastlane-1", "input": "hi"},
        )
        edge_cases["bad_auth_no_colon"] = await raw_post(
            h, "/responses", {"Authorization": "Bearer invalid_no_colon", "Content-Type": "application/json"},
            {"model": "moss/fastlane-1", "input": "hi"},
        )
        edge_cases["wrong_index_name"] = await raw_post(
            h, "/responses", {**hdrs, "X-Moss-Indexes": "nonexistent-index-xyz-999"},
            {"model": "moss/fastlane-1", "input": "hi", "max_output_tokens": 10},
        )
        too_many_idx = ",".join(f"idx-{i}" for i in range(33))
        edge_cases["over_32_indexes"] = await raw_post(
            h, "/responses", {**hdrs, "X-Moss-Indexes": too_many_idx},
            {"model": "moss/fastlane-1", "input": "hi", "max_output_tokens": 10},
        )
        edge_cases["retrieve_off_header"] = await raw_post(
            h, "/responses", {**hdrs, "X-Moss-Indexes": EVAL_INDEX, "X-Moss-Retrieve": "off", "X-Moss-Cache": "off"},
            {"model": "moss/fastlane-1", "input": "What is ESC-917?", "max_output_tokens": 30},
        )
        # Context length exceeded (>262,144 tokens -> ~1.1M chars) vs >4MB body (413)
        huge_text = "word " * 270_000  # ~1.35 MB, ~270k tokens
        edge_cases["context_length_exceeded"] = await raw_post(
            h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": huge_text, "max_output_tokens": 10}
        )
        over_4mb = "a" * (4_200_000)
        edge_cases["over_4mb_body"] = await raw_post(
            h, "/responses", hdrs, {"model": "moss/fastlane-1", "input": over_4mb}
        )
        report["edge_cases"] = edge_cases

        # 5. Concurrency & burst rate limit test (16 concurrent requests; burst is documented as 10)
        async def fire_one(i: int) -> dict[str, Any]:
            return await raw_post(
                h, "/responses", {**hdrs, "X-Moss-Cache": "off"},
                {"model": "moss/fastlane-1", "input": f"Burst ping {i}. Reply with {i}.", "max_output_tokens": 8},
            )

        t_b0 = time.perf_counter()
        burst_res = await asyncio.gather(*(fire_one(i) for i in range(16)))
        burst_wall_ms = round((time.perf_counter() - t_b0) * 1000, 2)
        status_counts: dict[int, int] = {}
        for br in burst_res:
            s = br["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
            if s == 200 and isinstance(br["body"], dict):
                u = br["body"].get("usage") or {}
                est_cost += estimate_inference_cost("moss/fastlane-1", u.get("input_tokens", 0), u.get("output_tokens", 0), 0, 0)
        report["burst_16_concurrent"] = {
            "wall_ms": burst_wall_ms,
            "status_counts": status_counts,
            "sample_429_or_last": [
                {"status": x["status"], "headers": x["headers"], "error": x["body"].get("error") if isinstance(x["body"], dict) else x["body"]}
                for x in burst_res
                if x["status"] != 200
            ][:3],
            "remaining_requests_seen": [x["headers"].get("x-ratelimit-remaining-requests") for x in burst_res],
        }

    # 6. SDK client-side validation + streaming early break
    sdk_checks: dict[str, Any] = {}
    try:
        MossInference("", pkey)
        sdk_checks["empty_project_id"] = "unexpectedly_succeeded"
    except InferenceError as e:
        sdk_checks["empty_project_id"] = {"type": e.type, "status": e.status, "message": e.message}

    async with MossInference(pid, pkey) as client:
        try:
            await client.responses.create(input="hi", indexes=["bad,name"])
            sdk_checks["comma_in_index"] = "unexpectedly_succeeded"
        except InferenceError as e:
            sdk_checks["comma_in_index"] = {"type": e.type, "status": e.status, "message": e.message}

        # Streaming early break
        stream = await client.responses.create(
            input="Count slowly from 1 to 50 with descriptions.",
            cache=False,
            stream=True,
        )
        chunks = []
        async with stream:
            async for ev in stream:
                if ev.type == "response.output_text.delta" and ev.delta:
                    chunks.append(ev.delta)
                    if len(chunks) >= 3:
                        break
        sdk_checks["early_stream_break"] = {"chunks_collected": len(chunks), "joined": "".join(chunks)}

    report["sdk_checks"] = sdk_checks
    report["estimated_cost_usd"] = round(est_cost, 6)
    path = save_log("03_inference_edge_cases.json", report, [pkey])
    print("Saved:", path)
    print("Baseline RTT /v1/models warm (ms):", report["baseline_rtt_warm_ms"])
    print("Chat Completions moss_tool_steps:", report["raw_chat_completions_moss_search"])
    print("Func tool retrieval diag:", report["func_tool_retrieval_diag"])
    for k, v in edge_cases.items():
        err = v["body"].get("error") if isinstance(v["body"], dict) else str(v["body"])[:100]
        print(f"Edge [{k}]: HTTP {v['status']} ({v['elapsed_ms']}ms) -> {err}")
    print("Burst 16 concurrent:", report["burst_16_concurrent"]["status_counts"], "wall_ms=", burst_wall_ms)
    print("SDK checks:", sdk_checks)


if __name__ == "__main__":
    asyncio.run(main())
