"""
Suite 2: Moss Inference Benchmark, Grounding Evals, Tool Calling, and OpenAI Compatibility.
"""
import asyncio
import json
import time
from typing import Any
import httpx
from openai import OpenAI
from moss import MossInference
from moss.inference import McpCall, ResponseFunctionToolCall
from moss_eval.common import (
    estimate_inference_cost,
    load_credentials,
    save_log,
    summarize_latencies,
)

BASE_URL = "https://inference.moss.dev/v1"
EVAL_INDEX = "eval-grounding-v1"

GROUNDING_EVAL_SET = [
    {"id": "g1", "q": "How many minutes does KoruMart Express take in Auckland CBD, and what is the order minimum to skip the express fee?", "must_contain_any": [["18"], ["35"]]},
    {"id": "g2", "q": "Above what Celsius temperature do cold-chain dairy items trigger an automatic instant refund?", "must_contain_any": [["8"]]},
    {"id": "g3", "q": "Within how many hours of delivery must perishable produce claims be submitted?", "must_contain_any": [["12"]]},
    {"id": "g4", "q": "What is the exact warehouse bin code for SKU MNG-4402?", "must_contain_any": [["NZ-AKL-04B"]]},
    {"id": "g5", "q": "What weekly drop count and on-time percentage activate Gold rider payout bonus?", "must_contain_any": [["120"], ["96.5"]]},
    {"id": "g6", "q": "What is the webhook partner API rate limit per minute?", "must_contain_any": [["240"]]},
    {"id": "g7", "q": "How many months until KoruMart gift cards expire?", "must_contain_any": [["36"]]},
    {"id": "g8", "q": "How much does KoruPass cost annually in NZD?", "must_contain_any": [["79"]]},
    {"id": "g9", "q": "Up to how much price difference does KoruMart absorb on substitutions?", "must_contain_any": [["5", "5.00"]]},
    {"id": "g10", "q": "What escalation code routes to the Auckland cold-chain duty manager?", "must_contain_any": [["ESC-917"]]},
    {"id": "g11_ml_pure", "q": "കേടായ പച്ചക്കറികൾക്ക് എത്ര മണിക്കൂറിനുള്ളിൽ റീഫണ്ട് ലഭിക്കും?", "must_contain_any": [["6"]]},
    {"id": "g12_trap_unanswerable", "q": "What is KoruMart's CEO's personal phone number?", "expect_refusal": True},
]


def check_facts(text: str, item: dict[str, Any]) -> bool:
    low = text.lower()
    if item.get("expect_refusal"):
        refusal_markers = ["don't know", "do not know", "not mentioned", "no information", "cannot find", "not provided", "not in", "don't have", "do not have", "unavailable", "no record"]
        return any(m in low for m in refusal_markers)
    groups = item.get("must_contain_any", [])
    return all(any(opt.lower() in low for opt in g) for g in groups)


async def stream_responses_httpx(
    h: httpx.AsyncClient,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ttft_ms = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    status_code = 0
    resp_headers: dict[str, str] = {}
    error_body = None

    async with h.stream("POST", f"{BASE_URL}/responses", headers=headers, json=payload) as resp:
        status_code = resp.status_code
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        if status_code != 200:
            raw = await resp.aread()
            error_body = raw.decode(errors="replace")
        else:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw_json = line[5:].strip()
                if not raw_json or raw_json == "[DONE]":
                    continue
                ev = json.loads(raw_json)
                etype = ev.get("type")
                if etype == "response.output_text.delta":
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    text_parts.append(ev.get("delta") or "")
                elif etype in ("response.completed", "response.incomplete"):
                    usage = (ev.get("response") or {}).get("usage") or {}
                elif etype == "response.failed":
                    error_body = ev

    total_ms = (time.perf_counter() - t0) * 1000
    retrieve_ms_raw = resp_headers.get("x-moss-retrieve-ms")
    retrieve_ms = float(retrieve_ms_raw) if retrieve_ms_raw is not None else None
    return {
        "status": status_code,
        "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
        "total_ms": round(total_ms, 2),
        "retrieve_ms": retrieve_ms,
        "box": resp_headers.get("x-moss-box"),
        "request_id": resp_headers.get("x-moss-request-id"),
        "ratelimit_remaining_req": resp_headers.get("x-ratelimit-remaining-requests"),
        "usage": usage,
        "text": "".join(text_parts),
        "error": error_body,
    }


async def main() -> None:
    pid, pkey = load_credentials()
    auth = f"Bearer {pid}:{pkey}"
    total_est_cost = 0.0
    results: dict[str, Any] = {}

    async with httpx.AsyncClient(timeout=60.0) as h:
        # 1. Latency benchmark: moss/fastlane-1 ungrounded (N=15, cache off)
        ungrounded_runs = []
        for i in range(15):
            res = await stream_responses_httpx(
                h,
                {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off"},
                {
                    "model": "moss/fastlane-1",
                    "stream": True,
                    "instructions": "Reply in one short sentence.",
                    "input": f"Say hello and confirm request #{i}.",
                    "max_output_tokens": 30,
                },
            )
            u = res["usage"]
            total_est_cost += estimate_inference_cost(
                "moss/fastlane-1",
                u.get("input_tokens", 0),
                u.get("output_tokens", 0),
                (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                0,
            )
            ungrounded_runs.append(res)
            await asyncio.sleep(0.15)

        results["fastlane_ungrounded_15x"] = {
            "ttft_ms": summarize_latencies([r["ttft_ms"] for r in ungrounded_runs if r["ttft_ms"] is not None]),
            "total_ms": summarize_latencies([r["total_ms"] for r in ungrounded_runs]),
            "runs": ungrounded_runs,
        }

        # 2. Latency benchmark: moss/fastlane-1 grounded on eval-grounding-v1 (N=15, cache off)
        grounded_runs = []
        prompts = [q["q"] for q in GROUNDING_EVAL_SET[:10]]
        for i in range(15):
            res = await stream_responses_httpx(
                h,
                {
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "X-Moss-Indexes": EVAL_INDEX,
                    "X-Moss-Cache": "off",
                },
                {
                    "model": "moss/fastlane-1",
                    "stream": True,
                    "instructions": "Answer concisely in 1 sentence using only the retrieved context.",
                    "input": prompts[i % len(prompts)],
                    "max_output_tokens": 60,
                },
            )
            u = res["usage"]
            total_est_cost += estimate_inference_cost(
                "moss/fastlane-1",
                u.get("input_tokens", 0),
                u.get("output_tokens", 0),
                (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                1,
            )
            grounded_runs.append(res)
            await asyncio.sleep(0.15)

        warm_grounded = grounded_runs[1:]  # exclude first cold call separately
        results["fastlane_grounded_15x"] = {
            "first_call_retrieve_ms": grounded_runs[0]["retrieve_ms"],
            "first_call_ttft_ms": grounded_runs[0]["ttft_ms"],
            "all_retrieve_ms": summarize_latencies([r["retrieve_ms"] for r in grounded_runs if r["retrieve_ms"] is not None]),
            "warm_retrieve_ms": summarize_latencies([r["retrieve_ms"] for r in warm_grounded if r["retrieve_ms"] is not None]),
            "all_ttft_ms": summarize_latencies([r["ttft_ms"] for r in grounded_runs if r["ttft_ms"] is not None]),
            "warm_ttft_ms": summarize_latencies([r["ttft_ms"] for r in warm_grounded if r["ttft_ms"] is not None]),
            "total_ms": summarize_latencies([r["total_ms"] for r in grounded_runs]),
            "runs": grounded_runs,
        }

        # 3. Reply cache test (cache=on, identical request 5 times)
        cache_runs = []
        for i in range(5):
            t0 = time.perf_counter()
            r = await h.post(
                f"{BASE_URL}/responses",
                headers={
                    "Authorization": auth,
                    "Content-Type": "application/json",
                    "X-Moss-Indexes": EVAL_INDEX,
                    "X-Moss-Cache": "on",
                },
                json={
                    "model": "moss/fastlane-1",
                    "instructions": "Answer concisely.",
                    "input": "What is the exact warehouse bin code for SKU MNG-4402? (cache probe 2026)",
                    "max_output_tokens": 40,
                },
            )
            dt = round((time.perf_counter() - t0) * 1000, 2)
            body = r.json()
            u = body.get("usage") or {}
            is_cache_hit = (u.get("input_tokens", -1) == 0)
            if not is_cache_hit:
                total_est_cost += estimate_inference_cost("moss/fastlane-1", u.get("input_tokens", 0), u.get("output_tokens", 0), 0, 1)
            cache_runs.append({
                "iter": i,
                "status": r.status_code,
                "total_ms": dt,
                "retrieve_ms": r.headers.get("x-moss-retrieve-ms"),
                "usage": u,
                "cache_hit_inferred": is_cache_hit,
            })
        results["reply_cache_5x"] = cache_runs

        # 4. Vendor model latency comparison (3 grounded calls each, cache off)
        vendor_results = {}
        for model_id in ["openai/gpt-6-luna", "anthropic/claude-haiku-5.5", "google/gemini-3.5-flash-lite"]:
            vruns = []
            for i in range(3):
                res = await stream_responses_httpx(
                    h,
                    {
                        "Authorization": auth,
                        "Content-Type": "application/json",
                        "X-Moss-Indexes": EVAL_INDEX,
                        "X-Moss-Cache": "off",
                    },
                    {
                        "model": model_id,
                        "stream": True,
                        "instructions": "Answer concisely in 1 sentence using the retrieved context.",
                        "input": prompts[i],
                        "max_output_tokens": 60,
                    },
                )
                u = res["usage"]
                total_est_cost += estimate_inference_cost(
                    model_id,
                    u.get("input_tokens", 0),
                    u.get("output_tokens", 0),
                    (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    1,
                )
                vruns.append(res)
                await asyncio.sleep(0.2)
            vendor_results[model_id] = {
                "ttft_ms": summarize_latencies([r["ttft_ms"] for r in vruns if r["ttft_ms"] is not None]),
                "retrieve_ms": summarize_latencies([r["retrieve_ms"] for r in vruns if r["retrieve_ms"] is not None]),
                "total_ms": summarize_latencies([r["total_ms"] for r in vruns]),
                "runs": vruns,
            }
        results["vendor_comparison"] = vendor_results

    # 5. Grounding accuracy ablation via MossInference SDK (retrieve=True vs retrieve=False)
    async with MossInference(pid, pkey) as client:
        eval_ablation = {"grounded": [], "ungrounded": []}
        for mode_name, use_retrieve in [("grounded", True), ("ungrounded", False)]:
            passed = 0
            for item in GROUNDING_EVAL_SET:
                t0 = time.perf_counter()
                resp = await client.responses.create(
                    model="moss/fastlane-1",
                    instructions="You are KoruMart support. Answer using only verified KoruMart policies. If the policy does not contain the answer, clearly say you do not have that information.",
                    input=item["q"],
                    indexes=[EVAL_INDEX],
                    retrieve=use_retrieve,
                    cache=False,
                    max_output_tokens=100,
                )
                dt = round((time.perf_counter() - t0) * 1000, 2)
                ok = check_facts(resp.output_text, item)
                if ok:
                    passed += 1
                u = resp.usage
                if u:
                    total_est_cost += estimate_inference_cost(
                        "moss/fastlane-1",
                        u.input_tokens,
                        u.output_tokens,
                        u.input_tokens_details.cached_tokens if u.input_tokens_details else 0,
                        1 if use_retrieve else 0,
                    )
                eval_ablation[mode_name].append({
                    "id": item["id"],
                    "passed": ok,
                    "latency_ms": dt,
                    "answer": resp.output_text,
                    "input_tokens": u.input_tokens if u else None,
                    "output_tokens": u.output_tokens if u else None,
                })
            eval_ablation[f"{mode_name}_accuracy"] = round(passed / len(GROUNDING_EVAL_SET), 4)
        results["grounding_ablation"] = eval_ablation

        # 6. Function tool calling loop + Structured Output + moss_search tool
        tools_func = [{
            "type": "function",
            "name": "lookup_sku_stock",
            "description": "Look up live warehouse stock for a SKU.",
            "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]},
        }]
        items: list[Any] = [{"role": "user", "content": "Check stock for SKU MNG-4402 and tell me its bin temperature policy."}]
        func_trace = []
        for step in range(3):
            resp = await client.responses.create(
                instructions="Use lookup_sku_stock for live stock and your index for storage policy.",
                input=items,
                tools=tools_func,
                indexes=[EVAL_INDEX],
                cache=False,
                max_output_tokens=120,
            )
            u = resp.usage
            if u:
                total_est_cost += estimate_inference_cost("moss/fastlane-1", u.input_tokens, u.output_tokens, 0, 1)
            items += resp.output
            calls = [i for i in resp.output if isinstance(i, ResponseFunctionToolCall)]
            func_trace.append({"step": step, "calls": [{"name": c.name, "args": c.arguments} for c in calls], "text": resp.output_text})
            if not calls:
                break
            for c in calls:
                items.append({"type": "function_call_output", "call_id": c.call_id, "output": json.dumps({"sku": "MNG-4402", "cases_in_stock": 42})})

        # Structured output (json_schema)
        schema = {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "bin_code": {"type": "string"},
                "storage_temp_c": {"type": "number"},
            },
            "required": ["sku", "bin_code", "storage_temp_c"],
            "additionalProperties": False,
        }
        struct_resp = await client.responses.create(
            instructions="Extract exact SKU storage facts from context.",
            input="What is the bin code and storage temperature for Alphonso mangoes?",
            indexes=[EVAL_INDEX],
            cache=False,
            text={"format": {"type": "json_schema", "name": "sku_storage", "schema": schema}},
        )
        struct_parsed = None
        struct_valid = False
        try:
            struct_parsed = json.loads(struct_resp.output_text)
            struct_valid = (struct_parsed.get("bin_code") == "NZ-AKL-04B" and struct_parsed.get("storage_temp_c") == 13)
        except Exception as e:
            struct_parsed = str(e)

        # moss_search tool on Responses
        ms_resp = await client.responses.create(
            instructions="Always use moss_search to verify policy facts before answering.",
            input="What is the escalation code for Auckland cold chain and how long are gift cards valid?",
            tools=[{"type": "moss_search", "indexes": [EVAL_INDEX], "top_k": 3}],
            tool_choice={"type": "mcp", "server_label": "moss", "name": "moss_search"},
            cache=False,
            max_output_tokens=120,
        )
        mcp_calls = [
            {"name": i.name, "status": i.status, "duration_ms": i.duration_ms, "args": i.arguments}
            for i in ms_resp.output if isinstance(i, McpCall)
        ]

        # Prefix caching check (~900 token repeated instructions, cache=False so reply cache off)
        long_inst = "Policy manual preamble: " + ("Always verify SKU codes, cold-chain rules, and Auckland delivery SLAs carefully. " * 60)
        pc1 = await client.responses.create(instructions=long_inst, input="What is ESC-917?", cache=False, indexes=[EVAL_INDEX], max_output_tokens=30)
        pc2 = await client.responses.create(instructions=long_inst, input="What is KoruPass annual price?", cache=False, indexes=[EVAL_INDEX], max_output_tokens=30)

        results["features"] = {
            "function_tool_loop": func_trace,
            "structured_output": {"raw_text": struct_resp.output_text, "parsed": struct_parsed, "valid": struct_valid},
            "moss_search_responses": {"mcp_calls": mcp_calls, "answer": ms_resp.output_text},
            "prefix_cache_probe": {
                "call1_input_tokens": pc1.usage.input_tokens if pc1.usage else None,
                "call1_cached_tokens": pc1.usage.input_tokens_details.cached_tokens if pc1.usage and pc1.usage.input_tokens_details else None,
                "call2_input_tokens": pc2.usage.input_tokens if pc2.usage else None,
                "call2_cached_tokens": pc2.usage.input_tokens_details.cached_tokens if pc2.usage and pc2.usage.input_tokens_details else None,
            },
        }

    # 7. OpenAI Python SDK compatibility (Responses + Chat Completions)
    oai = OpenAI(
        base_url=BASE_URL,
        api_key=f"{pid}:{pkey}",
        default_headers={"X-Moss-Indexes": EVAL_INDEX, "X-Moss-Cache": "off"},
    )
    oai_resp = oai.responses.create(model="moss/fastlane-1", instructions="Be brief.", input="How much is KoruPass per year?", max_output_tokens=40)
    oai_chat = oai.chat.completions.create(
        model="moss/fastlane-1",
        messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "How much is KoruPass per year?"}],
        max_tokens=40,
    )
    # Also test Chat Completions with moss_search tool
    oai_chat_ms = oai.chat.completions.create(
        model="moss/fastlane-1",
        messages=[{"role": "user", "content": "When is scheduled maintenance for dark-store inventory sync?"}],
        tools=[{"type": "moss_search", "indexes": [EVAL_INDEX], "top_k": 3}],
        max_tokens=60,
    )
    results["openai_sdk_compat"] = {
        "responses_output": oai_resp.output_text,
        "chat_completions_output": oai_chat.choices[0].message.content,
        "chat_moss_search_output": oai_chat_ms.choices[0].message.content,
        "chat_moss_tool_steps": getattr(oai_chat_ms, "moss_tool_steps", None),
    }

    results["estimated_inference_cost_usd"] = round(total_est_cost, 6)
    path = save_log("02_inference_bench_and_eval.json", results, [pkey])
    print("Saved:", path)
    print("Estimated cost USD:", results["estimated_inference_cost_usd"])
    print("Fastlane ungrounded TTFT:", results["fastlane_ungrounded_15x"]["ttft_ms"])
    print("Fastlane grounded warm retrieve_ms:", results["fastlane_grounded_15x"]["warm_retrieve_ms"], "| first_call_retrieve_ms:", results["fastlane_grounded_15x"]["first_call_retrieve_ms"])
    print("Fastlane grounded warm TTFT:", results["fastlane_grounded_15x"]["warm_ttft_ms"])
    print("Reply cache 5x:", [(c["iter"], c["total_ms"], c["cache_hit_inferred"]) for c in cache_runs])
    for m, v in vendor_results.items():
        print(f"Vendor {m}: TTFT={v['ttft_ms']} | retrieve={v['retrieve_ms']}")
    print("Grounding accuracy -> grounded:", results["grounding_ablation"]["grounded_accuracy"], "| ungrounded:", results["grounding_ablation"]["ungrounded_accuracy"])
    print("Structured output valid:", struct_valid, "| Function tool steps:", len(func_trace), "| moss_search calls:", mcp_calls)
    print("Prefix cache:", results["features"]["prefix_cache_probe"])


if __name__ == "__main__":
    asyncio.run(main())
