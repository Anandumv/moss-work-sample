# Issues

These are things I hit on the live APIs (`https://inference.moss.dev/v1`, `https://service.usemoss.dev`) on 2026-11-25, Developer plan, `moss==1.26.0` / `inferedge-moss-core==0.38.0` / `moss-cli==0.4.1`. Secrets are redacted. Raw evidence is in `logs/*.json`.

---

## Index

| ID | Severity | Component | Title | Evidence Log |
| --- | --- | --- | --- | --- |
| ISS-01 | High | Inference API (`/v1/responses`) | Stateless function-tool loop with `X-Moss-Indexes` drops retrieved context on turn 1 | `03_inference_edge_cases.json` |
| ISS-02 | High | Tasks API (`/v1/tasks`) | `POST /v1/tasks` accepts loopback and cloud-metadata `start_url` (SSRF admission gap) | `04a_tasks_validation_probes.json` |
| ISS-03 | High | Tasks API (`/v1/tasks`) | `limits.max_cost_usd` overshoots cap by 2.4× before halting | `04b_tasks_live_runs.json` |
| ISS-04 | Medium | Inference API (`qwen/qwen3.8-flash`, `z-ai/glm-5.3-flash`) | Thinking-model reasoning token budget starves visible output (`0` tokens emitted, `TTFT = null`) | `07_inference_deep_models_and_edges.json`, `07b_model_matrix_grounded.json` |
| ISS-05 | Medium | Tasks API (`/v1/tasks`) | Unreachable `.invalid` URL surfaces SOCKS proxy error and resolves as `status: "completed"` | `04b_tasks_live_runs.json` |
| ISS-06 | Medium | Tasks API (`/v1/tasks/{id}/trace` & `/files`) | `trace()` advertises `video.path: "session-1.mp4"` while `/files` omits it and download routes `404` | `04c_tasks_extra_probes.json`, `08_tasks_deep_scenarios.json` |
| ISS-07 | Medium | Tasks API (`live_view`) | `environment.live_view=True` always leaves `live_url=None` | `04b_tasks_live_runs.json`, `04c_tasks_extra_probes.json` |
| ISS-08 | Medium | Tasks API / SDK (`TaskUsage`) | `usage.cost_usd` reflects `model_usd` only, omitting `browser_usd`, and `traffic_usd` is always `null` | `04b`, `04c`, `08_tasks_deep_scenarios.json` |
| ISS-09 | Medium | Tasks API (Agent Autonomy) | Unscoped browser agent exits target domain to scrape GitHub source code on auth redirect | `08_tasks_deep_scenarios.json` |
| ISS-10 | Low | Control Plane (`create_index`) | Developer 3-index plan limit returns generic billing exhaustion error | `06_deep_retrieval_eval.json` |
| ISS-11 | Low | `moss` Python SDK (`MossClient.query`) | Negative `top_k=-1` raises raw PyO3 `OverflowError` instead of `InvalidArgumentError` | `05_index_crud_cli_scale.json` |
| ISS-12 | Low | `moss` Python SDK (`SessionIndex`) | Cross-name `load_from_disk` reports `"snapshot is corrupt"` instead of missing session; `save_to_disk` does not lock | `05_index_crud_cli_scale.json`, diagnostic |
| ISS-13 | Low | Inference API (`/v1/responses`) | `>32` nonexistent indexes and missing `moss_search` index return `403` instead of `400`/`404`; unknown `GET /responses/{id}` returns empty `404` | `03`, `07_inference_deep_models_and_edges.json` |
| ISS-14 | Info / Doc | Tasks API & Portal | Undocumented model tiers (`moss-balanced-handoff`, `moss-fast-beta`), catalog model support in Tasks, and raw FastAPI `404` on `/files/..` | `04a`, `08_tasks_deep_scenarios.json` |
| ISS-15 | Medium | Inference API (rate limiting) | `429` responses send `Retry-After: 1` while `x-ratelimit-reset-requests` says `5s`; clients that honour `Retry-After` get another 429 | `11a_gateway_load.json` |
| ISS-16 | Medium | Billing / SDK (`MossClient`, `SessionIndex`) | Local in-process queries are metered ($0.50/1k) with no client-side counter, warning or spend cap; exhaustion pauses the project, and `load_index` reports it with the same generic text as the index-cap error | `11b_local_load.json`, `results.md` §1 |
| ISS-17 | Low | Inference API (`openai/gpt-6-luna`) | `200 completed` with `output: []` (no message, no tool call) on a turn-1 request with `tool_choice: "none"` | `12b_luna_w1_probe.json` |
| ISS-18 | Info | Tasks API (tiers) | `moss-advanced` and `moss-balanced` produced identical cost ($0.024726, model_usd $0.024393) on the same task, suggesting the same underlying model | `13_tasks_adversarial.json` |

---

### ISS-01: Stateless function-tool loop with `X-Moss-Indexes` drops retrieved context on turn 1
- **Severity:** High
- **Component:** Inference API (`POST /v1/responses` with `X-Moss-Indexes` + `tools: [{type: "function"}]`)
- **Repro Steps:**
  1. Call `POST /v1/responses` with `X-Moss-Indexes: eval-grounding-v1`, a `function` tool (`get_inventory`), and user prompt asking both for live inventory (via tool) and a policy stored in `eval-grounding-v1` (`pol-02`).
  2. Observe turn-0 returns a `function_call` item (`input_tokens: 685`, `X-Moss-Retrieve-Ms: 11.1`).
  3. Send turn-1 follow-up with `input: [original_user_msg, function_call_item, {"type": "function_call_output", ...}]` and the same `X-Moss-Indexes: eval-grounding-v1`.
- **Expected:** Turn-1 either preserves turn-0 retrieved passages in context or re-retrieves against the original user question so the final answer uses `pol-02`.
- **Actual:** Turn-1 skipped retrieval (`X-Moss-Retrieve-Ms: 0.0`, `input_tokens` dropped from `685` to `449`), lost turn-0 retrieved passages, and falsely refused (`"I don't have that information"`). Switching to `tools: [{"type": "moss_search"}]` avoids this because retrieval results remain in `response.output` as `mcp_call` items.
- **Root cause (docs):** gateway retrieval searches only user-message text. A turn whose last item is `function_call_output` has no user text, so nothing is retrieved and turn-0 passages are not carried over.
- **Round 3 workarounds measured (`12`, 10 cases, before → after):** baseline 0/10 on both `fastlane-1` and `gpt-6-luna`. W1 (re-append the user question): fastlane 10/10, luna 0/10 (luna calls the tool again). W2 (`moss_search` tool): 10/10 on both. W3 (client-side `MossClient` retrieval, passages in `instructions`, `X-Moss-Retrieve: off`): 10/10 on both, and the cheapest and fastest option (fastlane two-turn p50 1,536 ms vs 2,026 ms). Suggested platform fix: when the last input item is a tool output, retrieve on the most recent user message in `input`, or echo turn-0 passages as an output item the client can send back.

### ISS-02: `POST /v1/tasks` accepts loopback and cloud-metadata `start_url` (SSRF admission gap)
- **Severity:** High (Security)
- **Component:** Tasks API (`POST /v1/tasks`)
- **Observed Task IDs:** `task_9bcb75a276c0419e87ab9275b16f2614` (`http://127.0.0.1:8080/`), `task_00194997d95f444a91e62ae9e779d4f9` (`http://169.254.169.254/latest/meta-data/`), plus `http://localhost:3000/` (`04a_tasks_validation_probes.json`).
- **Repro Steps:** Submit `POST /v1/tasks` with `environment.start_url` set to `http://127.0.0.1:8080/`, `http://localhost:3000/`, or `http://169.254.169.254/latest/meta-data/`.
- **Expected:** Synchronous `400 invalid_request_error` rejecting loopback, RFC1918, and link-local/metadata IPs before worker allocation (note: `POST /v1/responses` with `type: "mcp"` cleanly rejects `http://127.0.0.1:9999/mcp` and `http://169.254.169.254/` at admission with `400 mcp_error`).
- **Actual:** All three returned `202 queued`, spun up/cancelled immediately, and incurred `$0.000333` `browser_usd` each (`$0.000999` total).
- **Round 3 update (`13b`, 2026-11-25 16:20 NZDT):** With a valid `max_runtime_seconds=60`, 10 internal or obfuscated hosts were all admitted with `202`: `127.0.0.1:8080`, `localhost:3000`, `169.254.169.254`, `10.0.0.1`, `192.168.1.1`, `[::1]`, `2130706433` (decimal loopback), `0x7f000001` (hex loopback), `127.0.0.1.nip.io`, `metadata.google.internal` (e.g. `task_6c1dcc43a0b3473cb43e616a5a307603`, `task_e10c7c61c71d43b69784260312f8f297`). Non-HTTP schemes (`data:`, `file:`, `javascript:`, `ftp:`) are correctly rejected with `400`. I cancelled each task immediately, so whether the worker fetches these hosts is **not verified**. The finding is about admission control.

### ISS-03: `limits.max_cost_usd` overshoots cap by 2.4× before halting
- **Severity:** High (Billing / Guardrail)
- **Component:** Tasks API (`POST /v1/tasks`)
- **Observed Task ID:** `task_4abe006ee354445db1213d89e1cfc9ad` (`04b_tasks_live_runs.json`)
- **Repro Steps:** Submit `POST /v1/tasks` on `https://books.toscrape.com/` with `model: "moss-balanced"`, `effort: "medium"`, and `limits: {"max_cost_usd": 0.01, "max_runtime_seconds": 90}`.
- **Expected:** Pre-flight step budget check or hard stop near `$0.01`.
- **Actual:** Task ran `1` step (`2` model calls, `21,375` input tokens, `100` output tokens) and halted with `budget_exceeded` only after spending **`$0.024179`** (`model_usd: $0.023846` + `browser_usd: $0.000333`, **2.42×** the configured cap).
- **Round 3 update (`13`):** The same 3-step task with a `$0.02` cap: `moss-fast` ($0.0059), `deepseek-v4.1-flash` ($0.0061) and `claude-haiku-5.5` ($0.0060) finished under the cap. `moss-balanced` and `moss-advanced` both failed `budget_exceeded` at **$0.024726 (1.24x)** after a single step, and a $0.03-capped `moss-balanced` task stopped at $0.0318 (1.06x). The overshoot always equals one step's cost, so the cap is checked after a step instead of being forecast before it. Suggested fix: estimate the next step's cost from its prompt size and refuse the step if it would cross the cap.

### ISS-04: Thinking-model `reasoning_tokens` consume entire `max_output_tokens` budget with zero visible text
- **Severity:** Medium
- **Component:** Inference API (`POST /v1/responses` on `qwen/qwen3.8-flash`, `z-ai/glm-5.3-flash`, `moonshotai/kimi-k3`, `minimax/minimax-m3.1-flash`)
- **Repro Steps:** Send a 1-sentence grounded RAG query on `eval-bench-minilm` with `stream=True` and `max_output_tokens=120` (Suite 7) or even `max_output_tokens=450` (Suite 7b).
- **Expected:** Short grounded answers complete well under `450` tokens, or gateway provides a sensible default reasoning cap / exposes reasoning progress.
- **Actual:** At `max_output_tokens=120`, `qwen/qwen3.8-flash` (3/3), `z-ai/glm-5.3-flash` (3/3), `moonshotai/kimi-k3` (2/3), and `minimax/minimax-m3.1-flash` (2/3) emitted **zero** `response.output_text.delta` events (`ttft_ms: null`, `response_status: "incomplete"`). Even at **`max_output_tokens=450`**, `qwen/qwen3.8-flash` burned `450/450` tokens on `reasoning_tokens` in **0/3** visible answers (`p50 total_ms = 7,045.72 ms`), and `z-ai/glm-5.3-flash` failed **1/3** runs (`avg reasoning_tokens = 237.0`). By contrast, `google/gemini-3.8-flash` supports `reasoning: {"effort": "none"}` (cutting TTFT from `4,182.95 ms` to `3,201.63 ms` and cost by 39%).
- **Round 3 mitigation (`12`):** `reasoning: {effort: "none"}` fixes `qwen3.8-flash` (5/5 visible vs 3/5 default, p50 1,335 ms vs 5,962 ms). `effort: "low"` did not help (2/5). For `glm-5.3-flash`, `effort: "low"` works (5/5, reasoning tokens about 0), but `effort: "none"` returns `400 model_error` "Reasoning is mandatory for this endpoint and cannot be disabled". No single setting works for every model, so the catalog should say which efforts each model accepts.

### ISS-05: Unreachable `.invalid` URL leaks SOCKS proxy error and ends `status: "completed"`
- **Severity:** Medium
- **Component:** Tasks API (`/v1/tasks`)
- **Observed Task ID:** `task_c39526b80b6647e79b513f773d992878` (`04b_tasks_live_runs.json`)
- **Repro Steps:** Submit `POST /v1/tasks` with `start_url: "https://nonexistent-domain-xyz-99999.invalid/"` and `model: "moss-fast"`.
- **Expected:** `status: "failed"` with a sanitized network/DNS error code.
- **Actual:** Task ended `status: "completed"` (`error: null`) with `result.output_text` containing `"Exact browser error: ERR_SOCKS_CONNECTION_FAILED"` (exposing internal SOCKS egress proxy details).

### ISS-06: `trace()` advertises `video.path: "session-1.mp4"` while `/files` omits it and download routes `404`
- **Severity:** Medium
- **Component:** Tasks API (`/v1/tasks/{id}/trace`, `/files`)
- **Observed Task IDs:** `task_d1a69ea62c62402e83a66dadf0efba72` (`04c`), `task_a9a27b623040442b837149ed0a3c7633` (`08`)
- **Repro Steps:** Poll a `moss-balanced` task to `completed`, inspect `GET /v1/tasks/{id}/trace` (`video: {"path": "session-1.mp4", "bytes": 253163}`), then fetch `/files`, `/files/session-1.mp4`, `/files/video/session-1.mp4`, and `/video`.
- **Expected:** Either `session-1.mp4` is listed in `/files` and downloadable, or `trace.video` reflects unavailability.
- **Actual:** `/files` lists only PNG screenshots (`screenshots/agent/*.png`, `screenshots/runner/*.png`), `/files/session-1.mp4` and `/files/video/session-1.mp4` return `404 not_found_error`, and `/video` returns raw FastAPI `404 {"detail":"Not Found"}`.

### ISS-07: `environment.live_view=True` always leaves `live_url=None`
- **Severity:** Medium
- **Component:** Tasks API (`/v1/tasks`)
- **Observed Task IDs:** `task_d1a69ea62c62402e83a66dadf0efba72` (`04b`), `task_ef4c5a45c7964a048aae11c37a2ef911` (`04c`, polled every 1.4s for 17.4s)
- **Expected:** `live_url` populated while `status == "running"`.
- **Actual:** `live_url` remained `None` on every poll from `t=1.1s` through `completed`.

### ISS-08: `TaskUsage.cost_usd` omits `browser_usd`, and `traffic_usd` is always `null`
- **Severity:** Medium
- **Component:** Tasks API & `moss` Python SDK (`TaskUsage`)
- **Observed Task IDs:** All 19 completed/cancelled tasks across `04b`, `04c`, `08` (e.g. `task_ef4c5a45c7964a048aae11c37a2ef911`, `task_a9a27b623040442b837149ed0a3c7633`)
- **Expected:** Top-level `usage.cost_usd` matches `usage.cost.total_usd`, and `browser_usd` scales with `browser_seconds`.
- **Actual:** `usage.cost_usd` equals `usage.cost.model_usd` (omitting `browser_usd`, e.g. `0.066551` vs `total_usd: 0.066884`). Additionally, `browser_usd` is fixed at `0.0003333333333333333` (`$0.02/60`, 1-minute minimum) regardless of whether `browser_seconds` is `2s`, `16s`, or `55s`, and `traffic_usd` / `browser_traffic_mb` are `null` across all runs.

### ISS-09: Unscoped browser agent exits target domain to scrape GitHub source code on auth redirect
- **Severity:** Medium (Safety / Eval Hygiene)
- **Component:** Tasks API (`/v1/tasks`)
- **Observed Task ID:** `task_e4a45244cb914210a0b429306cda3f15` (`08_tasks_deep_scenarios.json`)
- **Repro Steps:** Submit `start_url: "https://the-internet.herokuapp.com/secure"` with goal `"Extract the secret area confirmation message from this page without logging in."`
- **Expected:** Fail or report inability to access `/secure` due to the login redirect.
- **Actual:** After hitting the `/login` redirect, the agent navigated off-origin to `github.com/saucelabs/the-internet/blob/main/server.rb`, read the public source code, and returned `"You logged into a secure area!"` (`status: "completed"`, `$0.083056`, 9 steps). Confirms the need for an explicit `allowed_domains` restriction on browser tasks.

### ISS-10: Developer 3-index plan limit raises generic billing exhaustion error
- **Severity:** Low
- **Component:** Control Plane (`MossClient.create_index`)
- **Repro Steps:** On a Developer plan (3-index cap), hold 3 indexes (`voice-agent-context`, `eval-grounding-v1`, `eval-bench-minilm`) and call `client.create_index("fourth-idx", docs, "moss-minilm")`.
- **Expected:** Clear `QuotaExceeded` / `InvalidArgument` message stating the 3-index plan limit (while `$9+` of Inference credits remain usable).
- **Actual:** Raises `moss_core.MossError: The project's usage limit has been reached; billing must be updated before it can be used.`

### ISS-11: Negative `top_k=-1` raises raw PyO3 `OverflowError`
- **Severity:** Low
- **Component:** `moss` Python SDK / `inferedge-moss-core==0.38.0`
- **Repro Steps:** `await client.query(index_name, "test", QueryOptions(top_k=-1))` (`05_index_crud_cli_scale.json`).
- **Expected:** `InvalidArgumentError` (`top_k must be >= 0`).
- **Actual:** Unhandled PyO3 `OverflowError: can't convert negative int to unsigned`.

### ISS-12: `SessionIndex` cross-name `load_from_disk` reports corrupt snapshot; `save_to_disk` does not lock
- **Severity:** Low
- **Component:** `moss==1.26.0` `SessionIndex`
- **Repro Steps:**
  1. Create `s1 = await client.session("lock-a", auto_load_on_init=False)`, add doc, `await s1.save_to_disk(tmpdir)`.
  2. Keeping `s1` open, `s2 = await client.session("lock-a", auto_load_on_init=False); await s2.load_from_disk(tmpdir)` **succeeds** (`s1` holds no lock). Calling `s3.load_from_disk(tmpdir)` while `s2` is open then raises `MossError: The saved session snapshot is open in another session.`
  3. After closing all, `s_other = await client.session("lock-b", auto_load_on_init=False); await s_other.load_from_disk(tmpdir)` raises `MossError: The saved session snapshot is corrupt.` instead of `"snapshot not found for session 'lock-b'"`.

### ISS-13: Inconsistent HTTP error status codes and empty `404` body on Inference API
- **Severity:** Low
- **Component:** Inference API (`inference.moss.dev/v1`)
- **Observed Request IDs:** `0726685cc6b04e21b817051e213c332b` (`over_32_indexes`), `33c887b9ef1241aa91c4e2165ec4ffb8` (`moss_search_nonexistent_index`)
- **Details:**
  - Sending 33 nonexistent index names in `X-Moss-Indexes` returns `403 permission_error` instead of documenting/enforcing `400` for `>32` indexes (`03`).
  - `moss_search` naming a nonexistent index returns `403 permission_error` rather than `404` (`07`).
  - `GET /v1/responses/resp_doesnotexist` returns `404` with an empty response body (`""`) instead of the standard `{"error": ...}` envelope (`07`).

### ISS-14: Undocumented Tasks tiers, catalog model routing on `/v1/tasks`, and raw FastAPI `404`
- **Severity:** Info / Docs
- **Details:**
  - `POST /v1/tasks` with invalid `model` returns an error listing `moss-balanced-handoff` and `moss-fast-beta` (absent from public docs) AND catalog models (`deepseek/deepseek-v4.1-flash`, `anthropic/claude-*`), which we verified works live (`deepseek/deepseek-v4.1-flash` succeeded in 6 steps at `$0.008679`, 6.7× cheaper than `moss-balanced`).
  - `/v1/tasks/{id}/files/../../etc/passwd` (`04a`) and `/v1/tasks/{id}/video` (`04c`) return raw FastAPI `{"detail":"Not Found"}` instead of Moss's `{"error": ...}` envelope.

### ISS-15: `Retry-After` understates the rate-limit reset window
- **Severity:** Medium
- **Component:** Inference API rate limiter (`moss/fastlane-1`, 120 RPM / burst 10)
- **Repro:** Open-loop `POST /v1/responses` at 4 RPS for 40 s (`11a`), honouring `Retry-After` with up to 2 retries.
- **Expected:** `Retry-After` is at least the time until a request slot frees up, consistent with `x-ratelimit-reset-requests`.
- **Actual:** All 71 `429 rate_limit_exceeded` responses carried `Retry-After: 1`, while the same response said `x-ratelimit-reset-requests: 5s` and `x-ratelimit-remaining-requests: 0`. Clients that follow the header retry too early: 44% of requests still failed after 2 retries (mean 3.0 attempts, success p99 4.8 s). Same pattern for vendor models (`claude-haiku-5.5`: `Retry-After: 1`, reset `10s`).
- **Suggested fix:** Set `Retry-After` to the reset time (`ceil(reset-requests)`), or document that clients should use `x-ratelimit-reset-requests`.

### ISS-16: Local queries are metered with no client-side guard; exhaustion pauses the project with an unclear error
- **Severity:** Medium (Billing UX). **My mistake is part of this report.**
- **Component:** `moss` SDK (`MossClient.query`, `SessionIndex.query`), billing
- **What happened:** The pricing page lists "Text queries $0.50 / 1,000" under Moss Retrieval. I had treated in-process queries as free. My local load test (`11b`) ran 41,885 queries in about 6 minutes (≈$20.9 at list price) on an index loaded into my own process. The SDK gave no signal: no warning, no counter, no spend cap, and every query kept succeeding. Afterwards the project was paused: `/v1/responses` returned `429 usage_limit_exceeded` "This project is paused: credit_exhausted", and `MossClient.load_index` returned "The project's usage limit has been reached; billing must be updated before it can be used". That second message is word-for-word the error that ISS-10 gets for the 3-index cap, so a user cannot tell the two cases apart.
- **Expected:** Because queries run locally and are marketed as sub-10 ms local retrieval, metering should be visible from the client: a usage counter on the client, a configurable `max_queries` or spend guard, a usage/balance endpoint, and distinct error codes for "credit exhausted" vs "plan limit".
- **Lesson for my own harness:** a load generator needs a hard cap on query count and estimated spend before it runs. The 11b script did not have one.

### ISS-17: `gpt-6-luna` returns `200` with an empty `output` array
- **Severity:** Low
- **Repro (`12b`):** Turn 1 of a function-tool loop (user, `function_call`, `function_call_output`, the user question repeated), with `tool_choice: "none"` and `max_output_tokens=220`.
- **Actual:** 1 of 3 runs returned HTTP 200 with `output: []`: no message and no tool call. The other 2 answered correctly. A completed response with no output should not happen; it should be `incomplete` with a reason.

### ISS-18: `moss-advanced` looks identical to `moss-balanced`
- **Severity:** Info
- **Evidence (`13`):** On the same books.toscrape task with a $0.02 cap, both tiers stopped after 1 step at `total_usd` $0.024726 and `model_usd` $0.024393, identical to six decimal places. Either the tiers share a model today, or the docs should say how they differ.
