import asyncio
import importlib
import time
import httpx
import pytest
from moss import MossClient, MossInference, QueryOptions
from moss.inference import InferenceError
from moss_eval.common import (
    estimate_inference_cost,
    load_credentials,
    scrub_secrets,
    summarize_latencies,
)

_m06 = importlib.import_module("moss_eval.06_deep_retrieval_eval")
_m06b = importlib.import_module("moss_eval.06b_rag_deep_ablation")


def test_secret_scrubber_redacts_keys_and_tokens():
    raw = {"auth": "Bearer pid123:moss_secret_abc", "nested": ["moss_secret_abc", "ok"]}
    cleaned = scrub_secrets(raw, ["moss_secret_abc"])
    assert "moss_secret_abc" not in str(cleaned)
    assert cleaned["nested"][0] == "[REDACTED]"


def test_summarize_latencies_percentiles():
    s = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["count"] == 5
    assert s["min"] == 10.0
    assert s["p50"] == 30.0
    assert s["max"] == 50.0
    assert s["mean"] == 30.0


def test_estimate_inference_cost_fastlane():
    cost = estimate_inference_cost(
        "moss/fastlane-1",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cached_tokens=0,
        retrievals=1000,
    )
    # $0.25 input + $1.75 output + $0.50/1k retrievals = $2.50
    assert cost == pytest.approx(2.50, rel=1e-4)


def test_ir_metrics_exact():
    m1 = _m06.compute_ir_metrics(["d2", "d1", "d3"], "d1", k=5)
    assert m1["r1"] == 0.0
    assert m1["r3"] == 1.0
    assert m1["mrr5"] == pytest.approx(0.5)
    m2 = _m06.compute_ir_metrics(["d1", "d2"], "d1", k=5)
    assert m2["r1"] == 1.0
    assert m2["ndcg5"] == pytest.approx(1.0)


def test_abstention_matcher_handles_curly_apostrophe():
    assert _m06b.norm_match("I don\u2019t have that information.", [], expect_refusal=True) is True
    assert _m06b.norm_match("The SLA is 18 minutes and $35.", ["18", "35"], expect_refusal=False) is True
    assert _m06b.norm_match("The SLA is 30 minutes.", ["18", "35"], expect_refusal=False) is False


def test_sdk_rejects_empty_credentials():
    with pytest.raises(InferenceError) as exc:
        MossInference("", "moss_fake")
    assert exc.value.type == "invalid_request_error"
    assert exc.value.status is None


def test_sdk_rejects_comma_in_index_name():
    pid, pkey = load_credentials()

    async def _run():
        async with MossInference(pid, pkey) as client:
            await client.responses.create(input="hi", indexes=["a,b"])

    with pytest.raises(InferenceError) as exc:
        asyncio.run(_run())
    assert exc.value.type == "invalid_request_error"


def test_models_endpoint_lists_fastlane():
    pid, pkey = load_credentials()
    r = httpx.get(
        "https://inference.moss.dev/v1/models",
        headers={"Authorization": f"Bearer {pid}:{pkey}"},
        timeout=15.0,
    )
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "moss/fastlane-1" in ids


def test_local_retrieval_sub_10ms():
    pid, pkey = load_credentials()

    async def _run():
        client = MossClient(pid, pkey)
        await client.load_index("voice-agent-context")
        latencies = []
        for _ in range(10):
            t0 = time.perf_counter()
            res = await client.query("voice-agent-context", "damaged order refund", QueryOptions(top_k=3))
            latencies.append((time.perf_counter() - t0) * 1000)
            assert len(res.docs) > 0
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        assert p50 < 10.0, f"Expected local query p50 < 10ms, got {p50:.2f}ms"
        await client.close()

    asyncio.run(_run())
