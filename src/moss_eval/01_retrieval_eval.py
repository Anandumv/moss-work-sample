"""
Suite 1: Local Retrieval & Index Eval (MossClient + SessionIndex)
Tests sub-10ms in-process retrieval claim, hybrid vs semantic vs keyword quality,
and creates the grounded eval index `eval-grounding-v1` used by Suite 2.
"""
import asyncio
import time
from moss import DocumentInfo, MossClient, QueryOptions
from moss_eval.common import load_credentials, save_log, summarize_latencies

EVAL_INDEX = "eval-grounding-v1"

EVAL_DOCS = [
    DocumentInfo(
        id="pol-01",
        text="KoruMart Express delivers groceries in 18 minutes across Auckland CBD, Newmarket, and Ponsonby for orders over NZ$35. Orders under NZ$35 incur a NZ$4.99 express fee.",
        metadata={"category": "delivery", "region": "auckland"},
    ),
    DocumentInfo(
        id="pol-02",
        text="Cold-chain dairy and frozen items ship in insulated thermal pouches rated for 90 minutes at 4 degrees Celsius. Temperature excursions above 8C trigger an automatic instant refund.",
        metadata={"category": "cold_chain", "region": "nz"},
    ),
    DocumentInfo(
        id="pol-03",
        text="Perishable produce claims must be submitted within 12 hours of delivery with a single photo. Non-perishable pantry items have a 14-day return window.",
        metadata={"category": "returns", "region": "nz"},
    ),
    DocumentInfo(
        id="pol-04",
        text="Warehouse bin code NZ-AKL-04B stores organic Alphonso mango crates (SKU: MNG-4402) at 13 degrees Celsius. Never store mangoes below 10C.",
        metadata={"category": "warehouse", "sku": "MNG-4402"},
    ),
    DocumentInfo(
        id="pol-05",
        text="Rider payout bonus tier Gold activates after 120 completed drops per week with an on-time rate >= 96.5%. Gold riders receive NZ$2.25 extra per peak-hour drop.",
        metadata={"category": "ops", "tier": "gold"},
    ),
    DocumentInfo(
        id="pol-06",
        text="KoruMart API rate limit for webhook partners is 240 requests per minute per tenant key, with burst allowance of 30 requests in 5 seconds.",
        metadata={"category": "api", "service": "webhooks"},
    ),
    DocumentInfo(
        id="pol-07",
        text="Gift cards expire 36 months from issue date and cannot be redeemed for tobacco, lottery tickets, or delivery pass subscriptions.",
        metadata={"category": "billing", "topic": "gift_cards"},
    ),
    DocumentInfo(
        id="pol-08",
        text="The KoruPass annual membership costs NZ$79/year (or NZ$8.99/month) and waives express delivery fees on orders above NZ$20.",
        metadata={"category": "membership", "plan": "korupass"},
    ),
    DocumentInfo(
        id="pol-09",
        text="If a substitution item is more expensive than the ordered SKU, KoruMart absorbs up to NZ$5.00 of the price difference automatically.",
        metadata={"category": "substitutions", "policy": "price_cap"},
    ),
    DocumentInfo(
        id="pol-10",
        text="Customer support escalation code ESC-917 routes directly to the Auckland cold-chain duty manager between 06:00 and 23:00 NZDT.",
        metadata={"category": "support", "code": "ESC-917"},
    ),
    DocumentInfo(
        id="pol-11",
        text="മലയാളം പിന്തുണ: കൊച്ചി വെയർഹൗസ് (KOC-02) രാവിലെ 5:30 മുതൽ രാത്രി 11:00 വരെ പ്രവർത്തിക്കുന്നു. കേടായ പച്ചക്കറികൾക്ക് 6 മണിക്കൂറിനുള്ളിൽ റീഫണ്ട് ലഭിക്കും.",
        metadata={"category": "regional", "lang": "ml"},
    ),
    DocumentInfo(
        id="pol-12",
        text="Scheduled maintenance window for dark-store inventory sync runs every Tuesday 02:15–02:45 UTC, during which slot reservation API returns HTTP 503 Retry-After: 1800.",
        metadata={"category": "ops", "maintenance": "weekly"},
    ),
]

EVAL_QA = [
    {"id": "q1", "query": "How fast is KoruMart Express delivery in Auckland and what is the minimum order for free delivery?", "expected_doc": "pol-01", "expected_fact": "18 minutes"},
    {"id": "q2", "query": "At what temperature threshold do cold-chain dairy items get an automatic refund?", "expected_doc": "pol-02", "expected_fact": "8C"},
    {"id": "q3", "query": "How long do I have to file a claim for spoiled produce?", "expected_doc": "pol-03", "expected_fact": "12 hours"},
    {"id": "q4", "query": "Which warehouse bin and temperature holds SKU MNG-4402?", "expected_doc": "pol-04", "expected_fact": "NZ-AKL-04B"},
    {"id": "q5", "query": "What on-time rate and weekly drops do riders need for Gold bonus tier?", "expected_doc": "pol-05", "expected_fact": "96.5%"},
    {"id": "q6", "query": "What is the webhook partner API rate limit?", "expected_doc": "pol-06", "expected_fact": "240"},
    {"id": "q7", "query": "How much does KoruPass cost per year and what is its order minimum?", "expected_doc": "pol-08", "expected_fact": "NZ$79"},
    {"id": "q8", "query": "What happens when a substitute grocery item costs more than what I ordered?", "expected_doc": "pol-09", "expected_fact": "NZ$5.00"},
    {"id": "q9", "query": "What escalation code reaches the Auckland cold-chain duty manager?", "expected_doc": "pol-10", "expected_fact": "ESC-917"},
    {"id": "q10", "query": "കൊച്ചി വെയർഹൗസ് KOC-02 പ്രവർത്തന സമയം എപ്പോഴാണ്?", "expected_doc": "pol-11", "expected_fact": "5:30"},
]


async def main() -> None:
    pid, pkey = load_credentials()
    client = MossClient(pid, pkey)

    existing = {idx.name for idx in await client.list_indexes()}
    t0 = time.perf_counter()
    if EVAL_INDEX in existing:
        await client.add_docs(EVAL_INDEX, EVAL_DOCS)
        index_action = "updated"
    else:
        await client.create_index(EVAL_INDEX, EVAL_DOCS, "moss-minilm")
        index_action = "created"
    create_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Load both voice-agent-context (50 docs) and eval-grounding-v1 (12 docs)
    t_load0 = time.perf_counter()
    await client.load_index("voice-agent-context")
    load_vac_ms = round((time.perf_counter() - t_load0) * 1000, 2)

    t_load1 = time.perf_counter()
    await client.load_index(EVAL_INDEX)
    load_eval_ms = round((time.perf_counter() - t_load1) * 1000, 2)

    # Local query latency benchmark (50 warm queries on voice-agent-context)
    vac_queries = [
        "I want a refund for a damaged item",
        "Where is my package right now?",
        "Cancel my subscription",
        "Change delivery address",
        "Speak to a human agent",
    ]
    warm_latencies_ms = []
    for i in range(50):
        q = vac_queries[i % len(vac_queries)]
        t_q = time.perf_counter()
        res = await client.query("voice-agent-context", q, QueryOptions(top_k=5, alpha=0.8))
        warm_latencies_ms.append((time.perf_counter() - t_q) * 1000)

    # Retrieval accuracy eval across alpha settings on eval-grounding-v1
    alpha_results = {}
    for alpha in [1.0, 0.6, 0.0]:
        hits_at_1 = 0
        hits_at_3 = 0
        mrr_sum = 0.0
        q_lat = []
        per_q = []
        for qa in EVAL_QA:
            t_q = time.perf_counter()
            res = await client.query(EVAL_INDEX, qa["query"], QueryOptions(top_k=5, alpha=alpha))
            dt = (time.perf_counter() - t_q) * 1000
            q_lat.append(dt)
            ids = [d.id for d in res.docs]
            rank = ids.index(qa["expected_doc"]) + 1 if qa["expected_doc"] in ids else None
            if rank == 1:
                hits_at_1 += 1
            if rank is not None and rank <= 3:
                hits_at_3 += 1
            if rank is not None:
                mrr_sum += 1.0 / rank
            per_q.append({
                "id": qa["id"],
                "expected_doc": qa["expected_doc"],
                "top_ids": ids[:3],
                "top_scores": [round(d.score, 4) for d in res.docs[:3]],
                "rank": rank,
                "latency_ms": round(dt, 2),
            })
        n = len(EVAL_QA)
        alpha_results[str(alpha)] = {
            "recall_at_1": hits_at_1 / n,
            "recall_at_3": hits_at_3 / n,
            "mrr": round(mrr_sum / n, 4),
            "latency_ms": summarize_latencies(q_lat),
            "per_query": per_q,
        }

    # SessionIndex local-first test (pure in-memory, no cloud push)
    t_s0 = time.perf_counter()
    session = await client.session("local-eval-session")
    await session.add_docs(EVAL_DOCS)
    sess_add_ms = round((time.perf_counter() - t_s0) * 1000, 2)
    sess_lat = []
    for qa in EVAL_QA:
        t_q = time.perf_counter()
        await session.query(qa["query"], QueryOptions(top_k=3))
        sess_lat.append((time.perf_counter() - t_q) * 1000)
    await session.close()

    report = {
        "index_action": index_action,
        "create_or_update_ms": create_ms,
        "load_voice_agent_context_ms": load_vac_ms,
        "load_eval_grounding_v1_ms": load_eval_ms,
        "warm_local_query_50x_ms": summarize_latencies(warm_latencies_ms),
        "session_add_12_docs_ms": sess_add_ms,
        "session_query_10x_ms": summarize_latencies(sess_lat),
        "eval_by_alpha": alpha_results,
    }
    path = save_log("01_retrieval_eval.json", report, [pkey])
    print("Saved:", path)
    print("Load VAC ms:", load_vac_ms, "| Load Eval ms:", load_eval_ms)
    print("Warm 50x local query (ms):", report["warm_local_query_50x_ms"])
    print("Session 10x query (ms):", report["session_query_10x_ms"])
    for a, r in alpha_results.items():
        print(f"alpha={a}: R@1={r['recall_at_1']:.2f} R@3={r['recall_at_3']:.2f} MRR={r['mrr']:.4f} p50={r['latency_ms']['p50']}ms")


if __name__ == "__main__":
    asyncio.run(main())
