"""
Suite 6: Deep Retrieval & End-to-End RAG Evaluation (60 Labelled Queries x 120 Docs)
Compares moss-minilm (cloud loaded index + SessionIndex) vs moss-mediumlm (SessionIndex)
across alpha={1.0, 0.6, 0.0} and candidate_depth=50, reporting Recall@1/3/5, MRR@5, and nDCG@5
overall and by stratum (direct, paraphrase, adversarial, multilingual).
Also evaluates end-to-end grounded RAG answer accuracy + hallucination rate across
moss/fastlane-1 (X-Moss-Indexes vs moss_search), openai/gpt-6-luna, and anthropic/claude-haiku-5.5.
"""
import asyncio
import json
import math
import time
from typing import Any
import httpx
from moss import DocumentInfo, MossClient, QueryOptions
from moss_eval.common import (
    estimate_inference_cost,
    load_credentials,
    save_log,
    summarize_latencies,
)

BASE_URL = "https://inference.moss.dev/v1"
IDX_MINI = "eval-bench-minilm"
IDX_MED = "eval-bench-mediumlm"


def build_corpus() -> list[DocumentInfo]:
    docs: list[DocumentInfo] = [
        # --- Grocery / Quick-Commerce (15 targets + 15 distractors) ---
        DocumentInfo(id="gr-01", text="KoruMart Auckland CBD express courier SLA is 18 minutes for baskets over NZ$35; baskets below NZ$35 pay a NZ$4.99 surcharge.", metadata={"domain": "grocery", "city": "auckland"}),
        DocumentInfo(id="gr-01d", text="KoruMart Wellington CBD express courier SLA is 24 minutes for baskets over NZ$40; baskets below NZ$40 pay a NZ$5.49 surcharge.", metadata={"domain": "grocery", "city": "wellington"}),
        DocumentInfo(id="gr-02", text="Organic Alphonso mangoes (SKU MNG-4402) are stored in Auckland bin NZ-AKL-04B at 13 degrees Celsius; chilling below 10C causes skin pitting.", metadata={"domain": "grocery", "sku": "MNG-4402"}),
        DocumentInfo(id="gr-02d", text="Kent mangoes (SKU MNG-4403) are stored in Auckland bin NZ-AKL-04C at 12 degrees Celsius; ripened batches move to ambient rack 09.", metadata={"domain": "grocery", "sku": "MNG-4403"}),
        DocumentInfo(id="gr-03", text="Active 2026 Policy: Fresh produce spoilage refunds require photo proof within 12 hours of drop-off for automatic wallet credit.", metadata={"domain": "grocery", "year": "2026"}),
        DocumentInfo(id="gr-03d", text="Deprecated 2025 Policy (superseded): Fresh produce spoilage claims previously allowed a 48-hour window with manual agent review.", metadata={"domain": "grocery", "year": "2025"}),
        DocumentInfo(id="gr-04", text="Cold-chain dairy pouches hold 4C for up to 90 minutes; any sensor reading exceeding 8C triggers an automatic instant refund.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-04d", text="Frozen gelato dry-ice packs hold -18C for up to 60 minutes; sensor excursions above -10C trigger a partial credit.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-05", text="Rider Gold payout tier requires 120 completed weekly deliveries and at least 96.5% on-time arrival, paying NZ$2.25 peak bonus per drop.", metadata={"domain": "grocery", "tier": "gold"}),
        DocumentInfo(id="gr-05d", text="Rider Silver payout tier requires 75 completed weekly deliveries and at least 93.0% on-time arrival, paying NZ$1.10 peak bonus per drop.", metadata={"domain": "grocery", "tier": "silver"}),
        DocumentInfo(id="gr-06", text="When a grocery picker substitutes an out-of-stock item with a pricier alternative, KoruMart absorbs up to NZ$5.00 of the upward price gap.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-06d", text="When a customer manually requests a brand upgrade in chat after checkout, the full price difference is charged to the saved payment card.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-07", text="KoruPass annual subscription costs NZ$79 per year and eliminates express delivery charges on orders above NZ$20.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-07d", text="KoruPass Business team plan costs NZ$299 per year for up to 10 office buyers and waives scheduled morning slot fees.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-08", text="Glass-bottle kombucha crates (SKU KMB-881) must never be stacked more than 3 tiers high on electric cargo bikes.", metadata={"domain": "grocery", "sku": "KMB-881"}),
        DocumentInfo(id="gr-08d", text="Aluminum-can sparkling water packs (SKU SPK-882) can be stacked up to 6 tiers high in standard delivery totes.", metadata={"domain": "grocery", "sku": "SPK-882"}),
        DocumentInfo(id="gr-09", text="KoruMart gift vouchers remain valid for 36 months from activation and exclude tobacco, lottery, and pass renewals.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-09d", text="Promotional referral coupons expire 14 days after issue and apply only to first-time app orders over NZ$50.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-10", text="Escalation code ESC-917 connects directly to the Auckland cold-chain duty supervisor from 06:00 to 23:00 NZDT.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-10d", text="Escalation code ESC-918 routes billing chargeback disputes to the Melbourne finance queue from 09:00 to 17:00 AEST.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-11", text="Dark-store inventory sync maintenance occurs every Tuesday 02:15-02:45 UTC, returning HTTP 503 with Retry-After: 1800.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-11d", text="Payment gateway reconciliation batch runs every Sunday 03:00-03:15 UTC without affecting checkout slot reservations.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-12", text="Christchurch depot accepts returned insulated tote bags via riders for a NZ$1.50 green-pack deposit credit per bag.", metadata={"domain": "grocery", "city": "christchurch"}),
        DocumentInfo(id="gr-12d", text="Dunedin partner lockers do not support reusable tote return credits; customers keep paper bags for curbside recycling.", metadata={"domain": "grocery", "city": "dunedin"}),
        DocumentInfo(id="gr-13", text="Allergy labeling rule: dark-store packers must place peanut-containing items in red-seal allergen sleeves separate from bakery goods.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-13d", text="Halal-certified meat SKUs are sealed in green-stripe butcher bags and stored on the top shelf of chiller bay B2.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-14", text="Kochi warehouse KOC-02 operates 05:30 to 23:00 IST and processes damaged vegetable refunds within 6 hours.", metadata={"domain": "grocery", "city": "kochi"}),
        DocumentInfo(id="gr-14d", text="Bengaluru warehouse BLR-07 operates 06:00 to 22:00 IST and processes dry-staple shortage claims within 24 hours.", metadata={"domain": "grocery", "city": "bengaluru"}),
        DocumentInfo(id="gr-15", text="Age-restricted alcohol deliveries require scanning a physical passport or NZ driver license; expired IDs are rejected.", metadata={"domain": "grocery"}),
        DocumentInfo(id="gr-15d", text="Non-alcoholic zero-beer multipacks do not require ID verification at doorstep drop-off.", metadata={"domain": "grocery"}),

        # --- SaaS Billing & APIs (15 targets + 15 distractors) ---
        DocumentInfo(id="api-01", text="Webhook partner API allows 240 requests per minute per tenant key with a 5-second burst cap of 30 requests.", metadata={"domain": "api"}),
        DocumentInfo(id="api-01d", text="Public catalog read API allows 1,200 requests per minute per IP with a 10-second burst cap of 100 requests.", metadata={"domain": "api"}),
        DocumentInfo(id="api-02", text="Idempotency keys on POST /v2/payouts persist for 24 hours; reusing a key with a mismatched JSON body returns HTTP 422.", metadata={"domain": "api"}),
        DocumentInfo(id="api-02d", text="Pagination cursors on GET /v2/ledger expire after 15 minutes and return HTTP 410 Gone when stale.", metadata={"domain": "api"}),
        DocumentInfo(id="api-03", text="Active 2026 Auth Spec: Service-to-service tokens use Ed25519 JWTs with a mandatory 300-second TTL.", metadata={"domain": "api", "year": "2026"}),
        DocumentInfo(id="api-03d", text="Deprecated 2024 Auth Spec: Legacy HMAC-SHA256 static bearer tokens had a 24-hour TTL before sunset.", metadata={"domain": "api", "year": "2024"}),
        DocumentInfo(id="api-04", text="Enterprise plan overage billing charges US$0.12 per 1,000 extra vector lookups beyond the 10M monthly quota.", metadata={"domain": "api"}),
        DocumentInfo(id="api-04d", text="Pro plan overage billing charges US$0.25 per 1,000 extra vector lookups beyond the 1M monthly quota.", metadata={"domain": "api"}),
        DocumentInfo(id="api-05", text="Webhook signatures are sent in X-Koru-Sig-256 and tolerate at most 180 seconds of clock skew.", metadata={"domain": "api"}),
        DocumentInfo(id="api-05d", text="Batch export S3 presigned URLs remain valid for 900 seconds after generation.", metadata={"domain": "api"}),
        DocumentInfo(id="api-06", text="Deleting a tenant workspace places all indexes in soft-delete quarantine for 7 days before permanent erasure.", metadata={"domain": "api"}),
        DocumentInfo(id="api-06d", text="Rotating an API key revokes the old secret immediately with zero grace period unless dual-key mode is enabled.", metadata={"domain": "api"}),
        DocumentInfo(id="api-07", text="Streaming SSE responses emit heartbeat comments every 15 seconds while upstream inference workers queue.", metadata={"domain": "api"}),
        DocumentInfo(id="api-07d", text="WebSocket voice bridges send binary PCM16 frames at 24 kHz with 20 ms packet intervals.", metadata={"domain": "api"}),
        DocumentInfo(id="api-08", text="SOC 2 audit logs are retained in immutable WORM buckets for 400 days and exported as NDJSON.", metadata={"domain": "api"}),
        DocumentInfo(id="api-08d", text="Debug trace payloads containing PII are scrubbed after 72 hours in non-production sandboxes.", metadata={"domain": "api"}),
        DocumentInfo(id="api-09", text="Custom SSO via SAML 2.0 requires SHA-256 assertions and maps the okta_role claim to RBAC Operator.", metadata={"domain": "api"}),
        DocumentInfo(id="api-09d", text="SCIM user deprovisioning syncs every 30 minutes and suspends inactive contractor seats automatically.", metadata={"domain": "api"}),
        DocumentInfo(id="api-10", text="Model router fallback switches from primary GPU pool ap-southeast-2 to us-west-2 if p99 TTFT exceeds 650 ms for 60s.", metadata={"domain": "api"}),
        DocumentInfo(id="api-10d", text="Canary traffic splitter caps experimental model weights at 5% of non-enterprise shadow requests.", metadata={"domain": "api"}),
        DocumentInfo(id="api-11", text="GraphQL introspection is disabled in production; schema SDL is published as an artifact per release tag.", metadata={"domain": "api"}),
        DocumentInfo(id="api-11d", text="OpenAPI 3.1 JSON specs are served unauthenticated at /openapi.json on staging clusters only.", metadata={"domain": "api"}),
        DocumentInfo(id="api-12", text="Invoice PDF generation fails with error INV-409 when a customer tax ID lacks the 9-digit NZBN prefix.", metadata={"domain": "api"}),
        DocumentInfo(id="api-12d", text="Credit-note issuance fails with error CRD-402 if the original settlement batch is still pending.", metadata={"domain": "api"}),
        DocumentInfo(id="api-13", text="TLS 1.2 is rejected on ingress gateway v3; clients must negotiate TLS 1.3 with ALPN h2.", metadata={"domain": "api"}),
        DocumentInfo(id="api-13d", text="Internal service mesh sidecars enforce mutual TLS using SPIFFE X.509 certificates rotated hourly.", metadata={"domain": "api"}),
        DocumentInfo(id="api-14", text="Bulk CSV import rejects rows exceeding 64 KB per cell and aborts the batch if malformed UTF-8 exceeds 1%.", metadata={"domain": "api"}),
        DocumentInfo(id="api-14d", text="Parquet export splits partitions at 128 MB uncompressed row groups using Zstandard level 3.", metadata={"domain": "api"}),
        DocumentInfo(id="api-15", text="Sandbox test card 4111-0000-0000-9911 simulates an issuer 3DS timeout after 12 seconds.", metadata={"domain": "api"}),
        DocumentInfo(id="api-15d", text="Sandbox test card 4111-0000-0000-0002 simulates immediate insufficient funds decline (code 51).", metadata={"domain": "api"}),

        # --- Cloud Infra & SRE (15 targets + 15 distractors) ---
        DocumentInfo(id="sre-01", text="Kubernetes node pool gpu-l40s-prod scales from 2 to 16 nodes when GPU KV-cache utilization stays above 78% for 90 seconds.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-01d", text="Kubernetes node pool cpu-worker-prod scales from 4 to 40 nodes when CPU load exceeds 70% for 120 seconds.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-02", text="Postgres primary failover via Patroni completes within 25 seconds using synchronous replication to one standby in AZ-b.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-02d", text="Read-replica lag above 500 ms automatically removes the replica from PgBouncer read pools.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-03", text="Redis cluster eviction policy is volatile-lru with maxmemory capped at 48 GB per shard.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-03d", text="Memcached session tier uses slab-rebalance with 16 GB per pod and 6-hour key TTL.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-04", text="Runbook RB-77: When Kafka consumer lag on order-events exceeds 50,000 messages, double partition concurrency to 32.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-04d", text="Runbook RB-78: When dead-letter topic dlq-emails exceeds 5,000 messages, pause SendGrid retry workers.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-05", text="Egress proxy firewall blocks RFC1918, loopback 127.0.0.0/8, and cloud metadata 169.254.169.254 at DNS and socket layers.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-05d", text="Ingress WAF rate-limits /auth/login to 10 POSTs per minute per /24 subnet.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-06", text="ClickHouse analytics retention drops raw request spans after 30 days while keeping 1-minute rollups for 18 months.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-06d", text="Prometheus metrics scrape interval is 15 seconds with 14-day local TSDB retention before Thanos upload.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-07", text="Container images must be signed with Cosign keyless OIDC before ArgoCD allows sync to prod-nz.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-07d", text="Trivy vulnerability scan blocks builds with unpatched CRITICAL CVEs older than 7 days.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-08", text="gRPC deadline propagation defaults to 1,500 ms for retrieval subcalls and 8,000 ms for browser orchestration.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-08d", text="HTTP keep-alive idle pool timeout on Envoy edge proxies is set to 75 seconds.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-09", text="PagerDuty Sev-1 incident bridge auto-pages the ML Platform on-call if 5xx error ratio exceeds 1.5% over 3 minutes.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-09d", text="PagerDuty Sev-2 Slack alert fires if p95 queue depth exceeds 200 jobs for 10 minutes.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-10", text="Terraform state bucket koru-tf-state-prod uses S3 versioning and DynamoDB table koru-tf-locks.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-10d", text="Ansible bastion inventory refreshes dynamically from AWS EC2 tags env=staging.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-11", text="Active 2026 Backup Rule: Cross-region snapshots replicate from ap-southeast-2 (Sydney) to ap-southeast-4 (Melbourne) every 4 hours.", metadata={"domain": "sre", "year": "2026"}),
        DocumentInfo(id="sre-11d", text="Deprecated 2024 Backup Rule: Nightly snapshots previously replicated from us-east-1 to us-west-2 once per 24 hours.", metadata={"domain": "sre", "year": "2024"}),
        DocumentInfo(id="sre-12", text="Vector index cold-start preloader warms top 50 active tenant indexes on pod startup via initContainer.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-12d", text="CronJob index-compactor merges tombstones every night at 01:00 UTC.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-13", text="NVIDIA driver 565.57 with CUDA 12.6 is pinned on all inference nodes running vLLM 0.7.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-13d", text="CPU embedding worker nodes run ONNX Runtime 1.20 on AWS Graviton4 c8g.4xlarge.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-14", text="DNS TTL on inference.koru.internal is lowered to 30 seconds prior to blue/green cutovers.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-14d", text="External CDN apex CNAME uses 300-second TTL behind Cloudflare Orange Cloud.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-15", text="Synthetic blackbox probe runs every 60 seconds from Auckland, Tokyo, and Oregon to measure end-to-end TTFT.", metadata={"domain": "sre"}),
        DocumentInfo(id="sre-15d", text="Chaos Mesh pod-kill experiments run Wednesdays 14:00 UTC in staging-chaos namespace only.", metadata={"domain": "sre"}),

        # --- Clinical & Logistics Compliance (15 targets + 15 distractors) ---
        DocumentInfo(id="cmp-01", text="NZ Customs air-freight clearance for fresh berries requires MPI phytosanitary certificate PHY-NZ-21 before wheels-up.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-01d", text="Sea-freight dry-grain imports require fumigation declaration FUM-NZ-09 at least 48 hours prior to port arrival.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-02", text="HIPAA Business Associate Agreement addendum v4 mandates AES-256-GCM encryption at rest with customer-managed KMS keys.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-02d", text="GDPR Standard Contractual Clauses Annex II covers EU resident support ticket pseudonymization.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-03", text="Clinical triage voice bots must play the non-diagnostic disclaimer within the first 4 seconds of call connection.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-03d", text="Retail order-status voice bots play a brief call-recording notice only when transferring to a human specialist.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-04", text="Lithium-ion battery replacements (UN3481) are capped at 2 cells per parcel via domestic NZ courier.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-04d", text="Dry-ice medical sample shippers (UN1845) must display diamond hazard labels and net weight in kilograms.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-05", text="Data deletion requests under NZ Privacy Act Principle 9 must be fulfilled and logged within 20 working days.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-05d", text="Payroll tax ledger records must be preserved for 7 years under Inland Revenue statutory rules.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-06", text="Medical device firmware updates require dual sign-off from QA Lead and Regulatory Officer before OTA rollout.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-06d", text="Mobile consumer app UI hotfixes require single peer-review approval and automated smoke pass.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-07", text="Tauranga port container demurrage free-time ends 72 hours after vessel discharge.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-07d", text="Auckland airport air-cargo storage fees begin 24 hours after flight manifest release.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-08", text="Forklift battery charging bay temperatures above 45C automatically trip the facility ventilation interlock.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-08d", text="Ambient dry-goods aisle humidity above 75% triggers portable dehumidifier dispatch.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-09", text="Active 2026 Telehealth Rule: Video consult recordings are disabled by default unless the patient gives explicit verbal opt-in.", metadata={"domain": "compliance", "year": "2026"}),
        DocumentInfo(id="cmp-09d", text="Deprecated 2023 Telehealth Rule: Audio transcripts were archived automatically for 90 days unless opted out.", metadata={"domain": "compliance", "year": "2023"}),
        DocumentInfo(id="cmp-10", text="Pharmaceutical cold-box courier sign-off requires recording both recipient signature and digital probe temp between 2C and 8C.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-10d", text="Over-the-counter vitamin bottles ship in uninsulated padded mailers up to 30C ambient.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-11", text="Biosecurity border dogs inspect incoming wooden pallets for untreated bark under ISPM-15 stamp verification.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-11d", text="Plastic Euro-pallets undergo high-pressure steam wash after every 10 warehouse turns.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-12", text="Incident disclosure to the Office of the Privacy Commissioner is mandatory within 72 hours for notifiable breaches.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-12d", text="Internal security near-miss retro must be drafted within 5 business days in Confluence.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-13", text="Drone last-mile trials in Northland operate strictly below 120 meters AGL under CAA Part 102 authorization.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-13d", text="Autonomous sidewalk bots in Wellington CBD are capped at 6 km/h on pedestrian footpaths.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-14", text="Vendor penetration test reports are valid for 12 months and must cover OWASP API Top 10.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-14d", text="Quarterly phishing simulations target 100% of corporate staff with mandatory 15-minute refresher for clicks.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-15", text="Infant formula export batches require MPI Lot Traceability Code LOT-IF-88 on every outer carton.", metadata={"domain": "compliance"}),
        DocumentInfo(id="cmp-15d", text="Whole milk powder bulk sacks carry export dairy mark NZ-DM-401 on pallet manifests.", metadata={"domain": "compliance"}),
    ]
    return docs


EVAL_60 = [
    {"id": "d01", "stratum": "direct", "query": "What is the KoruMart Auckland CBD express courier SLA and minimum basket?", "target": "gr-01", "fact": ["18", "35"]},
    {"id": "d02", "stratum": "direct", "query": "What triggers an automatic instant refund for cold-chain dairy pouches?", "target": "gr-04", "fact": ["8c"]},
    {"id": "d03", "stratum": "direct", "query": "How much does KoruPass annual subscription cost and when does it waive express fees?", "target": "gr-07", "fact": ["79", "20"]},
    {"id": "d04", "stratum": "direct", "query": "How long are KoruMart gift vouchers valid?", "target": "gr-09", "fact": ["36"]},
    {"id": "d05", "stratum": "direct", "query": "What is escalation code ESC-917 used for?", "target": "gr-10", "fact": ["auckland", "cold-chain"]},
    {"id": "d06", "stratum": "direct", "query": "What is the webhook partner API rate limit and burst cap?", "target": "api-01", "fact": ["240", "30"]},
    {"id": "d07", "stratum": "direct", "query": "How long do POST /v2/payouts idempotency keys persist?", "target": "api-02", "fact": ["24", "422"]},
    {"id": "d08", "stratum": "direct", "query": "What is the Enterprise plan overage fee per 1,000 extra vector lookups?", "target": "api-04", "fact": ["0.12"]},
    {"id": "d09", "stratum": "direct", "query": "Which header holds webhook signatures and what clock skew is tolerated?", "target": "api-05", "fact": ["x-koru-sig-256", "180"]},
    {"id": "d10", "stratum": "direct", "query": "When does Kubernetes node pool gpu-l40s-prod scale up?", "target": "sre-01", "fact": ["78%", "90"]},
    {"id": "d11", "stratum": "direct", "query": "What is Postgres primary failover time via Patroni?", "target": "sre-02", "fact": ["25"]},
    {"id": "d12", "stratum": "direct", "query": "What does Runbook RB-77 say about Kafka consumer lag?", "target": "sre-04", "fact": ["50,000", "32"]},
    {"id": "d13", "stratum": "direct", "query": "Which phytosanitary certificate is required for NZ Customs fresh berry air-freight?", "target": "cmp-01", "fact": ["phy-nz-21"]},
    {"id": "d14", "stratum": "direct", "query": "How quickly must NZ Privacy Act Principle 9 deletion requests be fulfilled?", "target": "cmp-05", "fact": ["20"]},
    {"id": "d15", "stratum": "direct", "query": "What is the maximum altitude for Northland drone last-mile trials?", "target": "cmp-13", "fact": ["120"]},

    {"id": "p01", "stratum": "paraphrase", "query": "How much extra cost does our store eat if a shopper gets a pricier replacement product?", "target": "gr-06", "fact": ["5.00"]},
    {"id": "p02", "stratum": "paraphrase", "query": "Are e-bike riders allowed to pile fizzy fermented drinks four boxes tall?", "target": "gr-08", "fact": ["3"]},
    {"id": "p03", "stratum": "paraphrase", "query": "When does the weekly dark-store stock job break slot bookings with a retry header?", "target": "gr-11", "fact": ["tuesday", "02:15"]},
    {"id": "p04", "stratum": "paraphrase", "query": "How do warehouse staff prevent cross-contamination between nuts and bread items?", "target": "gr-13", "fact": ["red-seal"]},
    {"id": "p05", "stratum": "paraphrase", "query": "Can a courier hand over wine if the customer only shows an out-of-date ID card?", "target": "gr-15", "fact": ["rejected"]},
    {"id": "p06", "stratum": "paraphrase", "query": "What happens to search data right after an enterprise customer closes their account?", "target": "api-06", "fact": ["7 days"]},
    {"id": "p07", "stratum": "paraphrase", "query": "How do streaming clients tell the difference between a dead socket and a queued LLM?", "target": "api-07", "fact": ["15"]},
    {"id": "p08", "stratum": "paraphrase", "query": "What triggers automatic cross-region routing away from Sydney GPUs during slowdowns?", "target": "api-10", "fact": ["650"]},
    {"id": "p09", "stratum": "paraphrase", "query": "Why would billing fail to render a New Zealand customer's tax receipt?", "target": "api-12", "fact": ["inv-409"]},
    {"id": "p10", "stratum": "paraphrase", "query": "Which fake Visa number triggers a 3D Secure hang in staging?", "target": "api-15", "fact": ["9911"]},
    {"id": "p11", "stratum": "paraphrase", "query": "How does our sandboxed browser stop internal network or AWS metadata probing?", "target": "sre-05", "fact": ["169.254.169.254"]},
    {"id": "p12", "stratum": "paraphrase", "query": "What cryptographic gate keeps unsigned containers out of New Zealand production?", "target": "sre-07", "fact": ["cosign"]},
    {"id": "p13", "stratum": "paraphrase", "query": "Who gets woken up if gateway 5xx errors spike past one and a half percent?", "target": "sre-09", "fact": ["ml platform"]},
    {"id": "p14", "stratum": "paraphrase", "query": "When must a hospital voice assistant warn callers it is not a doctor?", "target": "cmp-03", "fact": ["4 seconds"]},
    {"id": "p15", "stratum": "paraphrase", "query": "How soon do we have to notify the regulator after discovering a serious data leak?", "target": "cmp-12", "fact": ["72"]},

    {"id": "a01", "stratum": "adversarial", "query": "What is the exact storage bin and temperature for SKU MNG-4402 (not MNG-4403)?", "target": "gr-02", "fact": ["nz-akl-04b", "13"]},
    {"id": "a02", "stratum": "adversarial", "query": "Under the active 2026 rule (not deprecated 2025), how many hours do customers have for produce spoilage photos?", "target": "gr-03", "fact": ["12"]},
    {"id": "a03", "stratum": "adversarial", "query": "What weekly drops and on-time rate are needed for Gold rider bonus (not Silver)?", "target": "gr-05", "fact": ["120", "96.5%"]},
    {"id": "a04", "stratum": "adversarial", "query": "Which NZ city gives a NZ$1.50 rider credit for returning insulated tote bags?", "target": "gr-12", "fact": ["christchurch"]},
    {"id": "a05", "stratum": "adversarial", "query": "Under the current 2026 service auth spec (not legacy 2024), what token algorithm and TTL apply?", "target": "api-03", "fact": ["ed25519", "300"]},
    {"id": "a06", "stratum": "adversarial", "query": "How long are immutable SOC 2 WORM logs kept (not sandbox debug traces)?", "target": "api-08", "fact": ["400"]},
    {"id": "a07", "stratum": "adversarial", "query": "Which TLS version is required on ingress gateway v3, and which older version is rejected?", "target": "api-13", "fact": ["1.3", "1.2"]},
    {"id": "a08", "stratum": "adversarial", "query": "What is Redis maxmemory and eviction policy per shard (not Memcached)?", "target": "sre-03", "fact": ["48", "volatile-lru"]},
    {"id": "a09", "stratum": "adversarial", "query": "How long are raw ClickHouse request spans kept before dropping (not 1-minute rollups)?", "target": "sre-06", "fact": ["30"]},
    {"id": "a10", "stratum": "adversarial", "query": "What is the gRPC deadline for retrieval subcalls specifically (not browser orchestration)?", "target": "sre-08", "fact": ["1,500|1500"]},
    {"id": "a11", "stratum": "adversarial", "query": "Under the active 2026 backup rule, how often do snapshots replicate from Sydney to Melbourne?", "target": "sre-11", "fact": ["4 hours"]},
    {"id": "a12", "stratum": "adversarial", "query": "Which NVIDIA driver and CUDA version run on vLLM inference nodes (not CPU ONNX nodes)?", "target": "sre-13", "fact": ["565.57", "12.6"]},
    {"id": "a13", "stratum": "adversarial", "query": "When do Tauranga sea port container demurrage charges start (not Auckland airport)?", "target": "cmp-07", "fact": ["72"]},
    {"id": "a14", "stratum": "adversarial", "query": "Under the 2026 telehealth rule (not 2023), are video consult recordings on by default?", "target": "cmp-09", "fact": ["disabled"]},
    {"id": "a15", "stratum": "adversarial", "query": "Which MPI code is required on infant formula export cartons (not bulk milk powder)?", "target": "cmp-15", "fact": ["lot-if-88"]},

    {"id": "m01", "stratum": "multilingual", "query": "കൊച്ചി വെയർഹൗസ് പ്രവർത്തിക്കുന്ന സമയം എത്രയാണ്, കേടായ പച്ചക്കറികൾക്ക് എത്ര മണിക്കൂറിനുള്ളിൽ റീഫണ്ട് കിട്ടും?", "target": "gr-14", "fact": ["05:30|5:30", "6"]},
    {"id": "m02", "stratum": "multilingual", "query": "ഓക്ലൻഡ് സിബിഡിയിൽ എക്സ്പ്രസ് ഡെലിവറി എത്ര മിനിറ്റിൽ എത്തും?", "target": "gr-01", "fact": ["18"]},
    {"id": "m03", "stratum": "multilingual", "query": "അൽഫോൻസോ മാങ്ങകൾ സൂക്ഷിക്കേണ്ട താപനില എത്രയാണ്?", "target": "gr-02", "fact": ["13"]},
    {"id": "m04", "stratum": "multilingual", "query": "ग्राहक को महंगा विकल्प देने पर कंपनी अधिकतम कितना अतिरिक्त मूल्य वहन करती है?", "target": "gr-06", "fact": ["5.00|5"]},
    {"id": "m05", "stratum": "multilingual", "query": "गिफ्ट वाउचर कितने महीनों तक मान्य रहते हैं?", "target": "gr-09", "fact": ["36"]},
    {"id": "m06", "stratum": "multilingual", "query": "मेडिकल डिवाइस फर्मवेयर अपडेट के लिए किन दो अधिकारियों की मंजूरी आवश्यक है?", "target": "cmp-06", "fact": ["qa", "regulatory"]},
    {"id": "m07", "stratum": "multilingual", "query": "¿Cuántos requests por minuto permite la API de webhooks por tenant?", "target": "api-01", "fact": ["240"]},
    {"id": "m08", "stratum": "multilingual", "query": "¿Qué versión de TLS rechaza el gateway de ingreso v3?", "target": "api-13", "fact": ["1.2"]},
    {"id": "m09", "stratum": "multilingual", "query": "¿Cuántas baterías de litio UN3481 se permiten como máximo por paquete en Nueva Zelanda?", "target": "cmp-04", "fact": ["2"]},
    {"id": "m10", "stratum": "multilingual", "query": "フォークリフトのバッテリー充電室の温度が何度を超えると換気が作動しますか？", "target": "cmp-08", "fact": ["45"]},
    {"id": "m11", "stratum": "multilingual", "query": "シドニーからメルボルンへの2026年バックアップスナップショットは何時間ごとに複製されますか？", "target": "sre-11", "fact": ["4"]},
    {"id": "m12", "stratum": "multilingual", "query": "ベンチマーク用の合成プローブはどの3都市から何秒ごとに実行されますか？", "target": "sre-15", "fact": ["60"]},
    {"id": "m13", "stratum": "multilingual", "query": "Quel est le délai de basculement primaire Postgres avec Patroni ?", "target": "sre-02", "fact": ["25"]},
    {"id": "m14", "stratum": "multilingual", "query": "Quelle plage de température doit être enregistrée pour la livraison de médicaments réfrigérés ?", "target": "cmp-10", "fact": ["2c|2", "8c|8"]},
    {"id": "m15", "stratum": "multilingual", "query": "Combien de mois les rapports de test d'intrusion des fournisseurs restent-ils valides ?", "target": "cmp-14", "fact": ["12"]},
]

UNANSWERABLE_TRAPS = [
    {"id": "u01", "query": "What is KoruMart CEO's personal mobile number?", "expect_refusal": True},
    {"id": "u02", "query": "What is the Wi-Fi password for the Tauranga port security office?", "expect_refusal": True},
    {"id": "u03", "query": "How much did KoruMart spend on Google Ads in Q3 2026?", "expect_refusal": True},
    {"id": "u04", "query": "What is the secret admin override PIN for SKU MNG-4402?", "expect_refusal": True},
]


def compute_ir_metrics(ranked_ids: list[str], target: str, k: int = 5) -> dict[str, float]:
    top_k = ranked_ids[:k]
    r1 = 1.0 if (len(top_k) >= 1 and top_k[0] == target) else 0.0
    r3 = 1.0 if target in top_k[:3] else 0.0
    r5 = 1.0 if target in top_k[:5] else 0.0
    mrr = 0.0
    ndcg = 0.0
    if target in top_k:
        rank = top_k.index(target) + 1
        mrr = 1.0 / rank
        ndcg = 1.0 / math.log2(rank + 1)
    return {"r1": r1, "r3": r3, "r5": r5, "mrr5": mrr, "ndcg5": ndcg}


def agg_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    n = len(rows)
    return {
        "n": n,
        "recall_at_1": round(sum(r["r1"] for r in rows) / n, 4),
        "recall_at_3": round(sum(r["r3"] for r in rows) / n, 4),
        "recall_at_5": round(sum(r["r5"] for r in rows) / n, 4),
        "mrr_at_5": round(sum(r["mrr5"] for r in rows) / n, 4),
        "ndcg_at_5": round(sum(r["ndcg5"] for r in rows) / n, 4),
    }


def fact_match(answer: str, facts: list[str], expect_refusal: bool = False) -> bool:
    low = answer.lower()
    if expect_refusal:
        markers = [
            "don't know", "do not know", "not mentioned", "no information",
            "cannot find", "not provided", "not in", "don't have", "do not have",
            "not available", "does not contain", "doesn't contain", "unable to find",
        ]
        return any(m in low for m in markers)
    return all(any(alt.lower() in low for alt in f.split("|")) for f in facts)


async def eval_queryable(
    query_fn: Any,
    alpha: float,
    candidate_depth: int | None = None,
) -> dict[str, Any]:
    rows_all = []
    by_stratum: dict[str, list[dict[str, float]]] = {}
    lat_ms = []
    misses = []
    for item in EVAL_60:
        t0 = time.perf_counter()
        res = await query_fn(
            item["query"],
            QueryOptions(top_k=5, alpha=alpha, candidate_depth=candidate_depth),
        )
        lat_ms.append((time.perf_counter() - t0) * 1000)
        ids = [d.id for d in res.docs]
        m = compute_ir_metrics(ids, item["target"], k=5)
        rows_all.append(m)
        by_stratum.setdefault(item["stratum"], []).append(m)
        if m["r1"] == 0.0:
            misses.append({"id": item["id"], "stratum": item["stratum"], "target": item["target"], "got_top3": ids[:3]})
    return {
        "overall": agg_metrics(rows_all),
        "by_stratum": {s: agg_metrics(rs) for s, rs in by_stratum.items()},
        "latency_ms": summarize_latencies(lat_ms),
        "top1_misses_count": len(misses),
        "top1_misses_sample": misses[:15],
    }


async def run_rag_eval(pid: str, pkey: str, index_name: str) -> dict[str, Any]:
    hard_subset = (
        [q for q in EVAL_60 if q["stratum"] == "paraphrase"][:5]
        + [q for q in EVAL_60 if q["stratum"] == "adversarial"][:10]
        + [q for q in EVAL_60 if q["stratum"] == "multilingual"][:5]
    )
    all_cases = hard_subset + UNANSWERABLE_TRAPS
    auth = f"Bearer {pid}:{pkey}"
    sys_prompt = "You are a precise policy assistant. Answer in 1-2 short sentences using ONLY the retrieved context. If the answer is not in the context, explicitly say you do not have that information."

    configs = [
        ("fastlane_indexes", "moss/fastlane-1", "indexes"),
        ("fastlane_moss_search", "moss/fastlane-1", "moss_search"),
        ("gpt6_luna_indexes", "openai/gpt-6-luna", "indexes"),
        ("haiku55_indexes", "anthropic/claude-haiku-5.5", "indexes"),
    ]

    rag_report: dict[str, Any] = {}
    total_cost = 0.0

    async with httpx.AsyncClient(timeout=60.0) as h:
        for label, model_id, mode in configs:
            ans_pass = 0
            trap_pass = 0
            ttfts = []
            retrieves = []
            model_cost = 0.0
            failures = []

            for c in all_cases:
                hdrs = {"Authorization": auth, "Content-Type": "application/json", "X-Moss-Cache": "off"}
                body: dict[str, Any] = {
                    "model": model_id,
                    "stream": True,
                    "instructions": sys_prompt,
                    "input": c["query"],
                    "max_output_tokens": 110,
                }
                if mode == "indexes":
                    hdrs["X-Moss-Indexes"] = index_name
                else:
                    body["tools"] = [{"type": "moss_search", "indexes": [index_name], "top_k": 5}]

                t0 = time.perf_counter()
                ttft = None
                text_parts = []
                usage = {}
                retrieve_ms = None
                mcp_calls = 0

                async with h.stream("POST", f"{BASE_URL}/responses", headers=hdrs, json=body) as resp:
                    if resp.headers.get("x-moss-retrieve-ms") is not None:
                        retrieve_ms = float(resp.headers["x-moss-retrieve-ms"])
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        ev = json.loads(raw)
                        et = ev.get("type")
                        if et == "response.output_text.delta":
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            text_parts.append(ev.get("delta") or "")
                        elif et == "response.mcp_call.completed":
                            mcp_calls += 1
                        elif et in ("response.completed", "response.incomplete"):
                            usage = (ev.get("response") or {}).get("usage") or {}

                ans = "".join(text_parts)
                if ttft is not None:
                    ttfts.append(ttft)
                if retrieve_ms is not None and mode == "indexes":
                    retrieves.append(retrieve_ms)

                ret_count = mcp_calls if mode == "moss_search" else 1
                cost = estimate_inference_cost(
                    model_id,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    ret_count,
                )
                model_cost += cost
                total_cost += cost

                is_trap = bool(c.get("expect_refusal"))
                ok = fact_match(ans, c.get("fact", []), expect_refusal=is_trap)
                if is_trap:
                    if ok:
                        trap_pass += 1
                else:
                    if ok:
                        ans_pass += 1
                if not ok:
                    failures.append({"id": c["id"], "query": c["query"], "expected": c.get("fact", "REFUSAL"), "answer": ans[:220]})
                await asyncio.sleep(0.12)

            rag_report[label] = {
                "model": model_id,
                "mode": mode,
                "answerable_accuracy": round(ans_pass / len(hard_subset), 4),
                "answerable_passed": f"{ans_pass}/{len(hard_subset)}",
                "abstention_accuracy": round(trap_pass / len(UNANSWERABLE_TRAPS), 4),
                "abstention_passed": f"{trap_pass}/{len(UNANSWERABLE_TRAPS)}",
                "overall_accuracy": round((ans_pass + trap_pass) / len(all_cases), 4),
                "ttft_ms": summarize_latencies(ttfts),
                "retrieve_ms": summarize_latencies(retrieves) if retrieves else None,
                "est_cost_usd": round(model_cost, 6),
                "failures": failures,
            }

    rag_report["total_rag_est_cost_usd"] = round(total_cost, 6)
    return rag_report


async def main() -> None:
    pid, pkey = load_credentials()
    corpus = build_corpus()
    client = MossClient(pid, pkey)

    existing = {idx.name for idx in await client.list_indexes()}
    cloud_create_med_err = None
    if IDX_MINI not in existing:
        await client.create_index(IDX_MINI, corpus, "moss-minilm", wait=True)

    # Probe cloud moss-mediumlm (note: Free-tier indexing quota may reject if >~2k cumulative docs)
    try:
        if IDX_MED not in existing:
            await client.create_index(IDX_MED, corpus, "moss-mediumlm", wait=True)
    except Exception as e:
        cloud_create_med_err = f"{type(e).__name__}: {e}"

    t_l0 = time.perf_counter()
    await client.load_index(IDX_MINI)
    load_mini_ms = round((time.perf_counter() - t_l0) * 1000, 2)

    # Also evaluate moss-minilm and moss-mediumlm locally via SessionIndex (head-to-head)
    t_s_mini = time.perf_counter()
    sess_mini = await client.session("sess-bench-minilm", model_id="moss-minilm", auto_load_on_init=False)
    await sess_mini.add_docs(corpus)
    sess_mini_init_ms = round((time.perf_counter() - t_s_mini) * 1000, 2)

    t_s_med = time.perf_counter()
    sess_med = await client.session("sess-bench-mediumlm", model_id="moss-mediumlm", auto_load_on_init=False)
    await sess_med.add_docs(corpus)
    sess_med_init_ms = round((time.perf_counter() - t_s_med) * 1000, 2)

    retrieval_eval = {
        "corpus_size": len(corpus),
        "query_count": len(EVAL_60),
        "cloud_create_mediumlm_error": cloud_create_med_err,
        "load_cloud_minilm_ms": load_mini_ms,
        "session_build_ms": {"moss-minilm": sess_mini_init_ms, "moss-mediumlm": sess_med_init_ms},
        "configs": {
            "cloud_minilm_alpha_1.0": await eval_queryable(lambda q, o: client.query(IDX_MINI, q, o), 1.0),
            "cloud_minilm_alpha_0.6": await eval_queryable(lambda q, o: client.query(IDX_MINI, q, o), 0.6),
            "cloud_minilm_alpha_0.6_depth50": await eval_queryable(lambda q, o: client.query(IDX_MINI, q, o), 0.6, candidate_depth=50),
            "cloud_minilm_alpha_0.0_bm25": await eval_queryable(lambda q, o: client.query(IDX_MINI, q, o), 0.0),
            "session_minilm_alpha_1.0": await eval_queryable(sess_mini.query, 1.0),
            "session_minilm_alpha_0.6": await eval_queryable(sess_mini.query, 0.6),
            "session_mediumlm_alpha_1.0": await eval_queryable(sess_med.query, 1.0),
            "session_mediumlm_alpha_0.6": await eval_queryable(sess_med.query, 0.6),
            "session_mediumlm_alpha_0.0_bm25": await eval_queryable(sess_med.query, 0.0),
        },
    }

    await sess_mini.close()
    await sess_med.close()
    await client.unload_index(IDX_MINI)

    rag_eval = await run_rag_eval(pid, pkey, IDX_MINI)
    await client.close()

    report = {"retrieval_eval": retrieval_eval, "rag_eval": rag_eval}
    path = save_log("06_deep_retrieval_eval.json", report, [pkey])
    print("Saved:", path)
    if cloud_create_med_err:
        print("Cloud mediumlm note:", cloud_create_med_err)
    print("\n--- RETRIEVAL 60-QUERY BENCHMARK ---")
    for cfg_name, cfg in retrieval_eval["configs"].items():
        ov = cfg["overall"]
        st = cfg["by_stratum"]
        print(
            f"{cfg_name:30s} | R@1={ov['recall_at_1']:.3f} R@3={ov['recall_at_3']:.3f} MRR={ov['mrr_at_5']:.3f} nDCG={ov['ndcg_at_5']:.3f} "
            f"| p50={cfg['latency_ms']['p50']}ms | R@1: dir={st['direct']['recall_at_1']:.2f} para={st['paraphrase']['recall_at_1']:.2f} adv={st['adversarial']['recall_at_1']:.2f} ml={st['multilingual']['recall_at_1']:.2f}"
        )
    print("\n--- END-TO-END RAG 24-QUERY EVAL ---")
    for label, r in rag_eval.items():
        if isinstance(r, dict) and "overall_accuracy" in r:
            print(f"{label:22s} | ans={r['answerable_passed']} trap={r['abstention_passed']} overall={r['overall_accuracy']:.3f} | TTFT p50={r['ttft_ms']['p50']}ms | cost=${r['est_cost_usd']}")
    print("Total RAG est cost USD:", rag_eval["total_rag_est_cost_usd"])


if __name__ == "__main__":
    asyncio.run(main())
