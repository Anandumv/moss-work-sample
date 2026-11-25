# Moss Evaluation — Work Sample

**Anandu MV** · Auckland, NZ · Code, logs, charts & 18 issues: [https://github.com/Anandumv/moss-work-sample](https://github.com/Anandumv/moss-work-sample)

I built moss-evals, a CLI on Moss that turns production failures into regression datasets, runs retrieval plus an LLM faithfulness judge (calibrated at kappa 0.85–1.00 against fact checks), and gates CI on paired bootstrap drops. Across BEIR SciFact (5,183 docs, 300 queries) and load/security tests, I filed 18 reproducible issues. Local queries bill at $0.50/1k with no client counter or spend cap (ISS-16), so my 41,885-query load test exhausted the $10 credit before the final multilingual SciFact run could execute.

## 1. Portal
**What worked:** Google sign-in worked smoothly, 2FA phone setup was offered at sign-up (with a confirmation email), and onboarding issued $10 in credits, a one-time key, and a pre-seeded 50-doc voice-agent index.

**What didn't:** Usage and credit live only in the UI with no API. Moss's only usage email ("You've reached your Moss usage limit", billing@moss.dev) arrived at $0 with no 50%/80% alert, and the paused-project error matches the 3-index-limit error.

**What I'd improve:** 50%/80% threshold alerts, a usage endpoint, a spend cap, and distinct quota errors.

## 2. Inference API & Retrieval
**What worked:** Local retrieval is fast and steady: 2.7 ms median on 120 docs (359 QPS single-core) and under 60 ms at p99 on 5,183 docs before saturation; gateway retrieval stayed at 10–14 ms p99 up to 120 RPM. In the moss-evals demo (60 cases), translating non-Latin queries lifted hit@5 from 0.833 to 0.983 (9/9 non-Latin) and passed the gate, while translating every query failed because two English cases regressed. On SciFact, hybrid search (alpha 0.4) raised nDCG@10 from 0.653 to 0.725 (p<0.001); custom bge-small vectors hit 0.743.

**What didn't:** Dense minilm trailed keyword search on SciFact, mediumlm did not beat minilm, and single-process QPS stayed flat across concurrency. Header retrieval dropped passages after turn 0 in tool loops (0/10 vs 10/10 with client-side retrieval), and 429s sent Retry-After: 1 against a 5 s reset header (44% repeat failures).

**What I'd improve:** Default to hybrid search, keep passages across tool turns, align Retry-After with resets, and add a client query counter.

## 3. Tasks API
**What worked:** Agents resisted prompt injection, never leaked a canary token, stayed on-domain on valid pages, and produced identical outputs across 8 runs. moss-fast and DeepSeek matched moss-balanced at a tenth of the cost.

**What didn't:** Caps are checked post-step, so balanced and advanced overshot $0.02 by 1.24x ($0.024726, identical to six decimals). Ten private/obfuscated hosts (loopback, metadata, RFC1918, hex IPs) returned 202 before cancellation, live_url stayed null, and trace videos 404ed.

**What I'd improve:** Pre-check step cost against caps, block private/metadata URLs at admission, and fix trace/video links.
