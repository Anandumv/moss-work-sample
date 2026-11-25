# Candidate: translate-then-retrieve (non-Latin scripts) vs baseline

**Result: PASS**

| Gate | Status | Detail |
| --- | --- | --- |
| `hit@5` | pass | mean=0.9833 [95% CI 0.9500, 1.0000] n=60; vs baseline delta=+0.1500 [+0.0667, +0.2333] p=0.0 |
| `ndcg@5` | pass | mean=0.9485 [95% CI 0.8991, 0.9877] n=60; vs baseline delta=+0.1398 [+0.0564, +0.2296] p=0.0 |
| `mrr@5` | pass | mean=0.9367 [95% CI 0.8817, 0.9833] n=60; vs baseline delta=+0.1367 [+0.0533, +0.2233] p=0.0 |
| `faithfulness` | pass | mean=0.9694 [95% CI 0.9277, 1.0000] n=60 |

## By stratum

| Stratum | n | hit@5 | ndcg@5 |
| --- | ---: | ---: | ---: |
| stratum:multilingual | 9 | 1.000 [1.000, 1.000] | 0.932 [0.796, 1.000] |
| stratum:general | 32 | 0.969 [0.906, 1.000] | 0.946 [0.864, 1.000] |
| stratum:adversarial | 19 | 1.000 [1.000, 1.000] | 0.961 [0.903, 1.000] |
