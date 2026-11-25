# Baseline: dense moss-minilm (current prod config)

**Result: PASS**

| Gate | Status | Detail |
| --- | --- | --- |
| `hit@5` | pass | mean=0.8333 [95% CI 0.7333, 0.9167] n=60 |
| `ndcg@5` | pass | mean=0.8087 [95% CI 0.7131, 0.8938] n=60 |
| `mrr@5` | pass | mean=0.8000 [95% CI 0.7000, 0.8835] n=60 |
| `faithfulness` | pass | mean=0.9833 [95% CI 0.9583, 1.0000] n=60 |

## By stratum

| Stratum | n | hit@5 | ndcg@5 |
| --- | ---: | ---: | ---: |
| stratum:multilingual | 9 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| stratum:general | 32 | 0.969 [0.906, 1.000] | 0.946 [0.864, 1.000] |
| stratum:adversarial | 19 | 1.000 [1.000, 1.000] | 0.961 [0.903, 1.000] |
