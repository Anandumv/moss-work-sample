# Candidate: translate every query vs baseline

**Result: FAIL**

| Gate | Status | Detail |
| --- | --- | --- |
| `hit@5` | pass | mean=0.9667 [95% CI 0.9167, 1.0000] n=60; vs baseline delta=+0.1333 [+0.0333, +0.2500] p=0.016 |
| `ndcg@5` | pass | mean=0.9152 [95% CI 0.8568, 0.9651] n=60; vs baseline delta=+0.1064 [+0.0018, +0.2151] p=0.043 |
| `mrr@5` | pass | mean=0.8978 [95% CI 0.8289, 0.9556] n=60; vs baseline delta=+0.0978 [-0.0112, +0.2083] p=0.071 |
| `faithfulness` | **FAIL** | metric missing from run |

## By stratum

| Stratum | n | hit@5 | ndcg@5 |
| --- | ---: | ---: | ---: |
| stratum:multilingual | 9 | 1.000 [1.000, 1.000] | 0.944 [0.833, 1.000] |
| stratum:general | 32 | 0.969 [0.906, 1.000] | 0.911 [0.825, 0.977] |
| stratum:adversarial | 19 | 0.947 [0.842, 1.000] | 0.908 [0.784, 1.000] |

## Newly failing cases (2)

- `reg-3580eae48b`
- `reg-72cc780da3`
