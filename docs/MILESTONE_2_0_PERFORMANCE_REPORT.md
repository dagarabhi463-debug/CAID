# CAID Milestone 2.0 — Performance Report

Measured by `scripts/evaluate_holdout_2_0.py` on `data/independent_v2/holdout.csv` (1,500 rows). Latency is end-to-end pipeline time per variant on the same machine/session; memory is `tracemalloc` peak for that run. These are research measurements, not production SLAs.

| Variant | Seconds | Rows/sec | Peak memory (MB) |
|---|---:|---:|---:|
| v1_9_baseline | 4.2232 | 355.2 | 8.51 |
| v2_ml_rules_behavioral | 3.0191 | 496.8 | 7.45 |
| v2_session | 9.9284 | 151.1 | 13.00 |
| v2_sequence | 15.7612 | 95.2 | 12.37 |
| v2_complete | 12.8882 | 116.4 | 16.53 |

Platform: Python 3.14.6, pandas 3.1.1 (see `data/independent_v2/performance_results_v2.json`).

## Notes

- Sequence analysis is the most expensive channel because it reconstructs sessions and evaluates causal windows per event; session adds moderate overhead.
- The v2_complete pipeline is ~3.0× slower than the v1.9 baseline on this batch workload and uses ~2× peak traced memory, but remains tractable for research-scale synthetic datasets.
- No streaming/online latency is claimed; current implementation is batch/casual-prefix over split files.
