# ML Benchmark Results — Milestone 2.4

**Run:** 2026-11-12, `scripts/benchmark_models.py`
**Raw output:** `reports/ml_benchmark_results.json`
**Environment:** Python 3.14.6, pandas 3.1.1, numpy 2.5.4,
scikit-learn 1.9.1 — Windows, single machine.

> **Interpretation warning:** ~850 synthetic labeled rows with near-
> perfect class separability. These metrics rank models on THIS
> corpus only; they are not real-world detection accuracy figures.
>
> **Validation scope:** internal reproducibility on the current
> labeled CSVs — **not** independent validation. The recoverable
> `data/independent_v2/` datasets from git `bfdca25` remain the
> independent-validation option (see `ml_benchmark_methodology.md`).

## Splits (entity-disjoint, seed=42)

| Split | Rows | Purpose |
|---|---|---|
| unsup_train | 179 (normal only) | IF/LOF fit + feature baselines |
| calibration | 73 (normal only) | calibration reference |
| sup_train | 227 (all labels) | RF/GBM fit |
| validation | 123 | reporting |
| holdout | 133 | final eval, once |

## Holdout results — binary mode (suspicious+high_risk = positive)

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | Train s |
|---|---|---|---|---|---|---|---|
| isolation_forest | 0.9459 | 1.0000 | 0.9722 | 0.0204 | 1.0000 | 0.9714 | 1.68 |
| local_outlier | 0.2632 | 1.0000 | 0.4167 | 1.0000 | 1.0000 | 0.9714 | 0.08 |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9714 | 2.34 |
| gradient_boost | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.9714 | 0.24 |

## Holdout results — high_risk-only mode

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| isolation_forest | 0.3784 | 1.0000 | 0.5490 | 0.1933 | 0.9457 | 0.5049 |
| local_outlier | 0.1053 | 1.0000 | 0.1905 | 1.0000 | 1.0000 | 0.9286 |
| random_forest | 0.4000 | 1.0000 | 0.5714 | 0.1765 | 0.9412 | 0.4733 |
| gradient_boost | 0.4000 | 1.0000 | 0.5714 | 0.1765 | 0.9118 | 0.3494 |

## Validation split (sanity check, binary)

| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| isolation_forest | 0.9118 | 1.0000 | 0.9538 | 0.0326 |
| local_outlier | 0.2541 | 1.0000 | 0.4052 | 0.9891 |
| random_forest | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| gradient_boost | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

## Findings

1. **Isolation Forest (production baseline) is strong on this
   corpus:** binary holdout P=0.946 / R=1.000 / FPR=0.020 — only 2
   false positives on 98 normals. Validation agrees (P=0.912,
   FPR=0.033): no sign of validation→holdout degradation.
2. **LocalOutlierFactor is unusable at default settings:** recall
   is perfect but FPR ≈ 1.0 — it flags essentially everything.
   Its ranking (ROC-AUC 1.0) shows the *features* separate the
   classes; the novelty offset does not. LOF is rejected as a
   candidate channel without retuning (retuning was out of scope
   and would need its own calibration workflow).
3. **Supervised RF/GBM are perfect on binary** — expected given
   scripted synthetic labels and disjoint-but-similar users. They
   serve as an upper-bound reference only; they cannot be deployed
   without labeled attack data and would need a documented
   retraining workflow before any production consideration.
4. **High-risk-only mode is much harder for everyone:** best F1 ≈
   0.57 (supervised) / 0.55 (IF). The `suspicious` class absorbs
   most borderline cases; isolating `high_risk` at default
   thresholds yields low precision. This is a genuine detection
   limitation worth carrying into Milestone 2.6 (advanced
   detection), not a bug.
5. **Cost:** IF train 1.68 s / LOF 0.08 s / RF 2.34 s / GBM 0.24 s
   on ~180–230 rows; peak traced memory 2.54 MB. All models are
   trivially fast at research scale — inference latency is dominated
   by feature engineering, not the estimator.

## Decisions recorded

- **Isolation Forest remains the production ML channel.** No
  evidence justified switching; LOF rejected; supervised models
  documented as reference-only.
- Autoencoder comparison **not run**: would require new
  dependencies (torch/etc.) against the project's
  avoid-unnecessary-dependencies rule; marked PLANNED as an
  optional research experiment for a later milestone.
- No channel weights, thresholds, or features were changed based
  on these results (holdout was never used for tuning).
