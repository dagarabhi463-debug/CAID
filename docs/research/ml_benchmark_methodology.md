# ML Benchmark Methodology — Milestone 2.4

**Harness:** `scripts/benchmark_models.py`
**Results:** `docs/research/ml_benchmark_results.md` (+ `reports/ml_benchmark_results.json`)

## 1. Purpose

Compare anomaly-detection approaches on **consistent engineered
features** and **leakage-safe, entity-disjoint splits** to inform —
not replace — the production Isolation Forest baseline. Supervised
models are evaluated as *reference points*; no claim is made that
they are superior for deployment (they require labeled attack data
CAID does not have in production).

## 2. Data

| Source | Rows | Labels |
|---|---|---|
| `data/raw/security_logs.csv` | 650 | normal 500 / suspicious 100 / high_risk 50 |
| `data/processed/evaluation_data.csv` | 200 | normal 100 / suspicious 50 / high_risk 50 |

`security_logs.csv` is the concatenation of the three
`data/raw/*_activity.csv` files (verified identical user sets).
Duplicates across the exact feature columns are dropped once.
Scenario-generator data (`data/generate_scenarios.py`) is **not**
used: its labels are scripted scenario design, not independent
ground truth, and including it would leak scenario structure into
"evaluation".

**Known limitation:** ~850 labeled rows total, synthetic
distribution, near-perfect class separability (all models reach
ROC-AUC 1.0 on binary). Metrics indicate *relative* behavior, not
real-world accuracy. The removed `data/independent_v2/` entity-
disjoint datasets (git `bfdca25`) remain the gold standard for a
future, larger benchmark.

### Internal reproducibility vs independent validation

The deterministic splits in this harness are **internal
reproducibility** controls: they prove the benchmark is stable and
leakage-guarded on the *current* labeled CSVs. They are **not**
independent validation, because the corpus is small, synthetic, and
was already used to develop the pipeline.

Independent validation remains available from git history. The
Milestone 2.0 `data/independent_v2/` entity-disjoint datasets were
removed in commit `1744782` but are recoverable from `bfdca25`:

```powershell
# Restore the full independent_v2 dataset + generator (optional)
git checkout bfdca25 -- data/independent_v2 data/generate_independent_datasets_v2.py

# Or extract a single file without checkout
git show bfdca25:data/independent_v2/train.csv > data/independent_v2/train.csv
```

After recovery, re-run the dataset leakage audit before treating it
as independent validation:

```powershell
.venv/Scripts/python.exe scripts/leakage_audit_2_0.py
```

## 3. Splits — deterministic, entity-disjoint

Every `user_id` is assigned to exactly one split via
`sha1(seed:user_id)` (seed = 42), so **no user appears in two
splits** — the strongest leakage guard available on this corpus.

| Split | Contents | Rows (actual) | Users | Purpose |
|---|---|---|---|---|
| `unsup_train` | normal rows only | 179 | — | fit IF / LOF + feature baselines |
| `calibration` | normal rows only | 73 | — | score-calibration reference (not used for threshold tuning in this benchmark) |
| `sup_train` | all labels | 227 | — | fit supervised RF / HistGB |
| `validation` | all labels | 123 | — | model reporting / sanity check |
| `holdout` | all labels | 133 | — | **final evaluation, once** |

The script asserts pairwise user-set disjointness at runtime.

## 4. Feature pipeline — identical for all models

1. `FeatureBaselineManager().fit(unsup_train)` — baselines computed
   on **unsupervised-train normals only**.
2. `engineer_features(df, bm)` for every split with those frozen
   baselines (30 numeric features; `leave_one_out=False` at
   inference, matching production `predict()`).
3. `StandardScaler` fit on `unsup_train` features; transform all.

The production IsolationForest path additionally fits its own
scaler + score calibration inside `AnomalyDetector.fit()` — the
same contract the saved artifact uses.

## 5. Models & operating points

| Model | Type | Fit data | Score (higher = anomalous) | Prediction |
|---|---|---|---|---|
| `isolation_forest` | unsupervised | unsup_train normals | `anomaly_score_normalized` (fixed train-score calibration) | `is_anomaly` |
| `local_outlier` | unsupervised | unsup_train normals | `-decision_function` | `predict()==-1` (offset_) |
| `random_forest` | supervised | sup_train (labels) | `predict_proba[:,1]` | `predict()` |
| `gradient_boost` | supervised | sup_train (labels) | `predict_proba[:,1]` | `predict()` |

**No threshold tuning anywhere.** Operating points are each
model's documented default (IF contamination=0.1, LOF
contamination=0.1, RF/GBM 0.5 probability). Holdout results are
reported once and never fed back.

## 6. Metrics

Per model × split × mode (`binary`: suspicious+high_risk positive;
`high_risk`: high_risk only positive):

- Confusion matrix (tp/fp/tn/fn), precision, recall, F1, FPR
- ROC-AUC and PR-AUC (midpoint rank method, tie-averaged)
- Train wall-time; per-split predict wall-time; peak traced memory

## 7. Robustness probes

- **Unseen users/entities:** the holdout split *is* entity-disjoint
  by construction — its metrics ARE the unseen-entity result.
- **Missing features:** production `AnomalyDetector.predict()`
  raises `ValueError` on missing feature columns (verified by
  `tests/test_anomaly_detector.py`); benchmark models use the same
  frozen schema. Missing-feature behavior is therefore explicit
  failure, not silent-normal — tested, not re-measured here.
- **Class imbalance:** prevalence reported per split in the JSON
  (`data.splits.*.labels`); metrics are computed on the natural
  imbalance, no resampling at evaluation.
- **Distribution change / unseen scenarios:** out of scope for this
  harness; covered qualitatively by `core/drift_monitor.py`
  (Milestone 2.4 Phase 4) which measures drift observationally.

## 8. Reproducibility

```
.venv/Scripts/python scripts/benchmark_models.py
# optional: --out reports/ml_benchmark_results.json
```

Deterministic: fixed seed, hash-based splits, `random_state=42` on
all stochastic models, sort-stable ranking for AUC. Runtime on the
authoring machine: ~5 s total.
