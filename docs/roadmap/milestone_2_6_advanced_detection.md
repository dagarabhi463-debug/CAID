# Milestone 2.6 — Advanced Detection (Roadmap)

**Status:** PLANNED — follows Milestone 2.5 streaming ingestion.

## Scope candidates (prioritized by 2.4 findings)

1. **High-risk-only detection improvement** — weakest mode in the
   2.4 benchmark (best F1 ≈ 0.57). Options: dedicated high-risk
   channel, score calibration review, or enriched features —
   evaluated on entity-disjoint splits, holdout untouched until
   final.
2. **Session channel coverage** — produced no usable scores on the
   2.4 benchmark corpus (small per-user session counts). Investigate
   minimum-evidence thresholds and whether scenario-scale data
   unlocks the channel.
3. **Supervised model reconsideration** — RF/GBM were perfect on
   synthetic labels but need labeled attack data + a documented
   retraining workflow before any production role.
4. **Autoencoder experiment** — optional research track deferred
   from 2.4 (requires new dependencies; evaluate necessity first).
5. **Per-row performance work** — vectorize `iterrows` hotspots
   (`DetectionEngine.process_dataframe`, rule AST eval,
   `RiskEngine.apply`) as required by the 100 ev/s streaming target.
6. **v2 calibration regeneration** — controlled recalibration
   workflow to replace the stale artifact (2.4 finding).

## Rules carried forward

- No holdout tuning; no score-semantics changes; missing channels
  stay explicit; drift stays observational; synthetic data only.
