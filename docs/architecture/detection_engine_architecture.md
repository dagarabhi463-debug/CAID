# Detection Engine Architecture — Milestone 2.4

**Source:** `core/detection_engine.py` (+ channel producers:
`models/anomaly_detector.py`/`core/ml_integration.py`,
`core/detection_rules.py`, `core/behavioural_analyzer.py`,
`core/session_analyzer.py`, `core/sequence_detection.py`)

## 1. Channel model

Five evidence channels feed `DetectionEngine.process_event`:

| Channel | Producer | Score field | Semantics (higher = …) |
|---|---|---|---|
| `ml` | IsolationForest via `AnomalyDetector` | `anomaly_score_normalized` | more anomalous (0–1) |
| `rules` | `DetectionRulesEngine` (safe AST) | `rule_score` | more rule evidence (0–1) |
| `behavioral` | `BehavioralAnalyzer` | `behavioral_deviation_score` | more deviation (0–1) |
| `session` | `SessionAnalyzer` (causal prefix) | `session_deviation_score` | more deviation (0–1) |
| `sequence` | `SequenceAnalyzer` (causal window) | `sequence_deviation_score` | more anomalous (0–1) |

Missing channels are **never silently normal**: each carries an
explicit `*_missing` flag; `missing_channels`,
`channels_available`, `channel_weights`, `operating_mode`, and
per-channel `structured_evidence` are all surfaced on every
`Detection` and in the output frame.

## 2. Channel profiles & weights (fixed, documented — never tuned on holdout)

```python
CHANNEL_PROFILES = {
  'legacy':   ml .35 rules .35 behavioral .30
  'session':  ml .30 rules .30 behavioral .20 session .20
  'sequence': ml .30 rules .30 behavioral .20 sequence .20
  'v2_0':     ml .25 rules .25 behavioral .15 session .20 sequence .15
}
```

Custom `weights=` may be supplied; configured channels = weight keys.

## 3. Operating modes

- `FULL_*` — all configured channels produced real evidence.
- `DEGRADED_*` — ≥2 channels remain; weights **renormalize** over
  available channels only.
- `INSUFFICIENT_EVIDENCE` — <2 channels → `combined_score = NaN`,
  `risk_level = Insufficient Evidence`, no classification.

Legacy profile additionally distinguishes which channel is missing
(`DEGRADED_RULES_BEHAVIORAL` etc.).

## 4. Score semantics contract

- `anomaly_score_raw` / `anomaly_score`: raw IF
  `decision_function`, **higher = more normal** (deprecated alias
  `anomaly_score` is ambiguous — see Divergence below).
- `anomaly_score_normalized` / `ml_anomaly_score`: 0–1,
  **higher = more anomalous** (fixed training-score calibration —
  batch-invariant).
- `combined_score`: 0–1 weighted fusion, higher = riskier;
  `NaN` when insufficient evidence.
- `risk_level`: Low/Medium/High/Critical via `thresholds`
  (defaults .30/.55/.75; overridable by fingerprint-bound v2
  calibration artifact — currently unavailable for the persisted
  detector, see `docs/research/calibration_reliability.md`).

**None of these are probabilities.**

## 5. Determinism & isolation

- `detection_id` is counter-based (`DET-NNNNNN`, resets per
  `process_dataframe`) — deterministic within a run.
- The engine is a pure function of its inputs + weights +
  thresholds: identical input → identical scores/ordering.
- Retrospective layers (correlation, incidents, MITRE, AI analyst)
  consume outputs only; nothing feeds back into scores, weights,
  or thresholds (verified: no upward imports of correlation into
  detection paths).

## 6. Documented divergence (audit AUD-001)

The deprecated `anomaly_score` alias is interpreted differently by
two engines on in-range values:

- `DetectionEngine`: in-range [0,1] → treated as normalized
  (higher = anomalous); out-of-range → raw, inverted.
- `RiskEngine.calculate_risk_score`: always treated as raw
  `decision_function` → inverted.

Both behaviors are intentional backward-compatibility adapters and
are pinned by tests. New code must emit `anomaly_score_raw` or
`anomaly_score_normalized` — the alias will not be "fixed" because
that would silently change score semantics.

## 7. Fusion validation (Milestone 2.4, executed)

`scripts/evaluate_fusion.py` → `reports/fusion_evaluation.json`
on entity-disjoint validation/holdout splits (baselines fit on
unsup_train normals only; holdout evaluated once):

- Per-channel holdout (binary): ml P=.946 R=1.0; behavioral
  P=1.0 R=.943; rules P=1.0 R=.057 (high-precision/low-recall);
  sequence R=.343; session produced no usable scores on this
  corpus (small per-user session counts → missing/zero deviation).
- Combined v2_0: holdout P=1.0 R=.343 F1=.511, all rows
  `full_v2_0`; validation P=1.0 R=.452 F1=.622.
- Ablation (holdout F1): dropping sequence → .571, session → .542,
  ml/rules/behavioral → .511 (combined unchanged — channels
  compensate via renormalization).
- Threshold sensitivity reported on **validation only**.

Interpretation: fusion is precise but conservative at default v2_0
thresholds on this corpus; the high-risk-only mode remains the
hard case (see `ml_benchmark_results.md`). Channel weights were
**not** retuned — evaluation only.
