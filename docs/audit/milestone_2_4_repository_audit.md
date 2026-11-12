# Milestone 2.4 — Repository Audit

**Scope:** Full repository inspection for CAID Milestone 2.4
(Reliability, Detection Validation & Real-Time IDS Preparation).
**Date:** 2026-11-12
**Baseline:** `pytest tests -q` → **439 passed, 0 failed** (113s,
Python 3.14.6, reportlab 5.1.0, scikit-learn 1.9.1, pandas 3.1.1,
numpy 2.5.4, streamlit 1.67.0).

This audit covers `app.py`, `app_pages/`, `core/`, `models/`,
`modules/`, `data/`, `tests/`, `utils/`, `requirements.txt`,
`README.md`, configuration files, git history, model artifacts and
calibration artifacts.

---

## Summary

| Severity | Count |
|---|---|
| High     | 0     |
| Medium   | 3     |
| Low      | 6     |
| Info     | 5     |

The codebase is in good health: the leakage guarantees, score
contracts, calibration fingerprinting, explicit degraded operating
modes, shared-IP policy, and prompt-injection defenses documented in
`PROJECT_DOCUMENTATION.md` were all verified against source. No
broken imports, circular dependencies, `eval`/`exec`/`subprocess`
usage, or credential exposure were found. `joblib.load` is only used
on trusted local artifact paths.

---

## Findings

### AUD-001 — Divergent interpretation of the legacy `anomaly_score` field

- **Severity:** Medium
- **Affected files:** `core/risk_engine.py` (lines ~106–126),
  `core/detection_engine.py` (lines ~193–226, 925–944)
- **Evidence:** `RiskEngine.calculate_risk_score` treats a present
  `anomaly_score` **always** as a raw IsolationForest
  `decision_function` and inverts it (`-raw`). `DetectionEngine`
  treats the same field as *ambiguous*: values inside [0,1] are
  assumed already-normalized (higher = anomalous), values outside
  are treated as raw and inverted. `process_dataframe` adds a third
  path — a batch-level heuristic that decides "raw vs normalized"
  by whether the column's min/max leaves [0,1].
- **Root cause:** The `anomaly_score` field is a documented
  deprecated alias retained for backward compatibility; each engine
  grew its own adapter at a different milestone.
- **Reproduction:** Feed a row with `anomaly_score = 0.8` (in-range).
  RiskEngine interprets it as raw → normalized anomaly ≈ 0.2 (low
  risk contribution). DetectionEngine interprets it as normalized →
  anomaly 0.8 (high ML channel score). The two engines disagree on
  identical input.
- **Impact:** `risk_score` (heuristic) and `combined_score`
  (DetectionEngine) can rank the same event differently; dashboards
  show both. Not a correctness bug in the v2 pipeline (which emits
  explicit `anomaly_score_raw`/`anomaly_score_normalized`), but a
  latent trap for any consumer of the legacy field.
- **Recommended remediation:** Keep semantics unchanged (no silent
  score-semantics change allowed); document the divergence in
  `docs/architecture/detection_engine_architecture.md`; add a
  regression test pinning each engine's documented interpretation;
  prefer explicit fields in all new code.
- **Verification:** New test asserting both interpretations are
  intentional and stable.

### AUD-002 — Inconsistent artifact lifecycle for baseline components

- **Severity:** Medium
- **Affected files:** `core/behavioral_baseline.py:595-615`
  (`ensure_fitted_baseline`), `core/session_analyzer.py:89-99`
  (`ensure_fitted_session_analyzer`), `core/sequence_detection.py:608-618`
  (`ensure_fitted_sequence_analyzer`), `models/anomaly_detector.py:432-468`
  (`ensure_model_exists`)
- **Evidence:**
  - Behavioral: missing artifact **or load failure** → silently
    refits on `load_sample_dataset('normal')` **and persists** the
    new artifact (overwrites possibly-corrupt file).
  - Session/sequence: missing artifact → silently refits on the same
    sample normal data **in memory only** (never persisted, so the
    fit repeats every run and can differ if sample data changes).
  - Detector: missing file **or any load exception** (including
    corrupt/incompatible artifact) → retrains on normal data and
    saves, masking artifact corruption.
- **Root cause:** Convenience "ensure" helpers written per-milestone
  without a unified artifact-lifecycle policy.
- **Impact:** A corrupted `anomaly_detector.joblib` is silently
  replaced rather than reported; session/sequence baselines are
  non-deterministic across runs if the bundled sample data ever
  changes; operators cannot distinguish "loaded artifact" from
  "just-fit fallback" in the UI.
- **Recommended remediation:** Surface an explicit status
  (`loaded` / `refit_fallback`) in analysis results; keep fitting on
  designated training-normal data only (current behavior is
  leakage-safe); document the policy; optionally persist
  session/sequence refits for consistency — decision deferred to a
  controlled workflow, not this milestone.
- **Verification:** Tests asserting fallback status is reported and
  that no validation/holdout data is ever used for fitting.

### AUD-003 — Uploaded-data pipeline uses a weaker detection profile

- **Severity:** Medium
- **Affected file:** `modules/log_analysis.py:911-913`
- **Evidence:** `analyze_uploaded_data` constructs
  `DetectionEngine()` with the default `legacy` profile (ml/rules/
  behavioral only). The scenario pipeline
  (`modules/session_intelligence.py:53`) uses `channel_profile='v2_0'`
  (adds session + sequence channels and v2 calibration).
- **Root cause:** The upload page predates Milestone 2.0 channel
  profiles and was never upgraded.
- **Impact:** Events analyzed via Log Analysis → Upload never receive
  session or sequence deviation scores; `combined_score`/`risk_level`
  for the same event can differ between the two entry points.
- **Recommended remediation:** Route uploaded data through the same
  v2.0 pipeline (`analyze_session_intelligence`) or explicitly
  document the legacy profile as intentional for uploads. Prefer
  unifying — requires verifying upload schema compatibility
  (session_id/auth_result optional columns are supported).
- **Verification:** Test that uploaded CSV produces session/sequence
  columns when routed through the v2 pipeline.

### AUD-004 — Layering inversion: core imports modules

- **Severity:** Low
- **Affected file:** `core/incident_report.py:20`
- **Evidence:** `from modules.export import _sanitize_csv_frame` —
  a `core/` module depends on the Streamlit-facing `modules/` layer.
- **Impact:** No cycle today (`modules.export` imports only pandas),
  but it couples the report engine to the UI layer and would break
  headless/CLI use of `core`.
- **Recommended remediation:** Move `_sanitize_csv_frame` (or a
  `sanitize_csv_value` helper) into `utils/` and import from both
  places. Small, safe refactor.
- **Verification:** Existing CSV tests + import check.

### AUD-005 — Dead `sys.path.insert` in detection engine

- **Severity:** Low
- **Affected file:** `core/detection_engine.py:22-25`
- **Evidence:** Inserts its own directory on `sys.path` but imports
  only pandas/numpy/stdlib — no local imports.
- **Recommended remediation:** Remove the dead lines (cosmetic;
  `app.py` and tests already set the project root).

### AUD-006 — Unpinned dependencies

- **Severity:** Low
- **Affected file:** `requirements.txt`
- **Evidence:** All dependencies float (`streamlit`, `pandas`,
  `scikit-learn`, `reportlab`, …). Current env resolves to pandas
  3.1.1 / streamlit 1.67.0 / reportlab 5.1.0 — versions newer than
  the docs assume, already producing 211 deprecation warnings
  (`Pandas4Warning` for `dt.dayofweek`, `fillna` dtype changes).
- **Impact:** Reproducibility risk; a future pandas 4.x could break
  `fillna('unknown')` on float columns (already warned).
- **Recommended remediation:** Propose `>=` minimum pins matching the
  tested env after Phase-3/9 verification; do not pin exact versions
  without a full dependency audit. Document in README.

### AUD-007 — Runtime/state artifacts tracked in git

- **Severity:** Low
- **Affected files:** `data/incidents/incidents.json` (488K),
  `data/mitre_attack/enterprise-attack.json` (**24MB** STIX bundle),
  `models/saved_models/*.joblib`, `models/saved_models/calibration/*.joblib`,
  `data/processed/evaluation_data.csv`
- **Evidence:** `git ls-files` shows runtime-generated state
  committed to the repository.
- **Impact:** Repo bloat (24MB bundle), merge noise on incident
  updates, stale incidents shipped in clean checkouts, larger clone.
- **Recommended remediation:** Add `.gitignore` entries for
  `data/incidents/incidents.json` and keep the MITRE bundle either
  tracked (documented cache) or ignored with documented
  re-download. **Not changed in this milestone** — deleting tracked
  assets requires explicit justification per project rules; recorded
  as a recommendation.

### AUD-008 — MITRE bundle download lacks integrity verification

- **Severity:** Low
- **Affected file:** `core/mitre_attack.py:242-246, 268-286`
- **Evidence:** `urlopen` fetches the 24MB STIX bundle over HTTPS
  with no checksum/signature verification before parsing and
  caching into `data/mitre_attack/`.
- **Impact:** A compromised/mirror-served bundle would be parsed
  silently (JSON parse only). MITRE mappings could be poisoned.
- **Recommended remediation:** Document trust assumption; optionally
  pin expected SHA-256 of the official bundle for the version in
  `mitre_attack_meta.json`. Design-level note for deployment.

### AUD-009 — Upload handling edge cases

- **Severity:** Low
- **Affected files:** `modules/log_analysis.py:78-131, 770-855`
- **Evidence:** `validate_upload` enforces a 10MB cap and `.csv`
  extension and probes 5 rows; `process_uploaded_file` decodes
  strict UTF-8 — a non-UTF-8 file raises `UnicodeDecodeError`
  caught only by the generic `except` at the page level (shows raw
  traceback via `st.error`). No row-count cap beyond the byte cap.
- **Impact:** Poor UX on encoding failures; a 10MB CSV (~hundreds
  of thousands of rows) flows into per-row `iterrows` pipelines —
  potential UI freeze, bounded by Streamlit session.
- **Recommended remediation:** Catch decode errors with a friendly
  message; document resource expectations; add malformed-input
  tests (Phase 7/9).

### AUD-010 — `user_total_accounts` is a permanently-zero feature

- **Severity:** Info
- **Affected file:** `core/feature_engineering.py:301-304`
- **Evidence:** Column is in the ML feature schema but always set
  to `0.0` (documented backward-compatibility quirk).
- **Impact:** Dead feature consumed by the model; harmless but
  misleading. Changing it alters the feature schema/fingerprint —
  **do not change** without retraining.

### AUD-011 — `engineer_features` leakage-prone fallback path

- **Severity:** Info
- **Affected file:** `core/feature_engineering.py:552-611`
- **Evidence:** When `baseline_manager` is None or unfitted,
  `extract_user_features`/`extract_ip_features`/`extract_event_features`
  compute aggregations from the **current** DataFrame (the original
  leakage pattern). All production callers pass a fitted manager;
  only tests call the bare path.
- **Impact:** Footgun for future callers.
- **Recommended remediation:** Document; optionally emit a warning
  when the fallback path is used outside tests.

### AUD-012 — `Detection.detection_id` is counter-based, not content-derived

- **Severity:** Info
- **Affected file:** `core/detection_engine.py:181, 431`
- **Evidence:** IDs are `DET-000001…` reset per `process_dataframe`
  call — deterministic within a run but not stable identifiers like
  `corr_<sha1>`/`inc_<sha1>` used downstream.
- **Impact:** Cosmetic; detections aren't persisted by ID.

### AUD-013 — Per-row `iterrows`/`apply(axis=1)` hotspots

- **Severity:** Info (known, documented)
- **Affected files:** `core/detection_engine.py:948`,
  `core/risk_engine.py:415`, `core/detection_rules.py` (per-row AST),
  `core/cross_session_correlation.py`, `core/incident_manager.py`
- **Evidence:** Confirmed by grep (28 `iterrows`/`itertuples` uses
  across core/modules). M1.7 benchmarked ~3.8s @10k events.
- **Impact:** Fine at research scale; relevant to the 100 ev/s
  real-time target — recorded for Milestone 2.5 planning.

### AUD-014 — pandas 3.x deprecation warnings

- **Severity:** Info
- **Affected files:** `core/feature_engineering.py:455, 225, 374, 405`
- **Evidence:** 211 warnings in the test run:
  `Series.dt.dayofweek` deprecated → `day_of_week`;
  `fillna('unknown')` on float64 columns warns about future
  incompatibility.
- **Recommended remediation:** Safe mechanical fixes
  (`day_of_week`, explicit astype before fillna) — low risk,
  scheduled in Phase 9 cleanup. Not done yet to keep the audit
  baseline pristine.

### AUD-015 — `analyze_session_intelligence` reloads training data every run

- **Severity:** Info
- **Affected file:** `modules/session_intelligence.py:47-49`
- **Evidence:** `load_sample_dataset('normal')` is fetched on every
  analysis even when persisted baselines exist (passed to
  `ensure_fitted_*` which ignore it when artifacts load).
- **Impact:** Minor wasted IO per analysis.

---

## Positive verifications (no defect found)

- **Leakage:** `FeatureBaselineManager`, `BehavioralBaselineManager`,
  `SessionBaselineManager`, `SequenceAnalyzer` all fit on
  designated training-normal data only; inference never refits
  (verified in source and tests).
- **Calibration:** `core/calibration.py` enforces
  `dataset_role == 'calibration'`, fingerprint-binds detector +
  component fingerprints, rejects incompatible/missing artifacts
  with `CalibrationUnavailable`, and falls back to explicitly
  labeled uncalibrated modes.
- **Score contracts:** Documented and consistent for the explicit
  fields (`anomaly_score_raw` higher=normal, `*_normalized`/
  `*_deviation_score` 0–1 higher=anomalous/deviant, NaN+missing
  flags). Only the deprecated alias diverges (AUD-001).
- **Determinism:** Correlation IDs `corr_<sha1>`, incident IDs
  `inc_<sha1>`, seeded scenario generator, injectable incident
  clock, atomic JSON store writes (`tempfile`+`os.replace`).
- **Security:** No `eval`/`exec`/`subprocess`/`os.system`;
  rule engine uses restricted AST; CSV export sanitizes formula
  injection; AI prompts are wrapped/sanitized; no credentials in
  code or logs (`.env` pattern, settings page never displays
  secrets); Ollama is local-HTTP stdlib only.
- **Retrospective isolation:** Correlation/incidents/MITRE consume
  scored output only; nothing feeds back into per-event scores
  (verified in `modules/correlation_dashboard.py`,
  `core/incident_manager.py`).

## Audit conclusion

No blocking defects. Three medium findings (AUD-001..003) are
semantics/lifecycle inconsistencies to document and regression-test,
not rewrites. The milestone proceeds to Phase 2 (test baseline &
remediation) with the recorded baseline of 439 passed / 0 failed.
