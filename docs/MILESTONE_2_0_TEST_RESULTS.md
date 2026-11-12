# CAID Milestone 2.0 — Test Results

Command: `.venv/Scripts/python.exe -m pytest tests -q`
Result: **345 passed, 220 warnings in 75.89s** (warnings are pre-existing pandas deprecations/pytest fixture warnings).

Milestone 2.0 coverage added in `tests/test_milestone_2_0.py` (18 tests):

- Session reconstruction: explicit IDs, timeout splitting, missing timestamps, duplicate flags/preservation, causal prefix ordering, session summaries.
- Session baseline: fit/transform/save/load determinism, unseen-user global fallback, missing-value handling.
- Sequence detection: auth-failure→success→export scoring, missing `auth_result` non-fabrication, deterministic save/load.
- DetectionEngine: v2_0 weighted combined score, missing-channel exclusion/renormalization, legacy profile isolation, insufficient-evidence NaN.
- Calibration: v2 artifact fingerprints/channel profile and v1 artifact backward compatibility.

Regression: all pre-existing v1.9/v1.8.x tests continue to pass; v1.9 artifacts and `data/independent/` were not modified.
