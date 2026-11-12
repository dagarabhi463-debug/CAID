# CAID — Canvas AI Intrusion Detection System

**Complete Project Documentation**

> CAID is an independent, defensive security-research prototype for AI-powered
> security monitoring, anomaly detection, and cyber-incident investigation.
> It is **not** an official Canvas or Instructure product, uses **synthetic
> data only**, and must never be connected to real Canvas infrastructure.
> All scores are calibrated anomaly/deviation scores — **not probabilities** —
> and correlation strengths are **not** proof of attacker coordination.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack & Setup](#2-technology-stack--setup)
3. [Repository Layout](#3-repository-layout)
4. [Application Architecture](#4-application-architecture)
5. [Detection & Analytics Pipeline](#5-detection--analytics-pipeline)
6. [AI Analyst (Ollama)](#6-ai-analyst-ollama)
7. [Dashboard Pages](#7-dashboard-pages)
8. [Data Assets & Scenarios](#8-data-assets--scenarios)
9. [Testing & Validation](#9-testing--validation)
10. [Milestone History](#10-milestone-history)
11. [Current State, Limitations & Known Issues](#11-current-state-limitations--known-issues)

---

## 1. Project Overview

CAID ingests synthetic security-event logs (login activity, course/file/grade
access, API usage, after-hours activity, etc.) and runs a multi-channel
detection pipeline:

```
CSV logs → validation → feature engineering → Isolation Forest (ML)
        → rule engine (safe AST) → behavioral baselines → session engine
        → sequence detection → combined DetectionEngine → risk levels
        → cross-session correlation → incidents → MITRE ATT&CK mapping
        → AI analyst explanations (local Ollama)
```

Everything downstream of scoring — correlation, incidents, MITRE mapping, AI
explanations — is **retrospective only** and never feeds back into per-event
scores, channel weights, or thresholds.

### Design principles (from project steering)

- Synthetic data only; never connect to or scan real Canvas infrastructure.
- Explainable ML: every score ships with evidence; missing channels are
  explicit, never silently treated as "normal".
- No data leakage: baselines are fitted on designated training-normal data
  only; inference never refits.
- Determinism: identical input produces identical IDs, ordering, and scores.
- Graceful degradation: missing model/calibration/Ollama produces an explicit
  reported status, never fabricated results.
- Beginner-friendly, modular code; heavy commenting on important sections.

---

## 2. Technology Stack & Setup

| Layer | Technology |
|---|---|
| UI | Streamlit (multipage `st.navigation` app) |
| Data | pandas, NumPy |
| ML | scikit-learn `IsolationForest`, joblib-persisted artifacts |
| Visualization | Plotly, matplotlib |
| Reports | reportlab (PDF), sanitized CSV export |
| AI | Local Ollama over HTTP via stdlib `urllib` — no API keys, no cloud |
| Tests | pytest |

**Prerequisites:** Python 3.12+, Windows 10/11.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
pytest tests -q
```

Optional AI analyst configuration (`.env.example` → `.env`, never commit
secrets):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=60
```

Ollama is optional — install the app separately and `ollama pull llama3.1:8b`.
Without it, the AI analyst returns a deterministic evidence summary labelled
as non-AI fallback.

---

## 3. Repository Layout

```
web-project/
├── app.py                     # Entry point: sidebar + multipage navigation
├── requirements.txt           # Dependencies (see §2)
├── .env.example               # Ollama config template
├── README.md                  # Original README (structure section is stale —
│                              #   see §11)
│
├── app_pages/                 # Streamlit pages (10)
│   ├── overview.py            #   Monitor: KPIs, status, charts
│   ├── events.py              #   Monitor: searchable scored-event table
│   ├── anomaly_detection.py   #   Monitor: IF + rules + behavioral evidence
│   ├── session_intelligence.py#   Investigate: sessions, sequences, risk
│   ├── correlation.py         #   Investigate: cross-session findings
│   ├── investigation.py       #   Investigate: incidents, triage, reports
│   ├── mitre.py               #   Investigate: ATT&CK coverage/mappings
│   ├── ai_analyst.py          #   AI: Ollama-grounded Q&A & hunting
│   ├── log_analysis.py        #   Data & Admin: upload/explore raw logs
│   └── settings.py            #   Data & Admin: env, Ollama, MITRE status
│
├── core/                      # Detection & analytics engine (19 modules)
│   ├── log_processor.py       #   CSV load/validate, search, timelines
│   ├── feature_engineering.py #   Features + FeatureBaselineManager
│   ├── detection_rules.py     #   Safe AST rule engine (no eval/exec)
│   ├── risk_engine.py         #   Heuristic risk scoring + calibration
│   ├── detection_engine.py    #   Channel-weighted combined scoring
│   ├── behavioral_baseline.py #   Persisted per-user/global baselines
│   ├── behavioural_analyzer.py#   Deviation scoring wrapper
│   ├── session_reconstruction.py # Session grouping + causal prefix features
│   ├── session_baseline.py    #   Persisted session baselines
│   ├── session_analyzer.py    #   Session deviation scoring wrapper
│   ├── sequence_detection.py  #   Transition/pattern sequence analyzer
│   ├── cross_session_correlation.py # Retrospective correlation findings
│   ├── incident_manager.py    #   Incident grouping, lifecycle, JSON store
│   ├── incident_report.py     #   PDF/CSV incident report generation
│   ├── mitre_attack.py        #   ATT&CK STIX loader + technique mapping
│   ├── calibration.py         #   Versioned, fingerprint-bound thresholds
│   ├── ml_integration.py      #   AnomalyDetector artifact adapter
│   ├── ai_context.py          #   Grounded, injection-sanitized prompts
│   ├── ai_analyst.py          #   Explanation/hunt/summary orchestration
│   └── ollama_client.py       #   Typed-error local Ollama HTTP client
│
├── models/
│   ├── anomaly_detector.py    #   IsolationForest wrapper + evaluate()
│   └── saved_models/          #   Persisted artifacts
│       ├── anomaly_detector.joblib
│       ├── behavioral_baselines.joblib
│       └── calibration/       #   Versioned threshold artifacts (v1 + v2)
│
├── modules/                   # Streamlit UI/state layer (9 modules)
│   ├── analysis_state.py      #   Shared per-scenario analysis cache
│   ├── scenario_data.py       #   Scenario selector + frame loading
│   ├── session_intelligence.py#   Session dashboard renderer
│   ├── correlation_dashboard.py#  Correlation dashboard renderer
│   ├── investigation_dashboard.py# Incident dashboard renderer
│   ├── threat_detection.py    #   Anomaly-detection page renderer
│   ├── dashboard.py           #   Plotly chart components
│   ├── log_analysis.py        #   Log-analysis page renderer
│   └── export.py              #   CSV export w/ formula-injection sanitize
│
├── data/
│   ├── raw/                   #   Sample CSVs (normal/suspicious/high_risk,
│   │                          #   security_logs)
│   ├── processed/             #   evaluation_data.csv
│   ├── incidents/incidents.json # Persisted incident store
│   ├── mitre_attack/          #   Cached ATT&CK STIX bundle + metadata
│   └── generate_scenarios.py  #   Deterministic scenario generator (M2.3)
│
├── utils/validators.py        # CSV schema validation & cleaning
├── tests/                     # 25 pytest modules (~439 tests)
└── reports/generated_reports/ # (empty; reports generated on demand)
```

> Note: `README.md`'s structure section references files that no longer
> exist (`config.py`, `ai/`, `utils/helpers.py`, `models/train_model.py`).
> This document describes the actual tree.

---

## 4. Application Architecture

### Entry point — `app.py`

- `st.set_page_config` (wide layout), global CSS tweak.
- Sidebar: CAID branding, `render_scenario_selector()` (default sample
  dataset or a Milestone-2.3 scenario), and the currently-analyzed scenario
  status.
- `st.navigation` groups pages into four sections:
  - **Monitor** — Overview, Events, Anomaly Detection
  - **Investigate** — Session Intelligence, Correlation, Investigation, MITRE
  - **AI** — AI Analyst
  - **Data & Admin** — Log Analysis, Settings

### Shared analysis state — `modules/analysis_state.py`

Runs the full v2.x pipeline **once per selected scenario** and caches the
result in `st.session_state['caid_analysis']`, so every page sees identical
data. Also caches the `IncidentManager` (`caid_incident_manager`). Provides
`analysis_or_prompt()` — pages show a "load and analyze" prompt until a
scenario has been analyzed.

### Scenario selection — `modules/scenario_data.py`

Wraps `core.log_processor.load_sample_dataset` and
`data.generate_scenarios.generate_scenario` behind a sidebar selector with
options: the default sample dataset plus six scripted scenarios (see §8).

---

## 5. Detection & Analytics Pipeline

### 5.1 Ingestion & validation

- **`utils/validators.py`** — enforces the log schema: required columns
  (`timestamp, user_id, role, source_ip, event_type, courses_accessed,
  accounts_accessed, api_requests, data_mb, token_age_minutes, after_hours`),
  optional `label/session_id/auth_result`, a synthetic event-type
  vocabulary, and cleaning/summary helpers.
- **`core/log_processor.py`** — `load_security_log`, `load_sample_dataset`,
  statistics, `search_events`, `filter_events`, `get_user_activity`,
  `get_ip_activity`, `generate_timeline`, `detect_duplicates`.

### 5.2 Feature engineering — `core/feature_engineering.py`

- Time features (hour, day, weekend, after-hours) plus user/IP/event
  aggregation features.
- **`FeatureBaselineManager`** — the anti-leakage core: baselines (per-user,
  per-IP, event-type, global) are computed on **training data only** and
  persisted inside the detector artifact. Inference only *transforms*;
  unseen entities fall back to global stats; zero-variance uses a safe
  std floor; missing/non-finite inputs are explicit. `fingerprint()` binds
  baselines to calibration artifacts.

### 5.3 ML anomaly detection — `models/anomaly_detector.py` + `core/ml_integration.py`

- `IsolationForest` trained on normal-activity baselines; `fit()` /
  `predict()` / `save()` / `load()` with schema-validated joblib artifacts
  under `models/saved_models/` (currently `anomaly_detector.joblib`,
  `behavioral_baselines.joblib`, `calibration/`).
- **Score contract:** `anomaly_score` (raw `decision_function`, higher =
  *more normal*), `anomaly_score_normalized` / `ml_anomaly_score` (0–1,
  higher = *more anomalous*), `ml_confidence`, `is_anomaly`.
- `MLIntegration` adapts the artifact for the pipeline; a missing or
  incompatible model raises/flags errors rather than silently returning
  "all normal".
- `evaluate()` supports binary (suspicious+high_risk positive) and
  high-risk-only modes with precision/recall/FPR/ROC-AUC/PR-AUC.

### 5.4 Rule engine — `core/detection_rules.py`

- Configurable rules evaluated by a **restricted Python AST interpreter** —
  no `eval()`/`exec()`. Severities LOW→CRITICAL, actions ALERT/LOG.
- Emits `rule_score`/`rule_detection_score` (0–1 evidence) plus structured
  findings; also feeds MITRE mapping.

### 5.5 Risk engine — `core/risk_engine.py`

- Combines ML anomaly score with heuristic weighted factors into
  `risk_score`/`combined_score` (0–1, higher = riskier) and classifies into
  Low/Medium/High/Critical.
- `calibrate_thresholds()` fits **absolute** cutoffs on a separate
  calibration dataset; thresholds are versioned and dataset-relative
  percentiles are used only as an explicitly-labelled fallback.

### 5.6 Behavioral analysis — `core/behavioral_baseline.py` + `core/behavioural_analyzer.py`

- `BehavioralBaselineManager`: persisted per-user + global statistics fitted
  on training-normal data only; unseen users → global stats; canonical
  ordered feature list prevents schema drift.
- `BehavioralAnalyzer.analyze_user_behavior()` scores deviation per user;
  `behavioral_deviation_score` ∈ [0,1] (higher = more deviation), NaN +
  `behavioral_score_missing` when no usable baseline/input. Inference never
  refits; raises `RuntimeError` if unfitted. Chronological
  expanding-window evaluation is supported for research.

### 5.7 Session engine — `core/session_reconstruction.py`, `session_baseline.py`, `session_analyzer.py`

- **Session identity:** explicit non-empty `session_id` wins (scoped per
  user); otherwise per-user inactivity timeout grouping (default 1800 s);
  missing timestamps are isolated into deterministic per-event sessions;
  duplicates preserved and flagged by content hash.
- **Two feature views:** *causal session-prefix features* per event (only
  events up to and including the current one — future events can never
  influence earlier scores) and *completed-session summaries* for
  retrospective dashboards only.
- `SessionBaselineManager` mirrors the behavioral contract: fit on
  train-normal only, never mutates at inference, global fallback,
  deterministic versioned joblib artifact with `fingerprint()`.
- `SessionAnalyzer` adds `session_deviation_score` ∈ [0,1] (higher = more
  deviation; NaN + `session_score_missing` when missing) and retrospective
  session risk summaries.

### 5.8 Sequence detection — `core/sequence_detection.py`

- Interpretable statistical sequence analysis (no deep learning): per-user/
  global event-type transition frequencies with smoothing, causal in-session
  windows (`window_seconds`, `max_window_events`), event-rate and
  sensitive-operation deviations.
- Configurable patterns (defaults): `login_failure_burst_then_access`,
  `sensitive_operation_burst`, `export_after_broad_access`,
  `ip_switch_burst`.
- `sequence_deviation_score` ∈ [0,1] (**not a probability**); components:
  transition rarity, rate deviation, sensitive deviation, pattern match —
  all exposed via `sequence_factors`, `sequence_evidence`,
  `sequence_triggered_patterns` for explainability. Missing `auth_result`
  deactivates auth-dependent patterns rather than fabricating data.

### 5.9 Combined detection — `core/detection_engine.py`

- Merges channels — ML, rules, behavioral, session, sequence — into
  `combined_score` and `risk_level` with structured `evidence`.
- **Fixed channel profiles** (weights are documented, never tuned on
  holdout): `legacy` (ml .35 / rules .35 / behavioral .30), `session`,
  `sequence`, `v2_0` (five channels).
- **Operating modes:** `FULL_*` when all configured channels have evidence,
  `DEGRADED_*` when exactly two remain, `INSUFFICIENT_EVIDENCE` (<2) → NaN
  score; missing channels are excluded and weights renormalize.
  `missing_channels`, `channel_weights`, `channels_available`, and
  per-channel calibration status are all surfaced.

### 5.10 Calibration — `core/calibration.py`

- Versioned threshold artifacts (e.g. `cal-20261111T210735Z-…`,
  `v2_0_thresholds_…`) persisted under `models/saved_models/calibration/`.
- Each artifact is **fingerprint-bound** to the detector and every fitted
  component (feature baselines, behavioral baselines, session baselines,
  sequence model); loading with a mismatched fingerprint raises
  `CalibrationUnavailable`.
- Callers must fall back to an explicitly-reported
  `uncalibrated_dataset_relative` mode — never silently apply wrong
  thresholds. `apply_to_engine` / `apply_detection_calibration` wire
  artifacts into `RiskEngine`/`DetectionEngine`.
- **v2_0 detection artifact (closure):** regenerated via
  `scripts/calibrate_v2_detection.py` on the benchmark calibration
  split → `cal-20261112T020320Z-2962264f`, verified `calibrated`;
  stale artifacts archived under `calibration/archive/`;
  `tests/test_calibration_artifacts.py` guards fingerprint drift.

### 5.10a Artifact lifecycle — `core/artifact_lifecycle.py`

All `ensure_*` helpers (detector, behavioral/session/sequence
baselines) share one policy: `loaded` → use artifact;
missing → fit on `normal_activity.csv` + persist
(`refit_fallback_missing`); corrupt/incompatible → quarantine to
`*.corrupt-<UTC>` (never silently overwritten) + refit + persist
(`refit_fallback_corrupt`). Statuses surface via `artifact_status`,
`st.session_state.artifact_statuses`, and UI captions.

### 5.10b Upload pipeline parity (AUD-003)

`modules/log_analysis.py::analyze_uploaded_data` delegates to
`analyze_session_intelligence` — uploaded CSVs receive the identical
v2_0 channel pipeline (ML + rules + behavioral + session + sequence)
and calibration status as scenario analysis.

### 5.11 Cross-session correlation — `core/cross_session_correlation.py`

Retrospective layer (Milestone 2.1) over already-scored events:

- **Contract:** never feeds back into per-event scores; every finding
  carries `evidence` dicts (`signal`, `observed_value`,
  `expected_baseline_or_threshold`, `explanation`); deterministic IDs
  `corr_<sha1[:12]>`.
- **Finding types:** `repeated_pattern_across_sessions`,
  `similar_account_behavior` (cosine-similar profiles + shared pattern),
  `repeated_sensitive_operations`, `long_term_behavior_change`,
  `shared_ip_activity`.
- **Shared-IP policy:** shared IP is never proof — uncorroborated findings
  are capped (`shared_ip_max_score` = 0.4), flagged `shared_ip_only`, and
  escalate only with corroboration from another signal type.
- `account_profiles` exposes the per-user aggregates behind similarity.

### 5.12 Incident management — `core/incident_manager.py`

- Groups correlation findings into incidents via **union-find** over shared
  entities (users, sessions, IPs from evidence); incidents below
  `min_findings_per_incident` are skipped.
- Deterministic IDs `inc_<sha1[:12]>`; re-running analysis upserts while
  preserving analyst status/notes.
- **Lifecycle:** `create_from_findings`, `merge`, `split`, `add_note`,
  `set_status` (`Open`, `Investigating`, `Contained`, `Closed`,
  `FalsePositive`); `evidence_timeline` merges time-ordered event rows
  tagged with covering finding IDs.
- **Severity** derived from max `correlation_score` via configurable
  thresholds — a triage aid, not a probability.
- **Persistence:** atomic JSON store at `data/incidents/incidents.json`;
  corrupt/missing files recover empty with a warning flag.
- Shared-IP-only incidents keep their caveat through UI and reports.

### 5.13 Incident reports — `core/incident_report.py` + `modules/export.py`

- PDF via reportlab (`SimpleDocTemplate`): metadata, findings, per-finding
  evidence, shared-IP caveat block, evidence timeline, analyst notes,
  disclaimer.
- CSV in sanitized long format (section/item/detail); `modules/export.py`
  neutralizes spreadsheet formula injection (`=`, `+`, `-`, `@`, tab, CR
  prefixed with `'`).

### 5.14 MITRE ATT&CK — `core/mitre_attack.py`

- Downloads/caches the official ATT&CK Enterprise STIX 2.1 bundle under
  `data/mitre_attack/` (`enterprise-attack.json` + meta).
- Maps CAID detection rules, correlation findings, and event evidence to
  techniques with recorded technique ID, tactic, source, evidence,
  rationale, confidence (`confirmed` vs `potential`), mapping source, and
  framework version.
- **Contract:** potential mappings are never presented as confirmed
  behavior; unmapped sources are reported, never guessed;
  shared-IP-only findings produce no technique mapping.

---

## 6. AI Analyst (Ollama)

- **`core/ollama_client.py`** — stdlib-only HTTP client for a local Ollama
  server (`/api/generate`, `/api/tags`). Typed errors
  (`OllamaError`/`OllamaModelNotFound`/…) let the UI degrade gracefully.
- **`core/ai_context.py`** — builds every prompt exclusively from real CAID
  evidence: whitelisted event fact fields, structured finding/incident
  context, `wrap_untrusted` delimiters, and regex sanitization against
  prompt injection. Ships a registry of **safe predefined threat-hunting
  queries** — the AI never executes generated code, SQL, or shell.
- **`core/ai_analyst.py`** — `explain_event`, `explain_finding`,
  `summarize_incident`, `hunt`, `ask`. Always returns the evidence used
  alongside generated text; when Ollama is unavailable it returns a
  deterministic evidence summary labelled non-AI fallback. It never
  modifies scores, thresholds, or incident state.

---

## 7. Dashboard Pages

All "Investigate"/"Monitor" pages consume the shared `analysis_state`
pipeline result; each page prompts to load a scenario until analysis runs.

| Page | File | What it shows |
|---|---|---|
| Overview | `app_pages/overview.py` | Platform KPIs (events, risk mix, anomalies, open/critical incidents), Plotly charts |
| Events | `app_pages/events.py` | Searchable/filterable table of every scored event with ML, behavioral, sequence, and risk scores |
| Anomaly Detection | `app_pages/anomaly_detection.py` | Operating modes, missing-channel rate, calibration status, IF + rule + behavioral evidence via `modules/threat_detection.py` |
| Session Intelligence | `app_pages/session_intelligence.py` | Session timelines, suspicious sequences, session risk summaries, deviations, evidence, operating mode via `modules/session_intelligence.py` |
| Correlation | `app_pages/correlation.py` | Cross-session findings with per-signal evidence via `modules/correlation_dashboard.py` |
| Investigation | `app_pages/investigation.py` | Incident metrics/filters/table, manual grouping, merge/split, notes, PDF/CSV export via `modules/investigation_dashboard.py` |
| MITRE ATT&CK | `app_pages/mitre.py` | Framework load/status, tactic/technique coverage, confirmed vs potential mappings |
| AI Analyst | `app_pages/ai_analyst.py` | Ollama status, event/finding explanations, incident summaries, safe threat hunting |
| Log Analysis | `app_pages/log_analysis.py` | Upload/validate/explore raw logs (`modules/log_analysis.py`) |
| Settings | `app_pages/settings.py` | Python/Streamlit versions, Ollama config + connectivity test, MITRE dataset management |

---

## 8. Data Assets & Scenarios

### Raw datasets — `data/raw/`

`security_logs.csv`, `normal_activity.csv`, `suspicious_activity.csv`,
`high_risk_activity.csv` — the original sample corpus (14-column schema,
optional `label`/`session_id`/`auth_result`).

### Processed — `data/processed/evaluation_data.csv`

Labeled evaluation set (normal/suspicious/high_risk) used by
`AnomalyDetector.evaluate()`.

### Scenario generator — `data/generate_scenarios.py` (Milestone 2.3)

Deterministic, seeded scenario scripts in the raw 14-column schema:

| Key | Scenario |
|---|---|
| `default_sample` | Built-in sample dataset (all activity) |
| `baseline` | Baseline (mostly normal) |
| `insider_exfiltration` | Insider data exfiltration |
| `credential_stuffing_ring` | Credential stuffing ring |
| `after_hours_lab` | After-hours lab access |
| `low_and_slow` | Low-and-slow drift |
| `mixed_campaign` | Mixed campaign (demo) |

Guarantees: RFC 5737 documentation IP ranges only (192.0.2.0/24,
198.51.100.0/24, 203.0.113.0/24), explicit `session_id` and `auth_result`,
seeded determinism, and each scenario is designed to exercise specific
correlation types end-to-end.

### Other data

- `data/incidents/incidents.json` — persisted incident store (created at
  runtime).
- `data/mitre_attack/` — cached ATT&CK Enterprise STIX bundle + metadata.

> Earlier generators (`data/generate_independent_datasets.py`,
> `data/independent*/`) existed for the M1.9/M2.0 leakage audits and were
> removed in the repository cleanup; their methodology is preserved in the
> milestone history (§10) and git history.

---

## 9. Testing & Validation

`tests/` contains 25 pytest modules (~439 tests). Groups:

- **Core engines:** `test_detection_engine`, `test_detection_rules`,
  `test_risk_engine`, `test_log_analysis`, `test_ml_integration`,
  `test_anomaly_detector`, `test_score_consistency`
- **Baselines/session:** `test_behavioral_baseline`,
  `test_behavioral_analyzer`, `test_baseline_integration`
- **Milestone regressions:** `test_milestone_17`, `test_milestone_18_1`,
  `test_milestone_2_0`, `test_milestone_2_0_1_causal`,
  `test_milestone_2_1_correlation`, `test_milestone_2_2_incidents`,
  `test_milestone_2_3_scenarios`
- **AI/MITRE/system:** `test_ai_grounding`, `test_ollama_client`,
  `test_mitre_attack`, `test_system_reliability`, `test_dashboard_pages`

**Latest run (this workspace):** `436 passed, 3 failed` — see §11.

Historical validation also included standalone audit/evaluation scripts
(`scripts/leakage_audit_*.py`, `evaluate_holdout_*.py`,
`milestone_*_audit.py`, `benchmark_*.py`) — removed in cleanup; recoverable
from git history.

---

## 10. Milestone History

Condensed from the project's milestone reports (deleted `docs/` folder,
recoverable via `git show bfdca25:docs/`).

| Milestone | Delivered |
|---|---|
| **1.x Foundation** | Streamlit dashboard, log processor, validators, CSV export |
| **1.5** | Two evaluation modes (binary / high-risk), consistent train/predict features, model persistence — 60 tests validated |
| **1.7** | Performance: vectorized feature engineering (~50× `predict` speedup; 10k events ≈ 3.8 s median), absolute risk-threshold calibration on separate data |
| **1.8 / 1.8.1** | False-positive root cause found & fixed (unseen-user fallback used population count → ~500× outlier; fixed with per-user mean), artifact retrained; score-distribution & fallback audits |
| **1.9** | Independent seeded datasets (train/calibration/validation/holdout, entity-disjoint cohorts), leakage audit (legacy leaks found & fixed via `FeatureBaselineManager`), holdout metrics (binary: precision 1.000, recall 0.410, FPR 0.000, ROC-AUC 0.9177 — synthetic only), versioned fingerprint-bound calibration store |
| **2.0** | Session engine (`session_reconstruction`/`session_baseline`/`session_analyzer`), `SequenceAnalyzer` with causal windows & interpretable patterns, `DetectionEngine` channel profiles (`legacy`/`session`/`sequence`/`v2_0`) with explicit degraded/insufficient operating modes, calibration artifact v2 with component fingerprints |
| **2.0.1** | Causal validation (future-event exclusion, deterministic ordering, timeout/missing-ts/duplicate rules — 9 tests), feature-baseline fingerprints, calibration rejection tests, shared-IP user separation; **361 tests passed** at that checkpoint |
| **2.1** | `CrossSessionCorrelator` + correlation dashboard: 5 evidence-first finding types, deterministic IDs, shared-IP-never-proof policy |
| **2.2** | `IncidentManager` (union-find grouping, lifecycle, notes, merge/split) + JSON persistence + PDF/CSV reports + investigation dashboard |
| **2.3** | Deterministic scenario generator + sidebar scenario selector across dashboards |
| **Leakage-fix spec** | `.kiro/specs/caid-leakage-fix/requirements.md` — formal requirements for eliminating train/predict baseline leakage (all implemented in 1.9/2.0 work) |

---

## 11. Current State, Limitations & Known Issues

### What works today

- Full pipeline: sample CSV or scripted scenario → scored events →
  sessions → sequences → correlation findings → incidents → MITRE mapping →
  AI explanations, all visible across 10 dashboard pages.
- Persisted, fingerprint-bound calibration artifacts; explicit
  calibrated/uncalibrated operating status everywhere.
- Local-only AI analyst with prompt-injection sanitization and safe
  threat-hunting queries.

### Test status (v1.0, 2026-11-12)

`pytest tests -q` → **544 passed, 0 skipped, 4 warnings**. The earlier
"436 passed / 3 failed" note is obsolete (pre-release baseline).

### Documented limitations (by design)

> **v1.0 note:** this section predates the v2_0 stack and streaming
> pipeline. Authoritative limitation list: `docs/release/CAID_KNOWN_LIMITATIONS.md`.

- **Synthetic data only** — no real Canvas connectivity; metrics are not
  real-world accuracy figures. Independent synthetic holdout: recall 0.94,
  precision 0.50 (`docs/research/phase_4_3_independent_validation.md`).
- **Retrospective, batch-oriented dashboard** — streaming detection
  exists (`core/streaming/`) but correlation/incidents remain
  window-scoped.
- **Heuristic, uncalibrated correlation thresholds** — `CorrelationConfig`
  cutoffs are documented defaults; correlation precision/recall can't be
  measured (datasets label events, not relationships).
- **Shared IP ambiguity** — NAT/VPN/labs make IP a weak signal; corroboration
  required by policy.
- **Score semantics** — all scores are 0–1 deviations/strengths, never
  probabilities; `anomaly_score` raw direction is inverted vs normalized.
- **Performance** — fine at research scale (~3.8 s @10k events, M1.7);
  per-row rule AST eval and `iterrows` remain the hotspots.
- **README drift** — `README.md` project-structure section lists removed
  files (`config.py`, `ai/`, `utils/helpers.py`, etc.).

### Recovery notes

Milestone docs, audit scripts, independent datasets, and saved model
variants were removed in commit `1744782` ("Remove unused files…"). The
full history is intact in git: `git show bfdca25:<path>` recovers any
`docs/`, `scripts/`, or `data/independent*` artifact.
