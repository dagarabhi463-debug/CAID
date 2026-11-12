# CAID Phase 4.0 — Definitive Test Baseline

Date: 2026-11-12
Auditor: Devin (release audit, read-only — no code changes made)

## Environment

| Item | Value |
|---|---|
| OS | Windows (Git Bash), OneDrive-synced workspace |
| Python | 3.14.6 (`.venv/Scripts/python.exe`, venv == system python) |
| Streamlit | 1.67.0 |
| pandas | 3.1.1 |
| numpy | 2.5.4 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| plotly | 7.1.0 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |
| reportlab | 5.1.0 |
| redis (py) | 6.4.0 |
| Redis server | reachable — `redis-cli ping` → `PONG` (localhost:6379) |
| Ollama | CLI 0.40.16 installed |

## Commands run and results

### 1. Full pytest suite

```
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

**Result: 543 passed, 0 skipped, 4 warnings in 162.50s**

Warnings (all pre-existing, non-blocking):
- 4 × `PytestRemovedIn10Warning: Class-scoped fixtures defined as instance methods are deprecated`
  - `tests/test_milestone_2_3_scenarios.py::TestScenarioCorrelations::test_mixed_campaign_produces_multiple_types`
  - `tests/test_milestone_2_4_reliability.py::TestBenchmarkSplitIntegrity::test_entity_disjoint_splits`
  - `tests/test_redis_streams.py::TestRedisWorkerEndToEnd::test_worker_processs_and_acks`
  - `tests/test_score_consistency.py::TestEndToEndScenarios::test_pipeline_columns_present`

### 2. Compile check

```
.venv/Scripts/python.exe -m compileall -q app.py app_pages modules core utils tests scripts
```

**Result: clean, exit 0** (no output).

### 3. Streamlit startup verification

```
.venv/Scripts/python.exe -m streamlit run app.py --server.headless true --server.port 8765
curl -s -o /dev/null -w "%{http_code}" http://localhost:8765   → 200
```

App serves HTTP 200; shell page loads (verified HTML shell + health endpoint). Server terminated after check.

### 4. Registered dashboard page tests

`tests/test_dashboard_pages.py` parametrizes **all 13 pages** registered in `app.py` (verified against `st.navigation` block):

Overview, Events, Anomaly Detection, Streaming health, Detection Quality, Session Intelligence, Correlation, Investigation, MITRE ATT&CK, AI Analyst, Log Analysis, Settings, Reports & Exports — plus `app.py` entrypoint, MITRE cached-bundle regression, and Overview load-and-analyze functional test. All passed within the 543.

### 5. Redis integration tests

Redis was **available** during this run (`PONG`). `tests/test_redis_streams.py` and `tests/test_streaming_queue.py` executed their Redis paths (they `pytest.skip` only when Redis is unreachable or < 5.0). No skips occurred.

### 6. Streaming benchmark verification (saved reports)

Latest saved report: `reports/stream_benchmark_20261112T045510Z.json`

| Metric | Value |
|---|---|
| mode / queue | soak / asyncio |
| duration | 3600 s |
| events produced = processed | 360,000 = 360,000 |
| dead-lettered / rejected / duplicate / lost / inflight | 0 / 0 / 0 / 0 / 0 |
| DLQ depth | 0 |
| effective throughput | ~98.7 eps |

27 benchmark reports total (`reports/stream_benchmark_20261112T*.json`), all from 2026-11-12 benchmark sessions; 29 result JSONL + 29 DLQ JSONL pairs under `data/stream_results/` and `data/stream_dlq/`.

## Reconciliation: 528 passed/8 skipped vs 543 passed

Both numbers are consistent — they describe **different repository states**:

- **528 passed / 8 skipped** (`docs/audit/complete_dashboard_inventory.md`, `dashboard_functionality_audit.md`, `dashboard_test_report.md`, `milestone_2_5_1_reliability_report.md`): the pre–Milestone-3.2 baseline. The 8 skips were Redis-dependent tests skipping because no Redis server was reachable at that time; the suite then had ~536 collected tests.
- **543 passed** (`docs/audit/milestone_3_2_completion_report.md` and this audit): post-3.2 suite. Milestone 3.2 added page/regression/functional tests (streaming_health page coverage, MITRE cached-bundle regression, Overview load-and-analyze, Detection Quality, Reports & Exports pages) bringing collection to 543, and the report did not enumerate skips.
- **This audit (543 passed / 0 skipped)**: Redis was running, so the 8 formerly-skipped Redis tests executed and passed. Net: same 543-test suite; skip count is environment-dependent, not a regression.

**Definitive baseline for v1.0 RC: 543 passed, 0 skipped, 4 deprecation warnings, ~162s, Redis required for full pass count.**
