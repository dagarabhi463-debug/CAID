# CAID Installation Guide

**v1.0 · verified 2026-11-12 on Windows 11 / Python 3.14.6**

## Prerequisites

- Python 3.11+ (developed/verified on 3.14)
- Git
- Optional: [Ollama](https://ollama.com) for AI Analyst features
- Optional: Redis ≥ 5.0 for durable streaming queue (`redis://` URL)

## Install

```powershell
git clone <repo> caid && cd caid
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # optional; edit as needed
```

`requirements.txt` pins: streamlit, pandas, numpy, scikit-learn, joblib, plotly, matplotlib, pytest, reportlab, redis — all with bounded version ranges.

## Model artifacts

Detection artifacts ship in `models/saved_models/` (anomaly_detector, behavioral/session/sequence baselines, calibration thresholds). They are loaded lazily via `ensure_*` helpers; no training step is needed for inference. `reports/calibration_v2_manifest.json` binds artifact fingerprints.

## Run

```powershell
streamlit run app.py
```

Open http://localhost:8501. Select a scenario in the sidebar → **Load and analyze**.

## External dependencies

| Dependency | Required? | Behavior when absent |
|---|---|---|
| Ollama (`OLLAMA_BASE_URL`, default `http://localhost:11434`) | No | AI Analyst returns deterministic evidence summaries; detection unaffected |
| Redis (`REDIS_URL`) | No | Streaming falls back to in-process asyncio queue (dev-grade durability); `test_redis_streams.py` skips |
| MITRE dataset (`data/mitre_attack/`) | No | MITRE page shows "not loaded"; `ensure_data(allow_download=True)` can fetch the official STIX bundle once |

## Verify installation

```powershell
python -m compileall -q app.py core modules utils data app_pages
.venv\Scripts\python.exe -m pytest tests -q    # expect 543+ passed
```

## Python compatibility

Verified on CPython 3.14.6 (Windows). Code uses stdlib + the pinned deps only; pandas 3.x string-dtype semantics are already handled in `modules/export.py`.
