# CAID - Canvas AI Intrusion Detection System

AI-Powered Security Monitoring, Anomaly Detection and Cyber Incident Investigation Platform

## Overview

CAID is a defensive, AI-powered security monitoring and anomaly detection platform inspired by publicly reported cyber incidents involving the Canvas/Instructure learning management platform.

## Important Notes

- This is an independent defensive research prototype
- Not an official Canvas or Instructure product
- Uses synthetic data for all demonstrations
- Do not connect to or scan real Canvas infrastructure

## Installation

### Prerequisites

- Python 3.11+ (audited on 3.14)
- Windows 10/11

### Setup

1. Open PowerShell in the project directory

2. Create virtual environment:
```powershell
python -m venv .venv
```

3. Activate virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

4. Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Run the application:
```powershell
streamlit run app.py
```

### Configuration

Copy `.env.example` to `.env` for reference values. **The app does not
auto-load `.env`** — export the variables into your environment (or source
the file) before launching:

- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` — local Ollama server
- `REDIS_URL` — Redis Streams queue for streaming ingestion (optional;
  defaults to `redis://127.0.0.1:6379/0`)

## Project Structure

```
web-project/
├── app.py                  # Main application entry point (multipage)
├── requirements.txt        # Python dependencies
├── .env.example            # Ollama configuration template
│
├── app_pages/              # Streamlit pages (13)
├── core/                   # Detection & analytics engine
│   ├── detection_engine.py #   Channel-weighted fused scoring
│   ├── calibration.py      #   Fingerprint-bound threshold artifacts
│   ├── drift_monitor.py    #   Observational distribution drift (M2.4)
│   └── ...                 #   log/rules/risk/session/sequence/correlation/
│                           #   incidents/MITRE/AI modules
├── models/                 # anomaly_detector.py + saved_models/
├── modules/                # Streamlit UI/state layer
├── data/                   # raw/, processed/, scenarios, incidents, MITRE cache
├── utils/validators.py     # CSV schema validation & cleaning
├── scripts/                # benchmark_models.py, evaluate_fusion.py (M2.4)
├── tests/                  # pytest suite
├── docs/                   # audit/, architecture/, research/,
│                           # testing/, roadmap/
└── reports/                # generated_reports/ + benchmark JSON output
```

See `PROJECT_DOCUMENTATION.md` for the full architecture reference
and `docs/` for milestone deliverables.

## Development

### Running Tests

```powershell
pytest tests -q
```

### Benchmarks (Milestone 2.4)

```powershell
.venv\Scripts\python scripts\benchmark_models.py   # ML model comparison
.venv\Scripts\python scripts\evaluate_fusion.py    # channel fusion eval
```

### Project Milestones

- **Milestones 1.x–2.3**: dashboard, log processing, Isolation Forest,
  risk scoring, behavioral/session/sequence detection, correlation,
  incidents, MITRE mapping, AI analyst, scenarios
- **Milestone 2.4**: repository audit, test remediation, ML benchmark
  framework, drift monitoring, fusion validation, real-time IDS
  architecture design — see `docs/`

## License

This project is developed for educational and research purposes.
