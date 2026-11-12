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

- Python 3.12+
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

## Project Structure

```
CAID/
├── app.py                  # Main application entry point
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── data/                  # Data files
│   ├── raw/              # Raw synthetic datasets
│   └── processed/        # Processed data
│
├── models/                # ML models
│   ├── anomaly_detector.py
│   ├── train_model.py
│   └── saved_models/
│
├── core/                  # Core business logic
│   ├── log_processor.py
│   ├── feature_engineering.py
│   ├── risk_engine.py
│   └── incident_analyzer.py
│
├── ai/                    # AI/ML features
│   ├── ai_explainer.py
│   ├── mitre_mapper.py
│   └── prompt_templates.py
│
├── modules/               # Application modules
│   ├── dashboard.py
│   ├── log_analysis.py
│   ├── threat_detection.py
│   └── reports.py
│
├── utils/                 # Utility functions
│   ├── helpers.py
│   ├── validators.py
│   └── export_utils.py
│
├── tests/                 # Automated tests
│
├── docs/                  # Documentation
│
└── reports/               # Generated reports
    └── generated_reports/
```

## Development

### Running Tests

```powershell
pytest tests/ -v
```

### Project Milestones

- **Milestone 1**: Foundation and dashboard
- **Milestone 2**: Synthetic dataset and log processing
- **Milestone 3**: Isolation Forest anomaly detection
- **Milestone 4**: Risk scoring and threat classification
- **Milestone 5**: AI incident explanation
- **Milestone 6**: MITRE ATT&CK mapping
- **Milestone 7**: Incident reports and PDF export

## License

This project is developed for educational and research purposes.
