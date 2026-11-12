# CAID Backup & Restore Guide

## What to back up

| Asset | Path | Why |
|---|---|---|
| Incident store | `data/incidents/incidents.json` | Analyst state: statuses, notes, groupings |
| Model artifacts | `models/saved_models/` (~1 MB) | detector + baselines + calibration thresholds |
| Calibration manifest | `reports/calibration_v2_manifest.json` | artifact fingerprint binding |
| Calibration thresholds | `models/saved_models/calibration/` | v2_0 threshold joblibs |
| Reports | `reports/` (JSON evidence) | benchmark/evaluation history — optional |
| Streaming state | `data/stream_checkpoint/`, `data/stream_dlq/`, `data/stream_results/` | optional; DLQ is the recovery path for failed events |
| MITRE cache | `data/mitre_attack/` | optional; re-downloadable |

Core detection needs only `models/` + `data/raw` (shipped). Everything else is regenerable or optional.

## Backup procedure

```powershell
# Minimal backup (detection + analyst state)
xcopy /E /I /Y models "backup\models"
xcopy /E /I /Y data\incidents "backup\data\incidents"
copy reports\calibration_v2_manifest.json "backup\reports\"
# Full backup adds: reports\*.json, data\stream_*\, data\mitre_attack\
```

## Restore procedure (verified 2026-11-12)

1. Copy `models/` back into the repo root (preserving `saved_models/calibration/` subdir).
2. Copy `data/incidents/incidents.json` back.
3. Start the app — artifacts load lazily; incident store loads on first page visit.
4. Verify: Settings page shows model artifact "present"; Investigation page lists restored incidents.

**Smoke verification performed:** copied `models/`, `data/incidents/`, `data/raw/`, and the calibration manifest to a temp dir; `joblib.load` restored the detector dict and calibration artifact; `IncidentStore` restored **136 incidents, load_error=None**.

## Integrity notes

- Incident store and checkpoints use atomic `tempfile`+`os.replace` writes — a backup copied mid-write is still consistent (last complete version wins).
- Model artifacts are fingerprint-bound via `calibration_v2_manifest.json`; a mismatched artifact set surfaces as `uncalibrated` status rather than wrong scores.
- DLQ (`data/stream_dlq/*.jsonl`) is append-only — back it up before replay/cleanup operations.
- **Never delete originals during restore tests** — restore into a clean directory first (as verified above).
