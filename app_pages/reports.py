"""Reports & exports — SOC summary report generation and artifact inventory.

Generates a whole-analysis SOC summary (PDF/CSV) from the cached
pipeline output, then lists the offline report files CAID produces
(evaluation JSONs, calibration manifest, streaming benchmarks,
incident store) with download buttons, and documents where each
interactive export lives.
"""

import glob
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from utils.errors import safe_error_message

from core.summary_report import (
    generate_soc_summary_csv,
    generate_soc_summary_pdf,
    report_outline,
)
from modules.analysis_state import analysis_or_prompt, get_manager

st.subheader("Reports & Exports")
st.caption("Generate a SOC summary report for the analyzed scenario, "
           "or download offline evaluation artifacts. All data is "
           "synthetic/defensive research data.")

# ---- SOC summary report -----------------------------------------------------
st.markdown("### SOC summary report")
result = analysis_or_prompt()
if result is not None:
    incidents = get_manager().list_incidents()
    scored = result.get('scored_events')
    corr = result.get('correlations')
    scenario_key = str(result.get('scenario_key') or 'scenario')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    st.caption(
        f"Covers **{result.get('scenario_label', scenario_key)}** — "
        f"{len(scored) if scored is not None else 0} analyzed events, "
        f"{len(corr) if corr is not None else 0} correlation findings, "
        f"{len(incidents)} incidents.")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download SOC summary report (PDF)",
        data=generate_soc_summary_pdf(result, incidents),
        file_name=f"soc_summary_{scenario_key}_{stamp}.pdf",
        mime="application/pdf", width="stretch", key="rep_soc_pdf")
    c2.download_button(
        "Download SOC summary data (CSV)",
        data=generate_soc_summary_csv(result, incidents),
        file_name=f"soc_summary_{scenario_key}_{stamp}.csv",
        mime="text/csv", width="stretch", key="rep_soc_csv")
    st.caption("The PDF is a formatted report (scope, risk distribution, "
               "metrics, calibration, incidents, findings by type, top "
               "entities, highest-risk events). The CSV is the same data "
               "in tidy section/item/detail format. Per-incident reports "
               "are on the Investigation page.")
    with st.expander("Preview report contents"):
        st.dataframe(pd.DataFrame(report_outline(result, incidents)),
                     hide_index=True, width='stretch',
                     alt='SOC summary report outline')

# ---- Key evaluation reports -----------------------------------------------
KEY_REPORTS = [
    ('reports/detection_quality.json',
     'Unseen-user detection quality (Milestone 2.7)'),
    ('reports/fusion_evaluation.json',
     'Per-channel fusion evaluation'),
    ('reports/ml_benchmark_results.json',
     'ML model benchmark (Milestone 2.4)'),
    ('reports/calibration_v2_manifest.json',
     'v2.0 detection calibration manifest'),
]

st.markdown("### Evaluation & calibration reports")
rows = []
for path, desc in KEY_REPORTS:
    exists = os.path.exists(path)
    rows.append({
        'report': os.path.basename(path),
        'description': desc,
        'status': 'present' if exists else 'missing',
        'size_kb': (f"{os.path.getsize(path) / 1024:.1f}"
                    if exists else '—'),
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch',
             alt='Evaluation report inventory')

for path, _desc in KEY_REPORTS:
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            data = fh.read()
        st.download_button(
            f"Download {os.path.basename(path)}", data=data,
            file_name=os.path.basename(path),
            mime='application/json', key=f'rep_{os.path.basename(path)}')

# ---- Streaming benchmark reports -------------------------------------------
bench_files = sorted(glob.glob(os.path.join('reports', 'stream_benchmark_*.json')),
                     key=os.path.getmtime)
st.markdown("### Streaming benchmark reports")
if bench_files:
    latest = bench_files[-1]
    st.caption(f"{len(bench_files)} benchmark report(s) on disk; "
               f"latest: `{os.path.basename(latest)}`. Live telemetry is "
               "on the Streaming health page (Monitor group).")
    with open(latest, 'rb') as fh:
        st.download_button(
            f"Download latest benchmark ({os.path.basename(latest)})",
            data=fh.read(), file_name=os.path.basename(latest),
            mime='application/json', key='rep_latest_bench')
    with st.expander(f"All {len(bench_files)} benchmark reports"):
        for path in reversed(bench_files):
            with open(path, 'rb') as fh:
                st.download_button(
                    os.path.basename(path), data=fh.read(),
                    file_name=os.path.basename(path),
                    mime='application/json',
                    key=f'rep_bench_{os.path.basename(path)}')
else:
    st.info("No streaming benchmark reports found under `reports/` — "
            "run `scripts/stream_benchmark.py` to generate them.")

# ---- Calibration model artifacts --------------------------------------------
cal_dir = os.path.join('models', 'saved_models', 'calibration')
cal_files = sorted(glob.glob(os.path.join(cal_dir, '*.joblib')))
st.markdown("### Calibration artifacts")
if cal_files:
    st.dataframe(pd.DataFrame([{
        'file': os.path.basename(p),
        'size_kb': f"{os.path.getsize(p) / 1024:.1f}",
    } for p in cal_files]), hide_index=True, width='stretch',
        alt='Calibration artifacts')
    st.caption("Threshold artifacts are loaded automatically by the "
               "detection engine; download for offline inspection only.")
else:
    st.info(f"No calibration artifacts in `{cal_dir}`.")

# ---- Incident store ----------------------------------------------------------
st.markdown("### Incident store")
try:
    mgr = get_manager()
    n = len(mgr.list_incidents())
    st.metric("Stored incidents", n)
    st.caption(f"Store: `{mgr.store.path}`")
    if mgr.store.load_error:
        st.warning(f"Store load warning: {mgr.store.load_error}")
    if os.path.exists(mgr.store.path):
        with open(mgr.store.path, 'rb') as fh:
            st.download_button(
                "Download incident store (JSON)", data=fh.read(),
                file_name='incidents.json', mime='application/json',
                key='rep_incidents')
except Exception as exc:
    st.warning(safe_error_message(exc, "Incident store unavailable"))
st.caption("Per-incident PDF and CSV reports are generated on the "
           "Investigation page (Investigate group).")

# ---- Export map ---------------------------------------------------------------
st.markdown("### Where exports live")
st.dataframe(pd.DataFrame([
    {'page': 'Reports & Exports', 'export': 'SOC summary report (PDF + CSV)',
     'location': 'Data & Admin → Reports & Exports'},
    {'page': 'Events', 'export': 'Filtered events CSV',
     'location': 'Monitor → Events'},
    {'page': 'Session Intelligence', 'export': 'Session results CSV',
     'location': 'Investigate → Session Intelligence'},
    {'page': 'Correlation', 'export': 'Correlation findings CSV',
     'location': 'Investigate → Correlation'},
    {'page': 'Investigation', 'export': 'Incident PDF + CSV reports',
     'location': 'Investigate → Investigation (per incident)'},
    {'page': 'MITRE ATT&CK', 'export': 'Technique mappings CSV',
     'location': 'Investigate → MITRE ATT&CK'},
    {'page': 'AI Analyst', 'export': 'Threat-hunt results CSV',
     'location': 'AI → AI Analyst → Threat hunting tab'},
    {'page': 'Log Analysis', 'export': 'Processed/filtered/raw CSVs',
     'location': 'Data & Admin → Log Analysis'},
]), hide_index=True, width='stretch', alt='Export location map')
