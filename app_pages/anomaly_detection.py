"""Anomaly detection dashboard — Isolation Forest + rule + behavioral evidence."""

import pandas as pd
import streamlit as st

from modules.analysis_state import analysis_or_prompt
from modules.threat_detection import display_milestone_3_dashboard

result = analysis_or_prompt()
if result is not None:
    df = result.get('scored_events')
    if df is None or len(df) == 0:
        st.info("No scored events available.")
    else:
        # Operating mode / calibration / missing-channel status
        if 'operating_mode' in df.columns:
            modes = df['operating_mode'].value_counts().to_dict()
            st.markdown("**Operating modes:** " + ", ".join(
                f"{k}: {v}" for k, v in modes.items()))
        if 'missing_channels' in df.columns:
            missing_rate = float(
                df['missing_channels'].apply(len).gt(0).mean())
            st.markdown(f"**Missing-channel rate:** {missing_rate:.1%}")
        for label, col in [('Risk calibration', 'risk_calibration_status'),
                           ('Detection calibration',
                            'detection_calibration_status'),
                           ('Threshold version', 'threshold_version')]:
            if col in df.columns and df[col].notna().any():
                val = df[col].dropna().iloc[0]
                st.markdown(f"**{label}:** `{val}`")
        st.divider()
        display_milestone_3_dashboard(df)
