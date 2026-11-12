"""
Session & sequence intelligence dashboard for CAID Milestone 2.0.

Renders session timelines, suspicious sequences, session risk summaries,
behavioral/session/sequence deviations, structured evidence, operating mode,
calibration status, and missing-channel indicators.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st
from utils.errors import safe_error_message

from core.log_processor import load_sample_dataset
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine
from core.risk_engine import RiskEngine
from core.calibration import (
    apply_to_engine as apply_risk_calibration,
    apply_detection_calibration,
    component_fingerprint,
    detector_fingerprint,
)
from core.behavioural_analyzer import ensure_fitted_analyzer
from core.session_analyzer import ensure_fitted_session_analyzer
from core.sequence_detection import ensure_fitted_sequence_analyzer
from models.anomaly_detector import ensure_model_exists


def analyze_session_intelligence(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run the additive v2.0 channel pipeline on a raw security-log frame."""
    detector = ensure_model_exists()

    out = detector.predict(raw_df.copy())
    risk = RiskEngine()
    apply_risk_calibration(risk, detector=detector)
    out = risk.process_events(out)
    out = DetectionRulesEngine().process_events(out)
    behavioral = ensure_fitted_analyzer()
    out = behavioral.analyze_user_behavior(out)

    normal = load_sample_dataset('normal')
    session = ensure_fitted_session_analyzer(train_df=normal)
    sequence = ensure_fitted_sequence_analyzer(train_df=normal)
    out = session.analyze(out)
    out = sequence.analyze(out)

    # AUD-002: surface artifact lifecycle statuses (loaded vs refit
    # fallback) so operators can distinguish persisted artifacts from
    # just-fit fallbacks. Statuses are ephemeral runtime metadata.
    artifact_statuses = {
        'detector': getattr(detector, 'artifact_status', None),
        'feature_baseline': getattr(
            getattr(detector, 'baseline_manager', None),
            'artifact_status', None),
        'behavioral_baseline': getattr(
            getattr(behavioral, 'baseline_manager', None),
            'artifact_status', None),
        'session_baseline': getattr(
            getattr(session, 'baseline_manager', None),
            'artifact_status', None),
        'sequence_model': getattr(sequence, 'artifact_status', None),
    }
    try:
        st.session_state.artifact_statuses = artifact_statuses
    except Exception:
        pass
    out.attrs['artifact_statuses'] = artifact_statuses

    engine = DetectionEngine(channel_profile='v2_0')
    cal_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'models', 'saved_models', 'calibration')
    cal_path = os.path.join(
        cal_dir, f'v2_0_thresholds_{detector_fingerprint(detector)}.joblib')
    fingerprints = {
        'detector': detector_fingerprint(detector),
        'feature_baseline': component_fingerprint(
            detector.baseline_manager, 'feature_baseline'),
        'behavioral_baseline': component_fingerprint(
            behavioral.baseline_manager, 'behavioral_baseline'),
        'session_baseline': component_fingerprint(
            session.baseline_manager, 'session_baseline'),
        'sequence_model': component_fingerprint(sequence, 'sequence_model'),
    }
    apply_detection_calibration(engine, detector=detector, path=cal_path,
                                fingerprints=fingerprints)
    out = engine.process_dataframe(out)
    out['risk_calibration_status'] = risk.calibration_status
    out['detection_calibration_status'] = engine.calibration_status
    out['threshold_version'] = engine.threshold_version or risk.threshold_version
    return out


def _top_anomalies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if 'combined_score' not in df.columns:
        return pd.DataFrame()
    return df.sort_values('combined_score', ascending=False).head(n)


def _session_timeline(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    sub = df[df['user_id'] == user_id].copy()
    if 'session_id' not in sub.columns:
        return pd.DataFrame()
    sub['_event_idx'] = range(len(sub))
    return sub[['session_id', '_event_idx', 'timestamp', 'event_type',
                'combined_score', 'sequence_deviation', 'session_deviation',
                'risk_level']]


def render_session_intelligence_page() -> None:
    st.subheader("Session & Sequence Intelligence (Milestone 2.0)")
    st.caption("Synthetic defensive research prototype — scores are calibrated "
               "anomaly scores, not probabilities.")

    if 'session_df' not in st.session_state:
        st.session_state.session_df = None

    if st.session_state.session_df is None:
        if st.button("Load and Analyze Sample Dataset"):
            try:
                raw = load_sample_dataset('all')
                st.session_state.session_df = analyze_session_intelligence(raw)
                st.success("Session/sequence analysis complete.")
                st.rerun()
            except Exception as e:
                st.error(safe_error_message(e, "Session intelligence analysis failed"))
                return
        return

    render_session_results(st.session_state.session_df)


def render_session_results(df: pd.DataFrame) -> None:
    """Render session/sequence intelligence from a scored DataFrame.

    Shared by the standalone page and the multipage dashboard.
    """
    if df is None or len(df) == 0:
        st.info("No analyzed data available.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(df))
    c2.metric("Users", df['user_id'].nunique() if 'user_id' in df else 0)
    c3.metric("Sessions", df['session_id'].nunique() if 'session_id' in df else 0)
    high = df['risk_level'].isin(['High', 'Critical']).sum() if 'risk_level' in df else 0
    c4.metric("High/Critical", int(high))

    # Operating mode / calibration / missing channels
    mode_counts = df['operating_mode'].value_counts().to_dict() if 'operating_mode' in df else {}
    st.markdown("**Operating modes:** " + ", ".join(f"{k}: {v}" for k, v in mode_counts.items()))
    missing_rate = float(df['missing_channels'].apply(len).gt(0).mean()) \
        if 'missing_channels' in df else 0.0
    st.markdown(f"**Missing-channel rate:** {missing_rate:.1%}")
    risk_status = df['risk_calibration_status'].mode().iloc[0] \
        if 'risk_calibration_status' in df and len(df['risk_calibration_status'].mode()) else 'unknown'
    det_status = df['detection_calibration_status'].mode().iloc[0] \
        if 'detection_calibration_status' in df and len(df['detection_calibration_status'].mode()) else 'unknown'
    threshold_version = df['threshold_version'].dropna().iloc[0] \
        if 'threshold_version' in df and df['threshold_version'].notna().any() else 'none'
    st.markdown(f"**Calibration:** risk={risk_status}; detection={det_status}; "
                f"threshold_version={threshold_version}")
    artifact_statuses = getattr(df, 'attrs', {}).get('artifact_statuses') or {}
    if artifact_statuses:
        st.markdown("**Artifact status:** " + ", ".join(
            f"{k}={v}" for k, v in artifact_statuses.items()))

    # Retrospective session summary (not causal per-event alerts)
    st.subheader("Retrospective Session Risk Summary")
    st.caption("Completed-session aggregates are retrospective context only; "
               "causal per-event evidence is shown in timelines and sequence panels.")
    if {'user_id', 'session_id', 'combined_score'}.issubset(df.columns):
        sess = df.groupby(['user_id', 'session_id'], dropna=False).agg(
            events=('combined_score', 'size'),
            max_score=('combined_score', 'max'),
            avg_score=('combined_score', 'mean'),
            risk_level=('risk_level', lambda x: x.mode().iloc[0] if len(x.mode()) else ''),
        ).reset_index().sort_values('max_score', ascending=False)
        st.dataframe(sess.head(25), width="stretch")
    else:
        st.info("Session columns unavailable.")

    # Suspicious sequences
    st.subheader("Suspicious Event Sequences")
    if 'sequence_triggered_patterns' in df.columns:
        seq = df[df['sequence_triggered_patterns'].apply(lambda x: bool(x))].copy()
        if len(seq):
            show = seq[['timestamp', 'user_id', 'session_id', 'event_type',
                        'sequence_deviation', 'sequence_triggered_patterns',
                        'combined_score', 'risk_level']].head(50)
            st.dataframe(show, width="stretch")
        else:
            st.info("No sequence patterns triggered.")
    else:
        st.info("Sequence channel unavailable.")

    # User session timeline
    st.subheader("User Session Timeline")
    users = sorted(df['user_id'].dropna().unique()) if 'user_id' in df else []
    if users:
        user = st.selectbox("Select user", users)
        timeline = _session_timeline(df, user)
        if len(timeline):
            st.dataframe(timeline.head(100), width="stretch")
            chart = timeline.set_index('_event_idx')[['combined_score',
                                                       'sequence_deviation',
                                                       'session_deviation']]
            st.line_chart(chart)

    # Top anomalies + evidence
    st.subheader("Top Anomalies & Evidence")
    top = _top_anomalies(df, 10)
    if len(top):
        cols = [c for c in ['timestamp', 'user_id', 'session_id', 'event_type',
                            'combined_score', 'risk_level', 'operating_mode',
                            'missing_channels', 'detection_explanation'] if c in top.columns]
        st.dataframe(top[cols], width="stretch")
        for _, row in top.head(5).iterrows():
            with st.expander(f"Evidence — {row.get('user_id','?')} @ {row.get('timestamp','?')}"):
                evidence = row.get('structured_evidence', [])
                if evidence:
                    st.json(evidence)
                else:
                    st.write(row.get('detection_explanation', 'No explanation'))

    from modules.export import export_csv
    st.download_button(
        "Download Session Intelligence Results",
        export_csv(df),
        "session_intelligence_results.csv",
        "text/csv",
    )
