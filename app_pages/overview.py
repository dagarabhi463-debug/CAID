"""Main overview dashboard — platform-wide KPIs and charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.analysis_state import analysis_or_prompt, get_manager

RISK_COLORS = {"Low": "#00cc66", "Medium": "#ffaa00",
               "High": "#ff4444", "Critical": "#990000"}

result = analysis_or_prompt()
if result is None:
    st.stop()

df = result.get('scored_events')
correlations = result.get('correlations')
if df is None or len(df) == 0:
    st.info("No analyzed events available.")
    st.stop()

st.subheader("Overview")
st.caption("Platform KPIs and charts for the analyzed scenario. "
           "All metrics are computed from real pipeline output — "
           "scores are calibrated anomaly scores, not probabilities.")

incidents = get_manager().list_incidents()
open_inc = [i for i in incidents if i.get('status') == 'Open']
crit_inc = [i for i in incidents if i.get('severity') == 'Critical']

risk = df['risk_level'].astype(str).str.title() \
    if 'risk_level' in df.columns else pd.Series(dtype=str)
high_crit_events = int(risk.isin(['High', 'Critical']).sum())
susp_accounts = int(df.loc[risk.isin(['High', 'Critical']), 'user_id']
                    .nunique()) if 'user_id' in df.columns else 0
crit_accounts = int(df.loc[risk == 'Critical', 'user_id']
                    .nunique()) if 'user_id' in df.columns else 0
anomalies = int(df['is_anomaly'].fillna(False).astype(bool).sum()) \
    if 'is_anomaly' in df.columns else 0
n_corr = len(correlations) if correlations is not None else 0

# ---- KPI row -------------------------------------------------------------
st.markdown("### Security posture")
cols = st.columns(5)
kpis = [
    ("Events analyzed", f"{len(df):,}"),
    ("Users monitored", f"{df['user_id'].nunique():,}"
     if 'user_id' in df.columns else '—'),
    ("Sessions", f"{df['session_id'].nunique():,}"
     if 'session_id' in df.columns else '—'),
    ("Anomalies (ML)", f"{anomalies:,}"),
    ("High/Critical events", f"{high_crit_events:,}"),
]
for col, (label, val) in zip(cols, kpis):
    col.metric(label, val)

cols = st.columns(5)
kpis2 = [
    ("Suspicious accounts", f"{susp_accounts:,}"),
    ("High-risk accounts", f"{crit_accounts:,}"),
    ("Correlation findings", f"{n_corr:,}"),
    ("Open incidents", f"{len(open_inc):,}"),
    ("Critical incidents", f"{len(crit_inc):,}"),
]
for col, (label, val) in zip(cols, kpis2):
    col.metric(label, val)

# ---- Data freshness ---------------------------------------------------------
if 'timestamp' in df.columns:
    _ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
    if len(_ts):
        st.caption(
            f"Data window: {_ts.min()} → {_ts.max()} "
            f"({result.get('scenario_label', result.get('scenario_key', ''))})")

# ---- System status strip --------------------------------------------------
with st.expander("System status", expanded=False):
    s1, s2, s3, s4 = st.columns(4)
    risk_status = df['risk_calibration_status'].dropna().iloc[0] \
        if 'risk_calibration_status' in df.columns \
        and df['risk_calibration_status'].notna().any() else 'unknown'
    det_status = df['detection_calibration_status'].dropna().iloc[0] \
        if 'detection_calibration_status' in df.columns \
        and df['detection_calibration_status'].notna().any() else 'unknown'
    s1.metric("Risk calibration", str(risk_status))
    s2.metric("Detection calibration", str(det_status))
    s3.metric("Scenario",
              result.get('scenario_label', result.get('scenario_key', '')))
    try:
        from core.ollama_client import OllamaClient
        oc = OllamaClient()
        models = oc.list_models()
        s4.metric("Ollama", f"{len(models)} model(s)" if models
                  else "unavailable")
    except Exception:
        s4.metric("Ollama", "unavailable")
    try:
        from core.mitre_attack import MitreAttackStore
        ms = MitreAttackStore()
        ok = ms.ensure_data(allow_download=False)
        s4.caption(f"MITRE ATT&CK: {'loaded' if ok else 'not loaded'} "
                   f"({ms.framework_version or 'n/a'})")
    except Exception:
        pass

# ---- Charts ----------------------------------------------------------------
st.markdown("### Activity")
if 'timestamp' in df.columns:
    tdf = df.copy()
    tdf['_ts'] = pd.to_datetime(tdf['timestamp'], errors='coerce')
    tdf = tdf.dropna(subset=['_ts'])
    if len(tdf):
        daily = tdf.set_index('_ts').resample('D').size() \
            .rename('events').reset_index()
        fig = px.line(daily, x='_ts', y='events', markers=True,
                      title='Events over time',
                      labels={'_ts': 'Date', 'events': 'Events'})
        fig.update_layout(height=320)
        st.plotly_chart(fig, width='stretch')

c1, c2 = st.columns(2)
with c1:
    if 'risk_level' in df.columns:
        counts = risk.value_counts()
        labels = [l for l in ['Low', 'Medium', 'High', 'Critical']
                  if l in counts.index]
        fig = go.Figure(go.Pie(
            labels=labels, values=[int(counts[l]) for l in labels],
            hole=0.4,
            marker_colors=[RISK_COLORS[l] for l in labels]))
        fig.update_layout(title='Risk level distribution', height=340)
        st.plotly_chart(fig, width='stretch')
with c2:
    score_col = 'anomaly_score_normalized' if \
        'anomaly_score_normalized' in df.columns else 'combined_score'
    if score_col in df.columns:
        h = df.copy()
        h[score_col] = pd.to_numeric(h[score_col], errors='coerce')
        h = h.dropna(subset=[score_col])
        if len(h):
            fig = px.histogram(h, x=score_col, nbins=40,
                               title=f'{score_col} distribution',
                               labels={score_col: score_col})
            fig.update_layout(height=340)
            st.plotly_chart(fig, width='stretch')

# ---- Channel contribution ---------------------------------------------------
_CHANNEL_COLS = [
    ('ML anomaly score', 'anomaly_score_normalized', 'anomaly_score',
     'ml_anomaly_score'),
    ('Rules score', 'rule_score', 'rule_detection_score'),
    ('Behavioral deviation', 'behavioral_deviation_score',
     'behavioral_deviation'),
    ('Session deviation', 'session_deviation', 'session_deviation_score'),
    ('Sequence deviation', 'sequence_deviation', 'sequence_deviation_score'),
]
_chan_means = {}
for label, *cands in _CHANNEL_COLS:
    col = next((c for c in cands if c in df.columns), None)
    if col is None:
        continue
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(vals):
        _chan_means[label] = float(vals.mean())
if _chan_means:
    st.markdown("### Detection channel contribution")
    cap = ("Mean raw channel output across analyzed events "
           "(channel scores have different scales; not probabilities).")
    if 'missing_channels' in df.columns:
        cap += (f" Missing-channel rate: "
                f"{float(df['missing_channels'].apply(len).gt(0).mean()):.1%}.")
    st.caption(cap)
    fig = px.bar(x=list(_chan_means.values()),
                 y=list(_chan_means.keys()), orientation='h',
                 title='Mean channel score', height=280,
                 labels={'x': 'mean score', 'y': 'channel'})
    st.plotly_chart(fig, width='stretch')

# ---- Top entities ----------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top suspicious users**")
    if 'user_id' in df.columns and 'combined_score' in df.columns:
        tu = df.groupby('user_id')['combined_score'].mean() \
            .sort_values(ascending=False).head(10)
        fig = px.bar(x=tu.values, y=tu.index, orientation='h',
                     title='Mean combined score by user', height=320)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')
with c2:
    st.markdown("**Top suspicious IPs**")
    if 'source_ip' in df.columns and 'combined_score' in df.columns:
        ti = df.groupby('source_ip')['combined_score'].mean() \
            .sort_values(ascending=False).head(10)
        fig = px.bar(x=ti.values, y=ti.index, orientation='h',
                     title='Mean combined score by IP', height=320)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')

# ---- Recent alerts & incidents ----------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Recent high-risk events**")
    if 'combined_score' in df.columns:
        recent = df.sort_values('combined_score', ascending=False) \
            .head(10)
        cols = [c for c in ['timestamp', 'user_id', 'event_type',
                            'combined_score', 'risk_level']
                if c in recent.columns]
        st.dataframe(recent[cols], hide_index=True, width='stretch',
                     alt='Recent high-risk events')
with c2:
    st.markdown("**Recent incidents**")
    if incidents:
        rows = [{
            'incident_id': i.get('incident_id'),
            'severity': i.get('severity'),
            'status': i.get('status'),
            'findings': len(i.get('findings') or []),
            'updated': str(i.get('updated_at', ''))[:19],
        } for i in incidents[:10]]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width='stretch', alt='Recent incidents')
    else:
        st.info("No incidents recorded yet.")

st.caption("All metrics are computed from the currently analyzed "
           "synthetic scenario. Scores are calibrated anomaly scores, "
           "not probabilities.")
