"""
Cross-Session Threat Correlation dashboard for CAID Milestone 2.1.

Renders retrospective correlation findings linking sessions, accounts,
and shared-IP activity, with explicit evidence for every finding.
Shared IP is never presented as proof of coordination.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from modules.session_intelligence import analyze_session_intelligence
from modules.scenario_data import (
    load_scenario_frame,
    render_scenario_selector,
    scenario_label,
)
from core.cross_session_correlation import (
    CrossSessionCorrelator,
    CorrelationConfig,
    CORRELATION_TYPES,
)

_TYPE_LABELS = {
    'repeated_pattern_across_sessions': 'Repeated Pattern Across Sessions',
    'similar_account_behavior': 'Similar Account Behavior',
    'repeated_sensitive_operations': 'Repeated Sensitive Operations',
    'long_term_behavior_change': 'Long-Term Behavior Change',
    'shared_ip_activity': 'Shared IP Activity',
}


def analyze_correlation(raw_df: pd.DataFrame) -> Dict:
    """Run the v2.0 pipeline, then correlate across sessions/accounts/IPs."""
    scored = analyze_session_intelligence(raw_df)
    result = CrossSessionCorrelator(CorrelationConfig()).correlate(scored)
    result['scored_events'] = scored
    return result


def _evidence_json(evidence) -> str:
    import json
    return json.dumps(evidence, indent=2, default=str)


def render_correlation_page() -> None:
    st.subheader("Cross-Session Threat Correlation (Milestone 2.1)")
    st.caption(
        "Retrospective correlation over synthetic security logs — findings "
        "link sessions, accounts, and shared-IP activity with explicit "
        "evidence. Scores are correlation strengths, not probabilities. "
        "A shared IP alone is never treated as proof of coordination.")

    if 'corr_result' not in st.session_state:
        st.session_state.corr_result = None

    scenario_key = render_scenario_selector()
    result = st.session_state.corr_result
    if result is None or result.get('scenario_key') != scenario_key:
        if result is not None:
            st.info("Scenario changed — reload to analyze the new data.")
        if st.button("Load and Analyze"):
            try:
                raw = load_scenario_frame(scenario_key)
                res = analyze_correlation(raw)
                res['scenario_key'] = scenario_key
                res['scenario_label'] = scenario_label(scenario_key)
                st.session_state.corr_result = res
                st.success("Cross-session correlation complete.")
                st.rerun()
            except Exception as e:
                st.error(f"Correlation analysis failed: {e}")
                return
        return

    render_correlation_results(st.session_state.corr_result)


def render_correlation_results(result: Dict) -> None:
    """Render correlation findings from an analysis result dict.

    Shared by the standalone page and the multipage dashboard, which
    supplies the result via ``modules.analysis_state``.
    """
    corr = result['correlations']
    st.markdown(f"**Scenario:** {result.get('scenario_label', result.get('scenario_key', ''))}")
    if corr is None or len(corr) == 0:
        st.info("No cross-session correlations found.")
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correlation Findings", len(corr))
    c2.metric("Accounts Involved",
              len(set().union(*[set(e) for e in corr['entities']]))
              if 'entities' in corr else 0)
    c3.metric("Shared-IP Findings",
              int((corr['correlation_type'] == 'shared_ip_activity').sum()))
    c4.metric("Corroborated Shared-IP",
              int(((corr['correlation_type'] == 'shared_ip_activity') &
                   ~corr['shared_ip_only']).sum()))

    st.markdown("**Findings by type:** " + ", ".join(
        f"{_TYPE_LABELS.get(t, t)}: {int((corr['correlation_type'] == t).sum())}"
        for t in CORRELATION_TYPES))
    st.caption("Retrospective analysis only — correlations do not modify "
               "per-event risk scores.")

    # Filters
    f1, f2 = st.columns(2)
    types = f1.multiselect(
        "Correlation type",
        options=sorted(corr['correlation_type'].unique()),
        default=sorted(corr['correlation_type'].unique()))
    min_score = f2.slider("Minimum correlation score", 0.0, 1.0, 0.0, 0.05)
    view = corr[corr['correlation_type'].isin(types) &
                (corr['correlation_score'] >= min_score)]

    # Findings table
    st.subheader("Correlation Findings")
    show_cols = [c for c in ['correlation_type', 'entity_a', 'entity_b',
                             'entities', 'sessions', 'correlation_score',
                             'shared_ip_only', 'operating_note'] if c in view.columns]
    st.dataframe(view[show_cols].head(100), width="stretch")

    # Evidence expanders
    st.subheader("Evidence")
    for _, row in view.head(25).iterrows():
        label = (f"{_TYPE_LABELS.get(row['correlation_type'], row['correlation_type'])}"
                 f" — {row.get('entity_a', '?')}"
                 + (f" ↔ {row['entity_b']}" if row.get('entity_b') else '')
                 + f" (score {row['correlation_score']:.2f})")
        with st.expander(label):
            st.markdown(f"**Type:** `{row['correlation_type']}`  |  "
                        f"**ID:** `{row['correlation_id']}`")
            if row.get('sessions'):
                st.markdown(f"**Sessions:** {', '.join(map(str, row['sessions'][:12]))}"
                            + ('…' if len(row['sessions']) > 12 else ''))
            if row.get('operating_note'):
                st.markdown(f"**Note:** {row['operating_note']}")
            st.markdown("**Evidence:**")
            st.code(_evidence_json(row.get('evidence', [])), language='json')

    # Account profiles
    st.subheader("Account Behavioral Profiles")
    profiles = result.get('account_profiles')
    if profiles is not None and len(profiles):
        show = profiles[[c for c in profiles.columns
                         if not c.startswith('_')]].copy()
        st.dataframe(show.sort_values('suspicious_sessions', ascending=False)
                     .head(50), width="stretch")

    # Sensitive-operation tracks
    st.subheader("Sensitive Operations Over Time")
    tracks = result.get('sensitive_tracks')
    if tracks is not None and len(tracks):
        st.dataframe(tracks.sort_values(['user_id', 'day']).head(100),
                     width="stretch")
    else:
        st.info("No sensitive-operation activity detected.")

    from modules.export import export_csv
    st.download_button(
        "Download Correlation Findings",
        export_csv(corr),
        "cross_session_correlations.csv",
        "text/csv",
    )
