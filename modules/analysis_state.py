"""
Shared analysis state for the CAID multipage dashboard.

Runs the v2.1/v2.2 pipeline once per selected scenario and caches the
result in ``st.session_state`` so every page (overview, events, MITRE,
AI analyst, ...) sees identical data. Retrospective only — nothing here
feeds back into detection scores.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st
from utils.errors import safe_error_message

from modules.scenario_data import (
    DEFAULT_SCENARIO_KEY,
    load_scenario_frame,
    scenario_label,
)
from modules.correlation_dashboard import analyze_correlation
from core.incident_manager import (
    IncidentConfig,
    IncidentManager,
    IncidentStore,
)

ANALYSIS_KEY = 'caid_analysis'
MANAGER_KEY = 'incident_manager'


def current_scenario_key() -> str:
    """Return the scenario selected in the sidebar (set by app.py)."""
    return st.session_state.get('caid_scenario', DEFAULT_SCENARIO_KEY)


def get_manager() -> IncidentManager:
    """Session-scoped IncidentManager backed by the JSON incident store."""
    if MANAGER_KEY not in st.session_state:
        cfg = IncidentConfig()
        st.session_state[MANAGER_KEY] = IncidentManager(
            config=cfg, store=IncidentStore(cfg.store_path))
    return st.session_state[MANAGER_KEY]


def get_analysis() -> Optional[Dict]:
    """Return the cached analysis for the current scenario, or None."""
    cached = st.session_state.get(ANALYSIS_KEY)
    if cached and cached.get('scenario_key') == current_scenario_key():
        return cached
    return None


def run_analysis() -> Dict:
    """Run the full pipeline for the current scenario and cache it."""
    key = current_scenario_key()
    raw = load_scenario_frame(key)
    result = analyze_correlation(raw)
    result['scenario_key'] = key
    result['scenario_label'] = scenario_label(key)
    get_manager().build_incidents(result.get('correlations'))
    st.session_state[ANALYSIS_KEY] = result
    return result


def analysis_or_prompt() -> Optional[Dict]:
    """Return cached analysis, or render a Load button and return None."""
    result = get_analysis()
    if result is not None:
        return result
    key = current_scenario_key()
    if st.button(f"Load and analyze — {scenario_label(key)}",
                 key='caid_load_analysis', type='primary'):
        try:
            with st.spinner("Running detection pipeline "
                            "(ML + rules + behavioral + session + "
                            "correlation + incidents)…"):
                result = run_analysis()
            st.success("Analysis complete.")
            st.rerun()
        except Exception as exc:
            st.error(safe_error_message(exc, "Analysis failed"))
            return None
    else:
        st.info("Select a scenario in the sidebar, then load and analyze "
                "to populate this page.")
    return None
