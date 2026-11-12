"""
CAID - Canvas AI Intrusion Detection System

Main application entry point (multipage dashboard).
"""

import os
import sys

import streamlit as st

# Add project directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.scenario_data import render_scenario_selector, scenario_label
from modules.analysis_state import get_analysis

st.set_page_config(
    page_title="CAID - Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<style>"
    ".block-container {padding-top: 2rem;}"
    "</style>",
    unsafe_allow_html=True,
)

# ---- Sidebar: scenario selection ------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ CAID")
    st.caption("Canvas AI Intrusion Detection System — defensive "
               "research prototype (synthetic data only)")
    render_scenario_selector()
    ana = get_analysis()
    if ana is not None:
        st.success(f"Analyzed: {scenario_label(ana.get('scenario_key'))}")
    else:
        st.info("No analysis loaded — open a page and click "
                "“Load and analyze”.")

# ---- Navigation -------------------------------------------------------------
page = st.navigation(
    {
        "Monitor": [
            st.Page("app_pages/overview.py", title="Overview",
                    icon=":material/dashboard:",
                    help="Platform KPIs, system status, and activity "
                         "charts for the analyzed scenario."),
            st.Page("app_pages/events.py", title="Events",
                    icon=":material/table:",
                    help="Searchable, filterable view of every analyzed "
                         "event with ML, behavioral, and risk scores."),
            st.Page("app_pages/anomaly_detection.py",
                    title="Anomaly Detection",
                    icon=":material/query_stats:",
                    help="Isolation Forest results, detection rules, "
                         "behavioral deviations, and per-event evidence."),
        ],
        "Investigate": [
            st.Page("app_pages/session_intelligence.py",
                    title="Session Intelligence",
                    icon=":material/monitoring:",
                    help="Session timelines, sequence patterns, and "
                         "retrospective session risk summaries."),
            st.Page("app_pages/correlation.py",
                    title="Correlation",
                    icon=":material/account_tree:",
                    help="Cross-session correlation findings with "
                         "explicit evidence for every link."),
            st.Page("app_pages/investigation.py", title="Investigation",
                    icon=":material/troubleshoot:",
                    help="Incident management: triage, notes, merge/split, "
                         "and PDF/CSV reports."),
            st.Page("app_pages/mitre.py", title="MITRE ATT&CK",
                    icon=":material/policy:",
                    help="ATT&CK technique coverage and CAID detection "
                         "mappings with confidence levels."),
        ],
        "AI": [
            st.Page("app_pages/ai_analyst.py", title="AI Analyst",
                    icon=":material/smart_toy:",
                    help="Local Ollama assistant: event explanations, "
                         "incident summaries, and safe threat hunting."),
        ],
        "Data & Admin": [
            st.Page("app_pages/log_analysis.py", title="Log Analysis",
                    icon=":material/upload_file:",
                    help="Upload and explore raw security log files."),
            st.Page("app_pages/settings.py", title="Settings",
                    icon=":material/settings:",
                    help="System status, Ollama model configuration, and "
                         "MITRE dataset management."),
        ],
    },
    position="sidebar",
)

page.run()
