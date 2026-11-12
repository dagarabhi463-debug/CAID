"""Threat investigation & incident reports page (shared analysis state)."""

import streamlit as st

from modules.analysis_state import analysis_or_prompt, get_manager
from modules.investigation_dashboard import render_investigation_results

result = analysis_or_prompt()
if result is not None:
    render_investigation_results(get_manager(), result)
