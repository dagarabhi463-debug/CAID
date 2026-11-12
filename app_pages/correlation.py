"""Cross-session correlation page (shared analysis state)."""

import streamlit as st

from modules.analysis_state import analysis_or_prompt
from modules.correlation_dashboard import render_correlation_results

result = analysis_or_prompt()
if result is not None:
    render_correlation_results(result)
