"""Session & sequence intelligence page (shared analysis state)."""

import streamlit as st

from modules.analysis_state import analysis_or_prompt
from modules.session_intelligence import render_session_results

result = analysis_or_prompt()
if result is not None:
    render_session_results(result.get('scored_events'))
