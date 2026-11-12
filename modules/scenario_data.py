"""
Scenario data loading for CAID Milestone 2.3.

Shared helper used by the correlation and investigation dashboard pages
to offer a scenario selector: the built-in sample dataset or one of the
constructed Milestone 2.3 scenarios.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
import streamlit as st

from core.log_processor import load_sample_dataset
from data.generate_scenarios import (
    DEFAULT_SCENARIO_KEY,
    SCENARIOS,
    generate_scenario,
)

SCENARIO_WIDGET_KEY = 'caid_scenario'


def scenario_options() -> List[Tuple[str, str]]:
    """(key, label) pairs for the scenario selector."""
    opts = [(DEFAULT_SCENARIO_KEY, 'Default sample dataset (all)')]
    opts += [(name, f"{spec['title']} — {spec['description']}")
             for name, spec in SCENARIOS.items()]
    return opts


def scenario_label(key: str) -> str:
    for k, label in scenario_options():
        if k == key:
            return label
    return key


def render_scenario_selector() -> str:
    """Render the selector widget and return the chosen key."""
    return st.selectbox(
        'Scenario',
        options=[k for k, _ in scenario_options()],
        format_func=scenario_label,
        key=SCENARIO_WIDGET_KEY,
        help='Synthetic scenarios demonstrate specific correlation and '
             'incident patterns end-to-end.',
    )


def load_scenario_frame(key: str) -> pd.DataFrame:
    """Load the raw-log DataFrame for a scenario key."""
    if key == DEFAULT_SCENARIO_KEY:
        return load_sample_dataset('all')
    return generate_scenario(key)
