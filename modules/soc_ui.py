"""Shared CAID Security Operations Center UI helpers.

This module owns the visual language used across the Streamlit dashboard:
page headers, KPI cards, section headings, badges, and empty states. It does
not alter detection, correlation, incident, or report data.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import streamlit as st


CAID_COLORS = {
    "background": "#080B16",
    "secondary": "#0D1222",
    "sidebar": "#0A1020",
    "card": "#111A2D",
    "elevated": "#17223A",
    "primary": "#7957FF",
    "secondary_accent": "#27C8F5",
    "success": "#19D9A0",
    "warning": "#FFB547",
    "critical": "#FF4D70",
    "text": "#F4F6FF",
    "text_secondary": "#929CB5",
    "border": "#26324D",
}

SOC_CSS = """
<style>
:root {
  --caid-bg: #080B16;
  --caid-secondary: #0D1222;
  --caid-sidebar: #0A1020;
  --caid-card: #111A2D;
  --caid-elevated: #17223A;
  --caid-primary: #7957FF;
  --caid-cyan: #27C8F5;
  --caid-success: #19D9A0;
  --caid-warning: #FFB547;
  --caid-critical: #FF4D70;
  --caid-text: #F4F6FF;
  --caid-muted: #929CB5;
  --caid-border: rgba(130, 150, 200, 0.16);
}

html, body, [class*="css"] {
  background: var(--caid-bg);
  color: var(--caid-text);
}

[data-testid="stSidebar"] {
  background: var(--caid-sidebar) !important;
  border-right: 1px solid var(--caid-border) !important;
}

[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] button,
[data-testid="stSidebarNav"] [role="link"] {
  border-radius: 10px;
  transition: background-color 140ms ease, color 140ms ease, transform 140ms ease;
}

[data-testid="stSidebarNav"] a:hover,
[data-testid="stSidebarNav"] button:hover,
[data-testid="stSidebarNav"] [role="link"]:hover {
  background: rgba(121, 87, 255, 0.12);
}

[data-testid="stMetric"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"],
[data-testid="stTabs"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--caid-border);
  border-radius: 14px;
  background: linear-gradient(180deg, rgba(23, 34, 58, 0.42), rgba(17, 26, 45, 0.72));
}

[data-testid="stMetric"] {
  padding: 14px 16px;
  transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-1px);
  border-color: rgba(39, 200, 245, 0.32);
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
}

[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #7957FF 0%, #27C8F5 100%);
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 10px;
  color: #FFFFFF;
  font-weight: 650;
}

[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-tertiary"] {
  border-radius: 10px;
  border: 1px solid var(--caid-border);
  background: rgba(13, 18, 34, 0.86);
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stMultiSelect"] div[data-baseweb="select"],
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
  border-radius: 10px;
  border-color: var(--caid-border) !important;
  background: rgba(13, 18, 34, 0.88) !important;
  color: var(--caid-text) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stTabs"] button[role="tab"] {
  border-radius: 10px;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--caid-text);
  border-bottom-color: var(--caid-cyan);
}

.js-plotly-plot .plotly,
.js-plotly-plot .main-svg {
  background: transparent !important;
}

::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: var(--caid-bg);
}

::-webkit-scrollbar-thumb {
  background: #26324D;
  border-radius: 999px;
  border: 2px solid var(--caid-bg);
}
</style>
"""

SEVERITY_COLORS = {
    "critical": "red",
    "high": "orange",
    "medium": "yellow",
    "low": "green",
}

STATUS_COLORS = {
    "open": "red",
    "investigating": "blue",
    "contained": "orange",
    "closed": "green",
    "falsepositive": "gray",
    "resolved": "green",
    "in progress": "blue",
}


def configure_plotly_template() -> None:
    """Apply the CAID dark Plotly theme when Plotly is installed."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except Exception:
        return

    template = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": CAID_COLORS["text"], "family": "Inter, sans-serif"},
            colorway=[
                CAID_COLORS["primary"],
                CAID_COLORS["secondary_accent"],
                CAID_COLORS["success"],
                CAID_COLORS["warning"],
                CAID_COLORS["critical"],
                "#A78BFA",
                "#64748B",
            ],
            xaxis={
                "gridcolor": "rgba(130,150,200,0.10)",
                "zerolinecolor": "rgba(130,150,200,0.18)",
            },
            yaxis={
                "gridcolor": "rgba(130,150,200,0.10)",
                "zerolinecolor": "rgba(130,150,200,0.18)",
            },
            legend={"font": {"color": CAID_COLORS["text_secondary"]}},
            hoverlabel={"bgcolor": CAID_COLORS["elevated"],
                        "font": {"color": CAID_COLORS["text"]}},
        )
    )
    pio.templates["caid_dark"] = template
    pio.templates.default = "caid_dark"


def render_page_header(
    title: str,
    breadcrumb: str,
    subtitle: Optional[str] = None,
    badges: Optional[Iterable[str]] = None,
) -> None:
    """Render the consistent SOC page header and breadcrumb strip."""
    left, right = st.columns([5, 2], vertical_alignment="center")
    with left:
        st.markdown(f"### {title}")
        st.caption(f"CAID / {breadcrumb}")
        if subtitle:
            st.caption(subtitle)
    with right:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.badge("Synthetic data", icon=":material/science:", color="violet")
            st.badge("SOC workspace", icon=":material/shield:", color="blue")
            for badge in badges or []:
                st.badge(badge, color="primary")


def section_header(
    title: str,
    description: Optional[str] = None,
    icon: Optional[str] = None,
) -> None:
    """Render a consistent section heading with optional helper text."""
    kwargs = {"icon": icon} if icon else {}
    st.subheader(title, **kwargs)
    if description:
        st.caption(description)


def kpi_grid(items: List[Dict[str, Any]], columns: int = 5) -> None:
    """Render responsive KPI cards.

    Each item may contain ``label``, ``value``, ``icon``, ``delta``,
    ``delta_color``, ``chart_data``, ``chart_type``, and ``help``.
    """
    if not items:
        return
    with st.grid(min(columns, len(items)), gap="medium"):
        for item in items:
            st.metric(
                item.get("label", ""),
                item.get("value", "—"),
                delta=item.get("delta"),
                delta_color=item.get("delta_color", "off"),
                help=item.get("help"),
                icon=item.get("icon"),
                border=True,
                height="stretch",
                chart_data=item.get("chart_data"),
                chart_type=item.get("chart_type", "line"),
                delta_description=item.get("delta_description"),
            )


def severity_badge(value: Any, *, prefix: str = "") -> None:
    """Render a normalized severity/status badge."""
    label = str(value or "unknown").replace("_", " ").title()
    key = label.lower().replace(" ", "")
    color = SEVERITY_COLORS.get(key) or STATUS_COLORS.get(
        label.lower(), "blue")
    st.badge(f"{prefix}{label}", color=color, icon=":material/warning:")


def empty_state(message: str, icon: str = ":material/search_off:") -> None:
    """Render a consistent empty state inside a bordered panel."""
    with st.container(border=True):
        st.markdown(f":material/{icon.strip(':material/:')}" if False else "")
        st.markdown(f"**{message}**")
        st.caption("Adjust the filters or load the relevant analysis to continue.")


def status_badges(values: Iterable[Any]) -> None:
    """Render a compact row of badges for status-like values."""
    with st.container(horizontal=True, wrap=True):
        for value in values:
            severity_badge(value)
