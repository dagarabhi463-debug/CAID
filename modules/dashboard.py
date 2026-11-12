"""
CAID Security Dashboard
=======================

Streamlit dashboard components for visualising defensive security-log analysis.

This module expects an analysed pandas DataFrame. Typical columns include:
timestamp, user_id, role, event_type, risk_level, risk_score, is_anomaly,
anomaly_score, api_requests, data_mb, and risk_factors.

Evaluation metrics are optional and must be supplied in st.session_state under
"evaluation_metrics". Metrics must be calculated from a separate, labelled
synthetic evaluation dataset; they are not real-world accuracy estimates.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {
    "Low": "#00cc66",
    "Medium": "#ffaa00",
    "High": "#ff4444",
    "Critical": "#990000",
}


def _safe_number(value: Any, default: float = 0.0) -> float:
    """Convert a value to a finite float, returning a safe default otherwise."""
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


def _metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    """Display a Streamlit metric without injecting custom HTML."""
    st.metric(label=label, value=value, help=help_text)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.info("This visualisation needs these columns: " + ", ".join(missing))
        return False
    return True


def display_header() -> None:
    """Display the main CAID dashboard heading."""
    st.title("🛡️ CAID — Security Dashboard")
    st.caption("Canvas AI Intrusion Detection System | Defensive research prototype")


def display_metrics(df: pd.DataFrame) -> None:
    """Display event totals and risk/anomaly counts for the supplied DataFrame."""
    if df is None:
        st.warning("No analysis data is available.")
        return

    total_events = len(df)
    if total_events == 0:
        st.info("No events to summarise.")
        return

    risk_counts = (
        df["risk_level"].astype(str).str.strip().str.title().value_counts()
        if "risk_level" in df.columns
        else pd.Series(dtype="int64")
    )
    high_risk = int(risk_counts.get("High", 0) + risk_counts.get("Critical", 0))
    low_risk = int(risk_counts.get("Low", 0))
    anomaly_count = (
        int(df["is_anomaly"].fillna(False).astype(bool).sum())
        if "is_anomaly" in df.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Total Events", f"{total_events:,}")
    with col2:
        _metric_card("High/Critical Risk", f"{high_risk:,}")
    with col3:
        _metric_card("Low-Risk Events", f"{low_risk:,}")
    with col4:
        _metric_card("ML-Flagged Anomalies", f"{anomaly_count:,}")


def display_risk_distribution(df: pd.DataFrame) -> None:
    """Display a risk-level distribution chart."""
    st.subheader("Risk Distribution")
    if not _require_columns(df, ["risk_level"]) or df.empty:
        return

    levels = df["risk_level"].astype(str).str.strip().str.title()
    counts = levels.value_counts()
    labels = [level for level in RISK_ORDER if level in counts.index]

    if not labels:
        st.info("No recognised risk levels were found.")
        return

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=[int(counts[level]) for level in labels],
                hole=0.35,
                marker={"colors": [RISK_COLORS[level] for level in labels]},
            )
        ]
    )
    fig.update_layout(title="Risk Level Distribution", height=400)
    st.plotly_chart(fig, width='stretch')


def display_timeline(df: pd.DataFrame) -> None:
    """Display daily event volume using valid timestamps only."""
    st.subheader("Security Events Timeline")
    if not _require_columns(df, ["timestamp"]) or df.empty:
        return

    timeline = df.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce", utc=True)
    timeline = timeline.dropna(subset=["timestamp"])

    if timeline.empty:
        st.info("No valid timestamps are available for the timeline.")
        return

    daily_counts = (
        timeline.set_index("timestamp")
        .resample("D")
        .size()
        .rename("events")
        .reset_index()
    )
    fig = px.line(
        daily_counts,
        x="timestamp",
        y="events",
        markers=True,
        title="Daily Event Count",
        labels={"timestamp": "Date", "events": "Events"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')


def display_anomaly_chart(df: pd.DataFrame) -> None:
    """Display anomaly-score distributions, grouped by risk category.

    Prefers anomaly_score_normalized (0-1, higher = more anomalous, fixed
    calibration). Falls back to anomaly_score only when the normalized
    column is absent, and labels the axis accordingly.
    """
    st.subheader("Anomaly Score Distribution")
    score_col = (
        "anomaly_score_normalized"
        if "anomaly_score_normalized" in df.columns
        else "anomaly_score"
    )
    if not _require_columns(df, [score_col]) or df.empty:
        return

    chart_df = df.copy()
    chart_df[score_col] = pd.to_numeric(chart_df[score_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[score_col])

    if chart_df.empty:
        st.info("No valid anomaly scores are available.")
        return

    if "risk_level" not in chart_df.columns:
        chart_df["risk_level"] = "Unclassified"
    else:
        chart_df["risk_level"] = chart_df["risk_level"].astype(str)

    score_label = (
        "Anomaly Score (normalized, higher = more anomalous)"
        if score_col == "anomaly_score_normalized"
        else "Anomaly Score (raw decision_function, higher = more normal)"
    )
    fig = px.histogram(
        chart_df,
        x=score_col,
        color="risk_level",
        barmode="overlay",
        opacity=0.7,
        title="Anomaly Scores by Risk Level",
        labels={score_col: score_label, "count": "Number of Events"},
        color_discrete_map=RISK_COLORS,
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width='stretch')


def display_top_anomalies(df: pd.DataFrame, n: int = 10) -> None:
    """Show the highest-risk events, adapting to available columns."""
    st.subheader(f"Top {max(1, int(n))} Suspicious Events")
    if df is None or df.empty:
        st.info("No events are available.")
        return
    if "risk_score" not in df.columns:
        st.info("The data does not contain a risk_score column.")
        return

    display_df = df.copy()
    display_df["risk_score"] = pd.to_numeric(display_df["risk_score"], errors="coerce")
    display_df = display_df.sort_values("risk_score", ascending=False, na_position="last")
    preferred = [
        "timestamp", "user_id", "role", "event_type", "risk_level",
        "risk_score", "is_anomaly", "anomaly_score", "api_requests", "data_mb",
    ]
    columns = [column for column in preferred if column in display_df.columns]
    display_df = display_df[columns].head(max(1, int(n)))
    st.dataframe(display_df, width='stretch', hide_index=True)


def _normalise_risk_factors(value: Any) -> list[str]:
    """Accept common risk-factor representations without crashing the dashboard."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return []

    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("factor") or item.get("name")
            if name:
                names.append(str(name))
        elif item is not None:
            text = str(item).strip()
            if text:
                names.append(text)
    return names


def display_risk_factors(df: pd.DataFrame) -> None:
    """Display counts of explainable risk factors and ML anomaly flags."""
    st.subheader("Risk Factor Analysis")
    if df is None or df.empty:
        st.info("No events are available.")
        return
    if "risk_factors" not in df.columns and "is_anomaly" not in df.columns:
        st.info("No risk-factor or ML-anomaly information is available.")
        return

    counts: dict[str, int] = {}
    if "risk_factors" in df.columns:
        for value in df["risk_factors"]:
            for factor_name in _normalise_risk_factors(value):
                counts[factor_name] = counts.get(factor_name, 0) + 1

    if "is_anomaly" in df.columns:
        anomaly_count = int(df["is_anomaly"].fillna(False).astype(bool).sum())
        if anomaly_count:
            counts["ML-Detected Anomalies"] = anomaly_count

    if not counts:
        st.info("No risk factors or anomalies were recorded.")
        return

    factors_df = pd.DataFrame(
        [{"Factor": name, "Count": count} for name, count in counts.items()]
    ).sort_values("Count", ascending=True)

    fig = px.bar(
        factors_df,
        x="Count",
        y="Factor",
        orientation="h",
        title="Observed Risk Indicators",
        height=max(350, 45 * len(factors_df)),
    )
    st.plotly_chart(fig, width='stretch')


def display_model_info(df: pd.DataFrame) -> None:
    """Display model metadata and counts for the current analysed dataset."""
    st.subheader("Model Status")
    if df is None or "anomaly_score" not in df.columns:
        st.warning("No ML model output is present. Run analysis first.")
        return

    total_records = len(df)
    anomaly_count = (
        int(df["is_anomaly"].fillna(False).astype(bool).sum())
        if "is_anomaly" in df.columns
        else 0
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        _metric_card("Model Type", "Isolation Forest", "Unsupervised anomaly detection")
    with col2:
        _metric_card("Records Analysed", f"{total_records:,}")
    with col3:
        _metric_card("ML-Flagged Anomalies", f"{anomaly_count:,}")


def display_ml_evaluation_section(df: pd.DataFrame | None = None) -> None:
    """
    Display previously calculated synthetic evaluation metrics.

    Expected session-state shape:
      {
        "dataset_size": int,
        "class_distribution": {
          "normal": int, "suspicious": int, "high_risk": int
        },
        "binary": {
          "precision": float, "recall": float, "f1_score": float,
          "accuracy": float,
          "confusion_matrix": {"tn": int, "fp": int, "fn": int, "tp": int}
        },
        "high-risk": { ...same metric fields... }
      }
    """
    st.subheader("Machine Learning Evaluation — Synthetic Data")
    st.warning(
        "These metrics describe performance on the project's synthetic evaluation "
        "dataset only. They are not evidence of real-world Canvas detection accuracy."
    )

    metrics = st.session_state.get("evaluation_metrics")
    if not isinstance(metrics, dict) or not metrics:
        st.info("No evaluation metrics are available. Run the evaluation workflow first.")
        return

    distribution = metrics.get("class_distribution", {})
    col1, col2 = st.columns(2)
    with col1:
        _metric_card("Evaluation Dataset Size", metrics.get("dataset_size", "N/A"))
    with col2:
        normal = distribution.get("normal", "N/A")
        suspicious = distribution.get("suspicious", "N/A")
        high_risk = distribution.get("high_risk", "N/A")
        _metric_card(
            "Actual Class Distribution",
            f"{normal} / {suspicious} / {high_risk}",
            "Normal / Suspicious / High-Risk",
        )

    _display_one_evaluation_mode(
        "Binary Detection Mode",
        "Normal = Negative; Suspicious + High-Risk = Positive",
        metrics.get("binary"),
        "Blues",
    )
    _display_one_evaluation_mode(
        "High-Risk Detection Mode",
        "Normal + Suspicious = Negative; High-Risk = Positive",
        metrics.get("high-risk"),
        "Reds",
    )


def _display_one_evaluation_mode(
    title: str,
    definition: str,
    mode_metrics: Any,
    colorscale: str,
) -> None:
    """Render one evaluation mode, validating its expected fields."""
    st.markdown(f"#### {title}")
    st.caption(definition)

    if not isinstance(mode_metrics, dict):
        st.info(f"{title} metrics are unavailable.")
        return

    metric_names = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("Accuracy", "accuracy"),
    ]
    cols = st.columns(4)
    for column, (label, key) in zip(cols, metric_names):
        value = mode_metrics.get(key)
        with column:
            _metric_card(label, f"{_safe_number(value):.3f}" if value is not None else "N/A")

    cm = mode_metrics.get("confusion_matrix")
    required = {"tn", "fp", "fn", "tp"}
    if not isinstance(cm, dict) or not required.issubset(cm):
        st.info("Confusion matrix values are unavailable.")
        return

    try:
        matrix = [[int(cm["tn"]), int(cm["fp"])], [int(cm["fn"]), int(cm["tp"])]]
    except (TypeError, ValueError):
        st.info("Confusion matrix values are invalid.")
        return

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            colorscale=colorscale,
            text=matrix,
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(title=f"{title} — Confusion Matrix", height=350)
    st.plotly_chart(fig, width='stretch')


def render_dashboard(df: pd.DataFrame | None) -> None:
    """
    Render the complete dashboard page.

    Call this from app.py after the application's existing ingestion and
    analysis pipeline has produced the DataFrame. This function does not train
    or retrain a model and does not modify the supplied DataFrame.
    """
    display_header()

    if df is None:
        st.info("Upload or generate security-log data to view the dashboard.")
        return
    if not isinstance(df, pd.DataFrame):
        st.error("Dashboard input must be a pandas DataFrame.")
        return
    if df.empty:
        st.info("The current dataset contains no records.")
        return

    display_metrics(df)

    left, right = st.columns(2)
    with left:
        display_risk_distribution(df)
    with right:
        display_anomaly_chart(df)

    display_timeline(df)
    display_top_anomalies(df)
    left, right = st.columns(2)
    with left:
        display_risk_factors(df)
    with right:
        display_model_info(df)

    with st.expander("Machine Learning Evaluation (Synthetic Dataset)", expanded=False):
        display_ml_evaluation_section(df)
