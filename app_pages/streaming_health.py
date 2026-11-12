"""Streaming health dashboard — observational monitoring only."""

import glob
import os

import pandas as pd
import streamlit as st

from core.streaming.health import (
    checkpoint_meta, dlq_summary, health_snapshot, latest_benchmark_report,
    latest_file, scored_events_frame)

st.title("Streaming health")
st.caption("Observational pipeline telemetry for the streaming research prototype. "
           "These metrics never feed back into detection scores.")


def _safe_data_path(raw: str, label: str) -> str | None:
    """Restrict user-supplied paths to the project ``data/`` tree.

    Returns the resolved absolute path if it lives under the project
    directory, otherwise ``None`` (with a warning shown). This prevents
    the observational read-only inputs from being pointed at arbitrary
    files outside the project.
    """
    if not raw:
        return None
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 os.pardir))
    candidate = os.path.abspath(raw)
    try:
        common = os.path.commonpath([candidate, project_root])
    except ValueError:
        common = ''
    if common != project_root:
        st.warning(f"{label} path is outside the project directory and "
                   "will not be read for safety.")
        return None
    return candidate


checkpoint_path = st.text_input(
    "Checkpoint path", value="data/stream_checkpoint/stream_checkpoint.json")
dlq_path = st.text_input(
    "DLQ path", value="", placeholder="Leave blank to use the latest benchmark DLQ")

checkpoint_path = _safe_data_path(checkpoint_path, "Checkpoint")
dlq_path = _safe_data_path(dlq_path, "DLQ")

if not dlq_path:
    dlq_files = glob.glob(os.path.join("data", "stream_dlq", "bench_*.jsonl"))
    dlq_path = max(dlq_files, key=os.path.getmtime) if dlq_files else None

benchmark = latest_benchmark_report("reports")
snap = health_snapshot(
    benchmark=benchmark,
    checkpoint=checkpoint_meta(checkpoint_path),
    dlq=dlq_summary(dlq_path),
)
bench = snap["benchmark"]
latency = (benchmark or {}).get("latency") or {}

with st.container(border=True):
    cols = st.columns(4, vertical_alignment="bottom")
    cols[0].metric("Pipeline status", snap["status"].upper())
    cols[1].metric("Effective throughput", f"{bench.get('effective_eps') or 0:.1f} eps")
    cols[2].metric("Events lost", bench.get("events_lost") or 0)
    cols[3].metric("DLQ depth", snap["dlq"].get("depth") or 0)

with st.container(horizontal=True):
    st.metric("Processed", bench.get("events_processed") or 0, border=True)
    st.metric("Dead-lettered", bench.get("events_dead_lettered") or 0, border=True)
    st.metric("Rejected", bench.get("events_rejected") or 0, border=True)
    st.metric("Duplicates", bench.get("events_duplicate") or 0, border=True)
    lat_bits = []
    for p in ("p50", "p95", "p99"):
        v = latency.get(p)
        if v is not None:
            lat_bits.append(f"{p} {float(v):.3f}s")
    st.metric("Latency", " · ".join(lat_bits) if lat_bits else "n/a",
              border=True)

with st.expander("Benchmark details"):
    if benchmark:
        st.json({k: v for k, v in benchmark.items() if k != "monitor"})
    else:
        st.info("No stream_benchmark report found under `reports/`.")

with st.expander("Queue, checkpoint, and DLQ"):
    st.markdown(f"**Queue:** `{snap['queue'].get('impl') or 'unknown'}`  ")
    if snap["queue"].get("depth") is None and \
            snap["queue"].get("pending") is None:
        st.caption("Queue depth / pending gauges are unavailable in "
                   "observational mode — they require an in-process "
                   "MetricsRegistry attached during a live run. Live "
                   "producer/consumer and Redis connection status are "
                   "not probed by this page; telemetry comes from the "
                   "latest benchmark report on disk.")
    st.markdown(f"**Benchmark report:** `{bench.get('path') or 'none'}`")
    st.markdown(f"**Checkpoint present:** `{snap['checkpoint'].get('present')}`")
    if snap["checkpoint"].get("meta"):
        st.json(snap["checkpoint"]["meta"])
    st.markdown("**DLQ reasons**")
    reasons = snap["dlq"].get("reasons") or {}
    if reasons:
        st.dataframe(
            [{"reason": k, "count": v} for k, v in sorted(reasons.items())],
            hide_index=True, width="stretch")
    else:
        st.caption("No DLQ entries recorded.")

with st.expander("Streaming detection results"):
    results_dir = st.text_input(
        "Results directory", value="data/stream_results",
        key="sh_results_dir")
    results_dir = _safe_data_path(results_dir, "Results")
    res_path = (latest_file(results_dir or "", "*.jsonl")
                if results_dir else None)
    df_res, n_res = scored_events_frame(res_path, max_rows=500)
    st.markdown(f"**Sink file:** `{res_path or 'none'}`  ")
    if df_res.empty:
        st.info("No committed streaming results found. Run "
                "`scripts/stream_benchmark.py` or the streaming worker "
                "to produce scored output here.")
    else:
        n_late = int(df_res.get('is_late_event',
                                pd.Series([False] * len(df_res)))
                     .fillna(False).astype(bool).sum()) \
            if 'is_late_event' in df_res.columns else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored events (file)", n_res)
        c2.metric("Late events rescored", n_late)
        if 'combined_score' in df_res.columns:
            scores = pd.to_numeric(df_res['combined_score'],
                                   errors='coerce')
            c3.metric("Mean combined score",
                      f"{scores.mean():.3f}" if scores.notna().any()
                      else "n/a")
        show_cols = [c for c in (
            'timestamp', 'user_id', 'session_id', 'event_type',
            'combined_score', 'risk_level', 'is_anomaly',
            'is_late_event', 'detection_latency_ms')
            if c in df_res.columns]
        st.dataframe(df_res[show_cols].tail(100), hide_index=True,
                     width="stretch")
        st.caption("Rows are the durable committed output of the "
                   "streaming worker (result sink). 'is_late_event' "
                   "marks out-of-band rescored late arrivals — additive "
                   "evidence; causal positions are never rewritten.")

with st.expander("Registry counters"):
    counters = (snap.get("monitor") or {}).get("counters") or {}
    if counters:
        st.dataframe(
            [{"metric": k, "value": v} for k, v in sorted(counters.items())],
            hide_index=True, width="stretch")
    else:
        st.caption("No in-process metrics registry is attached to this page.")
