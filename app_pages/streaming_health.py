"""Streaming health dashboard — observational monitoring only."""

import glob
import os

import streamlit as st

from core.streaming.health import (
    checkpoint_meta, dlq_summary, health_snapshot, latest_benchmark_report)

st.title("Streaming health")
st.caption("Observational pipeline telemetry for the streaming research prototype. "
           "These metrics never feed back into detection scores.")

checkpoint_path = st.text_input(
    "Checkpoint path", value="data/stream_checkpoint/stream_checkpoint.json")
dlq_path = st.text_input(
    "DLQ path", value="", placeholder="Leave blank to use the latest benchmark DLQ")

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

with st.expander("Registry counters"):
    counters = (snap.get("monitor") or {}).get("counters") or {}
    if counters:
        st.dataframe(
            [{"metric": k, "value": v} for k, v in sorted(counters.items())],
            hide_index=True, width="stretch")
    else:
        st.caption("No in-process metrics registry is attached to this page.")
