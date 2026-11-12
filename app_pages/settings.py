"""Settings & system status page."""

import os
import sys

import streamlit as st
from utils.errors import safe_error_message

from core.ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    OllamaClient,
)

st.subheader("Settings & System Status")

# ---- Environment ----------------------------------------------------------
st.markdown("### Environment")
e1, e2 = st.columns(2)
e1.metric("Python", sys.version.split()[0])
try:
    import streamlit
    e2.metric("Streamlit", streamlit.__version__)
except Exception:
    e2.metric("Streamlit", "unknown")

# ---- ML detector ----------------------------------------------------------
st.markdown("### ML detector")
m1, m2, m3 = st.columns(3)
try:
    from models.anomaly_detector import get_model_path
    mpath = get_model_path()
    m1.metric("Model artifact",
              "present" if os.path.exists(mpath) else "missing")
    m1.caption(mpath)
except Exception as exc:
    m1.metric("Model artifact", f"error: {exc}")
try:
    import joblib
    m2.metric("joblib", "ok")
except Exception:
    m2.metric("joblib", "missing")
m3.metric("Saved model dir",
          os.path.join('models', 'saved_models'))

# Calibration status from last analysis, if any
ana = st.session_state.get('caid_analysis')
if ana and ana.get('scored_events') is not None:
    df = ana['scored_events']
    for label, col in [('Risk calibration', 'risk_calibration_status'),
                       ('Detection calibration',
                        'detection_calibration_status'),
                       ('Threshold version', 'threshold_version')]:
        if col in df.columns and df[col].notna().any():
            st.caption(f"{label}: `{df[col].dropna().iloc[0]}`")
else:
    st.caption("Run an analysis to view calibration status.")

# ---- Ollama ---------------------------------------------------------------

# ---- Ollama ---------------------------------------------------------------
st.markdown("### Ollama (remote AI)")

def get_config(key, default=None):
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(key)
        if value is not None:
            return value
    except Exception:
        pass

    return os.environ.get(key, default)


base_url = st.text_input(
    "Ollama base URL",
    value=get_config("OLLAMA_BASE_URL", DEFAULT_BASE_URL),
    key="set_ollama_url"
)

timeout = st.number_input(
    "Timeout (s)",
    min_value=5.0,
    max_value=300.0,
    value=float(get_config("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT)),
    key="set_ollama_timeout"
)

client = OllamaClient(
    base_url=base_url.strip().rstrip("/"),
    timeout=timeout
)

if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = get_config(
        "OLLAMA_MODEL", DEFAULT_MODEL
    )

c1, c2 = st.columns(2)

with c1:
    if st.button(
        "Test connection & refresh models",
        key="set_ollama_test"
    ):
        with st.spinner("Connecting to Ollama..."):
            health = client.health()

        st.session_state.ollama_health = health
        st.session_state.ollama_models = health.get("models", [])
        st.session_state.ollama_conn = health.get("reachable", False)

        if health.get("reachable"):
            st.success("Ollama connection successful.")
        else:
            st.error(
                "Ollama unavailable: "
                + str(health.get("error", "Connection failed"))
            )

with c2:
    health = st.session_state.get("ollama_health")

    if st.session_state.get("ollama_conn") and health:
        st.success(
            f"Connected — {len(health['models'])} model(s), "
            f"{health.get('latency_ms', 'N/A')} ms"
        )
    elif st.session_state.get("ollama_conn") is False:
        st.warning("Ollama connection failed.")

if st.session_state.ollama_models:
    current = st.session_state.ollama_model

    if current not in st.session_state.ollama_models:
        current = st.session_state.ollama_models[0]

    selected_model = st.selectbox(
        "Active model",
        options=st.session_state.ollama_models,
        index=st.session_state.ollama_models.index(current),
        key="set_ollama_model_pick"
    )

    st.session_state.ollama_model = selected_model

else:
    st.caption("No models available. Check your Ollama connection.")

st.caption(
    "AI features degrade gracefully when Ollama is offline; "
    "detection remains unaffected."
)


# ---- MITRE ----------------------------------------------------------------
st.markdown("### MITRE ATT&CK")
mitre1, mitre2 = st.columns(2)
try:
    from core.mitre_attack import MitreAttackStore
    ms = MitreAttackStore()
    loaded = ms.ensure_data(allow_download=False)
    if loaded:
        mitre1.metric("Status", "loaded")
        mitre1.caption(f"{ms.framework_version} · retrieved "
                       f"{str(ms.retrieved_at)[:10]}")
        mitre1.caption(ms.bundle_path)
    else:
        mitre1.metric("Status", "not loaded")
        mitre1.caption(ms.load_error or 'cache missing')
    if mitre2.button("Reload MITRE dataset", key='set_mitre_reload'):
        ms2 = MitreAttackStore()
        ok = ms2.ensure_data(allow_download=True)
        if ok:
            st.success("MITRE dataset reloaded.")
        else:
            st.error(f"Reload failed: {ms2.load_error}")
except Exception as exc:
    mitre1.metric("Status", f"error: {exc}")

# ---- Incident store --------------------------------------------------------
st.markdown("### Incident storage")
try:
    from modules.analysis_state import get_manager
    mgr = get_manager()
    n = len(mgr.store.incidents)
    st.metric("Stored incidents", n)
    st.caption(mgr.store.path)
    if mgr.store.load_error:
        st.warning(f"Store load warning: {mgr.store.load_error}")
except Exception as exc:
    st.warning(safe_error_message(exc, "Incident store unavailable"))

# ---- Configuration ---------------------------------------------------------
with st.expander("Configuration"):
    st.code(
        "OLLAMA_BASE_URL=%s\nOLLAMA_MODEL=%s\nOLLAMA_TIMEOUT=%s\n"
        "OLLAMA_MAX_RETRIES=%s\nOLLAMA_RETRY_BACKOFF=%s\n"
        "MITRE cache: data/mitre_attack/\nIncident store: "
        "data/incidents/incidents.json" % (
            base_url, st.session_state.ollama_model, timeout,
            client.max_retries, client.retry_backoff))
    st.caption("Secrets are never displayed. Set environment variables "
               "in a local .env file if needed.")
