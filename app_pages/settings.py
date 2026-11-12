"""Settings & system status page."""

import os
import sys

import streamlit as st

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
st.markdown("### Ollama (local AI)")
base_url = st.text_input("Ollama base URL",
                         value=os.environ.get('OLLAMA_BASE_URL',
                                              DEFAULT_BASE_URL),
                         key='set_ollama_url')
timeout = st.number_input("Timeout (s)", 5.0, 300.0,
                          float(os.environ.get('OLLAMA_TIMEOUT',
                                               DEFAULT_TIMEOUT)),
                          key='set_ollama_timeout')

client = OllamaClient(base_url=base_url, timeout=timeout)

if 'ollama_models' not in st.session_state:
    st.session_state.ollama_models = []
if 'ollama_model' not in st.session_state:
    st.session_state.ollama_model = os.environ.get('OLLAMA_MODEL',
                                                 DEFAULT_MODEL)

c1, c2 = st.columns(2)
with c1:
    if st.button("Test connection & refresh models", key='set_ollama_test'):
        try:
            st.session_state.ollama_models = client.list_models()
            st.session_state.ollama_conn = True
        except Exception as exc:
            st.session_state.ollama_models = []
            st.session_state.ollama_conn = False
            st.error(f"Ollama unavailable: {exc}")
with c2:
    if st.session_state.get('ollama_conn'):
        st.success(f"Connected — {len(st.session_state.ollama_models)} "
                   "model(s) installed")
    elif st.session_state.get('ollama_conn') is False:
        st.warning("Connection failed")

if st.session_state.ollama_models:
    current = st.session_state.ollama_model
    if current not in st.session_state.ollama_models:
        current = st.session_state.ollama_models[0]
    st.session_state.ollama_model = st.selectbox(
        "Active model", options=st.session_state.ollama_models,
        index=st.session_state.ollama_models.index(current),
        key='set_ollama_model_pick')
else:
    st.caption("No models listed — install one with "
               "`ollama pull <model>` and refresh.")
    st.session_state.ollama_model = st.text_input(
        "Model name", value=st.session_state.ollama_model,
        key='set_ollama_model_text')

st.caption("AI features degrade gracefully when Ollama is offline; "
           "detection is unaffected.")

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
    st.warning(f"Incident store unavailable: {exc}")

# ---- Configuration ---------------------------------------------------------
with st.expander("Configuration"):
    st.code(
        "OLLAMA_BASE_URL=%s\nOLLAMA_MODEL=%s\nOLLAMA_TIMEOUT=%s\n"
        "MITRE cache: data/mitre_attack/\nIncident store: "
        "data/incidents/incidents.json" % (
            base_url, st.session_state.ollama_model, timeout))
    st.caption("Secrets are never displayed. Set environment variables "
               "in a local .env file if needed.")
