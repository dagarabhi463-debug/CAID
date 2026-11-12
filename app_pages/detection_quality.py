"""Detection quality dashboard — offline evaluation artifacts + drift.

Surfaces the JSON reports produced by scripts/evaluate_detection_quality.py,
scripts/evaluate_fusion.py, scripts/benchmark_models.py, and
scripts/calibrate_v2_detection.py, plus an on-demand observational drift
measurement. All figures are internal-reproducibility results on synthetic
data — not real-world accuracy estimates. Nothing here modifies scores,
thresholds, or models.
"""

import json
import os

import pandas as pd
import streamlit as st
from utils.errors import safe_error_message

REPORTS = 'reports'


def _load_json(name: str):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return None, path
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh), path
    except Exception as exc:
        return {'_error': str(exc)}, path


def _metric_row(metrics: dict, keys=('precision', 'recall', 'f1', 'fpr')):
    cols = st.columns(len(keys))
    for col, k in zip(cols, keys):
        v = metrics.get(k)
        col.metric(k.upper(), f"{v:.4f}" if isinstance(v, (int, float)) else 'n/a')


st.subheader("Detection Quality")
st.caption("Offline evaluation artifacts for the v2_0 detection pipeline. "
           "Validation scope is internal reproducibility on labeled "
           "synthetic corpora — not independent or real-world validation. "
           "Scores are calibrated anomaly scores, not probabilities.")

# ---- Unseen-user evaluation (Milestone 2.7) -------------------------------
dq, dq_path = _load_json('detection_quality.json')
st.markdown("### Unseen-user evaluation")
if dq is None:
    st.info(f"No report at `{dq_path}` — run "
            "`scripts/evaluate_detection_quality.py` to generate it.")
elif '_error' in dq:
    st.error(f"Could not parse `{dq_path}`: {dq['_error']}")
else:
    uue = dq.get('unseen_user_evaluation') or {}
    st.caption(f"Report: `{dq_path}` · scope: "
               f"{dq.get('validation_scope', 'n/a')} · independent "
               f"validation: {dq.get('independent_validation', 'n/a')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation rows", uue.get('validation_rows', 'n/a'))
    m2.metric("Holdout rows", uue.get('holdout_rows', 'n/a'))
    m3.metric("Entity-disjoint splits",
              'yes' if uue.get('entity_disjoint') else 'no')
    auc = uue.get('holdout_score_auc') or {}
    m4.metric("Holdout ROC-AUC",
              f"{auc['roc_auc']:.4f}" if isinstance(auc.get('roc_auc'),
                                                 (int, float)) else 'n/a')

    st.markdown("**Holdout — default calibrated risk levels**")
    _metric_row(uue.get('holdout_default_risk_level') or {})
    st.markdown("**Holdout — validation-selected threshold applied**")
    _metric_row(uue.get('holdout_threshold_applied') or {})
    cal = uue.get('holdout_calibration_status') or {}
    st.caption("Calibration: " + ", ".join(
        f"{k}={v}" for k, v in cal.items()) or "n/a")

    scen = dq.get('synthetic_scenarios') or {}
    if scen:
        st.markdown("**Synthetic scenario checks** "
                    "(defensive research scenarios; FP = false positive)")
        rows = []
        for name, s in scen.items():
            dr = (s or {}).get('default_risk_level') or {}
            rows.append({
                'scenario': name,
                'rows': (s or {}).get('rows'),
                'FP': dr.get('fp'),
                'FPR': (f"{dr['fpr']:.3f}"
                        if isinstance(dr.get('fpr'), (int, float)) else 'n/a'),
                'threshold_version': (((s or {}).get('calibration_status')
                                       or {}).get('threshold_version')),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width='stretch', alt='Synthetic scenario FP checks')
    for note in dq.get('notes') or []:
        st.caption(f"Note: {note}")

# ---- Fusion / per-channel evaluation --------------------------------------
fq, fq_path = _load_json('fusion_evaluation.json')
st.markdown("### Per-channel fusion evaluation")
if fq is None:
    st.info(f"No report at `{fq_path}` — run `scripts/evaluate_fusion.py`.")
elif '_error' in fq:
    st.error(f"Could not parse `{fq_path}`: {fq['_error']}")
else:
    st.caption(f"Report: `{fq_path}` · scope: "
               f"{fq.get('validation_scope', 'n/a')}")
    channels = fq.get('channels') or {}
    if channels:
        rows = []
        for ch, splits in channels.items():
            for split_name, met in (splits or {}).items():
                met = met or {}
                rows.append({
                    'channel': ch, 'split': split_name,
                    'precision': met.get('precision'),
                    'recall': met.get('recall'),
                    'f1': met.get('f1'), 'fpr': met.get('fpr'),
                    'roc_auc': met.get('roc_auc'),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width='stretch', alt='Per-channel fusion metrics')

# ---- Model benchmark --------------------------------------------------------
mb, mb_path = _load_json('ml_benchmark_results.json')
st.markdown("### Model benchmark (Milestone 2.4)")
if mb is None:
    st.info(f"No report at `{mb_path}` — run `scripts/benchmark_models.py`.")
elif '_error' in mb:
    st.error(f"Could not parse `{mb_path}`: {mb['_error']}")
else:
    st.caption(f"Report: `{mb_path}` · scope: "
               f"{mb.get('validation_scope', 'n/a')}")
    results = mb.get('results') or {}
    rows = []
    for model, res in results.items():
        res = res or {}
        hb = res.get('holdout:binary') or {}
        hr = res.get('holdout:high_risk') or {}
        rows.append({
            'model': model,
            'holdout_binary_f1': hb.get('f1'),
            'holdout_binary_fpr': hb.get('fpr'),
            'holdout_highrisk_f1': hr.get('f1'),
            'train_time_s': res.get('train_time_s'),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     width='stretch', alt='Model benchmark results')
    with st.expander("Benchmark details"):
        st.json({k: v for k, v in mb.items() if k != 'results'})

# ---- Calibration manifest ---------------------------------------------------
cm, cm_path = _load_json('calibration_v2_manifest.json')
st.markdown("### Calibration manifest")
if cm is None:
    st.info(f"No report at `{cm_path}` — run "
            "`scripts/calibrate_v2_detection.py`.")
elif '_error' in cm:
    st.error(f"Could not parse `{cm_path}`: {cm['_error']}")
else:
    st.caption(f"Report: `{cm_path}`")
    det = cm.get('detection_thresholds') or {}
    risk = cm.get('risk_thresholds') or {}
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detection thresholds** (combined_score)")
        st.dataframe(pd.DataFrame([{
            'level': k, 'threshold': v}
            for k, v in det.items()]), hide_index=True,
            width='stretch', alt='Detection thresholds')
    with t2:
        st.markdown("**Risk thresholds**")
        st.dataframe(pd.DataFrame([{
            'level': k, 'threshold': v}
            for k, v in risk.items()]), hide_index=True,
            width='stretch', alt='Risk thresholds')
    fps = cm.get('fingerprints') or {}
    if fps:
        st.caption("Component fingerprints: " + ", ".join(
            f"{k}={str(v)[:12]}" for k, v in fps.items()))

# ---- Observational drift (on demand) ---------------------------------------
st.markdown("### Drift monitor (observational)")
st.caption("Measures distribution drift between the normal-activity "
           "baseline sample and the currently selected scenario's raw "
           "logs. Drift indicates distribution change only — it is not "
           "evidence of malicious activity and never changes scores or "
           "thresholds.")
if st.button("Measure drift vs normal baseline", key='dq_drift'):
    try:
        from core.drift_monitor import DriftMonitor
        from core.log_processor import load_sample_dataset
        from modules.analysis_state import current_scenario_key
        from modules.scenario_data import (
            load_scenario_frame, scenario_label)
        ref = load_sample_dataset('normal')
        cur = load_scenario_frame(current_scenario_key())
        rep = DriftMonitor().fit(ref).measure(cur)
        d = rep.to_dict()
        st.metric("Drift status", d['status'].upper())
        st.caption(d.get('summary', ''))
        st.caption(f"Reference rows: {d['reference_rows']} · "
                   f"current rows: {d['current_rows']} · "
                   f"scenario: {scenario_label(current_scenario_key())}")
        if d.get('feature_drift'):
            st.dataframe(pd.DataFrame(d['feature_drift']),
                         hide_index=True, width='stretch',
                         alt='Per-feature drift measurements')
        if d.get('columns_skipped'):
            with st.expander("Skipped columns"):
                st.dataframe(pd.DataFrame(d['columns_skipped']),
                             hide_index=True, width='stretch')
    except Exception as exc:
        st.error(safe_error_message(exc, "Drift measurement failed"))
