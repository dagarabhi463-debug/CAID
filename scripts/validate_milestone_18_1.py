"""
Milestone 1.8.1 validation: score distributions, channel decomposition,
detection metrics (record- and user-level), and calibration integrity.

Runs the SAME pipeline order as modules/log_analysis.py:
  detector.predict -> RiskEngine.process_events -> DetectionRulesEngine
  -> BehavioralAnalyzer.analyze_user_behavior -> DetectionEngine

Synthetic data only. Metrics describe this synthetic distribution and are
NOT real-world detection accuracy. anomaly_score_normalized is a calibrated
0-1 score (higher = more anomalous); 0.5 is NOT a classification threshold.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector, get_model_path
from core.log_processor import load_sample_dataset
from core.risk_engine import RiskEngine
from core.detection_rules import DetectionRulesEngine
from core.behavioural_analyzer import ensure_fitted_analyzer
from core.detection_engine import DetectionEngine

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)

SCORE_COLS = [
    'anomaly_score_normalized', 'anomaly_score_raw', 'risk_score',
    'rule_score', 'behavioral_deviation_score', 'combined_score',
    'ml_anomaly_score',
]


def run_pipeline(df, detector):
    """Replicate the application pipeline (log_analysis.py order)."""
    out = detector.predict(df)
    out = RiskEngine().process_events(out)
    out = DetectionRulesEngine().process_events(out)
    out = ensure_fitted_analyzer().analyze_user_behavior(out)
    out = DetectionEngine().process_dataframe(out)
    return out


def describe_scores(df, label):
    print(f"\n--- {label} (n={len(df)}) ---")
    for col in SCORE_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) == 0:
            print(f"  {col:32s} all-missing")
            continue
        print(f"  {col:32s} mean={s.mean():8.4f} median={s.median():8.4f} "
              f"std={s.std():8.4f} min={s.min():8.4f} max={s.max():8.4f} "
              f"p25={s.quantile(.25):8.4f} p75={s.quantile(.75):8.4f}")


def histogram(df, col='anomaly_score_normalized', bins=10):
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    counts, edges = np.histogram(s, bins=bins, range=(0.0, 1.0))
    print(f"\n  Histogram {col} ({len(s)} rows):")
    for i, c in enumerate(counts):
        print(f"    [{edges[i]:.1f},{edges[i+1]:.1f}): {'#' * int(60 * c / max(counts.max(), 1))} {c}")


def channel_means(df):
    print("\n--- Channel contributions (means) ---")
    for lab, sub in df.groupby('label'):
        ml = pd.to_numeric(sub.get('ml_anomaly_score'), errors='coerce').mean()
        rl = pd.to_numeric(sub.get('rule_score'), errors='coerce').mean()
        bh = pd.to_numeric(sub.get('behavioral_deviation'), errors='coerce').mean()
        cs = pd.to_numeric(sub.get('combined_score'), errors='coerce').mean()
        miss = float(sub.get('behavioral_score_missing', pd.Series(dtype=bool)).mean()) \
            if 'behavioral_score_missing' in sub.columns else float('nan')
        print(f"  {lab:12s} ml={ml:6.3f} rule={rl:6.3f} behavioral={bh:6.3f} "
              f"combined={cs:6.3f} beh_missing={miss:.2f}")


def binary_labels(df, mode):
    if mode == 'binary':
        return df['label'].isin(['suspicious', 'high_risk'])
    return df['label'] == 'high_risk'


def pred_from_level(levels, mode):
    if mode == 'binary':
        return levels.isin(['High', 'Critical'])
    return levels == 'Critical'


def metrics_block(name, y_true, y_pred, score=None):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    print(f"\n  [{name}] CM: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} "
          f"fpr={fpr:.4f} acc={(tp+tn)/max(len(y_true),1):.4f}")
    if score is not None:
        s = np.asarray(score, dtype=float)
        ok = np.isfinite(s)
        if ok.sum() == len(s) and len(np.unique(y_true)) == 2 \
                and len(np.unique(y_true[ok])) == 2:
            try:
                print(f"    ROC-AUC={roc_auc_score(y_true[ok], s[ok]):.4f} "
                      f"PR-AUC={average_precision_score(y_true[ok], s[ok]):.4f}")
            except ValueError as e:
                print(f"    ROC/PR-AUC unavailable: {e}")
        else:
            print("    ROC/PR-AUC unavailable: degenerate labels or scores")
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1, 'fpr': fpr}


def user_level_metrics(df, mode):
    """Aggregate per user_id: user score = MAX event normalized score
    (most anomalous event wins); user prediction = any event anomalous.
    Users with mixed labels are reported separately."""
    print(f"\n--- User-level aggregation ({mode} mode) ---")
    mixed = df.groupby('user_id')['label'].nunique()
    n_mixed = int((mixed > 1).sum())
    print(f"  users={df['user_id'].nunique()} mixed-label users={n_mixed}")
    g = df.groupby('user_id')
    user_score = g['anomaly_score_normalized'].max()
    user_any_anom = g['is_anomaly'].max().astype(bool)
    # user label: single label per user in this dataset; use mode
    user_label = g['label'].agg(lambda s: s.mode().iloc[0])
    y_true = binary_labels(pd.DataFrame({'label': user_label}), mode)
    score = user_score.reindex(user_label.index)
    metrics_block('user/is_anomaly(any)', np.asarray(y_true),
                  np.asarray(user_any_anom), score=np.asarray(score))
    # also max-score threshold sweep at the calibrated 'medium' cut is
    # covered by record-level metrics; here report AUC on max score only.


def record_metrics(df, mode):
    y_true = np.asarray(binary_labels(df, mode))
    score = pd.to_numeric(df['anomaly_score_normalized'], errors='coerce')
    metrics_block('record/is_anomaly(IF flag)', y_true,
                  np.asarray(df['is_anomaly'].astype(bool)), score=score)
    # detection engine combined score -> risk_level (High/Critical pos)
    lvl = df['risk_level'].astype(str)
    metrics_block('record/detection risk_level', y_true,
                  np.asarray(pred_from_level(lvl, mode)))
    # risk_score threshold sweep not used; calibrated eval below covers it


def calibration_section(normal_proc, eval_proc):
    print("\n" + "=" * 70)
    print("CALIBRATION INTEGRITY")
    print("=" * 70)
    # Calibrate on PROCESSED normal (baseline) data only — includes ML scores
    engine = RiskEngine()
    thr = engine.calibrate_thresholds(normal_proc)
    print(f"  fixed_thresholds={thr}")
    print(f"  calibration_info={engine.calibration_info}")
    # Thresholds must be identical after evaluating (no refit on eval)
    before = dict(engine.fixed_thresholds)
    m_bin = engine.evaluate_calibration(eval_proc, mode='binary')
    assert engine.fixed_thresholds == before, "thresholds mutated by eval!"
    print("  thresholds unchanged after evaluate_calibration: OK")
    m_hr = engine.evaluate_calibration(eval_proc, mode='high-risk')
    for k in ('tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1_score',
              'false_positive_rate'):
        print(f"  binary[{k}]={m_bin[k]:.4f}" if isinstance(m_bin[k], float)
              else f"  binary[{k}]={m_bin[k]}")
    print(f"  binary fpr={m_bin['false_positive_rate']:.4f} "
          f"high-risk fpr={m_hr['false_positive_rate']:.4f}")
    # Missing-ML handling: drop score columns -> neutral 0.5 path
    stripped = eval_proc.drop(columns=[
        c for c in ('anomaly_score_normalized', 'anomaly_score_raw',
                    'anomaly_score', 'risk_score', 'risk_level',
                    'risk_factors', 'risk_evidence') if c in eval_proc.columns])
    engine2 = RiskEngine()
    engine2.calibrate_thresholds(normal_proc)
    m_missing = engine2.evaluate_calibration(stripped, mode='binary')
    print(f"  missing-ML eval: fpr={m_missing['false_positive_rate']:.4f} "
          f"(neutral 0.5 ML path; compare vs {m_bin['false_positive_rate']:.4f})")
    # Unseen-entity representation in eval set
    train_users = set(load_sample_dataset('normal')['user_id'])
    ev_users = set(eval_proc['user_id'])
    print(f"  eval users unseen-in-training: "
          f"{len(ev_users - train_users)}/{len(ev_users)}")


def main():
    detector = AnomalyDetector().load(get_model_path())
    normal_raw = load_sample_dataset('normal')
    eval_raw = load_sample_dataset('evaluation')

    print("=" * 70)
    print("PIPELINE RUN: normal_activity.csv (training baseline)")
    print("=" * 70)
    normal_proc = run_pipeline(normal_raw, detector)
    describe_scores(normal_proc, 'normal_activity (label=normal)')
    histogram(normal_proc)
    channel_means(normal_proc)

    print("\n" + "=" * 70)
    print("PIPELINE RUN: evaluation_data.csv (labeled holdout)")
    print("=" * 70)
    eval_proc = run_pipeline(eval_raw, detector)
    for lab in ('normal', 'suspicious', 'high_risk'):
        describe_scores(eval_proc[eval_proc['label'] == lab], f'eval/{lab}')
    histogram(eval_proc)
    channel_means(eval_proc)

    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS (anomaly_score_normalized means)")
    print("=" * 70)
    means = eval_proc.groupby('label')['anomaly_score_normalized'].mean()
    for a, b in (('normal', 'suspicious'), ('normal', 'high_risk'),
                 ('suspicious', 'high_risk')):
        print(f"  {a} ({means[a]:.3f}) vs {b} ({means[b]:.3f})")

    print("\n" + "=" * 70)
    print("DETECTION METRICS — RECORD LEVEL")
    print("=" * 70)
    for mode in ('binary', 'high-risk'):
        print(f"\n>>> mode={mode}")
        record_metrics(eval_proc, mode)

    print("\n" + "=" * 70)
    print("DETECTION METRICS — USER LEVEL")
    print("=" * 70)
    for mode in ('binary', 'high-risk'):
        user_level_metrics(eval_proc, mode)

    calibration_section(normal_proc, eval_proc)


if __name__ == '__main__':
    main()
