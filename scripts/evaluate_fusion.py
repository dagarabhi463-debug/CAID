"""
Milestone 2.4 — Detection fusion validation.

Evaluates each detection channel and the combined DetectionEngine on
the SAME entity-disjoint splits as scripts/benchmark_models.py:

    validation : reporting / threshold-sensitivity analysis
    holdout    : final evaluation, once — never used for tuning

Channels (v2_0 profile): ml, rules, behavioral, session, sequence.
All baselines are fit on unsup_train NORMAL rows only — identical
data to the ML benchmark, so results are comparable.

Outputs reports/fusion_evaluation.json and prints a summary.
Scores are anomaly/deviation scores — NOT probabilities.

The combined engine is given the PRODUCTION calibrated detection
operating point (artifact cal-20261112T055222Z-2962264f,
medium/high/critical = 0.20/0.30/0.45; see
docs/research/phase_4_6_fusion_recalibration.md) so reported
risk_level predictions match the deployed threshold semantics. The
baselines here are refit on unsup_train, so this is an internal
reproducibility check, not the calibrated-artifact path.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from benchmark_models import load_labeled_data, make_splits  # noqa: E402
from benchmark_models import _auc, _prf  # noqa: E402

from models.anomaly_detector import AnomalyDetector  # noqa: E402
from core.detection_rules import DetectionRulesEngine  # noqa: E402
from core.behavioural_analyzer import BehavioralAnalyzer  # noqa: E402
from core.session_analyzer import SessionAnalyzer  # noqa: E402
from core.sequence_detection import SequenceAnalyzer  # noqa: E402
from core.detection_engine import DetectionEngine, CHANNEL_PROFILES  # noqa: E402

SCORE_COLS = {
    'ml': 'anomaly_score_normalized',
    'rules': 'rule_score',
    'behavioral': 'behavioral_deviation_score',
    'session': 'session_deviation_score',
    'sequence': 'sequence_deviation_score',
}

# Production calibrated detection operating point (Phase 4.6).
PRODUCTION_DETECTION_THRESHOLDS = {
    'medium': 0.20, 'high': 0.30, 'critical': 0.45,
}


def _binary_labels(s: pd.Series) -> np.ndarray:
    return s.isin(['suspicious', 'high_risk']).astype(int).to_numpy()


def _eval_score(y, score, pred=None):
    score = pd.to_numeric(pd.Series(score), errors='coerce').fillna(0.0)
    out = _auc(y, score.to_numpy())
    if pred is None:
        pred = (score >= 0.5).astype(int).to_numpy()
    out.update(_prf(y, np.asarray(pred)))
    return out


def _eval_combined(y, df_scored):
    """Evaluate combined_score (AUC) and risk_level High/Critical (pred)."""
    score = pd.to_numeric(df_scored['combined_score'],
                          errors='coerce').fillna(0.0).to_numpy()
    pred = df_scored['risk_level'].isin(['High', 'Critical']).astype(int)
    out = _auc(y, score)
    out.update(_prf(y, pred.to_numpy()))
    return out


def _operating_mode_counts(df):
    if 'operating_mode' not in df.columns:
        return {}
    return df['operating_mode'].value_counts().to_dict()


def run_pipeline(raw_df: pd.DataFrame, train_normal: pd.DataFrame,
                 profile: str = 'v2_0',
                 drop_channels: tuple = ()) -> pd.DataFrame:
    """Run the full v2 channel stack; returns scored frame."""
    det = AnomalyDetector().fit(train_normal)
    out = det.predict(raw_df.copy())
    out = DetectionRulesEngine().process_events(out)
    beh = BehavioralAnalyzer().fit(train_normal)
    out = beh.analyze_user_behavior(out)
    ses = SessionAnalyzer().fit(train_normal)
    out = ses.analyze(out)
    seq = SequenceAnalyzer().fit(train_normal)
    out = seq.analyze(out)

    weights = dict(CHANNEL_PROFILES[profile])
    for ch in drop_channels:
        weights.pop(ch, None)
    # Operating-mode logic keys off channel_profile; use v2_0 whenever
    # session/sequence channels remain configured.
    eff_profile = ('v2_0' if ('session' in weights or 'sequence' in weights)
                   else 'legacy')
    engine = DetectionEngine(
        channel_profile=eff_profile, weights=weights,
        thresholds=(PRODUCTION_DETECTION_THRESHOLDS
                    if eff_profile == 'v2_0' else None))
    return engine.process_dataframe(out)


def main() -> Dict:
    df = load_labeled_data()
    splits = make_splits(df)
    train_normal = splits['unsup_train']

    results: Dict[str, Dict] = {
        'splits': {k: len(v) for k, v in splits.items()},
        'validation_scope': 'internal_reproducibility',
        'independent_validation': 'not_performed_current_data_only',
    }

    scored = {}
    for split in ('validation', 'holdout'):
        t0 = time.perf_counter()
        scored[split] = run_pipeline(splits[split], train_normal)
        results.setdefault('timing', {})[f'{split}_pipeline_s'] = round(
            time.perf_counter() - t0, 2)

    y = {s: _binary_labels(splits[s]['label']) for s in scored}

    # --- Per-channel individual performance ---
    channel_results = {}
    for ch, col in SCORE_COLS.items():
        channel_results[ch] = {}
        for split in ('validation', 'holdout'):
            dfc = scored[split]
            if col not in dfc.columns:
                channel_results[ch][split] = {'error': 'column missing'}
                continue
            channel_results[ch][split] = _eval_score(
                y[split], dfc[col])
    results['channels'] = channel_results

    # --- Combined v2_0 profile ---
    combined = {}
    for split in ('validation', 'holdout'):
        combined[split] = _eval_combined(y[split], scored[split])
        combined[split]['operating_modes'] = _operating_mode_counts(
            scored[split])
        combined[split]['missing_channel_rate'] = float(
            scored[split]['missing_channels'].apply(len).gt(0).mean()) \
            if 'missing_channels' in scored[split].columns else None
    results['combined_v2_0'] = combined

    # --- Channel ablation (drop-one) on validation; holdout reported once ---
    ablation = {}
    for ch in SCORE_COLS:
        others = tuple(c for c in SCORE_COLS if c != ch)
        ablation[ch] = {}
        for split in ('validation', 'holdout'):
            t0 = time.perf_counter()
            dfc = run_pipeline(splits[split], train_normal,
                               drop_channels=(ch,))
            ablation[ch][split] = _eval_combined(y[split], dfc)
            ablation[ch][split]['pipeline_s'] = round(
                time.perf_counter() - t0, 2)
    results['ablation_drop_one'] = ablation

    # --- Legacy 3-channel profile for comparison ---
    legacy = {}
    for split in ('validation', 'holdout'):
        dfc = run_pipeline(splits[split], train_normal, profile='legacy')
        legacy[split] = _eval_combined(y[split], dfc)
        legacy[split]['operating_modes'] = _operating_mode_counts(dfc)
    results['combined_legacy'] = legacy

    # --- Threshold sensitivity on VALIDATION ONLY (no holdout tuning) ---
    val = scored['validation']
    yv = y['validation']
    score_col = pd.to_numeric(val['combined_score'],
                              errors='coerce').fillna(0.0).to_numpy()
    sens = []
    for thr in np.arange(0.30, 0.91, 0.05):
        pred = (score_col >= thr).astype(int)
        m = _prf(yv, pred)
        sens.append({'threshold': round(float(thr), 2), **m})
    results['threshold_sensitivity_validation_only'] = sens

    results['notes'] = [
        'All baselines fit on unsup_train normal rows only.',
        'Holdout evaluated once; never used for tuning or selection.',
        'Threshold sensitivity computed on validation split only.',
        'Channel scores are anomaly/deviation scores, not probabilities.',
        'Ablation removes a channel from configured weights; remaining '
        'weights renormalize per DetectionEngine policy.',
        'Validation scope is internal reproducibility on the current '
        'labeled CSVs, not independent validation.',
        'Independent validation option: recover data/independent_v2 '
        'from git commit bfdca25 (see docs/research/'
        'ml_benchmark_methodology.md).',
    ]

    out_path = os.path.join(ROOT, 'reports', 'fusion_evaluation.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2, default=str)

    print('== channels (holdout binary) ==')
    for ch in SCORE_COLS:
        m = channel_results[ch].get('holdout', {})
        print(f"  {ch:12s} roc={m.get('roc_auc')} pr={m.get('pr_auc')} "
              f"p={m.get('precision')} r={m.get('recall')} fpr={m.get('fpr')}")
    print('== combined ==')
    for split in ('validation', 'holdout'):
        m = combined[split]
        print(f"  {split}: roc={m.get('roc_auc')} p={m.get('precision')} "
              f"r={m.get('recall')} f1={m.get('f1')} fpr={m.get('fpr')} "
              f"modes={m.get('operating_modes')}")
    print('== ablation holdout f1 ==')
    for ch in SCORE_COLS:
        print(f"  -{ch:12s} f1={ablation[ch]['holdout'].get('f1')}")
    print(f'wrote {out_path}')
    return results


if __name__ == '__main__':
    main()
