"""
Milestone 2.4 — ML detection benchmark harness.

Compares anomaly detectors on CONSISTENT engineered features and
entity-disjoint data splits built deterministically from the current
labeled datasets:

    data/raw/security_logs.csv      (650 rows: normal/suspicious/high_risk)
    data/processed/evaluation_data.csv (200 rows, mostly disjoint users)

Splits (user-level, sha1-seeded — users never cross splits):
    unsup_train    : normal-only users  -> fit IsolationForest / LOF
    calibration    : normal-only users  -> score-calibration reference
    sup_train      : mixed users        -> fit supervised RF / GBM
    validation     : mixed users        -> model selection / reporting
    holdout        : mixed users        -> FINAL evaluation only

Holdout is NEVER used for training, feature selection, or threshold
tuning. All scores are anomaly/deviation scores — NOT probabilities.

Models:
    isolation_forest : existing AnomalyDetector artifact pipeline
                       (FeatureBaselineManager + StandardScaler + IF)
    local_outlier    : sklearn LocalOutlierFactor(novelty=True) on the
                       same engineered+scaled features
    random_forest    : supervised sklearn RandomForest on labeled
                       sup_train (normal=0, suspicious/high_risk=1)
    gradient_boost   : supervised HistGradientBoosting, same labels

Usage:
    .venv/Scripts/python scripts/benchmark_models.py [--out PATH]

Writes a JSON results file and prints a summary table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.feature_engineering import (
    FeatureBaselineManager,
    engineer_features,
    get_feature_matrix,
)
from models.anomaly_detector import AnomalyDetector

DATA_FILES = [
    os.path.join(ROOT, 'data', 'raw', 'security_logs.csv'),
    os.path.join(ROOT, 'data', 'processed', 'evaluation_data.csv'),
]
SEED = 42

# User-bucket quotas (fractions of distinct users, assigned by hash so
# the split is deterministic and entity-disjoint).
BUCKETS = {
    'validation': 0.15,
    'holdout': 0.15,
    # remaining ~0.70 split below between train/cal/sup
}
POOL_SPLIT = {'unsup_train': 0.50, 'calibration': 0.15, 'sup_train': 0.35}

MODELS = ['isolation_forest', 'local_outlier', 'random_forest',
          'gradient_boost']


# ----------------------------------------------------------------------
# Data loading & deterministic entity-disjoint splits
# ----------------------------------------------------------------------

def load_labeled_data() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in DATA_FILES]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(
        subset=['timestamp', 'user_id', 'event_type', 'source_ip',
                'api_requests', 'data_mb'])
    return df.reset_index(drop=True)


def _bucket(user_id: str) -> str:
    """Deterministic user -> split bucket via sha1."""
    h = hashlib.sha1(f'{SEED}:{user_id}'.encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < BUCKETS['validation']:
        return 'validation'
    if frac < BUCKETS['validation'] + BUCKETS['holdout']:
        return 'holdout'
    # pool users: subdivide deterministically
    h2 = hashlib.sha1(f'{SEED}:pool:{user_id}'.encode()).hexdigest()
    frac2 = int(h2[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, q in POOL_SPLIT.items():
        acc += q
        if frac2 < acc:
            return name
    return 'sup_train'


def make_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Assign rows to splits by user bucket.

    unsup_train/calibration keep ONLY normal rows (unsupervised
    fitting contract). sup_train keeps all labels (supervised
    training). validation/holdout keep all labels (evaluation).
    """
    buckets = df['user_id'].astype(str).map(_bucket)
    out = {k: df[buckets == k].copy() for k in
           ['unsup_train', 'calibration', 'sup_train',
            'validation', 'holdout']}
    out['unsup_train'] = out['unsup_train'][
        out['unsup_train']['label'] == 'normal']
    out['calibration'] = out['calibration'][
        out['calibration']['label'] == 'normal']
    return out


# ----------------------------------------------------------------------
# Feature extraction (shared, leakage-safe)
# ----------------------------------------------------------------------

def build_features(splits: Dict[str, pd.DataFrame]):
    """Fit FeatureBaselineManager on unsup_train normals ONLY, then
    engineer features for every split with those frozen baselines."""
    bm = FeatureBaselineManager().fit(splits['unsup_train'])
    feats, cols = {}, None
    for name, df in splits.items():
        eng, detected = engineer_features(df, bm)
        cols = cols or detected
        feats[name] = get_feature_matrix(eng, detected)
    return bm, feats, cols


def _scaled(X: np.ndarray, scaler) -> np.ndarray:
    return scaler.transform(X)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4), 'fpr': round(fpr, 4)}


def _auc(y_true: np.ndarray, scores: np.ndarray) -> Dict:
    """ROC-AUC and PR-AUC via rank statistics (no sklearn dependency
    surprises; handles ties deterministically)."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    # average ranks for ties
    sorted_s = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    P = int(y_true.sum())
    N = len(y_true) - P
    if P == 0 or N == 0:
        return {'roc_auc': None, 'pr_auc': None}
    roc = (ranks[y_true == 1].sum() - P * (P + 1) / 2) / (P * N)
    # PR-AUC: precision at each recall step (step interpolation)
    srt = np.argsort(-scores, kind='mergesort')
    ys = y_true[srt]
    tp_c = np.cumsum(ys)
    prec_i = tp_c / np.arange(1, len(ys) + 1)
    rec_i = tp_c / P
    pr = float(np.trapezoid(prec_i, rec_i)) if len(ys) > 1 else 0.0
    return {'roc_auc': round(float(roc), 4), 'pr_auc': round(pr, 4)}


def evaluate(y_true_labels: pd.Series, scores: np.ndarray,
             preds: np.ndarray, mode: str) -> Dict:
    if mode == 'binary':
        y = y_true_labels.isin(['suspicious', 'high_risk']).astype(int)
    else:  # high-risk only
        y = (y_true_labels == 'high_risk').astype(int)
    y = y.to_numpy()
    m = _prf(y, preds.astype(int))
    m.update(_auc(y, scores))
    return m


# ----------------------------------------------------------------------
# Model wrappers: uniform fit/predict -> (scores, preds) higher=anomalous
# ----------------------------------------------------------------------

class IFWrapper:
    name = 'isolation_forest'

    def __init__(self):
        self.det = AnomalyDetector()

    def fit(self, raw_train: pd.DataFrame):
        self.det.fit(raw_train)
        return self

    def predict(self, raw_df: pd.DataFrame):
        out = self.det.predict(raw_df)
        return (out['anomaly_score_normalized'].to_numpy(),
                out['is_anomaly'].astype(int).to_numpy())


class LOFWrapper:
    name = 'local_outlier'

    def __init__(self, scaler, feature_cols_count):
        from sklearn.neighbors import LocalOutlierFactor
        self.scaler = scaler
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True,
                                      contamination=0.1)

    def fit(self, X_train_scaled):
        self.lof.fit(X_train_scaled)

    def predict(self, X_eval_scaled):
        # novelty decision_function: higher = MORE NORMAL -> invert
        raw = self.lof.decision_function(X_eval_scaled)
        scores = -raw
        preds = (self.lof.predict(X_eval_scaled) == -1).astype(int)
        return scores, preds


class SupervisedWrapper:
    """RF / HistGB trained on labeled sup_train features."""

    def __init__(self, kind: str):
        self.kind = kind
        self.model = None

    def fit(self, X, y):
        if self.kind == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200, random_state=SEED, n_jobs=-1,
                class_weight='balanced_subsample')
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                random_state=SEED, class_weight='balanced')
        self.model.fit(X, y)

    def predict(self, X):
        if hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X)[:, 1]
        else:
            scores = self.model.decision_function(X)
        preds = self.model.predict(X)
        return scores, preds.astype(int)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _time_fit(fn) -> Tuple[float, object]:
    t0 = time.perf_counter()
    obj = fn()
    return time.perf_counter() - t0, obj


def _time_predict(fn) -> Tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def main() -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        ROOT, 'reports', 'ml_benchmark_results.json'))
    args = ap.parse_args()

    df = load_labeled_data()
    splits = make_splits(df)
    split_summary = {k: {'rows': len(v),
                         'labels': v['label'].value_counts().to_dict(),
                         'users': int(v['user_id'].nunique())}
                     for k, v in splits.items()}
    # entity-disjointness assertion
    users = {k: set(v['user_id']) for k, v in splits.items()}
    for a in users:
        for b in users:
            if a != b:
                assert not (users[a] & users[b]), f'user leak {a}/{b}'

    tracemalloc.start()
    bm, feats, cols = build_features(splits)
    feature_mem_mb = tracemalloc.get_traced_memory()[1] / 1e6

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(feats['unsup_train'])
    Xs = {k: _scaled(v, scaler) for k, v in feats.items()}

    results: Dict[str, Dict] = {}

    # --- IsolationForest (production artifact path) ---
    t_fit, ifw = _time_fit(lambda: IFWrapper().fit(splits['unsup_train']))
    det = ifw.det
    for split in ('validation', 'holdout'):
        t_pred, (scores, preds) = _time_predict(
            lambda s=split: tuple(
                det.predict(splits[s])[c].to_numpy()
                for c in ('anomaly_score_normalized', 'is_anomaly')))
        preds = preds.astype(int)
        for mode in ('binary', 'high_risk'):
            key = f'{split}:{mode}'
            results.setdefault('isolation_forest', {})[key] = evaluate(
                splits[split]['label'], scores, preds, mode)
            results['isolation_forest'].setdefault('timing', {})[
                f'{split}_predict_s'] = round(t_pred, 4)
    results['isolation_forest']['train_time_s'] = round(t_fit, 3)
    results['isolation_forest']['n_features'] = len(cols)
    results['isolation_forest']['feature_columns'] = cols

    # --- LOF novelty ---
    lof = LOFWrapper(scaler, len(cols))
    t_fit, _ = _time_fit(lambda: lof.fit(Xs['unsup_train']))
    for split in ('validation', 'holdout'):
        t_pred, (scores, preds) = _time_predict(
            lambda s=split: lof.predict(Xs[s]))
        for mode in ('binary', 'high_risk'):
            results.setdefault('local_outlier', {})[
                f'{split}:{mode}'] = evaluate(
                    splits[split]['label'], scores, preds, mode)
    results['local_outlier']['train_time_s'] = round(t_fit, 3)

    # --- Supervised models on labeled sup_train ---
    sup = splits['sup_train']
    y_sup = sup['label'].isin(['suspicious', 'high_risk']).astype(int)
    # sup_train features must be engineered with the SAME frozen
    # baselines — feats['sup_train'] already is.
    for kind, mname in [('random_forest', 'random_forest'),
                        ('gradient_boost', 'gradient_boost')]:
        sw = SupervisedWrapper(kind)
        t_fit, _ = _time_fit(lambda w=sw: w.fit(Xs['sup_train'],
                                                y_sup.to_numpy()))
        for split in ('validation', 'holdout'):
            t_pred, (scores, preds) = _time_predict(
                lambda w=sw, s=split: w.predict(Xs[s]))
            for mode in ('binary', 'high_risk'):
                results.setdefault(mname, {})[
                    f'{split}:{mode}'] = evaluate(
                        splits[split]['label'], scores, preds, mode)
        results[mname]['train_time_s'] = round(t_fit, 3)

    peak_mem_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    out = {
        'benchmark': 'milestone_2_4_ml_detection',
        'seed': SEED,
        'validation_scope': 'internal_reproducibility',
        'independent_validation': 'not_performed_current_data_only',
        'environment': {
            'python': sys.version.split()[0],
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'sklearn': __import__('sklearn').__version__,
        },
        'data': {
            'files': [os.path.relpath(p, ROOT) for p in DATA_FILES],
            'total_rows': int(len(df)),
            'splits': split_summary,
            'entity_disjoint': True,
        },
        'peak_traced_memory_mb': round(peak_mem_mb, 2),
        'feature_engineering_mem_mb': round(feature_mem_mb, 2),
        'results': results,
        'notes': [
            'Scores are anomaly scores, not probabilities.',
            'Holdout evaluated once; never used for training, feature '
            'selection, or threshold tuning.',
            'Supervised models trained on labeled sup_train users only '
            '(disjoint from all eval users).',
            'Default operating thresholds only — no threshold tuning on '
            'validation or holdout.',
            'Small labeled corpus (~850 rows): metrics are indicative, '
            'not statistically powerful.',
            'Validation scope is internal reproducibility on the current '
            'labeled CSVs, not independent validation.',
            'Independent validation option: recover data/independent_v2 '
            'from git commit bfdca25 (see docs/research/'
            'ml_benchmark_methodology.md).',
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, default=str)

    # Console summary
    print(f"rows={len(df)} splits="
          + ', '.join(f'{k}:{v["rows"]}' for k, v in split_summary.items()))
    for m in MODELS:
        r = results.get(m, {})
        hv = r.get('holdout:binary', {})
        hr = r.get('holdout:high_risk', {})
        print(f"{m:20s} holdout binary P={hv.get('precision')} "
              f"R={hv.get('recall')} F1={hv.get('f1')} "
              f"ROC={hv.get('roc_auc')} PR={hv.get('pr_auc')} | "
              f"high_risk R={hr.get('recall')} "
              f"train={r.get('train_time_s')}s")
    print(f"wrote {args.out}")
    return out


if __name__ == '__main__':
    main()
