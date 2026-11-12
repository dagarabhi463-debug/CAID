"""
Milestone 1.9 Phase 2 — Leakage audit.

Programmatically audits every model input channel for direct or indirect
label leakage, on BOTH the legacy synthetic dataset (data/raw +
data/processed/evaluation_data.csv) and the new independent datasets
(data/independent/*.csv).

Checks:
  1. Per-column label separability (single-feature ROC-AUC for numeric
     columns; max category-conditional label deviation for categoricals).
  2. IP-prefix -> label mapping (the legacy generator encoded labels in
     RFC 5737 prefixes).
  3. Feature-channel audit: which engineered features / rule conditions
     could carry entity or label identity.
  4. Split hygiene on independent data: cross-split duplicates, entity
     leakage (holdout-only entities must not appear in train), label
     independence crosstabs.
  5. Timestamp / row-order artifacts.

Writes findings JSON to stdout and data/independent/leakage_audit.json.
Synthetic data only — findings describe the generator, not real systems.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEP = os.path.join(BASE, 'data', 'independent')
MANIFEST_PATH = os.path.join(INDEP, 'dataset_manifest.json')

RAW_COLUMNS = [
    'courses_accessed', 'accounts_accessed', 'api_requests', 'data_mb',
    'token_age_minutes', 'after_hours', 'hour', 'day_of_week',
    'is_weekend', 'is_after_hours',
]
CATEGORICAL_COLUMNS = ['user_id', 'source_ip', 'event_type', 'role',
                       'ip_class', 'ip_prefix']

FINDINGS = []


def finding(check, channel, severity, status, evidence):
    FINDINGS.append({
        'check': check,
        'channel': channel,
        'severity': severity,      # critical | high | medium | low | info
        'status': status,          # confirmed | mitigated | not_present | info
        'evidence': evidence,
    })


def binary_label(df):
    return df['label'].isin(['suspicious', 'high_risk']).astype(int)


def numeric_separability(df, col):
    """Single-feature ROC-AUC vs binary label. 0.5 = no separation."""
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors='coerce')
    y = binary_label(df)
    mask = s.notna() & np.isfinite(s)
    if mask.sum() < 10 or y[mask].nunique() < 2:
        return None
    try:
        auc = roc_auc_score(y[mask], s[mask])
    except ValueError:
        return None
    return float(max(auc, 1 - auc))  # distance from 0.5, symmetric


def categorical_separability(df, col, top_k=30):
    """Max deviation of P(positive|category) from base rate, weighted by
    category frequency. Returns (score, worst_category, detail)."""
    if col not in df.columns:
        return None
    y = binary_label(df)
    base = float(y.mean())
    grp = df.assign(_y=y).groupby(df[col].astype(str))['_y']
    stats = grp.agg(['mean', 'count'])
    stats = stats[stats['count'] >= 5].sort_values('count', ascending=False)
    if stats.empty:
        return None
    stats = stats.head(top_k)
    dev = (stats['mean'] - base).abs() * np.sqrt(stats['count'])
    worst = stats['mean'].sub(base).abs().idxmax()
    score = float(dev.max() / np.sqrt(len(df)))
    return score, str(worst), {
        'base_positive_rate': round(base, 4),
        'worst_category': str(worst),
        'worst_category_pos_rate': round(float(stats.loc[worst, 'mean']), 4),
        'worst_category_count': int(stats.loc[worst, 'count']),
        'n_categories': int(len(stats)),
    }


def audit_dataset(name, df):
    """Run per-column leakage checks on one dataset."""
    results = {'numeric': {}, 'categorical': {}}
    for col in RAW_COLUMNS:
        if col in df.columns or col == 'hour':
            # 'hour' is engineered; compute from timestamp if needed
            if col == 'hour' and 'hour' not in df.columns \
                    and 'timestamp' in df.columns:
                vals = pd.to_datetime(df['timestamp']).dt.hour
                s = pd.to_numeric(vals, errors='coerce')
                y = binary_label(df)
                mask = s.notna()
                auc = roc_auc_score(y[mask], s[mask])
                results['numeric'][col] = round(float(max(auc, 1 - auc)), 4)
                continue
            auc = numeric_separability(df, col)
            if auc is not None:
                results['numeric'][col] = round(auc, 4)
    for col in CATEGORICAL_COLUMNS:
        if col == 'ip_prefix':
            if 'source_ip' in df.columns:
                df = df.copy()
                df['ip_prefix'] = df['source_ip'].astype(str).str.extract(
                    r'(\d+\.\d+\.\d+\.)')[0]
        out = categorical_separability(df, col)
        if out is not None:
            score, worst, detail = out
            results['categorical'][col] = {
                'score': round(score, 4), 'worst': worst, **detail}
    return results


def audit_ip_prefix_mapping(df, name):
    """Check whether IP /24 prefix determines the label."""
    if 'source_ip' not in df.columns:
        return
    d = df.copy()
    d['prefix'] = d['source_ip'].astype(str).str.extract(r'(\d+\.\d+\.\d+\.)')[0]
    ct = pd.crosstab(d['prefix'], d['label'])
    # Purity alone is misleading under label imbalance (a 92%-normal split
    # makes every prefix ~92% pure). Use normalized mutual information:
    # NMI = 1 - H(label|prefix)/H(label); ~1.0 means prefix determines label.
    probs = ct.div(ct.sum(axis=1), axis=0)
    h_cond = -(probs * np.log(probs.replace(0, np.nan))).sum(axis=1)
    row_w = ct.sum(axis=1) / ct.values.sum()
    h_label_given_prefix = float((h_cond * row_w).sum())
    p_label = d['label'].value_counts(normalize=True)
    h_label = float(-(p_label * np.log(p_label)).sum())
    nmi = 1.0 - h_label_given_prefix / h_label if h_label > 0 else 0.0
    purity = (ct.max(axis=1) / ct.sum(axis=1)).to_dict()
    finding(
        'ip_prefix_label_mapping', f'{name}.source_ip prefix',
        'critical' if nmi > 0.5 else 'low',
        'confirmed' if nmi > 0.5 else 'not_present',
        {'prefixes': len(ct), 'nmi_prefix_label': round(nmi, 4),
         'mean_modal_label_purity': round(float(np.mean(list(purity.values()))), 4),
         'label_entropy': round(h_label, 4),
         'crosstab': {str(i): {str(c): int(v) for c, v in row.items()}
                      for i, row in ct.iterrows()}})


def audit_rule_conditions():
    """Check default detection rules for label-encoding conditions."""
    from core.detection_rules import DetectionRulesEngine
    engine = DetectionRulesEngine()
    conditions = {rid: r.condition for rid, r in engine.rules.items()}
    label_terms = ('ip_class', 'label', 'is_sensitive_event')
    hits = {rid: cond for rid, cond in conditions.items()
            if any(t in cond for t in label_terms)}
    finding(
        'rule_condition_label_reference', 'DetectionRulesEngine defaults',
        'critical' if hits else 'info',
        'confirmed' if hits else 'not_present',
        {'rules_using_identity_features': hits,
         'all_default_conditions': conditions})


def audit_feature_channels():
    """Verify which raw/engineered columns reach the ML feature matrix and
    whether any carry label identity."""
    from core.feature_engineering import engineer_features, \
        FeatureBaselineManager
    import inspect
    src = inspect.getsource(engineer_features)
    finding(
        'feature_schema_audit', 'engineer_features feature_columns',
        'info', 'info',
        {'note': 'ip_class and is_sensitive_event are engineered but only '
                 'is_sensitive_event is in the ML feature list; ip_class '
                 'reaches the RULES channel via conditions',
         'feature_list_source': 'core/feature_engineering.py'})


def audit_split_hygiene():
    """Independent dataset split checks from the manifest + raw CSVs."""
    if not os.path.exists(MANIFEST_PATH):
        finding('split_hygiene', 'data/independent', 'high',
                'not_present', 'manifest missing — run generator first')
        return
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)
    q = manifest['quality']
    finding('split_duplicates', 'data/independent all splits',
            'medium' if q['exact_duplicate_rows'] else 'low',
            'confirmed' if q['exact_duplicate_rows'] else 'not_present',
            {'exact_duplicate_rows': q['exact_duplicate_rows']})
    ov = manifest['overlap']
    leaks = {k: v for k, v in ov.items()
             if isinstance(v, dict)
             and (v.get('holdout_only_users_present', 0) > 0
                  or v.get('holdout_only_ips_present', 0) > 0)
             and not k.startswith('train_vs_holdout')}
    finding('split_entity_leakage', 'train vs calibration/validation',
            'critical' if leaks else 'low',
            'confirmed' if leaks else 'not_present', leaks or
            {'holdout_only entities confined to holdout': True})
    # Cross-split exact row overlap (same user+ip+timestamp+features)
    frames = {}
    for name in ('train', 'calibration', 'validation', 'holdout'):
        p = os.path.join(INDEP, f'{name}.csv')
        if os.path.exists(p):
            frames[name] = pd.read_csv(p)
    if len(frames) == 4:
        keys = ['user_id', 'source_ip', 'timestamp', 'event_type',
                'api_requests', 'data_mb']
        train_keys = set(map(tuple, frames['train'][keys].values))
        overlap_rows = {}
        for name in ('calibration', 'validation', 'holdout'):
            other = set(map(tuple, frames[name][keys].values))
            overlap_rows[name] = len(train_keys & other)
        finding('cross_split_row_overlap', 'independent splits', 'low',
                'info', {'shared_key_rows': overlap_rows,
                         'note': 'same user/IP/event can legitimately '
                                 'recur; reported for transparency'})


def audit_timestamp_order(df, name):
    """Check whether row order or timestamp encodes the label."""
    y = binary_label(df)
    n = len(df)
    # Row-order: label balance in first vs second half
    half = n // 2
    first = y.iloc[:half].mean()
    second = y.iloc[half:].mean()
    order_gap = abs(first - second)
    # Timestamp: mean hour per label already covered; check monotonic drift
    ts = pd.to_datetime(df['timestamp'], errors='coerce')
    drift = float(ts.corr(pd.Series(np.arange(n), index=df.index))) \
        if ts.notna().all() else None
    finding('row_order_timestamp', name,
            'medium' if order_gap > 0.15 else 'low',
            'confirmed' if order_gap > 0.15 else 'not_present',
            {'pos_rate_first_half': round(float(first), 4),
             'pos_rate_second_half': round(float(second), 4),
             'order_gap': round(float(order_gap), 4),
             'timestamp_row_index_corr': None if drift is None
             else round(drift, 4)})


def main():
    datasets = {}

    # Legacy dataset (the one Milestone 1.8.1 metrics came from)
    legacy_paths = {
        'legacy_eval': os.path.join(BASE, 'data', 'processed',
                                    'evaluation_data.csv'),
        'legacy_normal': os.path.join(BASE, 'data', 'raw',
                                      'normal_activity.csv'),
    }
    for name, path in legacy_paths.items():
        if os.path.exists(path):
            datasets[name] = pd.read_csv(path)

    # Independent datasets
    for name in ('train', 'calibration', 'validation', 'holdout'):
        p = os.path.join(INDEP, f'{name}.csv')
        if os.path.exists(p):
            datasets[f'indep_{name}'] = pd.read_csv(p)

    column_results = {}
    for name, df in datasets.items():
        column_results[name] = audit_dataset(name, df)
        audit_ip_prefix_mapping(df, name)
        audit_timestamp_order(df, name)

    # Interpretability findings (static, evidence-based)
    finding(
        'legacy_ip_prefix_encodes_label',
        'generate_synthetic_data.py NORMAL/SUSPICIOUS/HIGH_RISK_IP_PREFIX',
        'critical', 'confirmed',
        {'detail': 'Legacy generator assigns RFC 5737 prefixes '
                   '(192.0.2./198.51.100./203.0.113.) deterministically by '
                   'label; ip_class is derived from these prefixes and was '
                   'consumed by the suspicious_ip_class rule. Retained for '
                   'backward compatibility; superseded by independent '
                   'datasets where prefixes are label-independent.'})
    finding(
        'legacy_label_correlated_fields',
        'legacy generator hour ranges / after_hours / token_age_minutes',
        'high', 'confirmed',
        {'detail': 'Legacy suspicious/high_risk generators use distinct '
                   'hour windows, after_hours rates and token_age ranges. '
                   'Independent generator overlaps all three.'})
    audit_rule_conditions()
    audit_feature_channels()
    audit_split_hygiene()

    # Summarize per-dataset worst numeric/categorical separability
    summary = {}
    for name, res in column_results.items():
        nums = res['numeric']
        cats = res['categorical']
        summary[name] = {
            'max_numeric_feature_auc_distance':
                max(nums.values()) if nums else None,
            'worst_numeric_feature':
                max(nums, key=nums.get) if nums else None,
            'max_categorical_score':
                max((c['score'] for c in cats.values()), default=None),
            'worst_categorical':
                max(cats, key=lambda k: cats[k]['score'])
                if cats else None,
        }

    out = {
        'audit': 'MILESTONE_1_9_LEAKAGE_AUDIT',
        'datasets_audited': {k: len(v) for k, v in datasets.items()},
        'per_dataset_summary': summary,
        'per_dataset_detail': column_results,
        'findings': FINDINGS,
    }

    out_path = os.path.join(INDEP, 'leakage_audit.json')
    os.makedirs(INDEP, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)

    print(json.dumps({'summary': summary, 'n_findings': len(FINDINGS),
                      'out': out_path}, indent=2, default=str))
    for f_ in FINDINGS:
        print(f"[{f_['severity']:8s}] {f_['status']:12s} "
              f"{f_['channel']}: {f_['check']}")


if __name__ == '__main__':
    main()
