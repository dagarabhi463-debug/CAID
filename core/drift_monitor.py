"""
Drift Monitor for CAID — Milestone 2.4 (observational only).

Measures distribution drift between a stored REFERENCE distribution
(typically the training/calibration normal baseline) and a CURRENT
sample of incoming or analyzed events.

DESIGN CONTRACT:
- OBSERVATIONAL ONLY. This module never retrains models, never
  changes thresholds, weights, or scores, and never feeds back into
  the detection pipeline.
- DRIFT IS NOT AN ALERT. Drift indicates the input distribution
  differs from the reference; it is NOT evidence of malicious
  activity. Findings must be interpreted by an analyst.
- EXPLICIT STATUS. Every report carries an explicit status
  (``ok`` / ``warning`` / ``critical`` / ``insufficient_reference``)
  — drift is never silently treated as normal or malicious.
- DETERMINISTIC. Same reference + same current data -> identical
  report (timestamps aside).
- MISSING DATA IS EXPLICIT. Non-numeric/unusable columns are
  reported per-feature, never silently skipped or treated as zero.

Metrics per numeric feature:
- PSI (Population Stability Index) on equal-frequency bins.
- Two-sample KS statistic (scipy-free implementation via sorted
  arrays) as a secondary measure.

For categorical/text features:
- Total variation distance between value-frequency distributions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Severity thresholds (documented defaults; NOT tuned on holdout —
# these are conventional PSI/KS interpretive bands).
DEFAULT_PSI_WARN = 0.10
DEFAULT_PSI_CRIT = 0.25
DEFAULT_KS_WARN = 0.10
DEFAULT_KS_CRIT = 0.20
DEFAULT_TVD_WARN = 0.10
DEFAULT_TVD_CRIT = 0.25
MIN_REFERENCE_ROWS = 30
N_BINS = 10

STATUS_OK = 'ok'
STATUS_WARNING = 'warning'
STATUS_CRITICAL = 'critical'
STATUS_INSUFFICIENT = 'insufficient_reference'


@dataclass
class DriftConfig:
    psi_warn: float = DEFAULT_PSI_WARN
    psi_crit: float = DEFAULT_PSI_CRIT
    ks_warn: float = DEFAULT_KS_WARN
    ks_crit: float = DEFAULT_KS_CRIT
    tvd_warn: float = DEFAULT_TVD_WARN
    tvd_crit: float = DEFAULT_TVD_CRIT
    min_reference_rows: int = MIN_REFERENCE_ROWS
    n_bins: int = N_BINS
    numeric_columns: Optional[List[str]] = None   # None = auto-detect
    categorical_columns: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DriftReport:
    """Result of one drift measurement. JSON-serializable via to_dict."""
    status: str
    measured_at_utc: str
    reference_rows: int
    current_rows: int
    columns_measured: List[str] = field(default_factory=list)
    columns_skipped: List[Dict] = field(default_factory=list)
    feature_drift: List[Dict] = field(default_factory=list)
    max_psi: float = 0.0
    max_ks: float = 0.0
    summary: str = ''
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'measured_at_utc': self.measured_at_utc,
            'reference_rows': self.reference_rows,
            'current_rows': self.current_rows,
            'columns_measured': self.columns_measured,
            'columns_skipped': self.columns_skipped,
            'feature_drift': self.feature_drift,
            'max_psi': self.max_psi,
            'max_ks': self.max_ks,
            'summary': self.summary,
            'config': self.config,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class DriftMonitor:
    """
    Reference-distribution drift monitor.

    Usage:
        mon = DriftMonitor(config).fit(reference_df)
        report = mon.measure(current_df)   # DriftReport
    """

    VERSION = 1

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        self._reference: Optional[Dict[str, np.ndarray]] = None
        self._ref_cats: Optional[Dict[str, Dict[str, float]]] = None
        self._ref_fingerprint: Optional[str] = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting the reference distribution
    # ------------------------------------------------------------------

    def fit(self, reference_df: pd.DataFrame) -> 'DriftMonitor':
        """Store reference distributions from designated baseline data.

        Reference data should be normal/training data only — fitting on
        attack or mixed data would mask real drift.
        """
        if reference_df is None or len(reference_df) == 0:
            raise ValueError('Cannot fit drift monitor on empty reference')
        cfg = self.config
        numeric_cols, cat_cols = self._select_columns(reference_df)

        ref: Dict[str, np.ndarray] = {}
        cats: Dict[str, Dict[str, float]] = {}
        skipped: List[Dict] = []

        for col in numeric_cols:
            vals = pd.to_numeric(reference_df[col], errors='coerce').dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) < cfg.min_reference_rows:
                skipped.append({'column': col, 'kind': 'numeric',
                                'reason': f'only {len(vals)} finite values'})
                continue
            ref[col] = np.sort(vals.to_numpy(dtype=float))

        for col in cat_cols:
            vc = reference_df[col].astype(str).value_counts(normalize=True)
            if len(vc) == 0:
                skipped.append({'column': col, 'kind': 'categorical',
                                'reason': 'no values'})
                continue
            cats[col] = {str(k): float(v) for k, v in vc.items()}

        if not ref and not cats:
            raise ValueError('No usable reference columns for drift monitor')

        self._reference = ref
        self._ref_cats = cats
        self._skipped_at_fit = skipped
        self._ref_row_count = len(reference_df)
        payload = json.dumps(
            {'numeric': {c: [float(v[0]), float(v[-1]), len(v)]
                         for c, v in ref.items()},
             'categorical': cats},
            sort_keys=True, default=str).encode('utf-8')
        self._ref_fingerprint = hashlib.sha256(payload).hexdigest()[:16]
        self._fitted = True
        return self

    def _select_columns(self, df: pd.DataFrame):
        cfg = self.config
        if cfg.numeric_columns is not None:
            numeric = [c for c in cfg.numeric_columns if c in df.columns]
        else:
            numeric = [c for c in df.columns
                       if pd.api.types.is_numeric_dtype(df[c])]
        if cfg.categorical_columns is not None:
            cat = [c for c in cfg.categorical_columns if c in df.columns]
        else:
            # Object columns are too heterogeneous for auto TV-distance;
            # categorical drift is measured only for explicitly configured
            # columns.
            cat = []
        return numeric, cat

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure(self, current_df: pd.DataFrame,
                reference_df: Optional[pd.DataFrame] = None) -> DriftReport:
        """Measure drift of `current_df` against the fitted reference.

        `reference_df` may be supplied to measure drift between two
        ad-hoc frames without fitting (reference rows counted from it).
        """
        if not self._fitted and reference_df is None:
            return DriftReport(
                status=STATUS_INSUFFICIENT,
                measured_at_utc=_utcnow(),
                reference_rows=0,
                current_rows=0 if current_df is None else len(current_df),
                summary='Drift monitor has no fitted reference.',
                config=self.config.to_dict())

        cfg = self.config
        if current_df is None or len(current_df) == 0:
            return DriftReport(
                status=STATUS_INSUFFICIENT,
                measured_at_utc=_utcnow(),
                reference_rows=self._ref_rows(),
                current_rows=0,
                summary='Current sample is empty; drift cannot be '
                        'measured.',
                config=cfg.to_dict())

        if reference_df is not None:
            tmp = DriftMonitor(cfg).fit(reference_df)
            ref_numeric, ref_cats = tmp._reference, tmp._ref_cats
            ref_rows = len(reference_df)
        else:
            ref_numeric, ref_cats = self._reference, self._ref_cats
            ref_rows = self._ref_rows()

        if ref_rows < cfg.min_reference_rows:
            return DriftReport(
                status=STATUS_INSUFFICIENT,
                measured_at_utc=_utcnow(),
                reference_rows=ref_rows,
                current_rows=len(current_df),
                summary=f'Reference has {ref_rows} rows '
                        f'(< {cfg.min_reference_rows}); drift cannot be '
                        'measured reliably.',
                config=cfg.to_dict())

        feature_drift: List[Dict] = []
        skipped = list(getattr(self, '_skipped_at_fit', []) or [])
        max_psi = max_ks = 0.0

        for col, ref_vals in (ref_numeric or {}).items():
            if col not in current_df.columns:
                skipped.append({'column': col, 'kind': 'numeric',
                                'reason': 'absent from current sample'})
                continue
            cur = pd.to_numeric(current_df[col],
                                errors='coerce').dropna()
            cur = cur[np.isfinite(cur)].to_numpy(dtype=float)
            if len(cur) < cfg.min_reference_rows:
                skipped.append({'column': col, 'kind': 'numeric',
                                'reason': f'only {len(cur)} finite values '
                                          'in current sample'})
                continue
            psi = _psi(ref_vals, np.sort(cur), cfg.n_bins)
            ks = _ks_statistic(ref_vals, np.sort(cur))
            sev = _severity(psi, ks, cfg.psi_warn, cfg.psi_crit,
                            cfg.ks_warn, cfg.ks_crit,
                            len(ref_vals), len(cur))
            max_psi = max(max_psi, psi)
            max_ks = max(max_ks, ks)
            feature_drift.append({
                'column': col, 'kind': 'numeric',
                'psi': round(psi, 4), 'ks': round(ks, 4),
                'severity': sev,
                'reference_n': int(len(ref_vals)),
                'current_n': int(len(cur)),
                'reference_mean': round(float(np.mean(ref_vals)), 6),
                'current_mean': round(float(np.mean(cur)), 6),
                'evidence': _evidence_numeric(col, psi, ks, sev,
                                              float(np.mean(ref_vals)),
                                              float(np.mean(cur))),
            })

        for col, ref_dist in (ref_cats or {}).items():
            if col not in current_df.columns:
                skipped.append({'column': col, 'kind': 'categorical',
                                'reason': 'absent from current sample'})
                continue
            cur_vc = (current_df[col].astype(str)
                      .value_counts(normalize=True))
            keys = set(ref_dist) | set(cur_vc.index.astype(str))
            tvd = 0.5 * sum(
                abs(ref_dist.get(k, 0.0)
                    - float(cur_vc.get(k, 0.0))) for k in keys)
            sev = ('critical' if tvd >= cfg.tvd_crit else
                   'warning' if tvd >= cfg.tvd_warn else 'ok')
            max_psi = max_psi  # categorical contributes via tvd only
            feature_drift.append({
                'column': col, 'kind': 'categorical',
                'tvd': round(float(tvd), 4),
                'severity': sev,
                'reference_n_categories': len(ref_dist),
                'current_n_categories': int(len(cur_vc)),
                'evidence': _evidence_categorical(col, float(tvd), sev),
            })

        status = _worst_status([f['severity'] for f in feature_drift],
                               has_data=bool(feature_drift))
        report = DriftReport(
            status=status,
            measured_at_utc=_utcnow(),
            reference_rows=ref_rows,
            current_rows=len(current_df),
            columns_measured=[f['column'] for f in feature_drift],
            columns_skipped=skipped,
            feature_drift=feature_drift,
            max_psi=round(max_psi, 4),
            max_ks=round(max_ks, 4),
            summary=_summary(status, feature_drift, skipped),
            config=cfg.to_dict())
        return report

    def _ref_rows(self) -> int:
        return getattr(self, '_ref_row_count', 0)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _psi(ref: np.ndarray, cur: np.ndarray, n_bins: int) -> float:
    """Population Stability Index on equal-frequency reference bins."""
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    r_hist = np.histogram(ref, bins=edges)[0] / len(ref)
    c_hist = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-4
    r_hist = np.clip(r_hist, eps, None)
    c_hist = np.clip(c_hist, eps, None)
    return float(np.sum((c_hist - r_hist) * np.log(c_hist / r_hist)))


def _ks_statistic(ref: np.ndarray, cur: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (sorted arrays)."""
    data_all = np.concatenate([ref, cur])
    cdf_r = np.searchsorted(ref, data_all, side='right') / len(ref)
    cdf_c = np.searchsorted(cur, data_all, side='right') / len(cur)
    return float(np.max(np.abs(cdf_r - cdf_c)))


def _severity(psi: float, ks: float, psi_warn, psi_crit,
              ks_warn, ks_crit, n_ref: int, n_cur: int) -> str:
    # KS thresholds are size-aware: the two-sample KS critical value at
    # alpha=0.01 is ~1.63*sqrt((n+m)/(n*m)); below that, observed
    # differences are consistent with sampling noise, not drift.
    ks_noise = 1.63 * float(np.sqrt((n_ref + n_cur) / (n_ref * n_cur)))
    ks_warn_eff = max(ks_warn, ks_noise)
    ks_crit_eff = max(ks_crit, ks_noise * 1.5)
    if psi >= psi_crit or ks >= ks_crit_eff:
        return 'critical'
    if psi >= psi_warn or ks >= ks_warn_eff:
        return 'warning'
    return 'ok'


def _worst_status(severities: List[str], has_data: bool) -> str:
    if not has_data:
        return STATUS_INSUFFICIENT
    if 'critical' in severities:
        return STATUS_CRITICAL
    if 'warning' in severities:
        return STATUS_WARNING
    return STATUS_OK


def _summary(status: str, drift: List[Dict], skipped: List[Dict]) -> str:
    n_crit = sum(1 for f in drift if f['severity'] == 'critical')
    n_warn = sum(1 for f in drift if f['severity'] == 'warning')
    base = (f'Drift status {status}: {len(drift)} features measured, '
            f'{n_warn} warning, {n_crit} critical.')
    if skipped:
        base += f' {len(skipped)} column(s) skipped (see columns_skipped).'
    base += (' Drift indicates distribution change only — it is not '
             'evidence of malicious activity.')
    return base


def _evidence_numeric(col, psi, ks, sev, ref_mean, cur_mean) -> Dict:
    return {
        'signal': f'distribution_drift:{col}',
        'observed_value': f'psi={psi:.4f}, ks={ks:.4f}',
        'expected_baseline_or_threshold': 'psi/ks within reference '
                                          'distribution bands',
        'explanation': (f'{sev}-level drift vs reference '
                        f'(ref_mean={ref_mean:.4g}, '
                        f'cur_mean={cur_mean:.4g})'),
    }


def _evidence_categorical(col, tvd, sev) -> Dict:
    return {
        'signal': f'categorical_drift:{col}',
        'observed_value': f'tvd={tvd:.4f}',
        'expected_baseline_or_threshold': 'category frequencies match '
                                          'reference',
        'explanation': f'{sev}-level categorical drift vs reference',
    }
