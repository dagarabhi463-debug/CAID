"""
Behavioral Baseline Manager for CAID (Milestone 1.5.4)

Persistent per-user and global behavioral baselines used by
BehavioralAnalyzer. Mirrors the FeatureBaselineManager /
AnomalyDetector reliability architecture from Milestones 1.5.2/1.5.3.

ARCHITECTURE:
- fit(): compute per-user and global statistics on TRAINING data only
- score_row()/transform(): inference uses stored baselines; NEVER refits
- save()/load(): persisted joblib artifact with schema validation
- evaluate_chronological(): expanding-window research evaluation where each
  event is scored only against strictly-prior events

GUARDS:
- Missing/non-finite input values are handled explicitly (never silently
  treated as normal activity).
- Zero-variance features use a safe std floor to avoid division errors.
- Users absent from training data fall back to global statistics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# Canonical ordered feature definitions. Order is preserved in the persisted
# artifact so feature semantics cannot drift between fit and inference.
BEHAVIORAL_FEATURES: List[str] = [
    'api_requests',
    'data_mb',
    'courses_accessed',
    'accounts_accessed',
    'token_age_minutes',
    'after_hours',
]

# Feature weights for the deviation score (must sum to 1.0).
# after_hours is categorical (0/1) and handled separately.
FEATURE_WEIGHTS: Dict[str, float] = {
    'api_requests': 0.25,
    'data_mb': 0.25,
    'courses_accessed': 0.20,
    'accounts_accessed': 0.20,
    'after_hours': 0.10,
}

# token_age_minutes is tracked in baselines but currently contributes no
# direct weight (kept for schema stability and future use).
BASELINE_STAT_FEATURES: List[str] = BEHAVIORAL_FEATURES

# Maps raw column names to legacy baseline stat suffixes, preserving the
# original BehavioralAnalyzer schema (avg_api, std_data_mb, ...).
STAT_KEY: Dict[str, str] = {
    'api_requests': 'api',
    'data_mb': 'data_mb',
    'courses_accessed': 'courses',
    'accounts_accessed': 'accounts',
    'token_age_minutes': 'token_age',
    'after_hours': 'after_hours',
}


def _avg_key(col: str) -> str:
    return f'avg_{STAT_KEY[col]}'


def _std_key(col: str) -> str:
    return f'std_{STAT_KEY[col]}'


class BehavioralBaselineManager:
    """
    Fit-once behavioral baseline store with persistence.

    Baselines are computed exclusively on designated training data.
    Inference never updates them.
    """

    ARTIFACT_VERSION = 1

    def __init__(self):
        self.user_baselines: Dict[str, Dict[str, float]] = {}
        self.global_stats: Dict[str, float] = {}
        self.feature_columns: List[str] = list(BEHAVIORAL_FEATURES)
        self.thresholds: Dict[str, float] = {
            'api_deviation_multiplier': 3.0,
            'data_deviation_multiplier': 3.0,
            'courses_deviation_multiplier': 3.0,
            'accounts_deviation_multiplier': 3.0,
            'timing_deviation_threshold': 2.0,
            'role_anomaly_multiplier': 2.5,
        }
        self._is_fitted = False
        self._fit_event_count = 0
        # Ephemeral runtime status set by ensure_fitted_baseline:
        # 'loaded' | 'refit_fallback_missing' | 'refit_fallback_corrupt'
        self.artifact_status: Optional[str] = None

    @staticmethod
    def _safe_std(value) -> float:
        """Return std, replacing NaN/zero/negative with 1.0 (zero-variance guard)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return v

    @staticmethod
    def _safe_mean(value) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return v if np.isfinite(v) else 0.0

    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, df: pd.DataFrame) -> 'BehavioralBaselineManager':
        """
        Fit baselines on TRAINING data only.

        Args:
            df: Training DataFrame (e.g. normal_activity.csv). Must NOT
                contain evaluation/inference events.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If DataFrame is empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("Cannot fit behavioral baselines on an empty DataFrame")

        df = df.copy()

        # Normalize key column
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].fillna('unknown').astype(str)
        else:
            df['user_id'] = 'unknown'

        # Coerce numeric features; missing/non-finite -> 0 for baseline stats
        for col in BASELINE_STAT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

        # Per-user baselines
        self.user_baselines = {}
        for user_id, user_data in df.groupby('user_id'):
            baseline = {}
            for col in BASELINE_STAT_FEATURES:
                baseline[_avg_key(col)] = self._safe_mean(user_data[col].mean())
                baseline[_std_key(col)] = self._safe_std(user_data[col].std())
            baseline['event_count'] = int(len(user_data))
            baseline['role'] = (
                str(user_data['role'].iloc[0])
                if 'role' in user_data.columns and len(user_data) else 'unknown'
            )
            self.user_baselines[str(user_id)] = baseline

        # Global fallback statistics (pooled across all training users)
        self.global_stats = {}
        for col in BASELINE_STAT_FEATURES:
            self.global_stats[_avg_key(col)] = self._safe_mean(df[col].mean())
            self.global_stats[_std_key(col)] = self._safe_std(df[col].std())
        self.global_stats['event_count'] = int(len(df))
        self.global_stats['user_count'] = int(df['user_id'].nunique())

        self._fit_event_count = int(len(df))
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _baseline_for_user(self, user_id) -> Dict[str, float]:
        """Return the user's baseline, or global stats for unseen users."""
        uid = str(user_id) if user_id is not None else 'unknown'
        return self.user_baselines.get(uid, self.global_stats)

    def _feature_value(self, row, col) -> Tuple[Optional[float], bool]:
        """
        Extract a numeric feature value from a row.

        Returns (value, is_valid). Missing/non-finite -> (None, False).
        """
        if col not in row.index:
            return None, False
        try:
            v = float(row[col])
        except (TypeError, ValueError):
            return None, False
        if not np.isfinite(v):
            return None, False
        return v, True

    def _calculate_deviation(self, value: float, baseline: float,
                             std: float, threshold: float) -> float:
        """
        Normalized deviation in [0, 1]; 0 when within `threshold` stds.
        """
        if std <= 0:
            std = 1.0
        if baseline == 0:
            # Zero baseline: any positive deviation is maximal; equal is normal
            return 1.0 if value > baseline else 0.0
        z_score = abs(value - baseline) / std
        if z_score > threshold:
            return float(np.clip((z_score - threshold) / threshold, 0.0, 1.0))
        return 0.0

    def score_row(self, row: pd.Series,
                  thresholds: Optional[Dict[str, float]] = None
                  ) -> Tuple[float, List[Dict], bool]:
        """
        Score a single event against stored baselines.

        Args:
            row: Single event row (raw security-log fields).
            thresholds: Optional threshold overrides (defaults to fitted ones).

        Returns:
            (score, factors, missing) where:
            - score: behavioral_deviation_score in [0, 1]; NaN if missing
            - factors: list of deviation factor dicts
            - missing: True when no usable baseline/input existed
        """
        if not self._is_fitted:
            return float('nan'), [], True

        thr = thresholds or self.thresholds
        user_id = row.get('user_id', 'unknown')
        baseline = self._baseline_for_user(user_id)

        # Map feature -> (baseline key suffix, threshold key)
        spec = [
            ('api_requests', 'api_deviation_multiplier', 0.25),
            ('data_mb', 'data_deviation_multiplier', 0.25),
            ('courses_accessed', 'courses_deviation_multiplier', 0.20),
            ('accounts_accessed', 'accounts_deviation_multiplier', 0.20),
        ]

        score = 0.0
        factors = []
        usable_features = 0

        for col, thr_key, weight in spec:
            value, valid = self._feature_value(row, col)
            if not valid:
                factors.append({
                    'factor': f'{col} Deviation',
                    'value': None,
                    'baseline': baseline.get(_avg_key(col)),
                    'deviation_score': 0.0,
                    'weight': weight,
                    'impact': 'missing',
                    'missing': True,
                })
                continue
            usable_features += 1
            mean = baseline.get(_avg_key(col), 0.0)
            std = baseline.get(_std_key(col), 1.0)
            deviation = self._calculate_deviation(
                value=value, baseline=mean, std=std,
                threshold=thr.get(thr_key, 3.0))
            if deviation > 0:
                score += deviation * weight
                factors.append({
                    'factor': f'{col} Deviation',
                    'value': value,
                    'baseline': mean,
                    'deviation_score': round(deviation, 3),
                    'weight': weight,
                    'impact': 'positive',
                })

        # after_hours categorical factor
        ah_value, ah_valid = self._feature_value(row, 'after_hours')
        if ah_valid and ah_value == 1:
            score += FEATURE_WEIGHTS['after_hours']
            factors.append({
                'factor': 'After Hours Activity',
                'value': ah_value,
                'weight': FEATURE_WEIGHTS['after_hours'],
                'impact': 'positive',
            })
            usable_features += 1
        elif not ah_valid:
            factors.append({
                'factor': 'After Hours Activity',
                'value': None,
                'weight': FEATURE_WEIGHTS['after_hours'],
                'impact': 'missing',
                'missing': True,
            })

        # Missing result: no fitted baseline or no usable numeric input.
        # Represent explicitly as NaN rather than silently "normal".
        if usable_features == 0:
            return float('nan'), factors, True

        return float(np.clip(score, 0.0, 1.0)), factors, False

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score every row of a DataFrame using stored baselines.

        Does NOT fit or update baselines.

        Args:
            df: Events to score (raw security-log columns).

        Returns:
            Copy of df with columns:
            - behavioral_deviation_score: float in [0,1], NaN when missing
            - behavioral_factors: list of factor dicts
            - behavioral_score_missing: bool
        """
        if df is None:
            raise ValueError("Cannot transform None DataFrame")
        df = df.copy()
        if df.empty:
            df['behavioral_deviation_score'] = pd.Series(dtype=float)
            df['behavioral_factors'] = pd.Series(dtype=object)
            df['behavioral_score_missing'] = pd.Series(dtype=bool)
            return df

        results = [self.score_row(row) for _, row in df.iterrows()]
        df['behavioral_deviation_score'] = [r[0] for r in results]
        df['behavioral_factors'] = [r[1] for r in results]
        df['behavioral_score_missing'] = [r[2] for r in results]
        return df

    # ------------------------------------------------------------------
    # Chronological (expanding-window) evaluation
    # ------------------------------------------------------------------

    def evaluate_chronological(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        min_history: int = 3,
    ) -> pd.DataFrame:
        """
        Chronological event detection via expanding windows.

        Events are sorted by timestamp. Each event is scored against a
        baseline computed ONLY from that user's strictly-prior events in
        this DataFrame. Users with fewer than `min_history` prior events
        fall back to global statistics computed from prior events only.
        If timestamps are missing/unparseable, the fitted baseline is used
        (batch mode) and the row is flagged via 'chronological' = False.

        This is RESEARCH evaluation only — it is not the same as production
        streaming detection, where baselines would come from a persisted
        store updated out-of-band.

        Args:
            df: Events with timestamps.
            timestamp_col: Timestamp column name.
            min_history: Minimum prior events before using per-user stats.

        Returns:
            DataFrame with behavioral_deviation_score,
            behavioral_factors, behavioral_score_missing, chronological.
        """
        if df is None or len(df) == 0:
            return self.transform(df if df is not None else pd.DataFrame())

        df = df.copy()
        if timestamp_col in df.columns:
            ts = pd.to_datetime(df[timestamp_col], errors='coerce')
        else:
            ts = pd.Series(pd.NaT, index=df.index)

        # Normalize user_id
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].fillna('unknown').astype(str)
        else:
            df['user_id'] = 'unknown'

        for col in BASELINE_STAT_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)

        # Process in timestamp order; ties keep original order
        order = np.lexsort((np.arange(len(df)), ts.fillna(pd.Timestamp.max).values))
        df_sorted = df.iloc[order].copy()
        ts_sorted = ts.iloc[order]

        # Running per-user history and global history
        user_hist: Dict[str, List[Dict[str, float]]] = {}
        global_hist: List[Dict[str, float]] = []

        scores, factors_list, missing_list, chrono_list = [], [], [], []

        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            uid = str(row['user_id'])
            has_ts = pd.notna(ts_sorted.iloc[i])

            if has_ts:
                hist = user_hist.get(uid, [])
                if len(hist) >= min_history:
                    arr = pd.DataFrame(hist)
                    baseline = {
                        _avg_key(c): self._safe_mean(arr[c].mean()) for c in BASELINE_STAT_FEATURES
                    }
                    baseline.update({
                        _std_key(c): self._safe_std(arr[c].std())
                        for c in BASELINE_STAT_FEATURES
                    })
                    baseline['event_count'] = len(hist)
                elif len(global_hist) >= min_history:
                    arr = pd.DataFrame(global_hist)
                    baseline = {
                        _avg_key(c): self._safe_mean(arr[c].mean()) for c in BASELINE_STAT_FEATURES
                    }
                    baseline.update({
                        _std_key(c): self._safe_std(arr[c].std())
                        for c in BASELINE_STAT_FEATURES
                    })
                    baseline['event_count'] = len(global_hist)
                else:
                    # Insufficient history anywhere: use fitted baseline if any
                    baseline = self._baseline_for_user(uid) if self._is_fitted else None
                    if baseline is None:
                        # No fitted baseline and no history: cannot score
                        scores.append(float('nan'))
                        factors_list.append([])
                        missing_list.append(True)
                        chrono_list.append(False)
                        _append_history(user_hist, global_hist, uid, row)
                        continue
                chrono = True
            else:
                baseline = self._baseline_for_user(uid) if self._is_fitted else self.global_stats
                if not self._is_fitted and not self.global_stats:
                    scores.append(float('nan'))
                    factors_list.append([])
                    missing_list.append(True)
                    chrono_list.append(False)
                    _append_history(user_hist, global_hist, uid, row)
                    continue
                chrono = False

            score, facs, miss = self._score_against_baseline(row, baseline)
            scores.append(score)
            factors_list.append(facs)
            missing_list.append(miss)
            chrono_list.append(chrono)

            # Append strictly AFTER scoring (no self-inclusion)
            _append_history(user_hist, global_hist, uid, row)

        df_sorted['behavioral_deviation_score'] = scores
        df_sorted['behavioral_factors'] = factors_list
        df_sorted['behavioral_score_missing'] = missing_list
        df_sorted['chronological'] = chrono_list

        # Restore original row order
        out = df_sorted.sort_index()
        for col in ('behavioral_deviation_score', 'behavioral_factors',
                    'behavioral_score_missing', 'chronological'):
            df[col] = out[col].values if len(out) == len(df) else out[col]
        return df

    def _score_against_baseline(self, row: pd.Series,
                                baseline: Dict[str, float]
                                ) -> Tuple[float, List[Dict], bool]:
        """Score one row against an explicit baseline dict (chronological use)."""
        spec = [
            ('api_requests', 'api_deviation_multiplier', 0.25),
            ('data_mb', 'data_deviation_multiplier', 0.25),
            ('courses_accessed', 'courses_deviation_multiplier', 0.20),
            ('accounts_accessed', 'accounts_deviation_multiplier', 0.20),
        ]
        thr = self.thresholds
        score = 0.0
        factors = []
        usable = 0
        for col, thr_key, weight in spec:
            value, valid = self._feature_value(row, col)
            if not valid:
                factors.append({
                    'factor': f'{col} Deviation', 'value': None,
                    'baseline': baseline.get(_avg_key(col)),
                    'deviation_score': 0.0, 'weight': weight,
                    'impact': 'missing', 'missing': True,
                })
                continue
            usable += 1
            mean = baseline.get(_avg_key(col), 0.0)
            std = baseline.get(_std_key(col), 1.0)
            dev = self._calculate_deviation(value, mean, std, thr.get(thr_key, 3.0))
            if dev > 0:
                score += dev * weight
                factors.append({
                    'factor': f'{col} Deviation', 'value': value,
                    'baseline': mean, 'deviation_score': round(dev, 3),
                    'weight': weight, 'impact': 'positive',
                })
        ah, ah_valid = self._feature_value(row, 'after_hours')
        if ah_valid and ah == 1:
            score += FEATURE_WEIGHTS['after_hours']
            factors.append({'factor': 'After Hours Activity', 'value': ah,
                            'weight': FEATURE_WEIGHTS['after_hours'],
                            'impact': 'positive'})
            usable += 1
        elif not ah_valid:
            factors.append({'factor': 'After Hours Activity', 'value': None,
                            'weight': FEATURE_WEIGHTS['after_hours'],
                            'impact': 'missing', 'missing': True})
        if usable == 0:
            return float('nan'), factors, True
        return float(np.clip(score, 0.0, 1.0)), factors, False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist baselines to a joblib artifact."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted behavioral baselines")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            'artifact_version': self.ARTIFACT_VERSION,
            'user_baselines': self.user_baselines,
            'global_stats': self.global_stats,
            'feature_columns': self.feature_columns,
            'thresholds': self.thresholds,
            'fit_event_count': self._fit_event_count,
        }, path)

    def load(self, path: str) -> 'BehavioralBaselineManager':
        """
        Restore baselines from a saved artifact.

        Raises:
            FileNotFoundError, ValueError on missing/invalid artifacts.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Baseline artifact not found: {path}")
        data = joblib.load(path)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid baseline artifact at {path}")
        if data.get('artifact_version') != self.ARTIFACT_VERSION:
            raise ValueError(
                f"Incompatible baseline artifact version "
                f"{data.get('artifact_version')!r} at {path}")
        required = ('user_baselines', 'global_stats', 'feature_columns',
                    'thresholds', 'fit_event_count')
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(
                f"Incomplete baseline artifact at {path}: missing {missing}")
        if data['feature_columns'] != self.feature_columns:
            raise ValueError(
                f"Baseline artifact feature schema mismatch at {path}: "
                f"{data['feature_columns']} != {self.feature_columns}")

        self.user_baselines = data['user_baselines']
        self.global_stats = data['global_stats']
        self.thresholds = data['thresholds']
        self._fit_event_count = data['fit_event_count']
        self._is_fitted = True
        return self


def _append_history(user_hist, global_hist, uid, row):
    """Append one event's feature vector to running histories."""
    vec = {c: float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
           for c in BASELINE_STAT_FEATURES}
    user_hist.setdefault(uid, []).append(vec)
    global_hist.append(vec)


def get_baseline_path() -> str:
    """Default path for the persisted behavioral baseline artifact."""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'saved_models', 'behavioral_baselines.joblib'
    )


def ensure_fitted_baseline(path: Optional[str] = None) -> BehavioralBaselineManager:
    """
    Load persisted behavioral baselines, training on the normal-activity
    baseline dataset if no usable artifact exists.

    Lifecycle policy (AUD-002, see core/artifact_lifecycle.py):
    - loads cleanly            -> artifact_status = 'loaded'
    - missing                  -> fit + persist, 'refit_fallback_missing'
    - corrupt/incompatible     -> quarantine corrupt file, fit + persist
                                  fresh, 'refit_fallback_corrupt'

    Returns:
        Fitted BehavioralBaselineManager with ``artifact_status`` set.
    """
    from core.artifact_lifecycle import (
        STATUS_LOADED, STATUS_REFIT_MISSING, STATUS_REFIT_CORRUPT,
        quarantine_corrupt)

    path = path or get_baseline_path()
    mgr = BehavioralBaselineManager()
    if os.path.exists(path):
        try:
            mgr.load(path)
            mgr.artifact_status = STATUS_LOADED
            return mgr
        except Exception as e:
            quarantine = quarantine_corrupt(path)
            print(f"Failed to load behavioral baselines: {e}. "
                  f"Quarantined corrupt artifact to {quarantine}; "
                  "retraining on normal baseline data...")
            from core.log_processor import load_sample_dataset
            df_normal = load_sample_dataset('normal')
            mgr.fit(df_normal)
            mgr.save(path)
            mgr.artifact_status = STATUS_REFIT_CORRUPT
            return mgr
    from core.log_processor import load_sample_dataset
    df_normal = load_sample_dataset('normal')
    mgr.fit(df_normal)
    mgr.save(path)
    mgr.artifact_status = STATUS_REFIT_MISSING
    return mgr
