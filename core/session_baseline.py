"""
SessionBaselineManager for CAID — Milestone 2.0

Persisted, training-only baselines for causal session-prefix features and
completed-session summaries. Mirrors BehavioralBaselineManager guarantees:

- fit() uses designated training-normal data only.
- transform()/score_row() never refit or mutate baselines.
- unseen users fall back to global statistics.
- missing/zero-variance inputs are explicit, never silent-normal.
- save()/load() are deterministic and schema-validated.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.session_reconstruction import (
    SessionConfig,
    SessionReconstructor,
    SESSION_PREFIX_FEATURES,
    SESSION_SUMMARY_FEATURES,
)


# Canonical session feature names (without the session_prefix_/session_
# prefix). These are the semantic features compared against baselines.
CANONICAL_SESSION_FEATURES: List[str] = [
    'event_count',
    'duration_seconds',
    'event_rate',
    'unique_event_types',
    'api_requests',
    'data_volume_mb',
    'unique_ips',
    'ip_switch_count',
    'sensitive_event_count',
    'after_hours_count',
    'auth_failures',
]

SESSION_FEATURE_WEIGHTS: Dict[str, float] = {
    'event_count': 0.12,
    'duration_seconds': 0.08,
    'event_rate': 0.12,
    'unique_event_types': 0.08,
    'api_requests': 0.10,
    'data_volume_mb': 0.14,
    'unique_ips': 0.08,
    'ip_switch_count': 0.06,
    'sensitive_event_count': 0.12,
    'after_hours_count': 0.05,
    'auth_failures': 0.05,
}

_PREFIX_MAP = {name: f'session_prefix_{name}' for name in CANONICAL_SESSION_FEATURES}
_SUMMARY_MAP = {name: f'session_{name}' for name in CANONICAL_SESSION_FEATURES}


def _mean_key(name: str) -> str:
    return f'mean_{name}'


def _std_key(name: str) -> str:
    return f'std_{name}'


class SessionBaselineManager:
    """Fit-once session baseline store with persistence."""

    ARTIFACT_VERSION = 1

    def __init__(self):
        self.user_prefix_baselines: Dict[str, Dict[str, float]] = {}
        self.global_prefix_stats: Dict[str, float] = {}
        self.user_session_stats: Dict[str, Dict[str, float]] = {}
        self.global_session_stats: Dict[str, float] = {}
        self.session_config: Dict = SessionConfig().to_dict()
        self.feature_columns: List[str] = list(SESSION_PREFIX_FEATURES)
        self.thresholds: Dict[str, float] = {
            'deviation_multiplier': 3.0,
            'min_prefix_events': 1,
        }
        self._is_fitted = False
        self._fit_event_count = 0
        self._fit_session_count = 0
        # Ephemeral runtime status set by ensure_fitted_session_analyzer:
        # 'loaded' | 'refit_fallback_missing' | 'refit_fallback_corrupt'
        self.artifact_status: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_std(value) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 1.0
        return v if np.isfinite(v) and v > 0 else 1.0

    @staticmethod
    def _safe_mean(value) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return v if np.isfinite(v) else 0.0

    def is_fitted(self) -> bool:
        return self._is_fitted

    @staticmethod
    def _norm_user(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'unknown'
        text = str(value).strip()
        return text if text else 'unknown'

    def _feature_value(self, row, col: str) -> Tuple[Optional[float], bool]:
        try:
            present = col in row.index if hasattr(row, 'index') else col in row
        except Exception:
            present = col in row
        if not present:
            return None, False
        try:
            v = float(row[col] if hasattr(row, 'index') else row.get(col))
        except (TypeError, ValueError):
            return None, False
        if not np.isfinite(v):
            return None, False
        return v, True

    @staticmethod
    def _deviation(value: float, mean: float, std: float, threshold: float) -> float:
        std = SessionBaselineManager._safe_std(std)
        if mean == 0:
            return 1.0 if value > 0 else 0.0
        z = abs(value - mean) / std
        if z > threshold:
            return float(np.clip((z - threshold) / threshold, 0.0, 1.0))
        return 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame,
            config: Optional[SessionConfig] = None) -> 'SessionBaselineManager':
        """Fit baselines on TRAINING-normal data only."""
        if df is None or len(df) == 0:
            raise ValueError('Cannot fit session baselines on an empty DataFrame')

        cfg = config or SessionConfig.from_dict(self.session_config)
        self.session_config = cfg.to_dict()
        recon = SessionReconstructor(cfg)
        events, sessions = recon.reconstruct_and_summarize(df)

        events = events.copy()
        events['_user_key'] = events.get(cfg.user_column, pd.Series('unknown', index=events.index)).map(self._norm_user)
        for name, col in _PREFIX_MAP.items():
            events[col] = pd.to_numeric(events.get(col, 0.0), errors='coerce')

        self.user_prefix_baselines = {}
        for uid, grp in events.groupby('_user_key', sort=True):
            stats = {}
            for name, col in _PREFIX_MAP.items():
                stats[_mean_key(name)] = self._safe_mean(grp[col].mean())
                stats[_std_key(name)] = self._safe_std(grp[col].std())
            stats['event_count'] = int(len(grp))
            stats['session_count'] = int(grp['session_id'].nunique())
            self.user_prefix_baselines[str(uid)] = stats

        self.global_prefix_stats = {}
        for name, col in _PREFIX_MAP.items():
            self.global_prefix_stats[_mean_key(name)] = self._safe_mean(events[col].mean())
            self.global_prefix_stats[_std_key(name)] = self._safe_std(events[col].std())
        self.global_prefix_stats['event_count'] = int(len(events))
        self.global_prefix_stats['session_count'] = int(events['session_id'].nunique())
        self.global_prefix_stats['user_count'] = int(events['_user_key'].nunique())

        # Completed-session stats (retrospective summaries; not used for
        # causal per-event scoring but persisted for session-level eval/UI).
        sessions = sessions.copy()
        sessions['_user_key'] = sessions.get('user_id', pd.Series('unknown', index=sessions.index)).map(self._norm_user)
        for name, col in _SUMMARY_MAP.items():
            if col in sessions.columns:
                sessions[col] = pd.to_numeric(sessions[col], errors='coerce')
            else:
                sessions[col] = 0.0

        self.user_session_stats = {}
        for uid, grp in sessions.groupby('_user_key', sort=True):
            stats = {}
            for name, col in _SUMMARY_MAP.items():
                stats[_mean_key(name)] = self._safe_mean(grp[col].mean())
                stats[_std_key(name)] = self._safe_std(grp[col].std())
            stats['session_count'] = int(len(grp))
            self.user_session_stats[str(uid)] = stats

        self.global_session_stats = {}
        for name, col in _SUMMARY_MAP.items():
            self.global_session_stats[_mean_key(name)] = self._safe_mean(sessions[col].mean())
            self.global_session_stats[_std_key(name)] = self._safe_std(sessions[col].std())
        self.global_session_stats['session_count'] = int(len(sessions))

        self._fit_event_count = int(len(events))
        self._fit_session_count = int(len(sessions))
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _baseline_for_user(self, user_id, kind: str = 'prefix') -> Dict[str, float]:
        uid = self._norm_user(user_id)
        if kind == 'session':
            return self.user_session_stats.get(uid, self.global_session_stats)
        return self.user_prefix_baselines.get(uid, self.global_prefix_stats)

    def _score_vector(self, row: pd.Series, col_map: Dict[str, str],
                      baseline: Dict[str, float]) -> Tuple[float, List[Dict], bool, str]:
        score = 0.0
        factors = []
        usable = 0
        threshold = float(self.thresholds.get('deviation_multiplier', 3.0))
        for name, weight in SESSION_FEATURE_WEIGHTS.items():
            col = col_map.get(name)
            value, valid = (self._feature_value(row, col) if col else (None, False))
            if not valid:
                factors.append({
                    'factor': f'{name} Deviation', 'value': None,
                    'baseline': baseline.get(_mean_key(name)),
                    'deviation_score': 0.0, 'weight': weight,
                    'impact': 'missing', 'missing': True,
                })
                continue
            usable += 1
            mean = baseline.get(_mean_key(name), 0.0)
            std = baseline.get(_std_key(name), 1.0)
            dev = self._deviation(value, mean, std, threshold)
            if dev > 0:
                score += dev * weight
                factors.append({
                    'factor': f'{name} Deviation', 'value': value,
                    'baseline': mean, 'deviation_score': round(dev, 3),
                    'weight': weight, 'impact': 'positive',
                })
        if usable == 0:
            return float('nan'), factors, True, 'insufficient'
        scope = 'user' if self._norm_user(row.get('user_id')) in (
            self.user_prefix_baselines if col_map is _PREFIX_MAP else self.user_session_stats) else 'global'
        return float(np.clip(score, 0.0, 1.0)), factors, False, scope

    def score_row(self, row: pd.Series) -> Tuple[float, List[Dict], bool, str]:
        """Score one event's causal session-prefix vector."""
        if not self._is_fitted:
            return float('nan'), [], True, 'unfitted'
        baseline = self._baseline_for_user(row.get('user_id'), 'prefix')
        return self._score_vector(row, _PREFIX_MAP, baseline)

    def score_session_row(self, row: pd.Series) -> Tuple[float, List[Dict], bool, str]:
        """Score one completed-session summary (retrospective)."""
        if not self._is_fitted:
            return float('nan'), [], True, 'unfitted'
        baseline = self._baseline_for_user(row.get('user_id'), 'session')
        return self._score_vector(row, _SUMMARY_MAP, baseline)

    def transform_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add session deviation columns to an events frame with prefix features."""
        if events_df is None:
            raise ValueError('Cannot transform None DataFrame')
        df = events_df.copy()
        if df.empty:
            for col in ('session_deviation_score', 'session_baseline_scope'):
                df[col] = pd.Series(dtype=object)
            df['session_score_missing'] = pd.Series(dtype=bool)
            df['session_factors'] = pd.Series(dtype=object)
            return df
        results = [self.score_row(row) for _, row in df.iterrows()]
        df['session_deviation_score'] = [r[0] for r in results]
        df['session_factors'] = [r[1] for r in results]
        df['session_score_missing'] = [r[2] for r in results]
        df['session_baseline_scope'] = [r[3] for r in results]
        return df

    def transform_sessions(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Add session deviation columns to completed-session summaries."""
        if sessions_df is None:
            raise ValueError('Cannot transform None DataFrame')
        df = sessions_df.copy()
        if df.empty:
            df['session_deviation_score'] = pd.Series(dtype=float)
            df['session_score_missing'] = pd.Series(dtype=bool)
            df['session_factors'] = pd.Series(dtype=object)
            df['session_baseline_scope'] = pd.Series(dtype=object)
            return df
        results = [self.score_session_row(row) for _, row in df.iterrows()]
        df['session_deviation_score'] = [r[0] for r in results]
        df['session_factors'] = [r[1] for r in results]
        df['session_score_missing'] = [r[2] for r in results]
        df['session_baseline_scope'] = [r[3] for r in results]
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic digest of fitted baseline state for calibration binding."""
        payload = {
            'artifact_version': self.ARTIFACT_VERSION,
            'feature_columns': self.feature_columns,
            'session_config': self.session_config,
            'thresholds': self.thresholds,
            'user_prefix_baselines': self.user_prefix_baselines,
            'global_prefix_stats': self.global_prefix_stats,
            'user_session_stats': self.user_session_stats,
            'global_session_stats': self.global_session_stats,
            'fit_event_count': self._fit_event_count,
            'fit_session_count': self._fit_session_count,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError('Cannot save unfitted session baselines')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            'artifact_version': self.ARTIFACT_VERSION,
            'user_prefix_baselines': self.user_prefix_baselines,
            'global_prefix_stats': self.global_prefix_stats,
            'user_session_stats': self.user_session_stats,
            'global_session_stats': self.global_session_stats,
            'session_config': self.session_config,
            'feature_columns': self.feature_columns,
            'thresholds': self.thresholds,
            'fit_event_count': self._fit_event_count,
            'fit_session_count': self._fit_session_count,
        }, path)

    def load(self, path: str) -> 'SessionBaselineManager':
        if not os.path.exists(path):
            raise FileNotFoundError(f'Session baseline artifact not found: {path}')
        data = joblib.load(path)
        if not isinstance(data, dict):
            raise ValueError(f'Invalid session baseline artifact at {path}')
        if data.get('artifact_version') != self.ARTIFACT_VERSION:
            raise ValueError(
                f'Incompatible session baseline artifact version '
                f'{data.get("artifact_version")!r} at {path}')
        required = ('user_prefix_baselines', 'global_prefix_stats',
                    'user_session_stats', 'global_session_stats',
                    'session_config', 'feature_columns', 'thresholds',
                    'fit_event_count')
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f'Incomplete session baseline artifact at {path}: missing {missing}')
        if data['feature_columns'] != self.feature_columns:
            raise ValueError(
                f'Session baseline feature schema mismatch at {path}: '
                f'{data["feature_columns"]} != {self.feature_columns}')
        self.user_prefix_baselines = data['user_prefix_baselines']
        self.global_prefix_stats = data['global_prefix_stats']
        self.user_session_stats = data['user_session_stats']
        self.global_session_stats = data['global_session_stats']
        self.session_config = SessionConfig.from_dict(data['session_config']).to_dict()
        self.thresholds = data['thresholds']
        self._fit_event_count = data['fit_event_count']
        self._fit_session_count = data.get('fit_session_count', 0)
        self._is_fitted = True
        return self


def get_session_baseline_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'saved_models', 'session_baselines.joblib')
