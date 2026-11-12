"""
SessionAnalyzer — thin inference wrapper for SessionBaselineManager.

Mirrors BehavioralAnalyzer: fit once on training-normal data, persist,
never mutate during inference. Adds causal per-event session deviation
scores and retrospective session-summary scores.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd

from core.session_baseline import SessionBaselineManager, get_session_baseline_path
from core.session_reconstruction import (
    SessionConfig,
    SessionReconstructor,
    SESSION_PREFIX_FEATURES,
)


class SessionAnalyzer:
    def __init__(self,
                 baseline_manager: Optional[SessionBaselineManager] = None,
                 config: Optional[SessionConfig] = None):
        self.baseline_manager = baseline_manager or SessionBaselineManager()
        self.config = config or SessionConfig.from_dict(self.baseline_manager.session_config)

    def is_fitted(self) -> bool:
        return self.baseline_manager.is_fitted()

    def fit(self, df: pd.DataFrame,
            config: Optional[SessionConfig] = None) -> 'SessionAnalyzer':
        """Fit on TRAINING-normal data only."""
        cfg = config or self.config
        self.config = cfg
        self.baseline_manager.fit(df, config=cfg)
        return self

    def _require_fitted(self):
        if not self.baseline_manager.is_fitted():
            raise RuntimeError(
                'Session baselines are not fitted. Call fit() on designated '
                'training-normal data first, or load() a persisted artifact.')

    def _ensure_prefix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'session_prefix_event_count' in df.columns and 'session_id' in df.columns:
            return df
        recon = SessionReconstructor(self.config)
        return recon.reconstruct(df)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add causal session deviation columns to an events DataFrame."""
        self._require_fitted()
        enriched = self._ensure_prefix_features(df)
        return self.baseline_manager.transform_events(enriched)

    def analyze_sessions(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (events_with_scores, scored_session_summaries)."""
        events = self.analyze(df)
        recon = SessionReconstructor(self.config)
        sessions = recon.session_summaries(events)
        scored = self.baseline_manager.transform_sessions(sessions)
        return events, scored

    def save(self, path: Optional[str] = None) -> str:
        path = path or get_session_baseline_path()
        self.baseline_manager.save(path)
        return path

    def load(self, path: Optional[str] = None) -> 'SessionAnalyzer':
        path = path or get_session_baseline_path()
        self.baseline_manager.load(path)
        self.config = SessionConfig.from_dict(self.baseline_manager.session_config)
        return self

    @classmethod
    def from_baseline_path(cls, path: Optional[str] = None) -> 'SessionAnalyzer':
        analyzer = cls()
        return analyzer.load(path)


def get_session_analyzer_path() -> str:
    return get_session_baseline_path()


def ensure_fitted_session_analyzer(path: Optional[str] = None,
                                   train_df: Optional[pd.DataFrame] = None
                                   ) -> SessionAnalyzer:
    """Load persisted session baselines, or fit on training-normal data.

    Lifecycle policy (AUD-002, see core/artifact_lifecycle.py):
    - loads cleanly            -> artifact_status = 'loaded'
    - missing                  -> fit + persist, 'refit_fallback_missing'
    - corrupt/incompatible     -> quarantine corrupt file, fit + persist
                                  fresh, 'refit_fallback_corrupt'

    ``train_df`` is used only when an artifact must be (re)fit; it must be
    designated training-normal data. Once a usable artifact exists it is
    loaded and ``train_df`` is ignored.
    """
    from core.artifact_lifecycle import (
        STATUS_LOADED, STATUS_REFIT_MISSING, STATUS_REFIT_CORRUPT,
        quarantine_corrupt)

    path = path or get_session_analyzer_path()
    if os.path.exists(path):
        try:
            analyzer = SessionAnalyzer.from_baseline_path(path)
            analyzer.baseline_manager.artifact_status = STATUS_LOADED
            return analyzer
        except Exception as e:
            quarantine = quarantine_corrupt(path)
            print(f"Failed to load session baselines: {e}. "
                  f"Quarantined corrupt artifact to {quarantine}; "
                  "refitting on training-normal data...")
            if train_df is None:
                from core.log_processor import load_sample_dataset
                train_df = load_sample_dataset('normal')
            analyzer = SessionAnalyzer().fit(train_df)
            analyzer.save(path)
            analyzer.baseline_manager.artifact_status = STATUS_REFIT_CORRUPT
            return analyzer
    if train_df is None:
        from core.log_processor import load_sample_dataset
        train_df = load_sample_dataset('normal')
    analyzer = SessionAnalyzer().fit(train_df)
    analyzer.save(path)
    analyzer.baseline_manager.artifact_status = STATUS_REFIT_MISSING
    return analyzer
