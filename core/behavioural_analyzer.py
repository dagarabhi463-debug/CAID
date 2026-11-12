"""
Behavioral Analysis Engine for CAID

This module implements behavioral analysis to identify unusual patterns
relative to established baselines.

ARCHITECTURE (Milestone 1.5.4):
- BehavioralBaselineManager: persisted per-user + global baselines fitted
  on designated TRAINING data only (see core/behavioral_baseline.py)
- User Baseline Modeling: normal behavior patterns per user
- Temporal Analysis: chronological expanding-window evaluation available
- Activity Deviation: compare current activity to stored baselines
- Risk Scoring: convert deviations into risk indicators

LEAKAGE GUARANTEE:
- fit() must be called on training data before analyze_user_behavior().
- Inference NEVER refits or updates baselines.
- analyze_user_behavior() raises RuntimeError if baselines are not fitted.

SCORE CONTRACT:
- behavioral_deviation_score: 0-1, HIGHER = MORE DEVIATION (NaN = missing)
- behavioral_score_missing: bool, True = no usable baseline/input
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

from core.behavioral_baseline import (
    BehavioralBaselineManager,
    get_baseline_path,
    ensure_fitted_baseline,
)


class BehavioralAnalyzer:
    """
    Behavioral analysis engine that identifies unusual patterns
    relative to established user baselines.

    Wraps a BehavioralBaselineManager. Baselines are fitted once on
    training data and persisted; inference never mutates them.
    """

    def __init__(self, baseline_manager: Optional[BehavioralBaselineManager] = None):
        """
        Initialize behavioral analysis.

        Args:
            baseline_manager: Optional pre-fitted manager (e.g. loaded from
                a persisted artifact). If None, a fresh unfitted manager is
                created and fit() must be called before inference.
        """
        self.baseline_manager = baseline_manager or BehavioralBaselineManager()

        # Behavioral thresholds (kept on the manager; exposed here for compat)
        self.thresholds = self.baseline_manager.thresholds

    # ------------------------------------------------------------------
    # Backward-compatible views over the baseline manager
    # ------------------------------------------------------------------

    @property
    def user_baselines(self) -> Dict[str, Dict[str, float]]:
        return self.baseline_manager.user_baselines

    @property
    def overall_stats(self) -> Dict[str, float]:
        return self.baseline_manager.global_stats

    # ------------------------------------------------------------------
    # Fitting (training data only)
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> 'BehavioralAnalyzer':
        """
        Fit behavioral baselines on TRAINING data only.

        IMPORTANT: Call this only on designated baseline/training data
        (e.g. normal_activity.csv). Never call it on evaluation or
        inference data — that reintroduces temporal leakage.

        Args:
            df: Training security-log DataFrame

        Returns:
            Self for method chaining
        """
        self.baseline_manager.fit(df)
        # Keep thresholds reference in sync (fit may replace dict contents)
        self.thresholds = self.baseline_manager.thresholds
        return self

    def is_fitted(self) -> bool:
        return self.baseline_manager.is_fitted()

    # ------------------------------------------------------------------
    # Inference (never refits)
    # ------------------------------------------------------------------

    def _require_fitted(self):
        if not self.baseline_manager.is_fitted():
            raise RuntimeError(
                "Behavioral baselines are not fitted. Call fit() on "
                "designated training data first, or load() a persisted "
                "baseline artifact."
            )

    def calculate_behavioral_deviation(self, row: pd.Series) -> Tuple[float, List[Dict]]:
        """
        Calculate behavioral deviation score for a single event.

        Uses stored baselines only — does NOT fit or update them.

        Args:
            row: Single row from security log DataFrame

        Returns:
            Tuple of (deviation_score, deviation_factors)
            deviation_score is NaN when the result is missing.
        """
        self._require_fitted()
        score, factors, _missing = self.baseline_manager.score_row(row)
        return score, factors

    def analyze_user_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze behavioral deviations for all events using STORED baselines.

        This is historical BATCH analysis: baselines are fixed from training
        data and applied to every event. It does NOT perform chronological
        (expanding-window) evaluation — use analyze_user_behavior_chronological()
        for that, and note that production streaming detection would require
        a persisted baseline store updated out-of-band.

        Args:
            df: Security log DataFrame to evaluate (inference data)

        Returns:
            DataFrame with behavioral_deviation_score,
            behavioral_factors, behavioral_score_missing columns

        Raises:
            RuntimeError: If fit() was not called on training data first.
        """
        self._require_fitted()
        return self.baseline_manager.transform(df)

    def analyze_user_behavior_chronological(
        self, df: pd.DataFrame, timestamp_col: str = 'timestamp',
        min_history: int = 3
    ) -> pd.DataFrame:
        """
        Chronological behavioral analysis via expanding windows.

        Each event is scored only against strictly-prior events, so future
        events cannot leak into baselines. See
        BehavioralBaselineManager.evaluate_chronological for semantics.

        Args:
            df: Security log DataFrame with timestamps
            timestamp_col: Timestamp column name
            min_history: Min prior events before per-user stats are used

        Returns:
            DataFrame with deviation columns plus 'chronological' flag
        """
        self._require_fitted()
        return self.baseline_manager.evaluate_chronological(
            df, timestamp_col=timestamp_col, min_history=min_history)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Persist fitted baselines to a joblib artifact.

        Args:
            path: Destination path (defaults to models/saved_models/
                  behavioral_baselines.joblib)

        Returns:
            The path written.
        """
        path = path or get_baseline_path()
        self.baseline_manager.save(path)
        return path

    def load(self, path: Optional[str] = None) -> 'BehavioralAnalyzer':
        """
        Restore fitted baselines from a persisted artifact.

        Args:
            path: Artifact path (defaults to models/saved_models/
                  behavioral_baselines.joblib)

        Returns:
            Self for method chaining
        """
        path = path or get_baseline_path()
        self.baseline_manager.load(path)
        self.thresholds = self.baseline_manager.thresholds
        return self

    @classmethod
    def from_baseline_path(cls, path: Optional[str] = None) -> 'BehavioralAnalyzer':
        """Construct an analyzer with baselines loaded from disk."""
        analyzer = cls()
        return analyzer.load(path)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_suspicious_users(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Identify users with suspicious behavioral patterns.

        Rows with missing behavioral scores (NaN) are excluded.

        Args:
            df: Security log DataFrame with behavioral analysis
            threshold: Deviation threshold for suspicious activity

        Returns:
            DataFrame with suspicious users and their metrics
        """
        if 'behavioral_deviation_score' not in df.columns:
            return pd.DataFrame(columns=['user_id', 'suspicious_event_count',
                                         'max_deviation', 'avg_deviation', 'risk_level'])

        scored = df.dropna(subset=['behavioral_deviation_score'])
        suspicious_events = scored[scored['behavioral_deviation_score'] >= threshold]

        if len(suspicious_events) == 0:
            return pd.DataFrame(columns=['user_id', 'suspicious_event_count',
                                         'max_deviation', 'avg_deviation', 'risk_level'])

        # Aggregate by user
        suspicious_users = suspicious_events.groupby('user_id').agg({
            'behavioral_deviation_score': ['count', 'max', 'mean']
        }).reset_index()

        suspicious_users.columns = ['user_id', 'suspicious_event_count',
                                    'max_deviation', 'avg_deviation']

        # Add risk level based on max deviation
        def get_risk_level(row):
            if row['max_deviation'] >= 0.75:
                return 'HIGH'
            elif row['max_deviation'] >= 0.5:
                return 'MEDIUM'
            else:
                return 'LOW'

        suspicious_users['risk_level'] = suspicious_users.apply(get_risk_level, axis=1)
        suspicious_users = suspicious_users.sort_values('max_deviation', ascending=False)

        return suspicious_users


def get_behavioral_baseline_path() -> str:
    """Default path for the persisted behavioral baseline artifact."""
    return get_baseline_path()


def ensure_fitted_analyzer(path: Optional[str] = None) -> BehavioralAnalyzer:
    """
    Return a BehavioralAnalyzer with persisted baselines, training on the
    normal-activity baseline dataset if no artifact exists.

    Args:
        path: Optional artifact path override.

    Returns:
        Fitted BehavioralAnalyzer ready for inference.
    """
    return BehavioralAnalyzer(ensure_fitted_baseline(path))
