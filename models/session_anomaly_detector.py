"""
SessionAwareAnomalyDetector — optional Milestone 2.0 ML experiment.

Extends AnomalyDetector (artifact v2) with causal session-prefix features.
This is a SEPARATE experiment artifact (version 3); it does not replace the
production/v2.0 baseline detector.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from models.anomaly_detector import AnomalyDetector
from core.session_reconstruction import (
    SessionConfig,
    SessionReconstructor,
    SESSION_PREFIX_FEATURES,
)


class SessionAwareAnomalyDetector(AnomalyDetector):
    """IsolationForest with additional causal session-prefix features."""

    ARTIFACT_VERSION = 3

    def __init__(self, contamination: float = 0.1, random_state: int = 42,
                 session_config: Optional[SessionConfig] = None):
        super().__init__(contamination=contamination, random_state=random_state)
        self.session_config = session_config or SessionConfig()
        self.session_feature_columns: List[str] = list(SESSION_PREFIX_FEATURES)
        self._extra_artifact_state = {}

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if 'session_prefix_event_count' in out.columns and 'session_id' in out.columns:
            return out
        recon = SessionReconstructor(self.session_config)
        return recon.reconstruct(out)

    def _numeric_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in SESSION_PREFIX_FEATURES:
            if col not in out.columns:
                out[col] = 0.0
            out[col] = pd.to_numeric(out[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame,
            feature_columns: Optional[List[str]] = None) -> 'SessionAwareAnomalyDetector':
        prepared = self._numeric_prefix(self._prepare(df))
        # Fit base engineered features first to learn detected schema.
        super().fit(prepared, feature_columns=feature_columns)
        # Append session prefix features to the model schema.
        base_cols = list(self.feature_columns)
        session_cols = [c for c in SESSION_PREFIX_FEATURES
                        if c in prepared.columns]
        self.feature_columns = base_cols + session_cols
        self.session_feature_columns = session_cols
        # Refit on the augmented matrix so scaler/model see all columns.
        from core.feature_engineering import engineer_features, get_feature_matrix
        df_engineered, _ = engineer_features(prepared, self.baseline_manager,
                                             leave_one_out=True)
        X = get_feature_matrix(df_engineered, self.feature_columns)
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        self._scaler_fitted = True
        train_scores = self.model.decision_function(X_scaled)
        self._score_calibration = {
            'score_min': float(np.percentile(train_scores, 1)),
            'score_max': float(np.percentile(train_scores, 99)),
        }
        if self._score_calibration['score_max'] <= self._score_calibration['score_min']:
            self._score_calibration['score_max'] = self._score_calibration['score_min'] + 1e-9
        self._extra_artifact_state = {
            'session_config': self.session_config.to_dict(),
            'session_feature_columns': self.session_feature_columns,
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = self._numeric_prefix(self._prepare(df))
        return super().predict(prepared)

    # ------------------------------------------------------------------
    # Persistence extras
    # ------------------------------------------------------------------

    def save(self, model_path: str) -> None:
        self._extra_artifact_state = {
            'session_config': self.session_config.to_dict(),
            'session_feature_columns': self.session_feature_columns,
        }
        super().save(model_path)

    def load(self, model_path: str) -> 'SessionAwareAnomalyDetector':
        super().load(model_path)
        state = getattr(self, '_extra_artifact_state', {}) or {}
        self.session_config = SessionConfig.from_dict(state.get('session_config'))
        self.session_feature_columns = list(state.get('session_feature_columns',
                                                       SESSION_PREFIX_FEATURES))
        return self
