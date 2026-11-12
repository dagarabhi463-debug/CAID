"""
ML Integration Module for CAID

This module provides ML detection capabilities by delegating to the
AnomalyDetector (Isolation Forest + FeatureBaselineManager) artifact.

ARCHITECTURE:
- Model Reuse: Load the persisted AnomalyDetector artifact (v1+)
- Consistent Feature Engineering: predict() uses stored training baselines
- Anomaly Detection: Generates predictions and calibrated scores
- Explicit Failure: Missing model or incompatible data raises/flags errors
  rather than silently returning "all normal"

SCORE CONTRACT (higher = more anomalous unless noted):
- is_anomaly: bool, True = anomalous
- anomaly_score: raw IsolationForest decision_function, HIGHER = MORE NORMAL
- anomaly_score_normalized: 0-1 calibrated score, HIGHER = MORE ANOMALOUS
- ml_anomaly_score: alias of anomaly_score_normalized for downstream engines
- ml_confidence: 0-1 distance-from-neutrality of the normalized score
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

from models.anomaly_detector import AnomalyDetector


class MLIntegration:
    """
    ML integration wrapper that reuses the persisted AnomalyDetector.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ML integration.
        
        Args:
            model_path: Path to saved model artifact (optional, uses default if not provided)
        """
        if model_path is None:
            self.model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'models', 'saved_models', 'anomaly_detector.joblib'
            )
        else:
            self.model_path = model_path
        
        self.detector: Optional[AnomalyDetector] = None
        self.model = None  # backward-compat alias for the loaded detector
        self.feature_columns = None
        self.fitted = False
    
    def load_model(self) -> bool:
        """
        Load the saved AnomalyDetector artifact.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                return False
            detector = AnomalyDetector()
            detector.load(self.model_path)
            self.detector = detector
            self.model = detector
            self.feature_columns = detector.feature_columns
            self.fitted = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.detector = None
            self.model = None
            self.fitted = False
            return False
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ML anomaly detection to a DataFrame.
        
        Args:
            df: Security log DataFrame (raw columns; the detector engineers
                features internally using the persisted training baselines)
            
        Returns:
            DataFrame with ML anomaly predictions.
            If no model artifact is available, fallback columns are added and
            'ml_model_available' is set to False so callers can detect that
            the result is NOT a real model prediction.
        """
        df = df.copy()
        
        # Ensure model is loaded
        if not self.fitted:
            if not self.load_model():
                # Explicit fallback: flag that no real prediction was made
                df['is_anomaly'] = False
                df['anomaly_score'] = np.nan
                df['anomaly_score_normalized'] = np.nan
                df['ml_anomaly_score'] = np.nan
                df['ml_confidence'] = 0.0
                df['ml_model_available'] = False
                return df
        
        try:
            result = self.detector.predict(df)
        except (ValueError, RuntimeError) as e:
            # Do not silently fake normal predictions on incompatible input
            raise ValueError(
                f"ML prediction failed: {e}. Ensure the input contains the "
                "required raw security-log columns."
            ) from e
        
        # Normalize column naming for downstream consumers.
        # anomaly_score_raw / anomaly_score: RAW decision_function
        #   (higher = more normal); anomaly_score is the deprecated alias.
        # anomaly_score_normalized / ml_anomaly_score: 0-1,
        #   higher = more anomalous.
        if 'anomaly_score_raw' in result.columns:
            result['anomaly_score'] = result['anomaly_score_raw']  # deprecated alias
        if 'anomaly_score_normalized' in result.columns:
            result['ml_anomaly_score'] = result['anomaly_score_normalized']
            result['ml_confidence'] = np.abs(result['anomaly_score_normalized'] - 0.5) * 2
        else:
            result['ml_anomaly_score'] = np.nan
            result['ml_confidence'] = 0.0
        result['ml_model_available'] = True
        
        return result
    
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure feature columns exist for prediction.
        
        NOTE: Kept for backward compatibility. AnomalyDetector.predict()
        performs its own feature engineering from raw columns, so this is
        only needed by legacy callers that bypass predict().
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with required features
        """
        required_features = [
            'courses_accessed', 'accounts_accessed', 'api_requests',
            'data_mb', 'token_age_minutes', 'after_hours'
        ]
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = 0
        
        return df
    
    def get_ml_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive ML analysis results.
        
        Args:
            df: DataFrame with ML predictions
            
        Returns:
            Dictionary with ML analysis results
        """
        # Prefer the calibrated normalized score for anomaly-ness metrics
        score_col = 'anomaly_score_normalized' if 'anomaly_score_normalized' in df.columns else 'anomaly_score'
        
        result = {
            'total_events': len(df),
            'anomalies_detected': int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
            'anomaly_rate': float(df['is_anomaly'].mean()) if 'is_anomaly' in df.columns else 0.0,
            'avg_anomaly_score': float(df[score_col].mean()) if score_col in df.columns else 0.0,
            'max_anomaly_score': float(df[score_col].max()) if score_col in df.columns else 0.0,
            'min_anomaly_score': float(df[score_col].min()) if score_col in df.columns else 0.0,
            'confident_anomalies': int((df['ml_confidence'] > 0.7).sum()) if 'ml_confidence' in df.columns else 0,
            'score_column_used': score_col,
        }
        
        return result
    
    def get_anomalies(self, df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """
        Get events classified as anomalies above threshold.
        
        Uses anomaly_score_normalized (0-1, higher = more anomalous) when
        available; falls back to is_anomaly flags.
        
        Args:
            df: DataFrame with ML predictions
            threshold: Anomaly score threshold on the normalized score
            
        Returns:
            DataFrame with anomalous events
        """
        if 'anomaly_score_normalized' in df.columns:
            return df[df['anomaly_score_normalized'] >= threshold].copy()
        if 'is_anomaly' in df.columns:
            return df[df['is_anomaly'].astype(bool)].copy()
        return df.iloc[0:0].copy()
    
    def explain_anomaly(self, row: pd.Series) -> Dict:
        """
        Provide explanation for an anomaly prediction.
        
        Args:
            row: Single event row with anomaly prediction
            
        Returns:
            Dictionary with anomaly explanation
        """
        # Prefer normalized score for interpretation; fall back to anomaly_score
        score = row.get('anomaly_score_normalized', row.get('anomaly_score', 0))
        try:
            score_val = float(score) if pd.notna(score) else 0.0
        except (TypeError, ValueError):
            score_val = 0.0
        
        explanation = {
            'is_anomaly': bool(row.get('is_anomaly', False)),
            'anomaly_score': score_val,
            'anomaly_score_normalized': (
                float(row['anomaly_score_normalized'])
                if 'anomaly_score_normalized' in row.index and pd.notna(row.get('anomaly_score_normalized'))
                else None
            ),
            'confidence': float(row.get('ml_confidence', 0)),
            'interpretation': self._interpret_anomaly_score(score_val),
            'key_features': self._get_key_features(row)
        }
        
        return explanation
    
    def _interpret_anomaly_score(self, score: float) -> str:
        """
        Interpret anomaly score for explanation.
        
        Args:
            score: Anomaly score (0-1, higher = more anomalous)
            
        Returns:
            Human-readable interpretation
        """
        if score >= 0.8:
            return "HIGH confidence anomaly - significant deviation from normal patterns"
        elif score >= 0.6:
            return "MODERATE confidence anomaly - notable deviation from normal patterns"
        elif score >= 0.4:
            return "LOW confidence anomaly - slight deviation from normal patterns"
        else:
            return "Normal activity - consistent with baseline patterns"
    
    def _get_key_features(self, row: pd.Series) -> List[Dict]:
        """
        Get key features contributing to anomaly score.
        
        Args:
            row: Single event row
            
        Returns:
            List of feature dictionaries with values and descriptions
        """
        features = []
        
        feature_descriptions = {
            'api_requests': 'High API request rate may indicate automation or scraping',
            'data_mb': 'Large data transfer may indicate exfiltration',
            'courses_accessed': 'Broad course access may indicate privilege abuse',
            'accounts_accessed': 'Broad account access may indicate privilege abuse',
            'token_age_minutes': 'Long token lifetime may indicate session hijacking',
            'after_hours': 'After-hours activity is unusual for this user'
        }
        
        for feature, description in feature_descriptions.items():
            if feature in row.index:
                value = row[feature]
                features.append({
                    'feature': feature,
                    'value': float(value) if pd.notna(value) else 0,
                    'description': description
                })
        
        return features
