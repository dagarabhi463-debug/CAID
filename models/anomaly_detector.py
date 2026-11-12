"""
Anomaly Detector for CAID

This module implements Isolation Forest for detecting anomalous security events.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple, Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.feature_engineering import (
    engineer_features,
    get_feature_matrix,
    FeatureBaselineManager,
)


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for security logs.
    
    Architecture:
    - fit(): Train on baseline data only (normal_activity.csv)
    - predict(): Use fitted model for predictions (no training)
    - save(): Persist model and preprocessing state
    - load(): Restore model and preprocessing state
    - evaluate(): Compute metrics using ground truth labels
    """
    
    # v2 (Milestone 1.8.1): training-time user_*_z features are now
    # leave-one-out (honest deviation) and unseen-user aggregate fallbacks
    # use typical-user statistics. Models trained on v1 semantics are
    # incompatible and must be retrained.
    ARTIFACT_VERSION = 2
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the training data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        self.baseline_manager: Optional[FeatureBaselineManager] = None
        self.is_fitted = False
        self.feature_columns: List[str] = []
        self._scaler_fitted = False
        self._score_calibration: Optional[dict] = None
    
    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model on training data.
        
        IMPORTANT: This method should only be called on baseline training data
        (normal_activity.csv) to avoid data leakage from evaluation sets.
        
        Args:
            df: Training DataFrame containing only baseline/normal activity
            feature_columns: List of feature columns (auto-detected if not provided)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If DataFrame is empty or has no valid features
        """
        if df.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        
        # Fit the feature baseline manager on the training data ONLY.
        # These baselines are reused (never refit) during prediction.
        self.baseline_manager = FeatureBaselineManager().fit(df)
        
        # Engineer features using the stored training baselines.
        # leave_one_out=True: training z-scores exclude the scored event
        # from its own baseline so they match prediction-time semantics.
        df_engineered, detected_features = engineer_features(
            df, self.baseline_manager, leave_one_out=True)
        
        # Use provided feature columns or use detected ones
        if feature_columns is None:
            self.feature_columns = detected_features
        else:
            # Validate that all requested columns exist
            missing = set(feature_columns) - set(detected_features)
            if missing:
                raise ValueError(f"Requested feature columns not found: {missing}")
            self.feature_columns = feature_columns
        
        if not self.feature_columns:
            raise ValueError("No valid feature columns available for training")
        
        # Get feature matrix
        X = get_feature_matrix(df_engineered, self.feature_columns)
        
        # Handle missing/invalid values
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit scaler on training data ONLY
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        self._scaler_fitted = True
        
        # Fixed score calibration from TRAINING decision scores.
        # Used for batch-invariant normalized anomaly scores at predict time.
        train_scores = self.model.decision_function(X_scaled)
        self._score_calibration = {
            'score_min': float(np.percentile(train_scores, 1)),
            'score_max': float(np.percentile(train_scores, 99)),
        }
        if self._score_calibration['score_max'] <= self._score_calibration['score_min']:
            self._score_calibration['score_max'] = self._score_calibration['score_min'] + 1e-9
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies using an already fitted model.
        
        IMPORTANT: This method performs prediction ONLY. It does NOT fit or
        re-fit the model. The model must be fitted before calling predict.
        
        Args:
            df: DataFrame to predict on (can be raw or already engineered)
            
        Returns:
            DataFrame with anomaly predictions and scores
            
        Raises:
            RuntimeError: If model has not been fitted yet
            ValueError: If feature columns don't match training
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before prediction. Call fit() first, "
                "or load() a saved model."
            )
        
        if self.baseline_manager is None or not self.baseline_manager.is_fitted():
            raise RuntimeError(
                "Feature baseline manager is not fitted. The model must be "
                "retrained or loaded from a compatible saved artifact."
            )
        
        df = df.copy()
        
        # ALWAYS engineer features using the stored training baselines.
        # This guarantees batch-invariant features: prediction data never
        # contributes to user/IP/event aggregations. Raw columns persist in
        # already-engineered frames, so re-engineering is safe and idempotent.
        df_engineered, _ = engineer_features(df, self.baseline_manager)
        
        # Validate feature schema matches training exactly
        available = set(df_engineered.columns)
        missing_cols = set(self.feature_columns) - available
        if missing_cols:
            raise ValueError(
                f"Missing expected feature columns: {missing_cols}\n"
                f"Expected columns: {set(self.feature_columns)}"
            )
        
        # Get feature matrix using EXACT same columns and order as training
        X = get_feature_matrix(df_engineered, self.feature_columns)
        
        # Handle missing/invalid values
        if np.any(~np.isfinite(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale using fitted scaler (NOT fit_transform)
        if self._scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            #Scaler should always be fitted when predict is called
            raise RuntimeError(
                "Scaler is not fitted during prediction. "
                "This indicates the model was not properly fitted or loaded. "
                "Call fit() first or use load() to restore from saved model."
            )
        
        # Get predictions (-1 = anomaly, 1 = normal)
        predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (lower = more anomalous)
        scores = self.model.decision_function(X_scaled)
        
        # anomaly_score_raw / anomaly_score: raw IsolationForest
        # decision_function (higher = more normal). anomaly_score is the
        # DEPRECATED ambiguous alias retained for backward compatibility;
        # new consumers should use anomaly_score_raw or
        # anomaly_score_normalized.
        df['anomaly_prediction'] = predictions
        df['anomaly_score_raw'] = scores
        df['anomaly_score'] = scores  # deprecated alias
        
        # Convert to more intuitive labels
        df['is_anomaly'] = predictions == -1
        
        # Fixed-calibration normalized score: maps training-score range to
        # [0, 1] and inverts so HIGHER = MORE ANOMALOUS. Because calibration
        # comes from training data, the same event gets the same normalized
        # score regardless of batch composition. This is NOT a probability.
        cal = self._score_calibration or {'score_min': float(scores.min()), 'score_max': float(scores.max())}
        denom = cal['score_max'] - cal['score_min']
        if denom <= 0:
            df['anomaly_score_normalized'] = 0.5
        else:
            df['anomaly_score_normalized'] = np.clip(
                1.0 - (scores - cal['score_min']) / denom, 0.0, 1.0
            )
        
        return df
    
    def save(self, model_path: str) -> None:
        """
        Save the fitted model and preprocessing objects.
        
        Args:
            model_path: Path to save the model (e.g., 'models/saved_models/anomaly_detector.joblib')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump({
            'artifact_version': self.ARTIFACT_VERSION,
            'model': self.model,
            'scaler': self.scaler,
            'baseline_manager': self.baseline_manager,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            '_scaler_fitted': self._scaler_fitted,
            'contamination': self.contamination,
            'random_state': self.random_state,
            'score_calibration': self._score_calibration,
            'extra_state': getattr(self, '_extra_artifact_state', {}),
        }, model_path)
    
    def load(self, model_path: str) -> 'AnomalyDetector':
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Self for method chaining
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        saved_data = joblib.load(model_path)
        
        # Validate artifact schema and version before restoring state
        if not isinstance(saved_data, dict):
            raise ValueError(
                f"Invalid model artifact at {model_path}: expected a dict, "
                f"got {type(saved_data).__name__}"
            )
        
        version = saved_data.get('artifact_version')
        if version != self.ARTIFACT_VERSION:
            raise ValueError(
                f"Incompatible model artifact version {version!r} at {model_path} "
                f"(expected {self.ARTIFACT_VERSION}). Retrain and resave the model."
            )
        
        required_keys = ('model', 'scaler', 'baseline_manager', 'feature_columns',
                         'is_fitted', '_scaler_fitted', 'score_calibration')
        missing_keys = [k for k in required_keys if k not in saved_data]
        if missing_keys:
            raise ValueError(
                f"Incomplete model artifact at {model_path}: missing keys {missing_keys}"
            )
        
        if not saved_data['is_fitted'] or not saved_data['_scaler_fitted']:
            raise ValueError(
                f"Model artifact at {model_path} was saved in an unfitted state"
            )
        
        baseline_manager = saved_data['baseline_manager']
        if baseline_manager is None or not getattr(baseline_manager, 'is_fitted', lambda: False)():
            raise ValueError(
                f"Model artifact at {model_path} has no fitted feature baseline manager"
            )
        
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.baseline_manager = baseline_manager
        self.feature_columns = saved_data['feature_columns']
        self.is_fitted = saved_data['is_fitted']
        self._scaler_fitted = saved_data['_scaler_fitted']
        self.contamination = saved_data.get('contamination', self.contamination)
        self.random_state = saved_data.get('random_state', self.random_state)
        self._score_calibration = saved_data['score_calibration']
        self._extra_artifact_state = saved_data.get('extra_state', {})
        
        return self
    
    def evaluate(
        self,
        df: pd.DataFrame,
        ground_truth_column: str = 'label',
        mode: str = 'binary'
    ) -> dict:
        """
        Evaluate model performance using ground truth labels.
        
        IMPORTANT: This evaluation is on synthetic data only. Performance
        metrics should not be interpreted as real-world detection accuracy.
        
        Args:
            df: DataFrame with predictions and ground truth labels
            ground_truth_column: Column name containing ground truth labels
            mode: Evaluation mode - 'binary' or 'high-risk'
                  - 'binary': normal = negative, suspicious+high_risk = positive
                  - 'high-risk': normal+suspicious = negative, high_risk = positive
            
        Returns:
            Dictionary containing evaluation metrics:
            - tp: True positives
            - fp: False positives
            - tn: True negatives
            - fn: False negatives
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1_score: 2 * (precision * recall) / (precision + recall)
            - accuracy: (TP + TN) / (TP + TN + FP + FN)
            - confusion_matrix: dict with tp, fp, tn, fn
            - mode: evaluation mode used
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        if df.empty:
            raise ValueError("Cannot evaluate on an empty DataFrame")
        
        if ground_truth_column not in df.columns:
            raise ValueError(f"Ground truth column '{ground_truth_column}' not found in DataFrame")
        
        # Ensure we have predictions
        if 'is_anomaly' not in df.columns:
            raise ValueError("Predictions not found. Call predict() first.")
        
        # Determine positive labels based on mode
        if mode == 'binary':
            # Binary detection: normal = negative, suspicious+high_risk = positive
            positive_labels = ['suspicious', 'high_risk']
        elif mode == 'high-risk':
            # High-risk detection: normal+suspicious = negative, high_risk = positive
            positive_labels = ['high_risk']
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'binary' or 'high-risk'.")
        
        y_true = df[ground_truth_column].isin(positive_labels).astype(int)
        y_pred = df['is_anomaly'].astype(int)
        
        # Calculate confusion matrix components
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'confusion_matrix': {
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            },
            'mode': mode,
            'ground_truth_distribution': {
                'positive': int(y_true.sum()),
                'negative': int(len(y_true) - y_true.sum())
            },
            'predictions_distribution': {
                'anomalies_detected': int(y_pred.sum()),
                'normal': int(len(y_pred) - y_pred.sum())
            }
        }


def get_model_path() -> str:
    """
    Get the default model save path.
    
    Returns:
        Path to the default model file
    """
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'saved_models',
        'anomaly_detector.joblib'
    )


def ensure_model_exists() -> AnomalyDetector:
    """
    Ensure a compatible model exists, loading or training as needed.
    
    On application startup:
    1. Check for saved model
    2. Load if available
    3. Otherwise train using baseline dataset
    4. Save the trained model
    
    Returns:
        Fitted AnomalyDetector instance
    """
    model_path = get_model_path()
    detector = AnomalyDetector()
    
    # Try to load existing model
    if os.path.exists(model_path):
        try:
            detector.load(model_path)
            print(f"Loaded existing model from {model_path}")
            return detector
        except Exception as e:
            print(f"Failed to load model: {e}. Training new model...")
    
    # Train new model using baseline dataset
    from core.log_processor import load_sample_dataset
    
    print("Training new model using normal_activity.csv as baseline...")
    df_normal = load_sample_dataset('normal')
    detector.fit(df_normal)
    
    # Save the trained model
    detector.save(model_path)
    print(f"Saved model to {model_path}")
    
    return detector
