"""
Risk Engine for CAID

This module implements a rule-based risk classification system that combines
machine learning anomaly scores with heuristic rules to classify events.

ARCHITECTURE:
- ML Anomaly Detection (Isolation Forest): Identifies statistical outliers
- Risk Scoring: Combines ML anomaly score with business rules
- Risk Classification: Categorizes risk scores into levels

IMPORTANT:
- ML Anomaly Detection and Risk Classification are SEPARATE processes
- Risk scores combine ML predictions with heuristic rules (weighted factors)
- Percentile-based thresholds are RELATIVE to the current dataset
- These metrics should NOT be interpreted as real-world accuracy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RiskEngine:
    """
    Rule-based risk classification engine that combines
    ML anomaly scores with heuristic rules.
    """
    
    def __init__(self):
        """Initialize risk thresholds and rules."""
        # Risk thresholds (adjust based on domain knowledge)
        self.thresholds = {
            'anomaly_score_high': 0.8,
            'anomaly_score_medium': 0.6,
            'api_threshold': 300,
            'data_threshold_mb': 10,
            'courses_threshold': 10,
            'accounts_threshold': 5
        }
        self.anomaly_min = None
        self.anomaly_max = None
        self.risk_min = None
        self.risk_max = None
        self._ml_explained = False  # Track if ML explanation was provided
        # Fixed calibrated thresholds (Milestone 1.7). When set, classify_risk
        # uses ABSOLUTE cutoffs derived from a separate calibration dataset
        # instead of dataset-relative percentiles. None = legacy mode.
        self.fixed_thresholds: Optional[Dict[str, float]] = None
        self.calibration_info: Optional[Dict] = None
        # Milestone 1.9: explicit calibration lifecycle state.
        # 'uncalibrated_dataset_relative' = legacy percentile mode, clearly
        # distinguished from fixed calibrated thresholds.
        self.calibration_status: str = 'uncalibrated_dataset_relative'
        self.threshold_version: Optional[str] = None
    
    def fit(self, df: pd.DataFrame) -> 'RiskEngine':
        """
        Fit the risk engine by computing anomaly score and risk score statistics.
        
        Args:
            df: DataFrame with anomaly scores and risk scores
            
        Returns:
            Self for method chaining
        """
        if 'anomaly_score' in df.columns:
            self.anomaly_min = df['anomaly_score'].min()
            self.anomaly_max = df['anomaly_score'].max()
        if 'risk_score' in df.columns:
            self.risk_min = df['risk_score'].min()
            self.risk_max = df['risk_score'].max()
        return self
    
    def calculate_risk_score(self, row: pd.Series) -> Tuple[float, List[Dict]]:
        """
        Calculate risk score and identify risk factors for a single event.
        
        Args:
            row: Single row from security log DataFrame
            
        Returns:
            Tuple of (risk_score, risk_factors)
        """
        risk_factors = []
        score = 0.0
        
        # Factor 1: ML Anomaly Score (40% weight)
        # Score semantics contract:
        #   - anomaly_score_normalized / ml_anomaly_score: 0-1,
        #     HIGHER = MORE ANOMALOUS (fixed calibration)
        #   - anomaly_score_raw: raw IsolationForest decision_function,
        #     HIGHER = MORE NORMAL (unbounded), must be inverted
        #   - anomaly_score: LEGACY ambiguous field (raw in production,
        #     0-1 normalized in older tests); treated as raw decision_function
        #   - missing ML output is flagged explicitly, never silently normal
        ml_missing = (
            'anomaly_score_normalized' not in row.index
            and 'anomaly_score_raw' not in row.index
            and 'anomaly_score' not in row.index
        )
        if 'anomaly_score_normalized' in row.index and pd.notna(row.get('anomaly_score_normalized')):
            normalized_anomaly = float(np.clip(row['anomaly_score_normalized'], 0.0, 1.0))
            ml_value = float(row['anomaly_score_normalized'])
            ml_source = 'anomaly_score_normalized'
        elif 'anomaly_score_raw' in row.index and pd.notna(row.get('anomaly_score_raw')):
            raw = float(row['anomaly_score_raw'])
            normalized_anomaly = float(np.clip(-raw, 0.0, 1.0))
            ml_value = raw
            ml_source = 'anomaly_score_raw (raw decision_function, inverted)'
        elif 'anomaly_score' in row.index and pd.notna(row.get('anomaly_score')):
            raw = float(row['anomaly_score'])
            # Legacy adapter: treat anomaly_score as raw decision_function
            # (higher = more normal) -> invert to anomaly-ness.
            normalized_anomaly = float(np.clip(-raw, 0.0, 1.0))
            ml_value = raw
            ml_source = 'anomaly_score (legacy, treated as raw decision_function, inverted)'
        else:
            normalized_anomaly = 0.5  # neutral when ML output is unavailable
            ml_value = None
            ml_source = 'unavailable'
        
        risk_factors.append({
            'factor': 'ML Anomaly Score',
            'value': round(ml_value, 3) if ml_value is not None else None,
            'normalized_value': round(normalized_anomaly, 3),
            'score_source': ml_source,
            'ml_score_missing': ml_missing,
            # Milestone 1.9: a neutral placeholder is NOT a genuine ML
            # observation — ml_validated is False whenever the model did
            # not produce a score for this row.
            'ml_validated': not ml_missing and ml_value is not None,
            'weight': 0.4,
            'impact': 'positive' if normalized_anomaly > 0.5 else 'negative'
        })
        score += normalized_anomaly * 0.4
        
        # Factor 2: API Request Volume (20% weight)
        api_requests = row.get('api_requests', 0)
        if api_requests > self.thresholds['api_threshold']:
            api_ratio = min(api_requests / 1000, 1.0)
            risk_factors.append({
                'factor': 'High API Requests',
                'value': api_requests,
                'threshold': self.thresholds['api_threshold'],
                'weight': 0.2,
                'impact': 'positive'
            })
            score += api_ratio * 0.2
        
        # Factor 3: Data Transfer Volume (15% weight)
        data_mb = row.get('data_mb', 0)
        if data_mb > self.thresholds['data_threshold_mb']:
            data_ratio = min(data_mb / 100, 1.0)
            risk_factors.append({
                'factor': 'High Data Transfer',
                'value': round(data_mb, 2),
                'threshold': self.thresholds['data_threshold_mb'],
                'weight': 0.15,
                'impact': 'positive'
            })
            score += data_ratio * 0.15
        
        # Factor 4: Course Access (10% weight)
        courses_accessed = row.get('courses_accessed', 0)
        if courses_accessed > self.thresholds['courses_threshold']:
            course_ratio = min(courses_accessed / 30, 1.0)
            risk_factors.append({
                'factor': 'Broad Course Access',
                'value': courses_accessed,
                'threshold': self.thresholds['courses_threshold'],
                'weight': 0.1,
                'impact': 'positive'
            })
            score += course_ratio * 0.1
        
        # Factor 5: Account Access (10% weight)
        accounts_accessed = row.get('accounts_accessed', 0)
        if accounts_accessed > self.thresholds['accounts_threshold']:
            account_ratio = min(accounts_accessed / 50, 1.0)
            risk_factors.append({
                'factor': 'Broad Account Access',
                'value': accounts_accessed,
                'threshold': self.thresholds['accounts_threshold'],
                'weight': 0.1,
                'impact': 'positive'
            })
            score += account_ratio * 0.1
        
        # Factor 6: After Hours Activity (5% weight)
        after_hours = row.get('after_hours', 0)
        if after_hours == 1:
            risk_factors.append({
                'factor': 'After Hours Activity',
                'value': after_hours,
                'weight': 0.05,
                'impact': 'positive'
            })
            score += 0.05
        
        # Normalize score to 0-1 range
        score = min(score, 1.0)
        
        return score, risk_factors
    
    def calibrate_thresholds(
        self,
        calibration_df: pd.DataFrame,
        percentiles: Tuple[float, float, float] = (30.0, 65.0, 85.0),
    ) -> Dict[str, float]:
        """
        Calibrate FIXED risk thresholds on a separate calibration dataset.

        Milestone 1.7: thresholds must be derived from designated calibration
        data (e.g. normal_activity.csv background), NEVER from evaluation
        data. The percentiles define what fraction of calibration (normal)
        activity lands in each band:
          - medium   : risk_score >= p30 of calibration distribution
          - high     : risk_score >= p65
          - critical : risk_score >= p85

        After calibration, classify_risk() uses these absolute cutoffs and
        results are comparable ACROSS datasets (unlike the legacy
        dataset-relative percentile mode, which is retained for backward
        compatibility when calibrate_thresholds() has not been called).

        Args:
            calibration_df: Calibration DataFrame (typically normal/baseline
                activity; must contain anomaly score columns if available)
            percentiles: (medium, high, critical) percentiles of the
                calibration risk_score distribution

        Returns:
            Dict with 'medium', 'high', 'critical' absolute thresholds
        """
        if calibration_df is None or len(calibration_df) == 0:
            raise ValueError("Cannot calibrate on an empty DataFrame")

        from utils.row_view import RowView
        scores = np.array([
            self.calculate_risk_score(RowView(r))[0]
            for r in calibration_df.to_dict('records')
        ], dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            raise ValueError("Calibration produced no finite risk scores")

        p_med, p_high, p_crit = np.percentile(scores, percentiles)
        # Ensure monotonic non-decreasing thresholds
        p_crit = max(p_crit, p_high, p_med)
        p_high = max(p_high, p_med)

        self.fixed_thresholds = {
            'medium': float(p_med),
            'high': float(p_high),
            'critical': float(p_crit),
        }
        # Direct (in-memory) calibration: mark status calibrated; persisted
        # artifacts additionally carry a threshold_version via
        # core.calibration.apply_to_engine.
        self.calibration_status = 'calibrated'
        self.calibration_info = {
            'calibration_rows': int(len(scores)),
            'percentiles': list(percentiles),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
        }
        return dict(self.fixed_thresholds)

    def classify_risk(self, risk_score: float) -> RiskLevel:
        """
        Classify risk level based on risk score.
        
        Args:
            risk_score: Calculated risk score (0-1)
            
        Returns:
            RiskLevel enum value
        """
        # Milestone 1.7: fixed calibrated thresholds take precedence.
        # These are ABSOLUTE cutoffs from a separate calibration dataset.
        if self.fixed_thresholds is not None:
            if risk_score >= self.fixed_thresholds['critical']:
                return RiskLevel.CRITICAL
            elif risk_score >= self.fixed_thresholds['high']:
                return RiskLevel.HIGH
            elif risk_score >= self.fixed_thresholds['medium']:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        # Use percentile-based thresholds for more realistic classification
        # These thresholds can be adjusted based on the actual data distribution
        if self.risk_max is not None and self.risk_min is not None:
            risk_range = self.risk_max - self.risk_min
            if risk_range > 0:
                # Normalize to 0-1 based on actual data range
                normalized = (risk_score - self.risk_min) / risk_range
                # Use quartile-based thresholds for balanced distribution
                # Q1 (25%) = Low, Q2 (50%) = Medium, Q3 (75%) = High, above = Critical
                if normalized >= 0.85:  # Top 15% as Critical
                    return RiskLevel.CRITICAL
                elif normalized >= 0.65:  # Next 20% as High
                    return RiskLevel.HIGH
                elif normalized >= 0.30:  # Next 35% as Medium
                    return RiskLevel.MEDIUM
                else:  # Bottom 30% as Low
                    return RiskLevel.LOW
        
        # Fallback to absolute thresholds
        if risk_score >= 0.75:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def evaluate_calibration(
        self,
        eval_df: pd.DataFrame,
        ground_truth_column: str = 'label',
        mode: str = 'binary',
    ) -> Dict:
        """
        Evaluate calibrated risk classification on a labeled evaluation set.

        IMPORTANT: synthetic labels only — metrics describe behavior on this
        synthetic distribution, NOT real-world detection accuracy.

        Args:
            eval_df: DataFrame with risk_score (or rows to score) and labels
            ground_truth_column: Label column name
            mode: 'binary' (suspicious+high_risk positive) or
                  'high-risk' (high_risk only positive)

        Returns:
            Dict with tp/fp/tn/fn, precision, recall, f1_score,
            false_positive_rate, thresholds_used, mode
        """
        if eval_df is None or len(eval_df) == 0:
            raise ValueError("Cannot evaluate on an empty DataFrame")
        if ground_truth_column not in eval_df.columns:
            raise ValueError(f"Ground truth column '{ground_truth_column}' missing")

        df = eval_df.copy()
        if 'risk_score' not in df.columns:
            from utils.row_view import RowView
            df['risk_score'] = [self.calculate_risk_score(RowView(r))[0]
                                for r in df.to_dict('records')]
        df['risk_level'] = df['risk_score'].apply(self.classify_risk)
        df['risk_level'] = df['risk_level'].apply(
            lambda x: x.value if isinstance(x, RiskLevel) else str(x))

        if mode == 'binary':
            pos_labels = {'suspicious', 'high_risk'}
            pred_pos = {'High', 'Critical'}
        elif mode == 'high-risk':
            pos_labels = {'high_risk'}
            pred_pos = {'Critical'}
        else:
            raise ValueError(f"Unknown mode: {mode}")

        y_true = df[ground_truth_column].isin(pos_labels)
        y_pred = df['risk_level'].isin(pred_pos)

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision, 'recall': recall,
            'f1_score': f1, 'false_positive_rate': fpr,
            'accuracy': (tp + tn) / max(len(df), 1),
            'mode': mode,
            'thresholds_used': dict(self.fixed_thresholds) if self.fixed_thresholds else None,
            'calibration_info': self.calibration_info,
            'calibration_status': self.calibration_status,
            'threshold_version': self.threshold_version,
            'note': 'Metrics computed on synthetic labeled data only.',
        }
    
    def process_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all events and add risk classification.
        
        Args:
            df: Security log DataFrame
            
        Returns:
            DataFrame with risk classifications
        """
        df = df.copy()
        
        # Fit the engine to get anomaly score statistics
        self.fit(df)
        
        # Calculate risk scores for each event. to_dict('records') +
        # RowView avoids the per-row Series cost of apply(axis=1).
        from utils.row_view import RowView
        records = df.to_dict('records')
        risk_results = [self.calculate_risk_score(RowView(r))
                        for r in records]

        # Extract results
        risk_scores = [r[0] for r in risk_results]
        risk_factors = [r[1] for r in risk_results]

        # Add risk scores
        df['risk_score'] = risk_scores
        df['risk_factors'] = risk_factors

        # Update risk_min and risk_max after scores are calculated
        self.risk_min = df['risk_score'].min()
        self.risk_max = df['risk_score'].max()

        # Classify risk levels
        df['risk_level'] = df['risk_score'].apply(self.classify_risk)

        # Convert RiskLevel enum to string value (e.g., "Low" instead of "RiskLevel.Low")
        df['risk_level'] = df['risk_level'].apply(lambda x: x.value if isinstance(x, RiskLevel) else str(x))

        # Add evidence-based classification
        if len(df) > 0:
            df['risk_evidence'] = [
                self._generate_evidence(RowView({**r, 'risk_factors': f}))
                for r, f in zip(records, risk_factors)
            ]
        else:
            df['risk_evidence'] = pd.Series(dtype=str)
        
        return df
    
    def _generate_evidence(self, row: pd.Series) -> str:
        """
        Generate human-readable evidence for risk classification.
        
        Args:
            row: Event row with risk factors
            
        Returns:
            Evidence string
        """
        factors = row.get('risk_factors', [])
        
        if not factors:
            return "No significant risk indicators detected"
        
        # Get high-impact factors
        high_factors = [f for f in factors if f.get('impact') == 'positive']
        
        if not high_factors:
            return "Activity appears consistent with baseline behavior"
        
        evidence_parts = []
        for factor in high_factors[:3]:  # Limit to top 3 factors
            name = factor.get('factor', 'Unknown')
            value = factor.get('value', 'N/A')
            threshold = factor.get('threshold', '')
            
            if threshold:
                evidence_parts.append(f"{name} ({value} > {threshold})")
            else:
                evidence_parts.append(f"{name} ({value})")
        
        return "; ".join(evidence_parts)
    
    def explain_classification(self, row: pd.Series) -> dict:
        """
        Provide explanation of the risk classification process for an event.
        
        Returns separate explanations for:
        1. ML Anomaly Prediction (Isolation Forest)
        2. Risk Scoring (weighted combination of factors)
        3. Risk Classification (percentile-based categorization)
        
        IMPORTANT: This shows how ML anomaly detection and risk classification
        are separate processes that work together.
        
        Args:
            row: Event row with anomaly_score, risk_score
            
        Returns:
            Dictionary with ML and risk classification explanations
        """
        explanation = {
            'ml_anomaly_prediction': {},
            'risk_scoring': {},
            'risk_classification': {}
        }
        
        # ML Anomaly Prediction explanation
        if 'anomaly_score' in row.index:
            anomaly_score = row['anomaly_score']
            is_anomaly = row.get('is_anomaly', False)
            
            explanation['ml_anomaly_prediction'] = {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'decision_boundary': 'Score < 0 indicates anomaly (statistical outlier)',
                'explanation': (
                    "The Isolation Forest model identified this event as "
                    + ("an " if is_anomaly else "a ") + "anomaly based on "
                    "statistical deviation from the baseline training data."
                )
            }
        
        # Risk Scoring explanation
        if 'risk_score' in row.index:
            risk_score = row['risk_score']
            risk_factors = row.get('risk_factors', [])
            
            weight_breakdown = {}
            for factor in risk_factors[:5]:  # Top 5 factors
                name = factor.get('factor', 'Unknown')
                weight = factor.get('weight', 0)
                if name not in weight_breakdown:
                    weight_breakdown[name] = weight
            
            explanation['risk_scoring'] = {
                'risk_score': float(risk_score),
                'risk_factors_used': [f.get('factor', 'Unknown') for f in risk_factors[:3]],
                'max_possible_score': 1.0,
                'weight_breakdown': weight_breakdown,
                'explanation': (
                    "Risk score combines ML anomaly score (40% weight) with "
                    "business rules: API volume (20%), data transfer (15%), "
                    "course access (10%), account access (10%), after-hours (5%)."
                )
            }
        
        # Risk Classification explanation
        if self.risk_min is not None and self.risk_max is not None:
            risk_score = row.get('risk_score', 0)
            risk_range = self.risk_max - self.risk_min
            
            if risk_range > 0:
                normalized = (risk_score - self.risk_min) / risk_range
                
                explanation['risk_classification'] = {
                    'risk_score': float(risk_score),
                    'dataset_min': float(self.risk_min),
                    'dataset_max': float(self.risk_max),
                    'normalized_score': float(normalized),
                    'percentile_based': True,
                    'explanation': (
                        "Risk level is assigned using PERCENTILE-based thresholds "
                        "relative to the current dataset's risk score distribution. "
                        "Thresholds: >=85th percentile = Critical, >=65th = High, "
                        ">=30th = Medium, <30th = Low."
                    )
                }
        
        return explanation
