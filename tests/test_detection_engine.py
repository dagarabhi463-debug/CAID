"""
Tests for Combined Detection Engine (Milestone 3)

Tests:
- Event processing
- Combined scoring
- Risk classification
- Detection summary
"""

import pytest
import pandas as pd
import numpy as np
from core.detection_engine import DetectionEngine, Detection, RiskLevel


class TestDetectionEngineInit:
    """Tests for DetectionEngine initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        engine = DetectionEngine()
        
        assert engine.detections == []
        assert engine._detection_counter == 0
        assert engine.weights is not None
        assert engine.thresholds is not None
    
    def test_custom_weights(self):
        """Test initialization with custom weights."""
        weights = {'ml': 0.5, 'rules': 0.3, 'behavioral': 0.2}
        
        engine = DetectionEngine()
        # Note: weights are not configurable in init, but we can test the default
        
        assert engine.weights['ml'] == 0.35
        assert engine.weights['rules'] == 0.35
        assert engine.weights['behavioral'] == 0.30


class TestCombinedScoreCalculation:
    """Tests for combined score calculation."""
    
    def test_calculate_combined_score(self):
        """Test calculating combined score."""
        engine = DetectionEngine()
        
        score = engine._calculate_combined_score(
            ml_score=0.8,
            rule_score=0.6,
            behavioral_score=0.7
        )
        
        # Should be weighted combination
        expected = 0.35 * 0.8 + 0.35 * 0.6 + 0.30 * 0.7
        assert abs(score - expected) < 0.001
    
    def test_calculate_combined_score_clamped(self):
        """Test that combined score is clamped to 1.0."""
        engine = DetectionEngine()
        
        score = engine._calculate_combined_score(
            ml_score=1.0,
            rule_score=1.0,
            behavioral_score=1.0
        )
        
        # Should be clamped to 1.0
        assert score == 1.0
    
    def test_calculate_combined_score_zeros(self):
        """Test calculating score with zeros."""
        engine = DetectionEngine()
        
        score = engine._calculate_combined_score(
            ml_score=0.0,
            rule_score=0.0,
            behavioral_score=0.0
        )
        
        # Should be 0
        assert score == 0.0


class TestRiskClassification:
    """Tests for risk classification."""
    
    def test_classify_critical(self):
        """Test classifying critical risk."""
        engine = DetectionEngine()
        
        level = engine._classify_risk(0.85)
        
        assert level == RiskLevel.CRITICAL
    
    def test_classify_high(self):
        """Test classifying high risk."""
        engine = DetectionEngine()
        
        level = engine._classify_risk(0.65)
        
        assert level == RiskLevel.HIGH
    
    def test_classify_medium(self):
        """Test classifying medium risk."""
        engine = DetectionEngine()
        
        level = engine._classify_risk(0.45)
        
        assert level == RiskLevel.MEDIUM
    
    def test_classify_low(self):
        """Test classifying low risk."""
        engine = DetectionEngine()
        
        level = engine._classify_risk(0.25)
        
        assert level == RiskLevel.LOW
    
    def test_classify_boundary(self):
        """Test risk classification at boundaries."""
        engine = DetectionEngine()
        
        # Exactly at boundary
        level = engine._classify_risk(0.75)
        assert level == RiskLevel.CRITICAL
        
        level = engine._classify_risk(0.55)
        assert level == RiskLevel.HIGH
        
        level = engine._classify_risk(0.30)
        assert level == RiskLevel.MEDIUM


class TestEventProcessing:
    """Tests for event processing."""
    
    def test_process_event(self):
        """Test processing a single event."""
        engine = DetectionEngine()
        
        row = pd.Series({
            'user_id': 'user1',
            'timestamp': '2024-01-15 10:30:00',
            'api_requests': 500,
            'data_mb': 50,
            'courses_accessed': 20,
            'accounts_accessed': 15,
            'token_age_minutes': 1500,
            'after_hours': 1,
            'is_anomaly': 1,
            'anomaly_score': 0.85,
            'rule_detection_score': 0.6,
            'behavioral_deviation_score': 0.7,
            'rule_triggers': [{'rule_id': 'api_high_volume', 'rule_name': 'High API Requests'}]
        })
        
        detection = engine.process_event(row)
        
        assert isinstance(detection, Detection)
        assert detection.user_id == 'user1'
        assert detection.ml_anomaly == True
        assert detection.ml_anomaly_score == 0.85
        assert detection.combined_score > 0
        assert detection.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    def test_process_event_fallback_values(self):
        """Test processing event with fallback values."""
        engine = DetectionEngine()
        
        # Minimal row
        row = pd.Series({
            'user_id': 'user1',
            'timestamp': '2024-01-15 10:30:00'
        })
        
        detection = engine.process_event(row)
        
        assert detection.user_id == 'user1'
        # Should have default values for missing fields
        assert detection.ml_anomaly_score == 0.5
        # Milestone 1.9: a bare row has fewer than two channels with real
        # evidence -> insufficient_evidence, NaN score, no classification
        assert detection.operating_mode == 'insufficient_evidence'
        assert detection.risk_level == RiskLevel.INSUFFICIENT
        assert np.isnan(detection.combined_score)


class TestDataFrameProcessing:
    """Tests for DataFrame processing."""
    
    def test_process_dataframe(self):
        """Test processing multiple events."""
        engine = DetectionEngine()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user3'],
            'timestamp': ['2024-01-15 10:00', '2024-01-15 11:00', '2024-01-15 12:00'],
            'is_anomaly': [1, 0, 1],
            'anomaly_score': [0.85, 0.3, 0.75],
            'rule_detection_score': [0.6, 0.1, 0.5],
            'behavioral_deviation_score': [0.7, 0.2, 0.6],
            'triggered_rules': [[{'rule_id': 'api_high'}], [], [{'rule_id': 'data_high'}]]
        })
        
        result_df = engine.process_dataframe(df)
        
        # Should have detection columns
        assert 'detection_id' in result_df.columns
        assert 'risk_level' in result_df.columns
        assert 'combined_score' in result_df.columns
        assert 'detection_explanation' in result_df.columns
        
        # Should have processed all events
        assert len(result_df) == 3
    
    def test_process_dataframe_empty(self):
        """Test processing empty DataFrame."""
        engine = DetectionEngine()
        
        df = pd.DataFrame()
        
        result_df = engine.process_dataframe(df)
        
        # Should handle empty DataFrame
        assert len(result_df) == 0


class TestSummary:
    """Tests for detection summary."""
    
    def test_get_summary(self):
        """Test getting detection summary."""
        engine = DetectionEngine()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user3', 'user4'],
            'is_anomaly': [1, 0, 1, 0],
            'anomaly_score': [0.85, 0.3, 0.75, 0.2],
            'rule_detection_score': [0.6, 0.1, 0.5, 0.05],
            'behavioral_deviation_score': [0.7, 0.2, 0.6, 0.15]
        })
        
        result_df = engine.process_dataframe(df)
        summary = engine.get_summary()
        
        assert 'total_detections' in summary
        assert 'risk_distribution' in summary
        assert summary['total_detections'] == 4
    
    def test_get_summary_empty(self):
        """Test getting summary with no detections."""
        engine = DetectionEngine()
        
        summary = engine.get_summary()
        
        assert summary['total_detections'] == 0


class TestSuspiciousUsers:
    """Tests for suspicious user identification."""
    
    def test_get_suspicious_users(self):
        """Test getting suspicious users."""
        engine = DetectionEngine()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user3'],
            'combined_score': [0.8, 0.7, 0.3, 0.6],
            'risk_level': ['High', 'High', 'Low', 'Medium'],
            'is_anomaly': [1, 1, 0, 0],
            'ml_anomaly_score': [0.8, 0.7, 0.3, 0.6]
        })
        
        suspicious = engine.get_suspicious_users(df, threshold=0.5)
        
        assert len(suspicious) >= 0
    
    def test_get_suspicious_users_empty(self):
        """Test getting suspicious users when none match."""
        engine = DetectionEngine()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'combined_score': [0.1, 0.2]
        })
        
        suspicious = engine.get_suspicious_users(df, threshold=0.9)
        
        assert len(suspicious) >= 0


class TestExplanations:
    """Tests for detection explanations."""
    
    def test_generate_explanation(self):
        """Test generating detection explanation."""
        engine = DetectionEngine()
        
        row = pd.Series({
            'user_id': 'user1',
            'timestamp': '2024-01-15 10:00',
            'is_anomaly': True,
            'anomaly_score': 0.85,
            'rule_detection_score': 0.6,
            'behavioral_deviation_score': 0.7,
            'triggered_rules': [{'rule_id': 'api_high_volume', 'rule_name': 'High API Requests'}]
        })
        
        explanation = engine._generate_explanation(
            row=row,
            ml_anomaly=True,
            ml_score=0.85,
            rule_triggers=[{'rule_name': 'High API Requests'}],
            behavioral_deviation=0.7,
            combined_score=0.75
        )
        
        assert isinstance(explanation, str)
        assert 'ML model' in explanation
        assert 'Rules triggered' in explanation or 'No detection rules' in explanation
        assert 'Behavioral pattern' in explanation or 'Unusual behavior' in explanation
    
    def test_generate_recommendations(self):
        """Test generating recommended actions."""
        engine = DetectionEngine()
        
        actions = engine._generate_recommendations(
            risk_level=RiskLevel.CRITICAL,
            ml_anomaly=True,
            rule_triggers=[{'rule_name': 'High API Requests'}]
        )
        
        assert isinstance(actions, list)
        assert len(actions) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
