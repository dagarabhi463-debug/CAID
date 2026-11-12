"""
Tests for Risk Engine

Tests:
- Initialization
- Risk score calculation
- Risk classification
- Evidence generation
- Explanation generation
"""

import pytest
import pandas as pd
import numpy as np
from core.risk_engine import RiskEngine, RiskLevel


class TestRiskEngineInit:
    """Tests for RiskEngine initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        engine = RiskEngine()
        
        assert engine.thresholds is not None
        assert 'api_threshold' in engine.thresholds
        assert engine.anomaly_min is None
        assert engine.anomaly_max is None
        assert engine.risk_min is None
        assert engine.risk_max is None
    
    def test_thresholds_exist(self):
        """Test that all expected thresholds exist."""
        engine = RiskEngine()
        
        expected_thresholds = [
            'anomaly_score_high',
            'anomaly_score_medium',
            'api_threshold',
            'data_threshold_mb',
            'courses_threshold',
            'accounts_threshold'
        ]
        
        for threshold in expected_thresholds:
            assert threshold in engine.thresholds


class TestRiskEngineFit:
    """Tests for RiskEngine fitting."""
    
    def test_fit_sets_stats(self):
        """Test that fit sets anomaly and risk statistics."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.2, 0.5, 0.8, 0.9],
            'risk_score': [0.3, 0.5, 0.7, 0.95]
        })
        
        engine.fit(df)
        
        assert engine.anomaly_min == 0.2
        assert engine.anomaly_max == 0.9
        assert engine.risk_min == 0.3
        assert engine.risk_max == 0.95
    
    def test_fit_handles_empty_data(self):
        """Test that fit handles empty DataFrame gracefully."""
        engine = RiskEngine()
        
        df = pd.DataFrame()
        
        result = engine.fit(df)
        
        # Should return self
        assert result is engine
        # Stats should remain None
        assert engine.anomaly_min is None


class TestRiskScoreCalculation:
    """Tests for risk score calculation."""
    
    def test_calculate_risk_score_basic(self):
        """Test basic risk score calculation."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.7,
            'api_requests': 200,
            'data_mb': 5,
            'courses_accessed': 5,
            'accounts_accessed': 3,
            'after_hours': 0
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(factors, list)
    
    def test_calculate_risk_score_with_high_api(self):
        """Test risk score with high API requests."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.5,
            'api_requests': 500,  # Above threshold
            'data_mb': 5,
            'courses_accessed': 5,
            'accounts_accessed': 3,
            'after_hours': 0
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        # Should have triggered API factor
        api_factors = [f for f in factors if f['factor'] == 'High API Requests']
        assert len(api_factors) > 0
    
    def test_calculate_risk_score_normalized(self):
        """Test that risk score is properly normalized."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.5,
            'api_requests': 500,
            'data_mb': 100,
            'courses_accessed': 30,
            'accounts_accessed': 50,
            'after_hours': 1
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        # Score should be capped at 1.0
        assert score <= 1.0


class TestRiskClassification:
    """Tests for risk classification."""
    
    def test_classify_risk_low(self):
        """Test classifying low risk."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'risk_score': [0.2, 0.5, 0.8, 0.95]
        })
        engine.fit(df)
        
        level = engine.classify_risk(0.25)
        
        assert level == RiskLevel.LOW
    
    def test_classify_risk_critical(self):
        """Test classifying critical risk."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'risk_score': [0.2, 0.5, 0.8, 0.95]
        })
        engine.fit(df)
        
        level = engine.classify_risk(0.95)
        
        assert level == RiskLevel.CRITICAL


class TestEvidenceGeneration:
    """Tests for evidence generation."""
    
    def test_generate_evidence(self):
        """Test generating evidence string."""
        engine = RiskEngine()
        
        row = pd.Series({
            'risk_factors': [
                {'factor': 'High API Requests', 'value': 500, 'threshold': 300},
                {'factor': 'After Hours Activity', 'value': 1}
            ]
        })
        
        evidence = engine._generate_evidence(row)
        
        assert isinstance(evidence, str)
        assert len(evidence) > 0
    
    def test_generate_evidence_no_factors(self):
        """Test evidence generation with no factors."""
        engine = RiskEngine()
        
        row = pd.Series({
            'risk_factors': []
        })
        
        evidence = engine._generate_evidence(row)
        
        # Should return default message
        assert 'No significant risk indicators' in evidence or 'baseline' in evidence.lower()


class TestClassificationExplanation:
    """Tests for classification explanation."""
    
    def test_explain_classification(self):
        """Test explaining risk classification."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.2, 0.5, 0.8, 0.95],
            'risk_score': [0.3, 0.5, 0.7, 0.95]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.8,
            'risk_score': 0.75,
            'risk_factors': [
                {'factor': 'High API Requests', 'weight': 0.2},
                {'factor': 'High Data Transfer', 'weight': 0.15}
            ]
        })
        
        explanation = engine.explain_classification(row)
        
        assert 'ml_anomaly_prediction' in explanation
        assert 'risk_scoring' in explanation
        assert 'risk_classification' in explanation


class TestProcessEvents:
    """Tests for processing events."""
    
    def test_process_events(self):
        """Test processing DataFrame through risk engine."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.3, 0.6, 0.9],
            'api_requests': [100, 350, 500],
            'data_mb': [2, 15, 30],
            'courses_accessed': [3, 12, 25],
            'accounts_accessed': [2, 8, 20],
            'after_hours': [0, 1, 1]
        })
        
        result_df = engine.process_events(df)
        
        # Should have added risk columns
        assert 'risk_score' in result_df.columns
        assert 'risk_factors' in result_df.columns
        assert 'risk_level' in result_df.columns
        assert 'risk_evidence' in result_df.columns
    
    def test_process_events_empty(self):
        """Test processing empty DataFrame."""
        engine = RiskEngine()
        
        df = pd.DataFrame()
        
        result_df = engine.process_events(df)
        
        assert len(result_df) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
class TestRiskScoreCalculation:
    """Tests for risk score calculation."""
    
    def test_calculate_risk_score_basic(self):
        """Test basic risk score calculation."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.7,
            'api_requests': 200,
            'data_mb': 5,
            'courses_accessed': 5,
            'accounts_accessed': 3,
            'after_hours': 0
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert isinstance(factors, list)
    
    def test_calculate_risk_score_with_high_api(self):
        """Test risk score with high API requests."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.5,
            'api_requests': 500,  # Above threshold
            'data_mb': 5,
            'courses_accessed': 5,
            'accounts_accessed': 3,
            'after_hours': 0
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        # Should have triggered API factor
        api_factors = [f for f in factors if f['factor'] == 'High API Requests']
        assert len(api_factors) > 0
    
    def test_calculate_risk_score_normalized(self):
        """Test that risk score is properly normalized."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.5, 0.5],
            'risk_score': [0.5, 0.5]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.5,
            'api_requests': 500,
            'data_mb': 100,
            'courses_accessed': 30,
            'accounts_accessed': 50,
            'after_hours': 1
        })
        
        score, factors = engine.calculate_risk_score(row)
        
        # Score should be capped at 1.0
        assert score <= 1.0
class TestRiskClassification:
    """Tests for risk classification."""
    
    def test_classify_risk_low(self):
        """Test classifying low risk."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'risk_score': [0.2, 0.5, 0.8, 0.95]
        })
        engine.fit(df)
        
        level = engine.classify_risk(0.25)
        
        assert level == RiskLevel.LOW
    
    def test_classify_risk_critical(self):
        """Test classifying critical risk."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'risk_score': [0.2, 0.5, 0.8, 0.95]
        })
        engine.fit(df)
        
        level = engine.classify_risk(0.95)
        
        assert level == RiskLevel.CRITICAL
class TestEvidenceGeneration:
    """Tests for evidence generation."""
    
    def test_generate_evidence(self):
        """Test generating evidence string."""
        engine = RiskEngine()
        
        row = pd.Series({
            'risk_factors': [
                {'factor': 'High API Requests', 'value': 500, 'threshold': 300},
                {'factor': 'After Hours Activity', 'value': 1}
            ]
        })
        
        evidence = engine._generate_evidence(row)
        
        assert isinstance(evidence, str)
        assert len(evidence) > 0
    
    def test_generate_evidence_no_factors(self):
        """Test evidence generation with no factors."""
        engine = RiskEngine()
        
        row = pd.Series({
            'risk_factors': []
        })
        
        evidence = engine._generate_evidence(row)
        
        # Should return default message
        assert 'No significant risk indicators' in evidence or 'baseline' in evidence.lower()
class TestClassificationExplanation:
    """Tests for classification explanation."""
    
    def test_explain_classification(self):
        """Test explaining risk classification."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.2, 0.5, 0.8, 0.95],
            'risk_score': [0.3, 0.5, 0.7, 0.95]
        })
        engine.fit(df)
        
        row = pd.Series({
            'anomaly_score': 0.8,
            'risk_score': 0.75,
            'risk_factors': [
                {'factor': 'High API Requests', 'weight': 0.2},
                {'factor': 'High Data Transfer', 'weight': 0.15}
            ]
        })
        
        explanation = engine.explain_classification(row)
        
        assert 'ml_anomaly_prediction' in explanation
        assert 'risk_scoring' in explanation
        assert 'risk_classification' in explanation
class TestProcessEvents:
    """Tests for processing events."""
    
    def test_process_events(self):
        """Test processing DataFrame through risk engine."""
        engine = RiskEngine()
        
        df = pd.DataFrame({
            'anomaly_score': [0.3, 0.6, 0.9],
            'api_requests': [100, 350, 500],
            'data_mb': [2, 15, 30],
            'courses_accessed': [3, 12, 25],
            'accounts_accessed': [2, 8, 20],
            'after_hours': [0, 1, 1]
        })
        
        result_df = engine.process_events(df)
        
        # Should have added risk columns
        assert 'risk_score' in result_df.columns
        assert 'risk_factors' in result_df.columns
        assert 'risk_level' in result_df.columns
        assert 'risk_evidence' in result_df.columns
    
    def test_process_events_empty(self):
        """Test processing empty DataFrame."""
        engine = RiskEngine()
        
        df = pd.DataFrame()
        
        result_df = engine.process_events(df)
        
        assert len(result_df) == 0