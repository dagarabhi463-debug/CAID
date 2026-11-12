"""
Tests for ML Integration Module (Milestone 3)

Tests:
- Model loading
- Anomaly prediction
- Feature handling
- Error handling
"""

import os
import pandas as pd
import numpy as np
from core.ml_integration import MLIntegration


def make_raw_df(n=5):
    """Minimal raw security-log rows for MLIntegration.predict()."""
    return pd.DataFrame({
        'timestamp': ['2024-01-01 10:00:00'] * n,
        'user_id': [f'user_{i}' for i in range(n)],
        'role': ['student'] * n,
        'source_ip': [f'192.0.2.{i}' for i in range(n)],
        'event_type': ['course_view'] * n,
        'courses_accessed': [1] * n,
        'accounts_accessed': [1] * n,
        'api_requests': [50, 500, 60, 70, 80][:n],
        'data_mb': [1.0] * n,
        'token_age_minutes': [30] * n,
        'after_hours': [0] * n,
    })


class TestMLIntegrationInit:
    """Tests for MLIntegration initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        integration = MLIntegration()
        
        assert integration.model is None
        assert integration.fitted == False
        assert integration.model_path is not None
    
    def test_custom_path_init(self):
        """Test initialization with custom model path."""
        integration = MLIntegration(model_path='/custom/path/model.joblib')
        
        assert integration.model_path == '/custom/path/model.joblib'


class TestModelLoading:
    """Tests for model loading."""
    
    def test_load_existing_model(self):
        """Test loading existing model from file."""
        integration = MLIntegration()
        
        # Try to load the existing model
        result = integration.load_model()
        
        # Model should exist and load
        assert result == True
        assert integration.fitted == True
        assert integration.model is not None
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model."""
        integration = MLIntegration(model_path='/nonexistent/path/model.joblib')
        
        result = integration.load_model()
        
        assert result == False
        assert integration.fitted == False


class TestAnomalyPrediction:
    """Tests for anomaly prediction."""
    
    def test_predict_with_loaded_model(self):
        """Test predicting anomalies with loaded model."""
        integration = MLIntegration()
        integration.load_model()
        
        df = make_raw_df(3)
        
        result_df = integration.predict(df)
        
        # Should have prediction columns
        assert 'is_anomaly' in result_df.columns
        assert 'anomaly_score' in result_df.columns
        assert 'ml_confidence' in result_df.columns
        
        # Should have predicted values
        assert len(result_df) == 3
        # is_anomaly should be boolean or convertible to int
        assert result_df['is_anomaly'].dtype in ['bool', 'int64', 'int32']
    
    def test_predict_fallback_when_model_not_loaded(self):
        """Test prediction fallback when model not loaded."""
        integration = MLIntegration(model_path='/nonexistent/path/model.joblib')
        # Don't load the model
        
        df = make_raw_df(3)
        
        result_df = integration.predict(df)
        
        # Should have fallback predictions, explicitly flagged as unavailable
        assert 'is_anomaly' in result_df.columns
        assert 'anomaly_score' in result_df.columns
        assert 'ml_confidence' in result_df.columns
        assert result_df['ml_model_available'].iloc[0] == False


class TestFeatureHandling:
    """Tests for feature handling."""
    
    def test_ensure_features(self):
        """Test feature column handling."""
        integration = MLIntegration()
        
        # Create DataFrame with minimal features
        df = pd.DataFrame({
            'courses_accessed': [5, 50],
            'accounts_accessed': [3, 20],
            'api_requests': [100, 500]
        })
        
        result_df = integration._ensure_features(df)
        
        # Should have added missing features
        for feature in ['data_mb', 'token_age_minutes', 'after_hours']:
            assert feature in result_df.columns
    
    def test_handle_missing_values(self):
        """Test handling of missing feature values."""
        integration = MLIntegration()
        integration.load_model()
        
        # Raw frame with missing values
        df = make_raw_df(4)
        df.loc[1, 'api_requests'] = np.nan
        df.loc[2, 'data_mb'] = np.nan
        
        result_df = integration.predict(df)
        
        # Should have predictions despite missing values
        assert 'is_anomaly' in result_df.columns


class TestMLAnalysis:
    """Tests for ML analysis results."""
    
    def test_get_ml_analysis(self):
        """Test getting ML analysis results."""
        integration = MLIntegration()
        integration.load_model()
        
        df = make_raw_df(5)
        
        result_df = integration.predict(df)
        analysis = integration.get_ml_analysis(result_df)
        
        assert 'total_events' in analysis
        assert 'anomalies_detected' in analysis
        assert 'anomaly_rate' in analysis
        assert 'avg_anomaly_score' in analysis
    
    def test_get_anomalies(self):
        """Test filtering anomalous events."""
        integration = MLIntegration()
        integration.load_model()
        
        df = make_raw_df(5)
        
        result_df = integration.predict(df)
        anomalies = integration.get_anomalies(result_df, threshold=0.5)
        
        # Should return subset of events
        assert len(anomalies) <= len(result_df)


class TestExplainAnomaly:
    """Tests for anomaly explanation."""
    
    def test_explain_anomaly(self):
        """Test explaining an anomaly."""
        integration = MLIntegration()
        
        row = pd.Series({
            'is_anomaly': True,
            'anomaly_score': 0.85,
            'ml_confidence': 0.72,
            'api_requests': 500,
            'data_mb': 50,
            'courses_accessed': 20,
            'accounts_accessed': 15,
            'token_age_minutes': 1500,
            'after_hours': 1
        })
        
        explanation = integration.explain_anomaly(row)
        
        assert explanation['is_anomaly'] == True
        assert explanation['anomaly_score'] == 0.85
        assert 'interpretation' in explanation
        assert 'key_features' in explanation
    
    def test_explain_normal_activity(self):
        """Test explaining normal activity."""
        integration = MLIntegration()
        
        row = pd.Series({
            'is_anomaly': False,
            'anomaly_score': 0.25,
            'ml_confidence': 0.5,
            'api_requests': 50,
            'data_mb': 2,
            'courses_accessed': 3,
            'accounts_accessed': 2,
            'token_age_minutes': 100,
            'after_hours': 0
        })
        
        explanation = integration.explain_anomaly(row)
        
        assert explanation['is_anomaly'] == False
        assert 'Normal activity' in explanation['interpretation']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
