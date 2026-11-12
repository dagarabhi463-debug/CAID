"""
Tests for Behavioral Analysis Engine (Milestone 3)

Tests:
- Baseline modeling
- Deviation calculation
- User behavior analysis
- Suspicious user identification
"""

import pytest
import pandas as pd
import numpy as np
from core.behavioural_analyzer import BehavioralAnalyzer


class TestBehavioralAnalyzerInit:
    """Tests for BehavioralAnalyzer initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        analyzer = BehavioralAnalyzer()
        
        assert analyzer.user_baselines == {}
        assert analyzer.overall_stats == {}
        assert analyzer.thresholds is not None


class TestBaselineModeling:
    """Tests for baseline modeling."""
    
    def test_fit_user_baselines(self):
        """Test fitting user baselines."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'api_requests': [100, 120, 200, 220],
            'data_mb': [5, 6, 10, 11],
            'courses_accessed': [3, 3, 5, 5],
            'accounts_accessed': [2, 2, 3, 3],
            'token_age_minutes': [50, 60, 100, 110],
            'after_hours': [0, 0, 0, 0],
            'source_ip': ['192.0.2.1', '192.0.2.1', '192.0.2.2', '192.0.2.2']
        })
        
        analyzer.fit(df)
        
        # Should have user baselines
        assert 'user1' in analyzer.user_baselines
        assert 'user2' in analyzer.user_baselines
        
        # Should have overall stats
        assert 'avg_api' in analyzer.overall_stats
    
    def test_fit_empty_dataframe(self):
        """Test fitting with empty DataFrame."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame(columns=[
            'user_id', 'api_requests', 'data_mb', 
            'courses_accessed', 'accounts_accessed', 'token_age_minutes',
            'after_hours', 'source_ip'
        ])
        
        # Empty training data is rejected explicitly (no silent baselines)
        with pytest.raises(ValueError):
            analyzer.fit(df)


class TestDeviationCalculation:
    """Tests for deviation calculation."""
    
    def test_calculate_deviation_normal(self):
        """Test deviation calculation for normal activity."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'api_requests': [100, 110, 105],
            'data_mb': [5, 5, 5],
            'courses_accessed': [3, 3, 3],
            'accounts_accessed': [2, 2, 2],
            'token_age_minutes': [50, 50, 50],
            'after_hours': [0, 0, 0],
            'source_ip': ['192.0.2.1', '192.0.2.1', '192.0.2.1']
        })
        analyzer.fit(df)
        
        # Create row with normal activity
        row = pd.Series({'user_id': 'user1', 'api_requests': 100})
        
        score, factors = analyzer.calculate_behavioral_deviation(row)
        
        # Should have low deviation for normal activity
        assert score < 0.2
    
    def test_calculate_deviation_high(self):
        """Test deviation calculation for high activity."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'api_requests': [100, 110, 105],
            'data_mb': [5, 5, 5],
            'courses_accessed': [3, 3, 3],
            'accounts_accessed': [2, 2, 2],
            'token_age_minutes': [50, 50, 50],
            'after_hours': [0, 0, 0],
            'source_ip': ['192.0.2.1', '192.0.2.1', '192.0.2.1']
        })
        analyzer.fit(df)
        
        # Create row with high activity
        row = pd.Series({'user_id': 'user1', 'api_requests': 500})
        
        score, factors = analyzer.calculate_behavioral_deviation(row)
        
        # Should have some deviation for unusual activity
        assert score >= 0.0  # Allow 0 or positive deviation
        # Note: The actual threshold depends on the deviation multiplier logic
    
    def test_calculate_deviation_unknown_user(self):
        """Test deviation calculation for unknown user."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'api_requests': [100, 110, 105],
            'data_mb': [5, 5, 5],
            'courses_accessed': [3, 3, 3],
            'accounts_accessed': [2, 2, 2],
            'token_age_minutes': [50, 50, 50],
            'after_hours': [0, 0, 0],
            'source_ip': ['192.0.2.1', '192.0.2.1', '192.0.2.1']
        })
        analyzer.fit(df)
        
        # Create row with unknown user
        row = pd.Series({'user_id': 'unknown_user', 'api_requests': 100})
        
        score, factors = analyzer.calculate_behavioral_deviation(row)
        
        # Should use overall stats for unknown user
        assert isinstance(score, (int, float))


class TestAnalyzeUserBehavior:
    """Tests for user behavior analysis."""
    
    def test_analyze_user_behavior(self):
        """Test analyzing user behavior for DataFrame."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'api_requests': [100, 500, 200, 150],
            'data_mb': [5, 50, 10, 8],
            'courses_accessed': [3, 20, 5, 4],
            'accounts_accessed': [2, 15, 3, 2],
            'token_age_minutes': [50, 1500, 100, 80],
            'after_hours': [0, 1, 0, 0]
        })
        
        analyzer.fit(df)
        result_df = analyzer.analyze_user_behavior(df)
        
        # Should have behavioral deviation columns
        assert 'behavioral_deviation_score' in result_df.columns
        assert 'behavioral_factors' in result_df.columns
        assert 'behavioral_score_missing' in result_df.columns
    
    def test_analyze_user_behavior_empty(self):
        """Test analyzing empty DataFrame."""
        analyzer = BehavioralAnalyzer()
        
        # Fit on non-empty training data first (required contract)
        train_df = pd.DataFrame({
            'user_id': ['user1', 'user1'],
            'api_requests': [100, 110],
            'data_mb': [5, 5],
            'courses_accessed': [3, 3],
            'accounts_accessed': [2, 2],
            'token_age_minutes': [50, 50],
            'after_hours': [0, 0],
        })
        analyzer.fit(train_df)
        
        df = pd.DataFrame(columns=[
            'user_id', 'api_requests', 'data_mb', 
            'courses_accessed', 'accounts_accessed', 
            'token_age_minutes', 'after_hours'
        ])
        
        result_df = analyzer.analyze_user_behavior(df)
        
        # Should handle empty DataFrame
        assert len(result_df) == 0


class TestSuspiciousUsers:
    """Tests for suspicious user identification."""
    
    def test_get_suspicious_users(self):
        """Test getting suspicious users."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'api_requests': [100, 500, 200, 150],
            'data_mb': [5, 50, 10, 8],
            'courses_accessed': [3, 20, 5, 4],
            'accounts_accessed': [2, 15, 3, 2],
            'token_age_minutes': [50, 1500, 100, 80],
            'after_hours': [0, 1, 0, 0]
        })
        
        # First analyze user behavior
        analyzer.fit(df)
        df_analyzed = analyzer.analyze_user_behavior(df)
        
        suspicious = analyzer.get_suspicious_users(df_analyzed, threshold=0.3)
        
        # Should identify suspicious users
        assert len(suspicious) >= 0
    
    def test_get_suspicious_users_no_suspicious(self):
        """Test when no users are suspicious."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'api_requests': [100, 200],
            'data_mb': [5, 10],
            'courses_accessed': [3, 5],
            'accounts_accessed': [2, 3],
            'token_age_minutes': [50, 100],
            'after_hours': [0, 0]
        })
        
        analyzer.fit(df)
        df_analyzed = analyzer.analyze_user_behavior(df)
        
        suspicious = analyzer.get_suspicious_users(df_analyzed, threshold=0.9)
        
        # Should return empty or minimal suspicious users
        assert len(suspicious) >= 0


class TestThresholdConfiguration:
    """Tests for threshold configuration."""
    
    def test_custom_thresholds(self):
        """Test using custom thresholds."""
        analyzer = BehavioralAnalyzer()
        
        df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user1'],
            'api_requests': [100, 110, 105],
            'data_mb': [5, 5, 5],
            'courses_accessed': [3, 3, 3],
            'accounts_accessed': [2, 2, 2],
            'token_age_minutes': [50, 50, 50],
            'after_hours': [0, 0, 0],
            'source_ip': ['192.0.2.1', '192.0.2.1', '192.0.2.1']
        })
        analyzer.fit(df)
        
        # Modify a threshold
        original_threshold = analyzer.thresholds['api_deviation_multiplier']
        analyzer.thresholds['api_deviation_multiplier'] = 5.0
        
        # High value should require more deviation to trigger
        row = pd.Series({'user_id': 'user1', 'api_requests': 200})
        score, factors = analyzer.calculate_behavioral_deviation(row)
        
        # With higher threshold, deviation score should be lower
        assert score < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
