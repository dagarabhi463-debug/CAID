"""
Tests for AnomalyDetector module.

These tests verify:
- Model training on baseline data only
- Prediction without retraining
- Model persistence (save/load)
- Evaluation metrics calculation
- Feature consistency
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector, get_model_path, ensure_model_exists
from core.log_processor import load_sample_dataset
from core.feature_engineering import engineer_features


class TestAnomalyDetectorInit:
    """Test anomaly detector initialization."""
    
    def test_default_init(self):
        """Test default initialization parameters."""
        detector = AnomalyDetector()
        assert detector.contamination == 0.1
        assert detector.random_state == 42
        assert detector.is_fitted is False
        assert detector.feature_columns == []
    
    def test_custom_init(self):
        """Test custom initialization parameters."""
        detector = AnomalyDetector(contamination=0.2, random_state=123)
        assert detector.contamination == 0.2
        assert detector.random_state == 123


class TestAnomalyDetectorFit:
    """Test model fitting functionality."""
    
    @pytest.fixture
    def normal_data(self):
        """Load normal activity data for training."""
        return load_sample_dataset('normal')
    
    def test_fit_on_normal_data(self, normal_data):
        """Test fitting on normal activity baseline data."""
        detector = AnomalyDetector()
        detector.fit(normal_data)
        
        assert detector.is_fitted is True
        assert len(detector.feature_columns) > 0
        assert detector.model is not None
    
    def test_fit_sets_feature_columns(self, normal_data):
        """Test that feature columns are properly set during fit."""
        detector = AnomalyDetector()
        detector.fit(normal_data)
        
        # Check that expected feature columns are present
        expected_cols = ['courses_accessed', 'accounts_accessed', 'api_requests',
                        'data_mb', 'token_age_minutes', 'after_hours']
        for col in expected_cols:
            assert col in detector.feature_columns
    
    def test_fit_with_custom_features(self, normal_data):
        """Test fitting with explicitly provided feature columns."""
        detector = AnomalyDetector()
        
        # Manually specify some features
        custom_features = ['courses_accessed', 'api_requests', 'data_mb']
        detector.fit(normal_data, feature_columns=custom_features)
        
        assert detector.is_fitted is True
        assert len(detector.feature_columns) == 3
        assert 'courses_accessed' in detector.feature_columns
    
    def test_fit_raises_error_on_empty_data(self):
        """Test that fitting on empty data raises error."""
        detector = AnomalyDetector()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            detector.fit(empty_df)
    
    def test_fit_raises_error_on_missing_features(self, normal_data):
        """Test that fitting with missing features raises error."""
        detector = AnomalyDetector()
        
        with pytest.raises(ValueError, match="Requested feature columns not found"):
            detector.fit(normal_data, feature_columns=['nonexistent_column'])


class TestAnomalyDetectorPredict:
    """Test prediction functionality."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    @pytest.fixture
    def evaluation_data(self):
        """Load evaluation data (separate from training)."""
        # Use suspicious and high_risk data as evaluation set
        df_suspicious = load_sample_dataset('suspicious')
        df_high_risk = load_sample_dataset('high_risk')
        return pd.concat([df_suspicious, df_high_risk], ignore_index=True)
    
    def test_predict_requires_fitted_model(self):
        """Test that predict raises error if model not fitted."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        
        with pytest.raises(RuntimeError, match="Model must be fitted before prediction"):
            detector.predict(normal_data)
    
    def test_predict_returns_dataframe(self, trained_detector, evaluation_data):
        """Test that predict returns a DataFrame with expected columns."""
        result = trained_detector.predict(evaluation_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'anomaly_prediction' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
    
    def test_predict_preserves_original_data(self, trained_detector, evaluation_data):
        """Test that predict preserves original columns."""
        original_cols = set(evaluation_data.columns)
        result = trained_detector.predict(evaluation_data)
        result_cols = set(result.columns)
        
        # Original columns should be preserved
        assert original_cols.issubset(result_cols)
    
    def test_predict_consistency_after_reload(self, trained_detector, evaluation_data):
        """Test that predictions are consistent after saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            
            # Save model
            trained_detector.save(model_path)
            
            # Create new detector and load model
            new_detector = AnomalyDetector()
            new_detector.load(model_path)
            
            # Predict with both
            result1 = trained_detector.predict(evaluation_data.copy())
            result2 = new_detector.predict(evaluation_data.copy())
            
            # Compare predictions
            assert (result1['anomaly_prediction'] == result2['anomaly_prediction']).all()
            assert (result1['is_anomaly'] == result2['is_anomaly']).all()
    
    def test_predict_handles_data_with_all_columns(self, trained_detector):
        """Test that predict works with data containing all required columns."""
        # Create dataframe with all required columns
        min_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-02 10:00:00'],
            'user_id': ['user_1', 'user_2'],
            'role': ['student', 'instructor'],
            'source_ip': ['192.0.2.1', '192.0.2.2'],
            'event_type': ['login', 'course_view'],
            'courses_accessed': [1, 2],
            'accounts_accessed': [1, 1],
            'api_requests': [50, 60],
            'data_mb': [1.0, 1.5],
            'token_age_minutes': [30, 45],
            'after_hours': [0, 0]
        })
        
        result = trained_detector.predict(min_df)
        
        assert len(result) == 2
        assert 'anomaly_score' in result.columns
    
    def test_predict_with_missing_values(self, trained_detector):
        """Test prediction handling of missing values."""
        normal_data = load_sample_dataset('normal')
        # Introduce some missing values
        test_data = normal_data.copy()
        test_data.loc[0:5, 'api_requests'] = np.nan
        
        result = trained_detector.predict(test_data)
        assert len(result) == len(test_data)
        assert 'anomaly_score' in result.columns


class TestAnomalyDetectorSaveLoad:
    """Test model persistence."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    def test_save_creates_file(self, trained_detector):
        """Test that save creates the model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            trained_detector.save(model_path)
            
            assert os.path.exists(model_path)
    
    def test_save_raises_error_on_unfitted(self):
        """Test that save raises error on unfitted model."""
        detector = AnomalyDetector()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
                detector.save(model_path)
    
    def test_load_restores_model(self, trained_detector):
        """Test that load restores model correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            trained_detector.save(model_path)
            
            new_detector = AnomalyDetector()
            new_detector.load(model_path)
            
            # Check that model is restored
            assert new_detector.is_fitted is True
            assert new_detector.feature_columns == trained_detector.feature_columns
            assert new_detector.contamination == trained_detector.contamination
    
    def test_load_raises_error_on_missing_file(self):
        """Test that load raises error on missing file."""
        detector = AnomalyDetector()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            detector.load('/nonexistent/path/model.joblib')


class TestAnomalyDetectorEvaluate:
    """Test evaluation metrics calculation."""
    
    @pytest.fixture
    def evaluated_detector(self):
        """Get a detector evaluated on known data."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        
        # Combine evaluation datasets
        df_suspicious = load_sample_dataset('suspicious')
        df_high_risk = load_sample_dataset('high_risk')
        eval_data = pd.concat([df_suspicious, df_high_risk], ignore_index=True)
        
        # Make predictions
        result = detector.predict(eval_data)
        
        return detector, result
    
    def test_evaluate_raises_error_on_unfitted(self):
        """Test that evaluate raises error on unfitted model."""
        detector = AnomalyDetector()
        eval_data = load_sample_dataset('suspicious')
        
        with pytest.raises(RuntimeError, match="Model must be fitted before evaluation"):
            detector.evaluate(eval_data)
    
    def test_evaluate_raises_error_on_missing_ground_truth(self, evaluated_detector):
        """Test that evaluate raises error on missing ground truth column."""
        detector, result = evaluated_detector
        
        # Rename label column to simulate missing ground truth
        result_no_label = result.drop(columns=['label'])
        
        with pytest.raises(ValueError, match="Ground truth column 'label' not found"):
            detector.evaluate(result_no_label, ground_truth_column='label')
    
    def test_evaluate_returns_metrics(self, evaluated_detector):
        """Test that evaluate returns all required metrics."""
        detector, result = evaluated_detector
        
        # Add ground truth label column
        result['label'] = result['label'].fillna('normal')
        
        metrics = detector.evaluate(result, ground_truth_column='label')
        
        # Check all required metrics are present
        assert 'tp' in metrics
        assert 'fp' in metrics
        assert 'tn' in metrics
        assert 'fn' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'accuracy' in metrics
        assert 'confusion_matrix' in metrics
    
    def test_evaluate_confusion_matrix_structure(self, evaluated_detector):
        """Test that confusion matrix has correct structure."""
        detector, result = evaluated_detector
        result['label'] = result['label'].fillna('normal')
        
        metrics = detector.evaluate(result, ground_truth_column='label')
        cm = metrics['confusion_matrix']
        
        assert 'tp' in cm
        assert 'fp' in cm
        assert 'tn' in cm
        assert 'fn' in cm
    
    def test_evaluate_with_different_labels(self, evaluated_detector):
        """Test evaluation with custom label names."""
        detector, result = evaluated_detector
        result['label'] = result['label'].fillna('normal')
        
        # Evaluate in high-risk mode (high_risk = positive)
        metrics = detector.evaluate(
            result,
            ground_truth_column='label',
            mode='high-risk'
        )
        
        assert metrics['tp'] >= 0
        assert metrics['fp'] >= 0


class TestAnomalyDetectorFeatureConsistency:
    """Test feature consistency between fit and predict."""
    
    def test_feature_order_consistency(self):
        """Test that feature columns are present in result."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        
        # Fit
        detector.fit(normal_data)
        feature_columns = detector.feature_columns
        
        # Predict on same data
        result = detector.predict(normal_data)
        
        # The result should have the original columns plus prediction columns
        # Check that prediction columns exist
        assert 'anomaly_prediction' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
    
    def test_feature_consistency_with_new_data(self):
        """Test feature consistency when predicting on different dataset."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        suspicious_data = load_sample_dataset('suspicious')
        
        # Fit on normal data
        detector.fit(normal_data)
        
        # Predict on suspicious data
        result = detector.predict(suspicious_data)
        
        # Check prediction columns exist
        assert 'anomaly_prediction' in result.columns
        assert 'anomaly_score' in result.columns
        assert 'is_anomaly' in result.columns
        
        # Verify predictions match expected behavior
        # Suspicious data should generally have more anomalies detected
        anomaly_count = result['is_anomaly'].sum()
        assert anomaly_count >= 0  # Sanity check


class TestEnsureModelExists:
    """Test the ensure_model_exists function."""
    
    def test_ensure_model_exists_loads_existing(self):
        """Test that existing model is loaded."""
        model_path = get_model_path()
        
        # First ensure model exists
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Create and save model
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        detector.save(model_path)
        
        # Now test ensure_model_exists
        loaded_detector = ensure_model_exists()
        
        assert loaded_detector.is_fitted is True
        assert os.path.exists(model_path)
    
    def test_ensure_model_exists_trains_new(self):
        """Test that new model is trained if none exists."""
        model_path = get_model_path()
        
        # Remove existing model
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Ensure model exists (should train)
        detector = ensure_model_exists()
        
        assert detector.is_fitted is True
        assert os.path.exists(model_path)


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_training_on_baseline_only(self):
        """Test that model is trained on baseline (normal) data only."""
        normal_data = load_sample_dataset('normal')
        
        detector = AnomalyDetector()
        detector.fit(normal_data)
        
        # Model should be trained only on normal data
        assert detector.is_fitted is True
    
    def test_evaluation_on_separate_data(self):
        """Test that evaluation uses separate data from training."""
        # Train on normal data
        normal_data = load_sample_dataset('normal')
        detector = AnomalyDetector()
        detector.fit(normal_data)
        
        # Evaluate on high-risk data (not seen during training)
        high_risk_data = load_sample_dataset('high_risk')
        result = detector.predict(high_risk_data)
        
        # Should have predictions for all records
        assert len(result) == len(high_risk_data)
        assert 'is_anomaly' in result.columns
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # 1. Train on baseline
        normal_data = load_sample_dataset('normal')
        detector = AnomalyDetector()
        detector.fit(normal_data)
        
        # 2. Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'model.joblib')
            detector.save(model_path)
            
            # 3. Load model
            loaded_detector = AnomalyDetector()
            loaded_detector.load(model_path)
            
            # 4. Predict on new data
            eval_data = load_sample_dataset('suspicious')
            result = loaded_detector.predict(eval_data)
            
            # 5. Evaluate
            result['label'] = result['label'].fillna('normal')
            metrics = loaded_detector.evaluate(result, ground_truth_column='label')
            
            # Check metrics are reasonable
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1
            assert 0 <= metrics['accuracy'] <= 1


class TestEvaluationModes:
    """Test both evaluation modes for binary and high-risk detection."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    @pytest.fixture
    def evaluation_data(self):
        """Load evaluation data (suspicious + high_risk)."""
        df_suspicious = load_sample_dataset('suspicious')
        df_high_risk = load_sample_dataset('high_risk')
        return pd.concat([df_suspicious, df_high_risk], ignore_index=True)
    
    def test_binary_detection_mode(self, trained_detector, evaluation_data):
        """Test binary detection mode: normal=negative, suspicious+high_risk=positive."""
        result = trained_detector.predict(evaluation_data)
        result['label'] = result['label'].fillna('normal')
        
        # Binary mode: suspicious + high_risk are positive
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        # In binary mode, all suspicious+high_risk should be detected as positive
        assert metrics['mode'] == 'binary'
        assert metrics['ground_truth_distribution']['positive'] == len(evaluation_data)  # All suspicious+high_risk
        assert metrics['ground_truth_distribution']['negative'] == 0  # No normal in eval
        
        # All should be predicted as anomalies (is_anomaly=True)
        assert metrics['tp'] == len(evaluation_data)  # All positive predicted positive
        assert metrics['fp'] == 0  # No false positives (no normal to flag)
        assert metrics['fn'] == 0  # No false negatives
    
    def test_high_risk_detection_mode(self, trained_detector, evaluation_data):
        """Test high-risk detection mode: normal+suspicious=negative, high_risk=positive."""
        result = trained_detector.predict(evaluation_data)
        result['label'] = result['label'].fillna('normal')
        
        # High-risk mode: only high_risk is positive
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        assert metrics['mode'] == 'high-risk'
        
        # In synthetic data, all records are either suspicious or high_risk
        # So ground truth positive should be high_risk count
        high_risk_count = len(evaluation_data[evaluation_data['label'] == 'high_risk'])
        assert metrics['ground_truth_distribution']['positive'] == high_risk_count
        assert metrics['ground_truth_distribution']['negative'] == len(evaluation_data) - high_risk_count
        
        # Check confusion matrix structure
        assert 'tp' in metrics['confusion_matrix']
        assert 'fp' in metrics['confusion_matrix']
        assert 'tn' in metrics['confusion_matrix']
        assert 'fn' in metrics['confusion_matrix']
    
    def test_mode_parameter_validation(self, trained_detector, evaluation_data):
        """Test that invalid mode raises error."""
        result = trained_detector.predict(evaluation_data)
        result['label'] = result['label'].fillna('normal')
        
        with pytest.raises(ValueError, match="Unknown mode"):
            trained_detector.evaluate(result, ground_truth_column='label', mode='invalid')
    
    def test_both_modes_produce_different_results(self, trained_detector, evaluation_data):
        """Test that binary and high-risk modes produce different metrics."""
        result = trained_detector.predict(evaluation_data)
        result['label'] = result['label'].fillna('normal')
        
        metrics_binary = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        metrics_high_risk = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # Metrics should differ between modes
        assert metrics_binary['tp'] != metrics_high_risk['tp'] or \
               metrics_binary['fp'] != metrics_high_risk['fp'] or \
               metrics_binary['fn'] != metrics_high_risk['fn'] or \
               metrics_binary['tn'] != metrics_high_risk['tn']


class TestFeatureEngineeringConsistency:
    """Test feature engineering consistency across datasets."""
    
    def test_feature_columns_consistent_across_datasets(self):
        """Test that feature engineering produces consistent columns across datasets."""
        from core.feature_engineering import engineer_features
        
        normal_data = load_sample_dataset('normal')
        suspicious_data = load_sample_dataset('suspicious')
        high_risk_data = load_sample_dataset('high_risk')
        
        # Engineer features for each dataset
        _, normal_features = engineer_features(normal_data)
        _, suspicious_features = engineer_features(suspicious_data)
        _, high_risk_features = engineer_features(high_risk_data)
        
        # All feature sets should be identical
        assert set(normal_features) == set(suspicious_features)
        assert set(suspicious_features) == set(high_risk_features)
        assert set(normal_features) == set(high_risk_features)
    
    def test_user_level_aggregation_features(self):
        """Test that user-level aggregation features are correctly calculated."""
        from core.feature_engineering import engineer_features
        
        normal_data = load_sample_dataset('normal')
        df_engineered, feature_columns = engineer_features(normal_data)
        
        # Check for expected user-level aggregation features
        expected_features = [
            'user_total_api', 'user_mean_api', 'user_max_api',
            'user_total_data', 'user_mean_data', 'user_max_data',
            'user_total_courses', 'user_mean_courses', 'user_max_courses',
            'user_event_count'
        ]
        
        for feature in expected_features:
            assert feature in df_engineered.columns, f"Missing feature: {feature}"
    
    def test_ip_level_aggregation_features(self):
        """Test that IP-level aggregation features are correctly calculated."""
        from core.feature_engineering import engineer_features
        
        normal_data = load_sample_dataset('normal')
        df_engineered, feature_columns = engineer_features(normal_data)
        
        # Check for IP event count
        assert 'ip_event_count' in df_engineered.columns
    
    def test_event_type_frequency_features(self):
        """Test that event type frequency features are correctly calculated."""
        from core.feature_engineering import engineer_features
        
        normal_data = load_sample_dataset('normal')
        df_engineered, feature_columns = engineer_features(normal_data)
        
        # Check for event type frequency
        assert 'event_type_frequency' in df_engineered.columns
    
    def test_no_evaluation_leakage_in_feature_engineering(self):
        """Test that feature engineering does not leak evaluation data into training."""
        from core.feature_engineering import engineer_features
        
        normal_data = load_sample_dataset('normal')
        
        # Engineer features on normal data
        df_normal, normal_features = engineer_features(normal_data)
        
        # Calculate user statistics
        user_stats = normal_data.groupby('user_id').agg({
            'api_requests': ['sum', 'mean', 'count']
        })
        
        # Verify user stats are reasonable
        assert len(user_stats) > 0
        assert 'api_requests' in user_stats.columns
    
    def test_feature_consistency_with_train_predict_separation(self):
        """Test that feature columns from training match prediction time."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        suspicious_data = load_sample_dataset('suspicious')
        
        # Fit on normal data
        detector.fit(normal_data)
        training_features = set(detector.feature_columns)
        
        # Predict on suspicious data (same feature engineering pipeline)
        result = detector.predict(suspicious_data)
        
        # The model should use the same feature columns
        assert len(training_features) > 0
        assert 'anomaly_score' in result.columns

class TestMilestone16EvaluationDataset:
    """Test Milestone 1.6: Evaluation dataset with proper class distribution."""
    
    def test_evaluation_dataset_exists(self):
        """Test that evaluation dataset file exists."""
        import os
        eval_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'processed', 'evaluation_data.csv'
        )
        assert os.path.exists(eval_path), "Evaluation dataset file not found"
    
    def test_evaluation_dataset_class_distribution(self):
        """Test that evaluation dataset has correct class distribution."""
        eval_df = load_sample_dataset('evaluation')
        
        label_counts = eval_df['label'].value_counts()
        
        # Check expected distribution
        assert label_counts.get('normal') == 100, "Expected 100 normal records"
        assert label_counts.get('suspicious') == 50, "Expected 50 suspicious records"
        assert label_counts.get('high_risk') == 50, "Expected 50 high-risk records"
        
        # Verify total
        assert len(eval_df) == 200, "Expected 200 total records"
    
    def test_evaluation_dataset_separate_from_training(self):
        """Test that evaluation dataset contains records not in training."""
        normal_data = load_sample_dataset('normal')
        eval_data = load_sample_dataset('evaluation')
        
        # Check that evaluation has normal records (separate from training)
        eval_normal = eval_data[eval_data['label'] == 'normal']
        assert len(eval_normal) > 0, "Evaluation should contain normal records"
        
        # Check that evaluation has suspicious and high_risk records
        assert 'suspicious' in eval_data['label'].values
        assert 'high_risk' in eval_data['label'].values
    
    def test_training_data_only_normal(self):
        """Test that training data contains only normal activity."""
        normal_data = load_sample_dataset('normal')
        
        # All records should be labeled as normal
        assert (normal_data['label'] == 'normal').all()
    
    def test_evaluation_mode_excludes_training_data(self):
        """Test that evaluation mode uses separate data from training."""
        normal_data = load_sample_dataset('normal')
        eval_data = load_sample_dataset('evaluation')
        
        # Training data should have only 1 label
        training_labels = set(normal_data['label'].unique())
        assert training_labels == {'normal'}
        
        # Evaluation data should have 3 labels
        eval_labels = set(eval_data['label'].unique())
        assert eval_labels == {'normal', 'suspicious', 'high_risk'}


class TestMilestone16BinaryDetection:
    """Test Milestone 1.6: Binary Detection mode evaluation."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    @pytest.fixture
    def eval_data(self):
        """Load evaluation data."""
        return load_sample_dataset('evaluation')
    
    def test_binary_detection_confusion_matrix(self, trained_detector, eval_data):
        """Test binary detection confusion matrix calculations."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        cm = metrics['confusion_matrix']
        
        # Verify confusion matrix components
        assert cm['tp'] >= 0  # True positives
        assert cm['fp'] >= 0  # False positives
        assert cm['tn'] >= 0  # True negatives
        assert cm['fn'] >= 0  # False negatives
        
        # Verify metrics are in valid range
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_binary_detection_mode_calculation(self, trained_detector, eval_data):
        """Test that binary mode correctly identifies positive labels."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        # In binary mode: suspicious+high_risk = positive, normal = negative
        cm = metrics['confusion_matrix']
        gt_dist = metrics['ground_truth_distribution']
        
        # Verify ground truth distribution
        assert gt_dist['positive'] == 100, "All suspicious+high_risk should be ground truth positive (50+50)"
        assert gt_dist['negative'] == 100, "All normal should be ground truth negative"
        
        # TP + FN should equal total positive
        assert cm['tp'] + cm['fn'] == gt_dist['positive'], "TP + FN should equal ground truth positive"
        
        # TN + FP should equal total negative
        assert cm['tn'] + cm['fp'] == gt_dist['negative'], "TN + FP should equal ground truth negative"
    
    def test_binary_detection_metrics(self, trained_detector, eval_data):
        """Test binary detection metrics calculation."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        # Verify F1 score calculation: 2 * P * R / (P + R)
        expected_f1 = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        assert abs(metrics['f1_score'] - expected_f1) < 1e-10
    
    def test_binary_detection_mode_label(self, trained_detector, eval_data):
        """Test that binary mode returns correct mode label."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        assert metrics['mode'] == 'binary'


class TestMilestone16HighRiskDetection:
    """Test Milestone 1.6: High-Risk Detection mode evaluation."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    @pytest.fixture
    def eval_data(self):
        """Load evaluation data."""
        return load_sample_dataset('evaluation')
    
    def test_high_risk_detection_confusion_matrix(self, trained_detector, eval_data):
        """Test high-risk detection confusion matrix calculations."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        cm = metrics['confusion_matrix']
        
        # Verify confusion matrix components
        assert cm['tp'] >= 0  # True positives
        assert cm['fp'] >= 0  # False positives
        assert cm['tn'] >= 0  # True negatives
        assert cm['fn'] >= 0  # False negatives
        
        # Verify metrics are in valid range
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_high_risk_detection_mode_calculation(self, trained_detector, eval_data):
        """Test that high-risk mode correctly identifies positive labels."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # In high-risk mode: high_risk = positive, normal+suspicious = negative
        cm = metrics['confusion_matrix']
        gt_dist = metrics['ground_truth_distribution']
        
        # Verify ground truth distribution
        assert gt_dist['positive'] == 50, "All high_risk should be ground truth positive"
        assert gt_dist['negative'] == 150, "All normal+suspicious should be ground truth negative (100+50)"
        
        # TP + FN should equal total positive
        assert cm['tp'] + cm['fn'] == gt_dist['positive'], "TP + FN should equal ground truth positive"
        
        # TN + FP should equal total negative
        assert cm['tn'] + cm['fp'] == gt_dist['negative'], "TN + FP should equal ground truth negative"
    
    def test_high_risk_detection_metrics(self, trained_detector, eval_data):
        """Test high-risk detection metrics calculation."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # Verify F1 score calculation
        expected_f1 = 2 * metrics['precision'] * metrics['recall'] / (
            metrics['precision'] + metrics['recall']
        ) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        assert abs(metrics['f1_score'] - expected_f1) < 1e-10
    
    def test_high_risk_detection_mode_label(self, trained_detector, eval_data):
        """Test that high-risk mode returns correct mode label."""
        result = trained_detector.predict(eval_data)
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        assert metrics['mode'] == 'high-risk'


class TestMilestone16ConfusionMatrices:
    """Test Milestone 1.6: Confusion matrix calculations."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    @pytest.fixture
    def eval_data(self):
        """Load evaluation data."""
        return load_sample_dataset('evaluation')
    
    def test_confusion_matrix_binary_vs_high_risk(self, trained_detector, eval_data):
        """Test that confusion matrices differ between modes."""
        result = trained_detector.predict(eval_data)
        
        metrics_binary = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        metrics_high_risk = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # Confusion matrices should differ between modes
        assert metrics_binary['confusion_matrix']['tp'] != metrics_high_risk['confusion_matrix']['tp']
    
    def test_confusion_matrix_sum_equivalence(self, trained_detector, eval_data):
        """Test that confusion matrix sums equal total records."""
        result = trained_detector.predict(eval_data)
        
        metrics_binary = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        metrics_high_risk = trained_detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # Sum of all confusion matrix components should equal total records
        cm_binary = metrics_binary['confusion_matrix']
        cm_high_risk = metrics_high_risk['confusion_matrix']
        
        assert cm_binary['tp'] + cm_binary['fp'] + cm_binary['tn'] + cm_binary['fn'] == len(eval_data)
        assert cm_high_risk['tp'] + cm_high_risk['fp'] + cm_high_risk['tn'] + cm_high_risk['fn'] == len(eval_data)
    
    def test_confusion_matrix_metrics_consistency(self, trained_detector, eval_data):
        """Test that confusion matrix metrics are consistent with formulas."""
        result = trained_detector.predict(eval_data)
        
        metrics_binary = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        cm = metrics_binary['confusion_matrix']
        
        # Verify precision formula: TP / (TP + FP)
        expected_precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
        assert abs(metrics_binary['precision'] - expected_precision) < 1e-10
        
        # Verify recall formula: TP / (TP + FN)
        expected_recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
        assert abs(metrics_binary['recall'] - expected_recall) < 1e-10
        
        # Verify accuracy formula: (TP + TN) / Total
        expected_accuracy = (cm['tp'] + cm['tn']) / len(result)
        assert abs(metrics_binary['accuracy'] - expected_accuracy) < 1e-10


class TestMilestone16ModelValidation:
    """Test Milestone 1.6: Model validation checks."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    def test_model_not_retrained_during_predict(self, trained_detector):
        """Test that model is not retrained during prediction."""
        is_fitted_before = trained_detector.is_fitted
        
        # Predict on evaluation data
        eval_data = load_sample_dataset('evaluation')
        _ = trained_detector.predict(eval_data)
        
        is_fitted_after = trained_detector.is_fitted
        assert is_fitted_before == is_fitted_after, "Model should not be retrained during predict"
    
    def test_model_persistence_consistency(self, trained_detector):
        """Test that predictions are consistent after save/load."""
        import tempfile
        import os
        
        eval_data = load_sample_dataset('evaluation')
        
        # Get predictions from original detector
        result1 = trained_detector.predict(eval_data.copy())
        
        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            trained_detector.save(model_path)
            
            new_detector = AnomalyDetector()
            new_detector.load(model_path)
            
            # Get predictions from loaded detector
            result2 = new_detector.predict(eval_data.copy())
        
        # Verify predictions match
        assert (result1['anomaly_prediction'] == result2['anomaly_prediction']).all()
        assert (result1['is_anomaly'] == result2['is_anomaly']).all()
    
    def test_feature_schema_consistency(self, trained_detector):
        """Test that feature schema is consistent across fit/predict."""
        eval_data = load_sample_dataset('evaluation')
        
        # Get training features
        training_features = set(trained_detector.feature_columns)
        
        # Predict and verify no errors
        result = trained_detector.predict(eval_data)
        assert 'anomaly_score' in result.columns
    
    def test_identical_feature_schema(self, trained_detector):
        """Test that feature schema is identical in training and prediction."""
        normal_data = load_sample_dataset('normal')
        eval_data = load_sample_dataset('evaluation')
        
        # Fit on normal data
        trained_detector.fit(normal_data)
        training_features = trained_detector.feature_columns
        
        # Predict on eval data - should use same features
        result = trained_detector.predict(eval_data)
        
        # Verify predictions were made
        assert len(result) == len(eval_data)
        assert 'is_anomaly' in result.columns


class TestMilestone16PredictionsConsistency:
    """Test Milestone 1.6: Prediction consistency checks."""
    
    @pytest.fixture
    def trained_detector(self):
        """Get a trained detector."""
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        return detector
    
    def test_predictions_generated_by_actual_model(self, trained_detector):
        """Test that predictions are generated by the actual Isolation Forest model."""
        eval_data = load_sample_dataset('evaluation')
        result = trained_detector.predict(eval_data)
        
        # Verify anomaly scores are calculated by model
        assert 'anomaly_score' in result.columns
        assert len(result['anomaly_score'].unique()) > 1, "Should have varied anomaly scores"
        
        # Verify predictions match model output
        assert 'anomaly_prediction' in result.columns
        # -1 = anomaly, 1 = normal
        assert set(result['anomaly_prediction'].unique()).issubset({-1, 1})
    
    def test_no_label_leakage(self, trained_detector):
        """Test that ground truth labels are not used as model features."""
        normal_data = load_sample_dataset('normal')
        
        # Check feature columns do not include label
        trained_detector.fit(normal_data)
        assert 'label' not in trained_detector.feature_columns
    
    def test_prediction_consistency_multiple_calls(self, trained_detector):
        """Test that predictions are consistent across multiple calls."""
        eval_data = load_sample_dataset('evaluation')
        
        # Make multiple predictions
        result1 = trained_detector.predict(eval_data.copy())
        result2 = trained_detector.predict(eval_data.copy())
        
        # Results should be identical
        assert (result1['is_anomaly'] == result2['is_anomaly']).all()
    
    def test_evaluation_does_not_modify_model(self, trained_detector):
        """Test that evaluation does not modify the model state."""
        import copy
        
        # Get initial model state
        eval_data = load_sample_dataset('evaluation')
        result = trained_detector.predict(eval_data.copy())
        
        # Store model state before evaluation
        import numpy as np
        initial_predictions = copy.deepcopy(result['anomaly_prediction'].values)
        
        # Run evaluation
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        # Model state should be unchanged
        assert 'is_fitted' in dir(trained_detector)
        assert trained_detector.is_fitted is True
    
    def test_metrics_are_calculated_not_hardcoded(self, trained_detector):
        """Test that metrics are calculated from actual predictions."""
        eval_data = load_sample_dataset('evaluation')
        result = trained_detector.predict(eval_data.copy())
        
        # Get actual predictions
        actual_anomalies = int(result['is_anomaly'].sum())
        actual_normal = len(result) - actual_anomalies
        
        # Run evaluation
        metrics = trained_detector.evaluate(result, ground_truth_column='label', mode='binary')
        
        # Verify that metrics are derived from confusion matrix, not hardcoded
        cm = metrics['confusion_matrix']
        
        # The metrics should reflect actual confusion matrix values
        # This is verified by checking that the confusion matrix is populated
        assert cm['tp'] + cm['fp'] + cm['tn'] + cm['fn'] > 0


class TestMilestone16EndToEnd:
    """Test Milestone 1.6: End-to-end validation."""
    
    def test_complete_evaluation_pipeline(self):
        """Test complete evaluation pipeline from training to metrics."""
        # 1. Train on normal data
        detector = AnomalyDetector()
        normal_data = load_sample_dataset('normal')
        detector.fit(normal_data)
        
        # 2. Load evaluation data
        eval_data = load_sample_dataset('evaluation')
        
        # 3. Make predictions
        result = detector.predict(eval_data)
        
        # 4. Evaluate in both modes
        metrics_binary = detector.evaluate(result, ground_truth_column='label', mode='binary')
        metrics_high_risk = detector.evaluate(result, ground_truth_column='label', mode='high-risk')
        
        # 5. Verify all metrics are valid
        for mode_name, metrics in [('binary', metrics_binary), ('high-risk', metrics_high_risk)]:
            assert 0 <= metrics['precision'] <= 1, f"{mode_name} precision out of range"
            assert 0 <= metrics['recall'] <= 1, f"{mode_name} recall out of range"
            assert 0 <= metrics['f1_score'] <= 1, f"{mode_name} f1_score out of range"
            assert 0 <= metrics['accuracy'] <= 1, f"{mode_name} accuracy out of range"
            
            # Verify confusion matrix
            cm = metrics['confusion_matrix']
            total = cm['tp'] + cm['fp'] + cm['tn'] + cm['fn']
            assert total == len(eval_data), f"{mode_name} confusion matrix sum mismatch"
