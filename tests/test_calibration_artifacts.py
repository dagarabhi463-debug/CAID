"""Milestone 2.4 closure — v2_0 calibration artifact regression guard.

Verifies that a fingerprint-bound v2 detection-calibration artifact
exists for the CURRENT production components and loads cleanly. If any
component artifact is refit/re-saved, this test fails until
``scripts/calibrate_v2_detection.py`` is re-run — preventing silent
staleness like the pre-closure ``uncalibrated_default_thresholds`` mode.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope='module')
def components():
    from core.behavioural_analyzer import ensure_fitted_analyzer
    from core.log_processor import load_sample_dataset
    from core.sequence_detection import ensure_fitted_sequence_analyzer
    from core.session_analyzer import ensure_fitted_session_analyzer
    from models.anomaly_detector import ensure_model_exists
    detector = ensure_model_exists()
    behavioral = ensure_fitted_analyzer()
    normal = load_sample_dataset('normal')
    session = ensure_fitted_session_analyzer(train_df=normal)
    sequence = ensure_fitted_sequence_analyzer(train_df=normal)
    return detector, behavioral, session, sequence


def _fingerprints(detector, behavioral, session, sequence):
    from core.calibration import component_fingerprint, detector_fingerprint
    return {
        'detector': detector_fingerprint(detector),
        'feature_baseline': component_fingerprint(
            detector.baseline_manager, 'feature_baseline'),
        'behavioral_baseline': component_fingerprint(
            behavioral.baseline_manager, 'behavioral_baseline'),
        'session_baseline': component_fingerprint(
            session.baseline_manager, 'session_baseline'),
        'sequence_model': component_fingerprint(sequence, 'sequence_model'),
    }


class TestV2CalibrationArtifact:
    def test_v2_artifact_loads_for_current_fingerprints(self, components):
        from core.calibration import load_calibration
        detector, behavioral, session, sequence = components
        fps = _fingerprints(detector, behavioral, session, sequence)
        cal_path = os.path.join(
            ROOT, 'models', 'saved_models', 'calibration',
            f'v2_0_thresholds_{fps["detector"]}.joblib')
        assert os.path.exists(cal_path), (
            f'v2 calibration artifact missing for current detector '
            f'fingerprint {fps["detector"]} — re-run '
            f'scripts/calibrate_v2_detection.py')
        artifact = load_calibration(path=cal_path, detector=detector,
                                    fingerprints=fps)
        assert artifact['channel_profile'] == 'v2_0'
        assert artifact['dataset_role'] == 'calibration'
        assert artifact.get('detection_thresholds')

    def test_apply_detection_calibration_reaches_calibrated(
            self, components):
        from core.calibration import apply_detection_calibration
        from core.detection_engine import DetectionEngine
        detector, behavioral, session, sequence = components
        fps = _fingerprints(detector, behavioral, session, sequence)
        cal_path = os.path.join(
            ROOT, 'models', 'saved_models', 'calibration',
            f'v2_0_thresholds_{fps["detector"]}.joblib')
        engine = DetectionEngine(channel_profile='v2_0')
        artifact = apply_detection_calibration(
            engine, detector=detector, path=cal_path, fingerprints=fps)
        assert artifact is not None
        assert engine.calibration_status == 'calibrated'
        assert engine.threshold_version
