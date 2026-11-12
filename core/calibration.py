"""
Calibration Store for CAID — Milestone 1.9

Persists RiskEngine threshold calibrations as versioned artifacts and binds
each artifact to the detector fingerprint and calibration dataset identity
that produced it.

LIFECYCLE:
    Training -> Calibration (calibration split only) -> Threshold
    persistence -> Application loading -> Evaluation (holdout never used)

GUARANTEES:
- Thresholds are usable only with the detector they were calibrated for
  (fingerprint match), preventing silent application to a different model.
- Every artifact carries a threshold version id, calibration dataset role,
  dataset manifest hash and calibration statistics.
- Loading a missing/incompatible artifact raises CalibrationUnavailable —
  callers must fall back to an EXPLICITLY REPORTED uncalibrated mode
  (dataset-relative percentiles), never silently.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import joblib


class CalibrationUnavailable(RuntimeError):
    """Raised when no usable calibrated thresholds exist for a detector."""


ARTIFACT_VERSION = 1
SUPPORTED_ARTIFACT_VERSIONS = {1, 2}
V2_ARTIFACT_VERSION = 2
DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'models', 'saved_models', 'calibration')


def detector_fingerprint(detector) -> str:
    """
    Stable fingerprint identifying a fitted detector's decision surface:
    artifact version + feature schema + training hyperparameters + a
    content hash of the fitted model state. Two detectors trained on
    different data MUST NOT share a fingerprint (their score distributions
    differ, so thresholds calibrated on one do not transfer to the other).
    """
    payload = {
        'artifact_version': getattr(detector, 'ARTIFACT_VERSION', None),
        'feature_columns': list(getattr(detector, 'feature_columns', []) or []),
        'contamination': getattr(detector, 'contamination', None),
        'random_state': getattr(detector, 'random_state', None),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    base = hashlib.sha256(blob).hexdigest()[:16]

    # Canonical digest of fitted state. Pickle/joblib bytes are NOT
    # deterministic across save/load, so we hash the model's numeric
    # arrays directly (deterministic across loads of the same artifact).
    state_hash = _fitted_state_digest(detector)
    return f'{base}-{state_hash}'


def _fitted_state_digest(detector) -> str:
    """Deterministic digest of a fitted detector's decision surface."""
    import numpy as np
    h = hashlib.sha256()

    def feed_array(name, arr):
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    try:
        model = getattr(detector, 'model', None)
        for i, est in enumerate(getattr(model, 'estimators_', []) or []):
            tree = getattr(est, 'tree_', None)
            if tree is None:
                continue
            feed_array(f'tree{i}.children_left', tree.children_left)
            feed_array(f'tree{i}.children_right', tree.children_right)
            feed_array(f'tree{i}.feature', tree.feature)
            feed_array(f'tree{i}.threshold', tree.threshold)
            feed_array(f'tree{i}.value', tree.value)
        scaler = getattr(detector, 'scaler', None)
        feed_array('scaler.mean_', getattr(scaler, 'mean_', []))
        feed_array('scaler.scale_', getattr(scaler, 'scale_', []))
        cal = getattr(detector, '_score_calibration', None) or {}
        h.update(json.dumps(cal, sort_keys=True, default=str).encode())
        return h.hexdigest()[:16]
    except Exception:
        return 'unfitted'


def dataset_manifest_hash(manifest: Optional[Dict]) -> Optional[str]:
    """Hash a dataset manifest dict (or manifest file path) for binding."""
    if manifest is None:
        return None
    if isinstance(manifest, str) and os.path.exists(manifest):
        with open(manifest, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    blob = json.dumps(manifest, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def default_calibration_path(detector=None) -> str:
    """Default artifact path; fingerprint-scoped when detector is given."""
    os.makedirs(DEFAULT_DIR, exist_ok=True)
    if detector is not None:
        return os.path.join(DEFAULT_DIR,
                            f'thresholds_{detector_fingerprint(detector)}.joblib')
    return os.path.join(DEFAULT_DIR, 'thresholds.joblib')


def component_fingerprint(component, name: str = 'component') -> Optional[str]:
    """Best-effort deterministic fingerprint for a fitted component."""
    if component is None:
        return None
    fp = getattr(component, 'fingerprint', None)
    if callable(fp):
        try:
            return fp()
        except Exception:
            pass
    if hasattr(component, 'model') or hasattr(component, 'feature_columns'):
        try:
            return detector_fingerprint(component)
        except Exception:
            return None
    return None


def save_calibration(
    thresholds: Dict[str, float],
    calibration_info: Optional[Dict],
    detector=None,
    dataset_role: str = 'calibration',
    manifest_hash: Optional[str] = None,
    path: Optional[str] = None,
    percentiles=None,
    artifact_version: int = ARTIFACT_VERSION,
    fingerprints: Optional[Dict[str, str]] = None,
    channel_profile: Optional[str] = None,
    detection_thresholds: Optional[Dict[str, float]] = None,
    component_versions: Optional[Dict[str, str]] = None,
    session_config: Optional[Dict] = None,
    sequence_config: Optional[Dict] = None,
) -> Dict:
    """
    Persist a calibrated threshold set.

    Args:
        thresholds: {'medium','high','critical'} absolute risk_score cutoffs
        calibration_info: stats recorded by RiskEngine.calibrate_thresholds
        detector: the AnomalyDetector the thresholds apply to (fingerprint)
        dataset_role: manifest-declared role of the calibration data
                      (must be 'calibration' — enforced by callers/scripts)
        manifest_hash: hash of the dataset manifest for traceability
        path: explicit output path (default: fingerprint-scoped)
        percentiles: percentile tuple used

    Returns:
        The saved artifact metadata dict (includes 'threshold_version').
    """
    if dataset_role != 'calibration':
        raise ValueError(
            f"Refusing to persist thresholds calibrated on role "
            f"'{dataset_role}': only 'calibration' data may set thresholds.")

    path = path or default_calibration_path(detector)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    version = 'cal-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') \
        + '-' + hashlib.sha256(
            json.dumps({'t': thresholds, 'i': calibration_info},
                       sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

    artifact = {
        'artifact_version': artifact_version,
        'threshold_version': version,
        'thresholds': dict(thresholds),
        'percentiles': list(percentiles) if percentiles else None,
        'calibration_info': dict(calibration_info or {}),
        'detector_fingerprint': detector_fingerprint(detector)
        if detector is not None else None,
        'dataset_role': dataset_role,
        'dataset_manifest_hash': manifest_hash,
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    if artifact_version >= V2_ARTIFACT_VERSION:
        artifact['fingerprints'] = dict(fingerprints or {})
        artifact['channel_profile'] = channel_profile
        artifact['detection_thresholds'] = (
            dict(detection_thresholds) if detection_thresholds else None)
        artifact['component_versions'] = dict(component_versions or {})
        artifact['session_config'] = session_config
        artifact['sequence_config'] = sequence_config
    joblib.dump(artifact, path)
    return artifact


def load_calibration(path: Optional[str] = None,
                     detector=None,
                     fingerprints: Optional[Dict[str, str]] = None) -> Dict:
    """
    Load a calibration artifact.

    Args:
        path: explicit artifact path, or None for the fingerprint-scoped
              default (requires `detector`)
        detector: when given, the artifact's detector_fingerprint must
              match or CalibrationUnavailable is raised.

    Returns:
        Artifact dict with 'thresholds', 'threshold_version', etc.

    Raises:
        CalibrationUnavailable: missing file, wrong artifact version,
            incomplete schema, or detector fingerprint mismatch.
    """
    if path is None:
        if detector is None:
            raise CalibrationUnavailable(
                'No calibration path given and no detector for default lookup')
        path = default_calibration_path(detector)

    if not os.path.exists(path):
        raise CalibrationUnavailable(f'Calibration artifact not found: {path}')

    try:
        data = joblib.load(path)
    except Exception as e:
        raise CalibrationUnavailable(
            f'Failed to load calibration artifact {path}: {e}') from e

    if not isinstance(data, dict):
        raise CalibrationUnavailable(
            f'Invalid calibration artifact at {path}: expected dict')
    version = data.get('artifact_version')
    if version not in SUPPORTED_ARTIFACT_VERSIONS:
        raise CalibrationUnavailable(
            f'Incompatible calibration artifact version '
            f'{version!r} at {path} '
            f'(expected one of {sorted(SUPPORTED_ARTIFACT_VERSIONS)})')
    required = ('thresholds', 'threshold_version', 'calibration_info',
                'dataset_role')
    if version >= V2_ARTIFACT_VERSION:
        required = required + ('fingerprints', 'channel_profile')
    missing = [k for k in required if k not in data]
    if missing:
        raise CalibrationUnavailable(
            f'Incomplete calibration artifact at {path}: missing {missing}')
    if data['dataset_role'] != 'calibration':
        raise CalibrationUnavailable(
            f'Calibration artifact at {path} was produced from role '
            f'{data["dataset_role"]!r}, not calibration data')

    if detector is not None:
        expected = detector_fingerprint(detector)
        actual = data.get('detector_fingerprint')
        if actual is not None and actual != expected:
            raise CalibrationUnavailable(
                f'Calibration artifact at {path} was built for detector '
                f'fingerprint {actual}, not {expected}')

    if fingerprints:
        artifact_fps = data.get('fingerprints') or {}
        for name, expected_fp in fingerprints.items():
            if expected_fp is None:
                continue
            actual_fp = artifact_fps.get(name)
            if actual_fp is None:
                raise CalibrationUnavailable(
                    f'Calibration artifact at {path} lacks fingerprint for '
                    f'component {name!r}')
            if actual_fp != expected_fp:
                raise CalibrationUnavailable(
                    f'Calibration artifact at {path} component {name!r} '
                    f'fingerprint {actual_fp} != {expected_fp}')

    return data


def apply_to_engine(engine, detector=None, path: Optional[str] = None) -> Dict:
    """
    Load persisted thresholds into a RiskEngine.

    On success sets engine.fixed_thresholds, engine.calibration_info and
    engine.calibration_status='calibrated' (+ threshold_version).
    On failure leaves the engine in explicit uncalibrated mode:
    calibration_status='uncalibrated_dataset_relative'.

    Returns the artifact dict when calibrated, else None.
    """
    engine.calibration_status = 'uncalibrated_dataset_relative'
    engine.threshold_version = None
    try:
        artifact = load_calibration(path=path, detector=detector)
    except CalibrationUnavailable:
        return None
    engine.fixed_thresholds = dict(artifact['thresholds'])
    engine.calibration_info = dict(artifact.get('calibration_info') or {})
    engine.calibration_status = 'calibrated'
    engine.threshold_version = artifact.get('threshold_version')
    return artifact


def apply_detection_calibration(engine, detector=None, path: Optional[str] = None,
                                fingerprints: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """
    Load persisted v2 calibration into a DetectionEngine.

    Sets detection thresholds, channel profile, calibration status and
    threshold version when a compatible artifact is available. Leaves the
    engine explicitly uncalibrated otherwise.
    """
    engine.calibration_status = 'uncalibrated_default_thresholds'
    engine.threshold_version = None
    try:
        artifact = load_calibration(path=path, detector=detector,
                                    fingerprints=fingerprints)
    except CalibrationUnavailable:
        return None
    det_thresholds = artifact.get('detection_thresholds') or {}
    if det_thresholds:
        engine.thresholds.update({k: float(v) for k, v in det_thresholds.items()})
    profile = artifact.get('channel_profile')
    if profile and hasattr(engine, 'channel_profile'):
        engine.channel_profile = profile
        if hasattr(engine, 'configured_channels'):
            from core.detection_engine import CHANNEL_PROFILES
            engine.weights = dict(CHANNEL_PROFILES.get(profile, engine.weights))
            engine.configured_channels = tuple(engine.weights.keys())
    engine.calibration_status = 'calibrated'
    engine.threshold_version = artifact.get('threshold_version')
    return artifact
