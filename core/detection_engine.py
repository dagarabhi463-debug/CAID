"""
Combined Detection Engine for CAID

This module implements the central detection engine that combines
ML anomaly detection, rule-based detection, and behavioral analysis
into a unified threat assessment.

ARCHITECTURE:
- ML Results: Input from Isolation Forest model
- Rule Results: Input from Detection Rules Engine
- Behavioral Results: Input from Behavioral Analyzer
- Risk Calculation: Weighted combination with explainable scoring
- Detection Output: Unified threat assessment with explanations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class RiskLevel(Enum):
    """Risk classification levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"
    INSUFFICIENT = "Insufficient Evidence"


class OperatingMode(Enum):
    """
    Detection operating modes (Milestone 1.9 missing-ML policy, extended in
    Milestone 2.0 for configurable channel profiles).

    A detection channel 'has evidence' when it produced a real observation:
      - ml:        a genuine anomaly score (not the missing placeholder)
      - rules:     a rule evaluation result (rule_score, 0 is genuine)
      - behavioral: a non-missing behavioral deviation score
      - session:   a non-missing session-prefix deviation score
      - sequence:  a non-missing sequence deviation score

    Modes:
      FULL_ML_RULES_BEHAVIORAL — legacy three-channel profile, all evidence.
      DEGRADED_*                — legacy exactly-two-channel degraded modes.
      INSUFFICIENT_EVIDENCE     — fewer than two channels produced evidence.
      FULL_/DEGRADED_<PROFILE>  — configured v2 channel profiles.
    """
    FULL_ML_RULES_BEHAVIORAL = "full_ml_rules_behavioral"
    DEGRADED_RULES_BEHAVIORAL = "degraded_rules_behavioral"
    DEGRADED_ML_RULES = "degraded_ml_rules"
    DEGRADED_ML_BEHAVIORAL = "degraded_ml_behavioral"
    FULL_SESSION = "full_session"
    DEGRADED_SESSION = "degraded_session"
    FULL_SEQUENCE = "full_sequence"
    DEGRADED_SEQUENCE = "degraded_sequence"
    FULL_V2_0 = "full_v2_0"
    DEGRADED_V2_0 = "degraded_v2_0"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_DEGRADED_MODE_BY_MISSING = {
    'ml': OperatingMode.DEGRADED_RULES_BEHAVIORAL,
    'behavioral': OperatingMode.DEGRADED_ML_RULES,
    'rules': OperatingMode.DEGRADED_ML_BEHAVIORAL,
}

_LEGACY_CHANNELS = ('ml', 'rules', 'behavioral')

_FULL_OPERATING_MODES = {
    OperatingMode.FULL_ML_RULES_BEHAVIORAL,
    OperatingMode.FULL_SESSION,
    OperatingMode.FULL_SEQUENCE,
    OperatingMode.FULL_V2_0,
}

# Fixed, documented channel profiles. v2_0 weights were selected on the
# pooled VALIDATION splits of the internal and independent_v2 corpora
# (never on holdout data) to suppress single-channel false positives:
# behavioral/session baselines are the most precise channels on OOD
# synthetic traffic, while ml/sequence are the noisiest.
CHANNEL_PROFILES: Dict[str, Dict[str, float]] = {
    'legacy': {'ml': 0.35, 'rules': 0.35, 'behavioral': 0.30},
    'session': {'ml': 0.30, 'rules': 0.30, 'behavioral': 0.20, 'session': 0.20},
    'sequence': {'ml': 0.30, 'rules': 0.30, 'behavioral': 0.20, 'sequence': 0.20},
    'v2_0': {'ml': 0.05, 'rules': 0.15, 'behavioral': 0.35,
            'session': 0.30, 'sequence': 0.15},
}

_PROFILE_MODES = {
    'session': (OperatingMode.FULL_SESSION, OperatingMode.DEGRADED_SESSION),
    'sequence': (OperatingMode.FULL_SEQUENCE, OperatingMode.DEGRADED_SEQUENCE),
    'v2_0': (OperatingMode.FULL_V2_0, OperatingMode.DEGRADED_V2_0),
}


@dataclass
class Detection:
    """A single detection event."""
    detection_id: str
    user_id: str
    timestamp: str
    risk_level: RiskLevel
    ml_anomaly: bool
    ml_anomaly_score: float
    rule_triggers: List[Dict]
    behavioral_deviation: float
    combined_score: float
    explanation: str
    recommended_actions: List[str]
    behavioral_score_missing: bool = False
    ml_score_missing: bool = False
    session_deviation: float = float('nan')
    sequence_deviation: float = float('nan')
    session_score_missing: bool = True
    sequence_score_missing: bool = True
    missing_channels: List[str] = field(default_factory=list)
    channel_weights: Dict[str, float] = field(default_factory=dict)
    operating_mode: str = OperatingMode.FULL_ML_RULES_BEHAVIORAL.value
    channels_available: List[str] = field(default_factory=list)
    evidence: List[Dict] = field(default_factory=list)


class DetectionEngine:
    """
    Combined detection engine that unifies ML, rule-based, and
    behavioral analysis into a single threat assessment.
    """
    
    def __init__(self, channel_profile: str = 'legacy',
                 weights: Dict[str, float] = None,
                 thresholds: Dict[str, float] = None):
        """Initialize the detection engine.

        Args:
            channel_profile: 'legacy' (default, ml/rules/behavioral),
                'session', 'sequence', or 'v2_0'.
            weights: Optional explicit channel weights (must cover the
                configured channels; normalized at combine time).
            thresholds: Optional {'critical','high','medium'} overrides.
        """
        self.detections = []
        self._detection_counter = 0
        self.channel_profile = channel_profile if channel_profile in CHANNEL_PROFILES \
            else 'legacy'
        if weights is not None:
            self.weights = {k: float(v) for k, v in weights.items()}
        else:
            self.weights = dict(CHANNEL_PROFILES[self.channel_profile])
        self.configured_channels = tuple(self.weights.keys())

        # Minimum thresholds for classification
        self.thresholds = {
            'critical': 0.75,
            'high': 0.55,
            'medium': 0.30,
        }
        if thresholds:
            self.thresholds.update({k: float(v) for k, v in thresholds.items()})
        self.calibration_status = 'uncalibrated_default_thresholds'
        self.threshold_version = None
    
    def process_event(self, row: pd.Series, ml_result: Dict = None,
                      rule_result: Dict = None, behavioral_result: Dict = None,
                      session_result: Dict = None,
                      sequence_result: Dict = None) -> Detection:
        """
        Process a single event and generate a detection.
        
        Args:
            row: Event data row
            ml_result: ML detection results (optional)
            rule_result: Rule-based results (optional)
            behavioral_result: Behavioral analysis results (optional)
            
        Returns:
            Detection object with assessment
        """
        self._detection_counter += 1
        
        user_id = row.get('user_id', 'unknown')
        timestamp = row.get('timestamp', '')
        
        # Get ML anomaly result.
        # Score contract (higher = more anomalous unless noted):
        #   anomaly_score_normalized / ml_anomaly_score: 0-1 calibrated
        #   anomaly_score_raw: raw IsolationForest decision_function,
        #     HIGHER = MORE NORMAL (explicit field, preferred over legacy)
        #   anomaly_score: LEGACY ambiguous field — kept for backward
        #     compatibility; interpreted via the range adapter below
        if ml_result is not None and ml_result.get('anomaly_score') is not None:
            ml_anomaly = ml_result.get('is_anomaly', False)
            ml_anomaly_score = float(ml_result['anomaly_score'])
        elif 'anomaly_score_normalized' in row.index and pd.notna(row.get('anomaly_score_normalized')):
            ml_anomaly = bool(row.get('is_anomaly', 0))
            ml_anomaly_score = float(np.clip(row['anomaly_score_normalized'], 0.0, 1.0))
        elif 'anomaly_score_raw' in row.index and pd.notna(row.get('anomaly_score_raw')):
            # Explicit raw decision_function: higher = more normal -> invert
            ml_anomaly = bool(row.get('is_anomaly', 0))
            ml_anomaly_score = float(np.clip(-float(row['anomaly_score_raw']), 0.0, 1.0))
        elif 'anomaly_score' in row.index and pd.notna(row.get('anomaly_score')):
            raw = float(row['anomaly_score'])
            # LEGACY adapter: anomaly_score is ambiguous (raw decision_function
            # in production, 0-1 normalized in older tests). Values outside
            # [0,1] are treated as raw and inverted; in-range values are
            # treated as already-normalized. New code should emit
            # anomaly_score_raw or anomaly_score_normalized instead.
            if raw < 0 or raw > 1:
                ml_anomaly_score = float(np.clip(-raw, 0.0, 1.0))
            else:
                ml_anomaly_score = raw
            ml_anomaly = bool(row.get('is_anomaly', 0))
        else:
            ml_anomaly = bool(
                ml_result.get('is_anomaly', False) if ml_result is not None
                else row.get('is_anomaly', 0)
            )
            ml_anomaly_score = 0.5  # neutral; flagged as missing in explanation
        ml_score_missing = not (
            (ml_result is not None and ml_result.get('anomaly_score') is not None)
            or ('anomaly_score_normalized' in row.index and pd.notna(row.get('anomaly_score_normalized')))
            or ('anomaly_score_raw' in row.index and pd.notna(row.get('anomaly_score_raw')))
            or ('anomaly_score' in row.index and pd.notna(row.get('anomaly_score')))
        )
        
        # Get rule triggers and rule score.
        # Production emits 'rule_score' (0-1); 'rule_detection_score' is a
        # legacy/test alias. Accept both.
        rules_channel_present = (
            rule_result is not None
            or 'triggered_rules' in row.index
            or 'rule_score' in row.index
            or 'rule_detection_score' in row.index
        )
        if rule_result is not None:
            rule_triggers = rule_result.get('triggered_rules', [])
            rule_score_value = rule_result.get(
                'rule_score',
                rule_result.get('rule_detection_score',
                                row.get('rule_detection_score', row.get('rule_score', 0)))
            )
        else:
            rule_triggers = row.get('triggered_rules', [])
            rule_score_value = row.get('rule_detection_score', row.get('rule_score', 0))
        rules_missing = not rules_channel_present or rule_score_value is None \
            or (isinstance(rule_score_value, float) and np.isnan(rule_score_value))
        
        # Get behavioral deviation.
        # behavioral_deviation_score: 0-1, higher = more deviation.
        # Missing results are NaN + behavioral_score_missing=True — they are
        # NEVER silently represented as normal (0.0) activity.
        if behavioral_result is not None:
            behavioral_raw = behavioral_result.get('behavioral_deviation_score')
            behavioral_missing = bool(
                behavioral_result.get('behavioral_score_missing', False))
        else:
            behavioral_raw = (
                row.get('behavioral_deviation_score')
                if 'behavioral_deviation_score' in row.index else None
            )
            behavioral_missing = bool(
                row.get('behavioral_score_missing', False)
            ) if 'behavioral_score_missing' in row.index else False

        if behavioral_raw is None or pd.isna(behavioral_raw):
            behavioral_deviation = float('nan')
            behavioral_missing = True
        else:
            behavioral_deviation = float(np.clip(float(behavioral_raw), 0.0, 1.0))
            behavioral_missing = behavioral_missing or behavioral_deviation != behavioral_deviation

        # Session channel (Milestone 2.0): causal session-prefix deviation.
        # Missing results are NaN + session_score_missing=True — never silent.
        if session_result is not None:
            session_raw = session_result.get('session_deviation_score',
                                             session_result.get('session_score'))
            session_missing = bool(session_result.get('session_score_missing', False))
            session_evidence = session_result.get('session_evidence',
                                                  session_result.get('session_factors', []))
        else:
            session_raw = (
                row.get('session_deviation_score')
                if 'session_deviation_score' in row.index else None
            )
            session_missing = bool(
                row.get('session_score_missing', False)
            ) if 'session_score_missing' in row.index else False
            session_evidence = (
                row.get('session_evidence')
                if 'session_evidence' in row.index else
                row.get('session_factors', [])
                if 'session_factors' in row.index else []
            )
        if session_raw is None or pd.isna(session_raw):
            session_deviation = float('nan')
            session_missing = True
        else:
            session_deviation = float(np.clip(float(session_raw), 0.0, 1.0))
            session_missing = session_missing or session_deviation != session_deviation

        # Sequence channel (Milestone 2.0): causal sequence deviation.
        if sequence_result is not None:
            sequence_raw = sequence_result.get('sequence_deviation_score',
                                               sequence_result.get('sequence_score'))
            sequence_missing = bool(sequence_result.get('sequence_score_missing', False))
            sequence_evidence = sequence_result.get('sequence_evidence',
                                                    sequence_result.get('sequence_factors', []))
            sequence_patterns = sequence_result.get('sequence_triggered_patterns', [])
        else:
            sequence_raw = (
                row.get('sequence_deviation_score')
                if 'sequence_deviation_score' in row.index else None
            )
            sequence_missing = bool(
                row.get('sequence_score_missing', False)
            ) if 'sequence_score_missing' in row.index else False
            sequence_evidence = (
                row.get('sequence_evidence')
                if 'sequence_evidence' in row.index else
                row.get('sequence_factors', [])
                if 'sequence_factors' in row.index else []
            )
            sequence_patterns = (
                row.get('sequence_triggered_patterns', [])
                if 'sequence_triggered_patterns' in row.index else []
            )
        if sequence_raw is None or pd.isna(sequence_raw):
            sequence_deviation = float('nan')
            sequence_missing = True
        else:
            sequence_deviation = float(np.clip(float(sequence_raw), 0.0, 1.0))
            sequence_missing = sequence_missing or sequence_deviation != sequence_deviation

        # Milestone 1.9/2.0 operating-mode policy: count channels that
        # produced REAL evidence within the configured channel profile.
        channel_scores = {
            'ml': (ml_anomaly_score, ml_score_missing),
            'rules': (float(rule_score_value or 0), rules_missing),
            'behavioral': (behavioral_deviation, behavioral_missing),
            'session': (session_deviation, session_missing),
            'sequence': (sequence_deviation, sequence_missing),
        }
        configured = set(self.configured_channels)
        channels_available = [
            ch for ch in self.configured_channels
            if ch in channel_scores and not channel_scores[ch][1]
        ]
        missing_channels = [ch for ch in self.configured_channels
                            if ch not in channels_available]

        if self.channel_profile == 'legacy' or set(self.configured_channels) == set(_LEGACY_CHANNELS):
            if len(channels_available) >= 3:
                operating_mode = OperatingMode.FULL_ML_RULES_BEHAVIORAL
            elif len(channels_available) == 2:
                missing = (set(_LEGACY_CHANNELS) - set(channels_available)).pop()
                operating_mode = _DEGRADED_MODE_BY_MISSING[missing]
            else:
                operating_mode = OperatingMode.INSUFFICIENT_EVIDENCE
        else:
            full_mode, degraded_mode = _PROFILE_MODES.get(
                self.channel_profile, (OperatingMode.FULL_V2_0, OperatingMode.DEGRADED_V2_0))
            if len(channels_available) >= len(self.configured_channels):
                operating_mode = full_mode
            elif len(channels_available) >= 2:
                operating_mode = degraded_mode
            else:
                operating_mode = OperatingMode.INSUFFICIENT_EVIDENCE

        # Calculate combined score over available channels only; weights
        # renormalize. Insufficient evidence -> NaN (no classification).
        if operating_mode == OperatingMode.INSUFFICIENT_EVIDENCE:
            combined_score = float('nan')
        else:
            combined_score = self._combined_score(channel_scores, channels_available)

        # Classify risk level (NaN combined score -> INSUFFICIENT)
        risk_level = self._classify_risk(combined_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            row=row,
            ml_anomaly=ml_anomaly,
            ml_score=ml_anomaly_score,
            rule_triggers=rule_triggers,
            behavioral_deviation=behavioral_deviation,
            combined_score=combined_score,
            ml_score_missing=ml_score_missing,
            behavioral_missing=behavioral_missing,
            session_deviation=session_deviation,
            session_missing=session_missing,
            sequence_deviation=sequence_deviation,
            sequence_missing=sequence_missing,
            sequence_patterns=sequence_patterns,
            missing_channels=missing_channels,
            operating_mode=operating_mode
        )

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            risk_level=risk_level,
            ml_anomaly=ml_anomaly,
            rule_triggers=rule_triggers
        )

        # Structured evidence objects (Milestone 1.9 Phase 7, extended 2.0)
        evidence = self._build_evidence(
            row=row,
            ml_anomaly=ml_anomaly,
            ml_score=ml_anomaly_score,
            ml_score_missing=ml_score_missing,
            rule_triggers=rule_triggers,
            rule_score=float(rule_score_value or 0),
            rules_missing=rules_missing,
            behavioral_deviation=behavioral_deviation,
            behavioral_missing=behavioral_missing,
            session_deviation=session_deviation,
            session_missing=session_missing,
            session_evidence=session_evidence,
            sequence_deviation=sequence_deviation,
            sequence_missing=sequence_missing,
            sequence_evidence=sequence_evidence,
            combined_score=combined_score,
            operating_mode=operating_mode,
            channels_available=channels_available,
            missing_channels=missing_channels,
        )

        detection = Detection(
            detection_id=f"DET-{self._detection_counter:06d}",
            user_id=user_id,
            timestamp=str(timestamp),
            risk_level=risk_level,
            ml_anomaly=ml_anomaly,
            ml_anomaly_score=ml_anomaly_score,
            rule_triggers=rule_triggers,
            behavioral_deviation=behavioral_deviation,
            combined_score=combined_score,
            explanation=explanation,
            recommended_actions=recommended_actions,
            behavioral_score_missing=behavioral_missing,
            ml_score_missing=ml_score_missing,
            session_deviation=session_deviation,
            sequence_deviation=sequence_deviation,
            session_score_missing=session_missing,
            sequence_score_missing=sequence_missing,
            missing_channels=list(missing_channels),
            channel_weights=dict(self.weights),
            operating_mode=operating_mode.value,
            channels_available=list(channels_available),
            evidence=evidence
        )
        
        self.detections.append(detection)
        
        return detection
    
    def _combined_score(self, channel_scores: Dict[str, Tuple[float, bool]],
                        channels_available: List[str]) -> float:
        """Weighted combined score over available configured channels."""
        available = {}
        for ch in channels_available:
            score, missing = channel_scores.get(ch, (float('nan'), True))
            if missing or pd.isna(score):
                continue
            weight = float(self.weights.get(ch, 0.0))
            if weight > 0:
                available[ch] = (weight, float(score))
        total_w = sum(w for w, _ in available.values())
        if total_w <= 0:
            return float('nan')
        combined = sum(w * s for w, s in available.values()) / total_w
        return float(np.clip(combined, 0.0, 1.0))

    def _calculate_combined_score(self, ml_score: float, rule_score: float,
                                   behavioral_score: float,
                                   ml_missing: bool = False,
                                   rules_missing: bool = False,
                                   behavioral_missing: bool = False) -> float:
        """Legacy three-channel combined score (backward-compatible wrapper)."""
        channel_scores = {
            'ml': (ml_score, ml_missing),
            'rules': (rule_score, rules_missing),
            'behavioral': (behavioral_score, behavioral_missing),
        }
        available = [ch for ch in _LEGACY_CHANNELS if not channel_scores[ch][1]]
        return self._combined_score(channel_scores, available)
    
    def _classify_risk(self, combined_score: float) -> RiskLevel:
        """
        Classify risk level based on combined score.
        
        Args:
            combined_score: Combined detection score (0-1)
            
        Returns:
            RiskLevel enum value
        """
        if pd.isna(combined_score):
            return RiskLevel.INSUFFICIENT
        if combined_score >= self.thresholds['critical']:
            return RiskLevel.CRITICAL
        elif combined_score >= self.thresholds['high']:
            return RiskLevel.HIGH
        elif combined_score >= self.thresholds['medium']:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _generate_explanation(self, row: pd.Series, ml_anomaly: bool,
                               ml_score: float, rule_triggers: List[Dict],
                               behavioral_deviation: float, combined_score: float,
                               ml_score_missing: bool = False,
                               behavioral_missing: bool = False,
                               session_deviation: float = float('nan'),
                               session_missing: bool = True,
                               sequence_deviation: float = float('nan'),
                               sequence_missing: bool = True,
                               sequence_patterns: List[str] = None,
                               missing_channels: List[str] = None,
                               operating_mode: 'OperatingMode' = None) -> str:
        """
        Generate human-readable explanation for detection.
        
        Args:
            row: Event data
            ml_anomaly: Whether ML flagged as anomaly
            ml_score: ML anomaly score
            rule_triggers: List of triggered rules
            behavioral_deviation: Behavioral deviation score
            combined_score: Final combined score
            behavioral_missing: Whether behavioral result is unavailable
            
        Returns:
            Human-readable explanation string
        """
        parts = []
        
        # ML explanation
        if ml_score_missing:
            parts.append("ML anomaly result unavailable (excluded from combined score)")
        elif ml_anomaly:
            if ml_score >= 0.8:
                parts.append(f"ML model identified anomaly (score: {ml_score:.2f})")
            else:
                parts.append(f"ML model noted deviation (score: {ml_score:.2f})")
        else:
            parts.append(f"ML model assessed normal activity (score: {ml_score:.2f})")
        
        # Rule explanation
        if rule_triggers:
            rule_names = [r.get('rule_name', 'Unknown') for r in rule_triggers]
            parts.append(f"Rules triggered: {', '.join(rule_names[:3])}")
        else:
            parts.append("No detection rules triggered")
        
        # Behavioral explanation
        if behavioral_missing:
            parts.append("Behavioral analysis unavailable (weight redistributed to ML/rule scores)")
        elif behavioral_deviation >= 0.5:
            parts.append(f"Unusual behavior pattern detected (deviation: {behavioral_deviation:.2f})")
        else:
            parts.append(f"Behavioral pattern consistent with baseline (deviation: {behavioral_deviation:.2f})")

        # Session channel explanation (only when configured)
        if 'session' in (self.weights or {}):
            if session_missing or pd.isna(session_deviation):
                parts.append("Session analysis unavailable (excluded from combined score)")
            elif session_deviation >= 0.5:
                parts.append(f"Session behavior deviates from baseline (deviation: {session_deviation:.2f})")
            else:
                parts.append(f"Session behavior consistent with baseline (deviation: {session_deviation:.2f})")

        # Sequence channel explanation (only when configured)
        if 'sequence' in (self.weights or {}):
            if sequence_missing or pd.isna(sequence_deviation):
                parts.append("Sequence analysis unavailable (excluded from combined score)")
            else:
                if sequence_patterns:
                    parts.append(
                        f"Suspicious sequence pattern(s): {', '.join(sequence_patterns[:3])} "
                        f"(deviation: {sequence_deviation:.2f})")
                elif sequence_deviation >= 0.5:
                    parts.append(f"Unusual event sequence detected (deviation: {sequence_deviation:.2f})")
                else:
                    parts.append(f"Event sequence consistent with baseline (deviation: {sequence_deviation:.2f})")

        # Final assessment
        if pd.isna(combined_score):
            parts.append("Overall risk score: unavailable "
                         "(insufficient detection channels)")
        else:
            parts.append(f"Overall risk score: {combined_score:.2f}")
            if operating_mode is not None and (
                    self.channel_profile != 'legacy'
                    or operating_mode not in _FULL_OPERATING_MODES):
                if operating_mode in _FULL_OPERATING_MODES:
                    parts.append(f"Operating mode: {operating_mode.value}")
                elif operating_mode == OperatingMode.INSUFFICIENT_EVIDENCE:
                    parts.append(f"Operating mode: {operating_mode.value}")
                else:
                    parts.append(f"Operating mode: {operating_mode.value} "
                                 "(degraded — not all detection channels available)")

        return "; ".join(parts)
    
    def _build_evidence(self, row: pd.Series, ml_anomaly: bool,
                        ml_score: float, ml_score_missing: bool,
                        rule_triggers: List[Dict], rule_score: float,
                        rules_missing: bool,
                        behavioral_deviation: float,
                        behavioral_missing: bool,
                        session_deviation: float = float('nan'),
                        session_missing: bool = True,
                        session_evidence: List[Dict] = None,
                        sequence_deviation: float = float('nan'),
                        sequence_missing: bool = True,
                        sequence_evidence: List[Dict] = None,
                        combined_score: float = float('nan'),
                        operating_mode: 'OperatingMode' = None,
                        channels_available: List[str] = None,
                        missing_channels: List[str] = None) -> List[Dict]:
        """
        Build structured per-channel evidence objects (Milestone 1.9).

        Each object identifies: channel, evidence_name, observed_value,
        expected_baseline_or_threshold, contribution_or_severity,
        availability and a plain-language explanation. The legacy
        `explanation` string is unchanged — this is additive metadata.
        """
        evidence = []

        # --- ML channel ---
        if ml_score_missing:
            evidence.append({
                'channel': 'ml',
                'evidence_name': 'anomaly_score',
                'observed_value': None,
                'expected_baseline_or_threshold':
                    'IsolationForest calibrated score in [0,1] '
                    '(higher = more anomalous)',
                'contribution_or_severity': 'excluded',
                'availability': 'missing',
                'explanation': 'ML model produced no score for this event; '
                               'channel excluded from combined score.',
            })
        else:
            evidence.append({
                'channel': 'ml',
                'evidence_name': 'anomaly_score',
                'observed_value': round(float(ml_score), 4),
                'expected_baseline_or_threshold':
                    'calibrated anomaly score; weight 0.35 when all '
                    'channels present',
                'contribution_or_severity':
                    'anomaly' if ml_anomaly else 'normal',
                'availability': 'available',
                'explanation': (
                    'ML model flagged this event as anomalous.'
                    if ml_anomaly else
                    'ML model assessed this event as consistent with '
                    'baseline activity.'),
            })

        # --- Rules channel ---
        if rules_missing:
            evidence.append({
                'channel': 'rules',
                'evidence_name': 'rule_score',
                'observed_value': None,
                'expected_baseline_or_threshold': 'no rules triggered',
                'contribution_or_severity': 'excluded',
                'availability': 'missing',
                'explanation': 'Rule evaluation produced no result.',
            })
        elif rule_triggers:
            for trig in rule_triggers[:5]:
                evidence.append({
                    'channel': 'rules',
                    'evidence_name': trig.get('name', trig.get('rule_id',
                                                              'unknown_rule')),
                    'observed_value': trig.get('condition', ''),
                    'expected_baseline_or_threshold':
                        f"severity={trig.get('severity', 'unknown')}, "
                        f"weight={trig.get('weight', 0)}",
                    'contribution_or_severity': trig.get('severity',
                                                         'unknown'),
                    'availability': 'available',
                    'explanation': trig.get('description',
                                            'Detection rule triggered.'),
                })
        else:
            evidence.append({
                'channel': 'rules',
                'evidence_name': 'rule_score',
                'observed_value': round(float(rule_score), 4),
                'expected_baseline_or_threshold': 'no rules triggered',
                'contribution_or_severity': 'none',
                'availability': 'available',
                'explanation': 'No detection rules triggered for this event.',
            })

        # --- Behavioral channel ---
        if behavioral_missing or pd.isna(behavioral_deviation):
            evidence.append({
                'channel': 'behavioral',
                'evidence_name': 'behavioral_deviation_score',
                'observed_value': None,
                'expected_baseline_or_threshold':
                    'per-user baseline deviation in [0,1] '
                    '(higher = more deviation)',
                'contribution_or_severity': 'excluded',
                'availability': 'missing',
                'explanation': 'Behavioral baseline unavailable or input '
                               'insufficient; channel excluded.',
            })
        else:
            dev = float(behavioral_deviation)
            evidence.append({
                'channel': 'behavioral',
                'evidence_name': 'behavioral_deviation_score',
                'observed_value': round(dev, 4),
                'expected_baseline_or_threshold':
                    'user baseline; deviation 0 = consistent, '
                    '1 = maximal deviation',
                'contribution_or_severity':
                    'deviation' if dev >= 0.5 else 'consistent',
                'availability': 'available',
                'explanation': (
                    'Behavior deviates from the user baseline.'
                    if dev >= 0.5 else
                    'Behavior is consistent with the user baseline.'),
            })

        # --- Session channel (Milestone 2.0, only when configured) ---
        if 'session' in (self.weights or {}):
            if session_missing or pd.isna(session_deviation):
                evidence.append({
                    'channel': 'session',
                    'evidence_name': 'session_deviation_score',
                    'observed_value': None,
                    'expected_baseline_or_threshold':
                        'causal session-prefix deviation in [0,1] '
                        '(higher = more deviation)',
                    'contribution_or_severity': 'excluded',
                    'availability': 'missing',
                    'explanation': 'Session baseline unavailable or prefix '
                                   'features insufficient; channel excluded.',
                })
            else:
                dev = float(session_deviation)
                evidence.append({
                    'channel': 'session',
                    'evidence_name': 'session_deviation_score',
                    'observed_value': round(dev, 4),
                    'expected_baseline_or_threshold':
                        'user/global session-prefix baseline; 0 = consistent, '
                        '1 = maximal deviation',
                    'contribution_or_severity':
                        'deviation' if dev >= 0.5 else 'consistent',
                    'availability': 'available',
                    'explanation': (
                        'Session prefix deviates from the stored baseline.'
                        if dev >= 0.5 else
                        'Session prefix is consistent with the stored baseline.'),
                })
                for item in (session_evidence or [])[:5]:
                    if isinstance(item, dict):
                        evidence.append({
                            'channel': 'session',
                            'evidence_name': item.get('factor', 'session_factor'),
                            'observed_value': item.get('value'),
                            'expected_baseline_or_threshold': item.get('baseline'),
                            'contribution_or_severity': item.get('impact', 'unknown'),
                            'availability': 'available',
                            'explanation': item.get('factor', 'Session factor'),
                        })

        # --- Sequence channel (Milestone 2.0, only when configured) ---
        if 'sequence' in (self.weights or {}):
            if sequence_missing or pd.isna(sequence_deviation):
                evidence.append({
                    'channel': 'sequence',
                    'evidence_name': 'sequence_deviation_score',
                    'observed_value': None,
                    'expected_baseline_or_threshold':
                        'causal sequence deviation in [0,1] '
                        '(higher = more anomalous)',
                    'contribution_or_severity': 'excluded',
                    'availability': 'missing',
                    'explanation': 'Sequence model unavailable or causal window '
                                   'insufficient; channel excluded.',
                })
            else:
                dev = float(sequence_deviation)
                evidence.append({
                    'channel': 'sequence',
                    'evidence_name': 'sequence_deviation_score',
                    'observed_value': round(dev, 4),
                    'expected_baseline_or_threshold':
                        'user/global transition + pattern model; 0 = typical, '
                        '1 = maximally anomalous',
                    'contribution_or_severity':
                        'deviation' if dev >= 0.5 else 'consistent',
                    'availability': 'available',
                    'explanation': (
                        'Event sequence deviates from expected patterns.'
                        if dev >= 0.5 else
                        'Event sequence is consistent with expected patterns.'),
                })
                for item in (sequence_evidence or [])[:8]:
                    if isinstance(item, dict):
                        evidence.append({
                            'channel': 'sequence',
                            'evidence_name': item.get('evidence_name',
                                                       item.get('pattern', 'sequence_factor')),
                            'observed_value': item.get('observed_value', item.get('score')),
                            'expected_baseline_or_threshold':
                                item.get('expected_baseline_or_threshold',
                                         item.get('kind')),
                            'contribution_or_severity':
                                item.get('contribution_or_severity', item.get('score')),
                            'availability': item.get('availability', 'available'),
                            'explanation': item.get('explanation',
                                                    item.get('detail', 'Sequence evidence')),
                        })

        # --- Combined / operating mode ---
        evidence.append({
            'channel': 'combined',
            'evidence_name': 'operating_mode',
            'observed_value': (
                None if pd.isna(combined_score)
                else round(float(combined_score), 4)),
            'expected_baseline_or_threshold':
                'channels available: ' + ','.join(channels_available),
            'contribution_or_severity': operating_mode.value,
            'availability': 'available',
            'explanation': (
                'All configured detection channels contributed evidence.'
                if operating_mode in _FULL_OPERATING_MODES else
                ('Insufficient evidence: fewer than two detection channels '
                 'produced real observations; no risk classification.'
                 if operating_mode == OperatingMode.INSUFFICIENT_EVIDENCE else
                 'Degraded mode: score computed over available channels '
                 'only.')),
        })
        return evidence

    def _generate_recommendations(self, risk_level: RiskLevel,
                                   ml_anomaly: bool, 
                                   rule_triggers: List[Dict]) -> List[str]:
        """
        Generate recommended investigation steps.
        
        Args:
            risk_level: Classified risk level
            ml_anomaly: Whether ML flagged as anomaly
            rule_triggers: List of triggered rules
            
        Returns:
            List of recommended actions
        """
        actions = []

        if risk_level == RiskLevel.INSUFFICIENT:
            actions.append("INSUFFICIENT EVIDENCE: manual review recommended "
                           "(required detection channels unavailable)")
        elif risk_level == RiskLevel.CRITICAL:
            actions.append("IMMEDIATE: Escalate to security team")
            actions.append("Review user activity timeline")
            actions.append("Check for data exfiltration indicators")
        elif risk_level == RiskLevel.HIGH:
            actions.append("Review user access patterns")
            actions.append("Verify user authorization")
            actions.append("Check for account compromise indicators")
        elif risk_level == RiskLevel.MEDIUM:
            actions.append("Monitor user activity closely")
            actions.append("Review recent access patterns")
            actions.append("Consider temporary access restrictions")
        else:
            actions.append("Continue routine monitoring")
            actions.append("Log activity for audit trail")
        
        # Add specific recommendations based on rules
        for rule in rule_triggers:
            rule_name = rule.get('rule_name', '')
            if 'API' in rule_name:
                if "Rate limit user API requests" not in actions:
                    actions.append("Rate limit user API requests")
            elif 'Data' in rule_name or 'Export' in rule_name:
                if "Review data access logs" not in actions:
                    actions.append("Review data access logs")
            elif 'Account' in rule_name or 'Course' in rule_name:
                if "Audit user permissions" not in actions:
                    actions.append("Audit user permissions")
        
        return actions
    
    def process_dataframe(self, df: pd.DataFrame,
                          ml_column: str = 'is_anomaly',
                          rule_column: str = 'triggered_rules',
                          behavioral_column: str = 'behavioral_deviation_score',
                          session_column: str = 'session_deviation_score',
                          sequence_column: str = 'sequence_deviation_score') -> pd.DataFrame:
        """
        Process entire DataFrame through detection engine.
        
        Args:
            df: Security log DataFrame
            ml_column: Column name for ML anomaly result
            rule_column: Column name for rule triggers
            behavioral_column: Column name for behavioral deviation
            
        Returns:
            DataFrame with detection results
        """
        df = df.copy()
        
        # Clear previous detections
        self.detections = []
        self._detection_counter = 0
        
        # If the frame carries only the raw IsolationForest decision_function
        # (anomaly_score, higher = more normal) and no normalized score,
        # detect it at batch level: raw scores span negative values, normalized
        # scores are always within [0, 1]. Convert once for all rows.
        raw_score_col = None
        # Prefer the explicit anomaly_score_raw field; fall back to the
        # deprecated ambiguous anomaly_score only when raw is absent.
        raw_candidates = [c for c in ('anomaly_score_raw', 'anomaly_score')
                          if c in df.columns]
        if ('anomaly_score_normalized' not in df.columns
                and raw_candidates):
            for cand in raw_candidates:
                scores = pd.to_numeric(df[cand], errors='coerce')
                if scores.min() < 0 or scores.max() > 1:
                    raw_score_col = cand
                    break
                elif cand == 'anomaly_score_raw':
                    # explicit raw field: always treat as decision_function
                    raw_score_col = cand
                    break
        
        # Process each event, honoring the configured column names.
        # to_dict('records') + RowView avoids the per-row Series
        # materialization cost of iterrows() (hot-path optimization).
        from utils.row_view import RowView
        detection_results = []
        for rec in df.to_dict('records'):
            row = RowView(rec)
            ml_payload = {}
            if ml_column in row.index:
                ml_payload['is_anomaly'] = row.get(ml_column)
            if raw_score_col is not None:
                raw_val = pd.to_numeric(row.get(raw_score_col), errors='coerce')
                if pd.notna(raw_val):
                    # Invert: raw decision_function higher = more normal
                    ml_payload['anomaly_score'] = float(np.clip(-float(raw_val), 0.0, 1.0))
            ml_result = ml_payload or None
            rule_result = None
            if (rule_column in row.index
                    or 'triggered_rules' in row.index
                    or 'rule_score' in row.index
                    or 'rule_detection_score' in row.index):
                rule_result = {
                    'triggered_rules': row.get(rule_column, row.get('triggered_rules', [])),
                    'rule_score': row.get('rule_detection_score', row.get('rule_score', 0)),
                }
            behavioral_result = (
                {
                    'behavioral_deviation_score': row.get(behavioral_column),
                    'behavioral_score_missing': bool(
                        row.get('behavioral_score_missing', False)
                    ) if 'behavioral_score_missing' in row.index else False,
                }
                if behavioral_column in row.index else None
            )
            session_result = (
                {
                    'session_deviation_score': row.get(session_column),
                    'session_score_missing': bool(
                        row.get('session_score_missing', False)
                    ) if 'session_score_missing' in row.index else False,
                    'session_evidence': row.get('session_evidence',
                                                row.get('session_factors', [])),
                }
                if session_column in row.index else None
            )
            sequence_result = (
                {
                    'sequence_deviation_score': row.get(sequence_column),
                    'sequence_score_missing': bool(
                        row.get('sequence_score_missing', False)
                    ) if 'sequence_score_missing' in row.index else False,
                    'sequence_evidence': row.get('sequence_evidence',
                                                 row.get('sequence_factors', [])),
                    'sequence_triggered_patterns': row.get(
                        'sequence_triggered_patterns', []),
                }
                if sequence_column in row.index else None
            )
            detection = self.process_event(
                row,
                ml_result=ml_result,
                rule_result=rule_result,
                behavioral_result=behavioral_result,
                session_result=session_result,
                sequence_result=sequence_result,
            )
            detection_results.append(detection)
        
        # Add detection columns to DataFrame
        df['detection_id'] = [d.detection_id for d in detection_results]
        df['risk_level'] = [d.risk_level.value for d in detection_results]
        df['ml_anomaly'] = [d.ml_anomaly for d in detection_results]
        df['ml_anomaly_score'] = [d.ml_anomaly_score for d in detection_results]
        df['rule_triggers'] = [d.rule_triggers for d in detection_results]
        df['behavioral_deviation'] = [d.behavioral_deviation for d in detection_results]
        df['behavioral_score_missing'] = [d.behavioral_score_missing for d in detection_results]
        df['ml_score_missing'] = [d.ml_score_missing for d in detection_results]
        df['combined_score'] = [d.combined_score for d in detection_results]
        df['operating_mode'] = [d.operating_mode for d in detection_results]
        df['channels_available'] = [d.channels_available for d in detection_results]
        df['structured_evidence'] = [d.evidence for d in detection_results]
        df['detection_explanation'] = [d.explanation for d in detection_results]
        df['recommended_actions'] = [d.recommended_actions for d in detection_results]
        df['session_deviation'] = [d.session_deviation for d in detection_results]
        df['sequence_deviation'] = [d.sequence_deviation for d in detection_results]
        df['session_score_missing'] = [d.session_score_missing for d in detection_results]
        df['sequence_score_missing'] = [d.sequence_score_missing for d in detection_results]
        df['missing_channels'] = [d.missing_channels for d in detection_results]
        df['channel_weights'] = [d.channel_weights for d in detection_results]
        
        return df
    
    def get_summary(self) -> Dict:
        """
        Get detection summary statistics.
        
        Returns:
            Dictionary with detection summary
        """
        if not self.detections:
            return {
                'total_detections': 0,
                'risk_distribution': {},
                'ml_anomalies': 0,
                'high_risk_events': 0
            }
        
        risk_counts = {}
        ml_anomaly_count = 0
        high_risk_count = 0
        
        for detection in self.detections:
            risk_str = detection.risk_level.value
            risk_counts[risk_str] = risk_counts.get(risk_str, 0) + 1
            
            if detection.ml_anomaly:
                ml_anomaly_count += 1
            
            if detection.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                high_risk_count += 1
        
        return {
            'total_detections': len(self.detections),
            'risk_distribution': risk_counts,
            'ml_anomalies': ml_anomaly_count,
            'high_risk_events': high_risk_count,
            'critical_count': risk_counts.get(RiskLevel.CRITICAL.value, 0),
            'high_count': risk_counts.get(RiskLevel.HIGH.value, 0),
            'medium_count': risk_counts.get(RiskLevel.MEDIUM.value, 0),
            'low_count': risk_counts.get(RiskLevel.LOW.value, 0)
        }
    
    def get_suspicious_users(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get list of suspicious users based on detections.
        
        Args:
            df: DataFrame with detection results
            threshold: Combined score threshold
            
        Returns:
            DataFrame with suspicious user summary
        """
        suspicious = df[df['combined_score'] >= threshold].copy()
        
        if len(suspicious) == 0:
            return pd.DataFrame()
        
        # Aggregate by user
        user_summary = suspicious.groupby('user_id').agg(
            detection_count=('combined_score', 'count'),
            max_score=('combined_score', 'max'),
            avg_score=('combined_score', 'mean'),
            total_risk_score=('combined_score', 'sum'),
        ).reset_index()
        
        # Highest risk level per user (first mode is fine for synthetic data)
        if 'risk_level' in suspicious.columns:
            risk_modes = suspicious.groupby('user_id')['risk_level'].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) else None
            )
            user_summary['highest_risk'] = user_summary['user_id'].map(risk_modes)
        else:
            user_summary['highest_risk'] = None
        
        if 'ml_anomaly' in suspicious.columns:
            user_summary['ml_detection_count'] = suspicious.groupby('user_id')['ml_anomaly'].sum().values
        else:
            user_summary['ml_detection_count'] = 0
        
        # Sort by risk
        user_summary = user_summary.sort_values('total_risk_score', ascending=False)
        
        return user_summary
