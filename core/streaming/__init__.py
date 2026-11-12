"""
Streaming ingestion package for CAID — Milestone 2.5.

Additive architecture: the batch pipeline (``modules/session_intelligence.py``)
is unchanged and remains the operational path. This package implements
continuous ingestion -> reliable queueing -> causal per-event detection
reusing the SAME persisted artifacts and v2_0 calibration.

Scores emitted here are calibrated anomaly scores, NOT probabilities.
"""
