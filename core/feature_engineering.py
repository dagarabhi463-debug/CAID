"""
Feature Engineering for CAID - FIXED VERSION

This module provides functionality for extracting and engineering features
from security logs for machine learning analysis.

FIXED: Uses pre-computed baselines from training data to prevent data leakage.
"""

import hashlib
import json

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler


def _str_col_with_unknown(series: pd.Series) -> pd.Series:
    """NaN -> 'unknown' then str, without pandas-4 fillna dtype warnings."""
    return series.astype('object').where(series.notna(), 'unknown').astype(str)


class FeatureBaselineManager:
    """
    Manages pre-computed feature baselines from training data.
    
    FIX: Prevents data leakage by storing baselines computed on training data only
    and reusing them during inference instead of computing from current data.
    """

    ARTIFACT_VERSION = 1

    def __init__(self):
        """Initialize baseline manager."""
        self.user_baselines: Dict[str, Dict[str, float]] = {}
        self.user_totals: Dict[str, Dict[str, float]] = {}
        self.ip_frequency: Dict[str, int] = {}
        self.event_type_frequency: Dict[str, int] = {}
        self.global_stats: Dict[str, float] = {}
        self._global_ip_freq = 1.0
        self._global_event_freq = 1.0
        self._is_fitted = False
    
    @staticmethod
    def _safe_std(value: float) -> float:
        """Return std, replacing NaN/zero with 1.0 to avoid division errors."""
        if value is None or not np.isfinite(value) or value == 0:
            return 1.0
        return float(value)
    
    def fit(self, df: pd.DataFrame) -> 'FeatureBaselineManager':
        """
        Fit the baseline manager on training data.
        
        FIX: Computes and stores baselines from training data ONLY.
        These baselines will be reused during inference instead of
        computing new aggregations from the current DataFrame.
        
        Args:
            df: Training DataFrame (should be normal_activity.csv)
            
        Returns:
            Self for method chaining

        Raises:
            ValueError: If DataFrame is None or empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("Cannot fit feature baselines on an empty DataFrame")
        df = df.copy()
        
        # Normalize key columns so missing values aggregate safely
        for col in ('user_id', 'source_ip', 'event_type'):
            if col in df.columns:
                df[col] = _str_col_with_unknown(df[col])
            else:
                df[col] = 'unknown'
        
        for col in ('api_requests', 'data_mb', 'courses_accessed', 'accounts_accessed'):
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Compute user baselines from training data
        user_agg = df.groupby('user_id').agg({
            'api_requests': ['mean', 'std', 'sum', 'count'],
            'data_mb': ['mean', 'std', 'sum'],
            'courses_accessed': ['mean', 'std', 'sum'],
            'accounts_accessed': ['mean', 'std', 'sum']
        }).reset_index()
        
        # Flatten column names
        user_agg.columns = [
            'user_id', 
            'user_mean_api', 'user_std_api', 'user_sum_api', 'user_count_api',
            'user_mean_data', 'user_std_data', 'user_sum_data',
            'user_mean_courses', 'user_std_courses', 'user_sum_courses',
            'user_mean_accounts', 'user_std_accounts', 'user_sum_accounts'
        ]
        
        # Store user baselines and totals as dicts
        for _, row in user_agg.iterrows():
            event_count = int(row['user_count_api']) if pd.notna(row['user_count_api']) else 0
            self.user_baselines[row['user_id']] = {
                'mean_api': float(row['user_mean_api']),
                'std_api': self._safe_std(row['user_std_api']),
                'mean_data': float(row['user_mean_data']),
                'std_data': self._safe_std(row['user_std_data']),
                'mean_courses': float(row['user_mean_courses']),
                'std_courses': self._safe_std(row['user_std_courses']),
                'mean_accounts': float(row['user_mean_accounts']),
                'std_accounts': self._safe_std(row['user_std_accounts']),
            }
            self.user_totals[row['user_id']] = {
                'total_api': float(row['user_sum_api']),
                'total_data': float(row['user_sum_data']),
                'total_courses': float(row['user_sum_courses']),
                'event_count': event_count,
            }
        
        # Compute IP frequency from training data
        ip_counts = df['source_ip'].value_counts()
        self.ip_frequency = ip_counts.to_dict()
        self._global_ip_freq = float(ip_counts.mean()) if len(ip_counts) else 1.0
        
        # Compute event type frequency from training data
        event_counts = df['event_type'].value_counts()
        self.event_type_frequency = event_counts.to_dict()
        self._global_event_freq = float(event_counts.mean()) if len(event_counts) else 1.0
        
        # Compute global statistics
        self.global_stats = {
            'mean_api': float(df['api_requests'].mean()) if 'api_requests' in df.columns else 0.0,
            'std_api': self._safe_std(df['api_requests'].std()) if 'api_requests' in df.columns else 1.0,
            'mean_data': float(df['data_mb'].mean()) if 'data_mb' in df.columns else 0.0,
            'std_data': self._safe_std(df['data_mb'].std()) if 'data_mb' in df.columns else 1.0,
            'mean_courses': float(df['courses_accessed'].mean()) if 'courses_accessed' in df.columns else 0.0,
            'std_courses': self._safe_std(df['courses_accessed'].std()) if 'courses_accessed' in df.columns else 1.0,
            'mean_accounts': float(df['accounts_accessed'].mean()) if 'accounts_accessed' in df.columns else 0.0,
            'std_accounts': self._safe_std(df['accounts_accessed'].std()) if 'accounts_accessed' in df.columns else 1.0,
            'total_api': float(df['api_requests'].sum()) if 'api_requests' in df.columns else 0.0,
            'total_data': float(df['data_mb'].sum()) if 'data_mb' in df.columns else 0.0,
            'total_courses': float(df['courses_accessed'].sum()) if 'courses_accessed' in df.columns else 0.0,
            'event_count': int(len(df)),
            # Mean events per user — used for unseen-user fallback so that
            # user_event_count/user_total_* reflect a TYPICAL user, not the
            # whole training population (Milestone 1.8 FP fix)
            'per_user_event_count': float(len(df)) / max(int(df['user_id'].nunique()), 1),
        }

        # Typical-user profile statistics (Milestone 1.8.1): for each
        # per-user aggregate feature, the MEAN across training users of that
        # feature. Unseen users fall back to these values so they present
        # like a randomly chosen training user rather than a population
        # aggregate. z-score features have no per-user baseline equivalent
        # and keep the population (global_stats) fallback instead.
        n_users = max(int(df['user_id'].nunique()), 1)
        for c in ('api', 'data', 'courses', 'accounts'):
            means = [b[f'mean_{c}'] for b in self.user_baselines.values()]
            maxs = [b[f'mean_{c}'] + b[f'std_{c}'] for b in self.user_baselines.values()]
            self.global_stats[f'typical_mean_{c}'] = (
                float(np.mean(means)) if means else self.global_stats[f'mean_{c}'])
            self.global_stats[f'typical_max_{c}'] = (
                float(np.mean(maxs)) if maxs else
                self.global_stats[f'mean_{c}'] + self.global_stats[f'std_{c}'])
        for c in ('api', 'data', 'courses'):
            totals = [t['total_' + c] for t in self.user_totals.values()]
            self.global_stats[f'typical_total_{c}'] = (
                float(np.mean(totals)) if totals else
                self.global_stats[f'total_' + c] / n_users)
        self.global_stats['typical_event_count'] = self.global_stats['per_user_event_count']

        self._is_fitted = True
        return self
    
    def _zscore(self, value: float, mean: float, std: float) -> float:
        """Compute clipped z-score deviation from a baseline."""
        if not np.isfinite(value):
            return 0.0
        z = (value - mean) / std
        return float(np.clip(z, -10.0, 10.0))
    
    def transform_user_features(self, df: pd.DataFrame,
                                leave_one_out: bool = False) -> pd.DataFrame:
        """
        Transform user features using stored baselines.

        FIX: Uses pre-computed baselines instead of computing from current DataFrame.
        This prevents data leakage where future events influence baselines.

        Feature definitions (all derived from TRAINING data only):
        - user_total_*: historical totals for the user across the training set
        - user_event_count: number of events the user had in the training set
        - user_mean_*: per-event mean for the user in the training set
        - user_max_*: per-event mean + 1 std for the user in the training set
        - user_*_z: per-row z-score deviation of this event's value from the
          user's training mean/std; unseen users get a population z-score
          vs global mean/std (missing values -> 0)
        - Unseen users receive TYPICAL-user statistics (Milestone 1.8.1):
          user_mean_*/user_max_*/user_total_* fall back to the mean across
          training users of each per-user feature, and user_event_count to
          the mean events per user — never to pooled population totals.
          user_*_z falls back to a population z-score vs global mean/std
          because an unseen user has no personal baseline.

        Args:
            df: DataFrame to transform
            leave_one_out: When True (TRAINING-time use only), user_*_z is
                computed leave-one-out: each event is scored against the
                user's baseline EXCLUDING that event, so training z-scores
                have the same honest-deviation semantics as prediction.
                Users with fewer than 2 events get z = 0 (no baseline).
                This flag must NEVER be used at prediction time.

        Returns:
            DataFrame with user features added
        """
        df = df.copy()
        
        for col in ('api_requests', 'data_mb', 'courses_accessed', 'accounts_accessed'):
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'user_id' not in df.columns:
            df['user_id'] = 'unknown'
        df['user_id'] = _str_col_with_unknown(df['user_id'])
        
        # Initialize columns
        for col in ('user_total_api', 'user_mean_api', 'user_max_api',
                    'user_total_data', 'user_mean_data', 'user_max_data',
                    'user_total_courses', 'user_mean_courses', 'user_max_courses',
                    'user_total_accounts', 'user_mean_accounts', 'user_max_accounts',
                    'user_event_count',
                    'user_api_z', 'user_data_z', 'user_courses_z', 'user_accounts_z'):
            df[col] = 0.0
        
        # VECTORIZED (Milestone 1.7): identical semantics to the previous
        # per-user loop, but baselines are applied via vectorized maps.
        # For each unique user, get their baseline stats
        unique_users = df['user_id'].unique()

        # Build lookup maps once (known users) and global fallbacks
        mean_maps = {c: {} for c in ('api', 'data', 'courses', 'accounts')}
        std_maps = {c: {} for c in ('api', 'data', 'courses', 'accounts')}
        max_maps = {c: {} for c in ('api', 'data', 'courses', 'accounts')}
        total_maps = {c: {} for c in ('api', 'data', 'courses')}
        count_map = {}
        for uid in unique_users:
            if uid in self.user_baselines:
                b = self.user_baselines[uid]
                t = self.user_totals[uid]
                for c in mean_maps:
                    mean_maps[c][uid] = b[f'mean_{c}']
                    std_maps[c][uid] = b[f'std_{c}']
                    max_maps[c][uid] = b[f'mean_{c}'] + b[f'std_{c}']
                total_maps['api'][uid] = t['total_api']
                total_maps['data'][uid] = t['total_data']
                total_maps['courses'][uid] = t['total_courses']
                count_map[uid] = t['event_count']

        g = self.global_stats
        g_mean = {'api': g['mean_api'], 'data': g['mean_data'],
                  'courses': g['mean_courses'], 'accounts': g['mean_accounts']}
        g_std = {'api': g['std_api'], 'data': g['std_data'],
                 'courses': g['std_courses'], 'accounts': g['std_accounts']}
        # Unseen-user aggregate fallbacks: TYPICAL-user statistics (mean
        # across training users of each per-user feature), never pooled
        # population totals. .get() fallbacks preserve compatibility with
        # artifacts fitted before Milestone 1.8.1.
        g_typical_mean = {c: g.get(f'typical_mean_{c}', g_mean[c])
                          for c in g_mean}
        g_typical_max = {c: g.get(f'typical_max_{c}', g_mean[c] + g_std[c])
                         for c in g_mean}
        g_typical_n = g.get('per_user_event_count', g['event_count'])

        uid = df['user_id']
        known = uid.isin(self.user_baselines)

        col_map = {'api': 'api_requests', 'data': 'data_mb',
                   'courses': 'courses_accessed', 'accounts': 'accounts_accessed'}

        for c in ('api', 'data', 'courses', 'accounts'):
            src = col_map[c]
            mean_s = uid.map(mean_maps[c]).where(known, g_typical_mean[c])
            max_s = uid.map(max_maps[c]).where(known, g_typical_max[c])
            df[f'user_mean_{c}'] = mean_s
            df[f'user_max_{c}'] = max_s
            # Per-row z-score deviation, clipped to [-10, 10]; non-finite -> 0.
            # Unseen users have no personal baseline, so z uses the
            # population mean/std (documented fallback semantics).
            vals = pd.to_numeric(df[src], errors='coerce')
            std_s = uid.map(std_maps[c]).where(known, g_std[c])
            z = (vals - mean_s) / std_s.replace(0, 1.0)
            z = z.clip(-10.0, 10.0)
            z[~np.isfinite(vals)] = 0.0
            df[f'user_{c}_z'] = z

        for c in ('api', 'data', 'courses'):
            total_s = uid.map(total_maps[c]).where(
                known, g.get(f'typical_total_{c}', g_mean[c] * g_typical_n))
            df[f'user_total_{c}'] = total_s
        # NOTE: user_total_accounts is part of the feature schema but was
        # never populated by the original implementation (always 0.0).
        # Preserved for exact backward compatibility.
        df['user_total_accounts'] = 0.0
        df['user_event_count'] = uid.map(count_map).where(known, g_typical_n)

        if leave_one_out:
            # TRAINING-TIME ONLY: recompute user_*_z leave-one-out so each
            # event is scored against the user's baseline EXCLUDING itself.
            # This gives training z-scores the same honest-deviation
            # semantics as prediction (Milestone 1.8.1 defect fix: the old
            # self-referential z was near-degenerate ~0, so any honest
            # prediction-time deviation looked maximally anomalous).
            for c, src in col_map.items():
                df[f'user_{c}_z'] = self._loo_zscores(df, src)

        return df

    @staticmethod
    def _loo_zscores(df: pd.DataFrame, src_col: str) -> pd.Series:
        """
        Leave-one-out per-user z-scores for one raw feature column.

        For each user group of size k >= 2, event i is scored against the
        mean/std of the OTHER k-1 events (std ddof=1, floored to 1.0 via the
        same zero-variance guard as _safe_std). Groups of size 1 and
        non-finite values get z = 0 (no usable baseline / no evidence).
        """
        z = pd.Series(0.0, index=df.index, dtype=float)
        raw = pd.to_numeric(df[src_col], errors='coerce')
        for _, sub in df.groupby('user_id', sort=False):
            k = len(sub)
            if k < 2:
                continue
            v = raw.loc[sub.index].fillna(0.0).to_numpy(dtype=float)
            valid = np.isfinite(raw.loc[sub.index].to_numpy(dtype=float))
            m = k - 1  # leave-one-out count
            total = v.sum()
            total_sq = float((v * v).sum())
            mean_o = (total - v) / m
            if m > 1:
                ss_o = (total_sq - v * v) - m * mean_o * mean_o
                var_o = ss_o / (m - 1)
                std_o = np.where(np.isfinite(var_o) & (var_o > 0),
                                 np.sqrt(var_o), 1.0)
            else:
                std_o = np.full(k, 1.0)
            zv = np.clip((v - mean_o) / std_o, -10.0, 10.0)
            zv[~valid] = 0.0
            z.loc[sub.index] = zv
        return z
    
    def transform_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform IP features using stored baselines.
        
        FIX: Uses pre-computed IP frequency instead of computing from current DataFrame.
        
        NOTE: ip_class is derived from RFC 5737 documentation TEST-NET ranges
        (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) used by the synthetic
        dataset generator. These are SIMULATION-ONLY labels and do NOT represent
        real-world IP reputation.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with IP features added
        """
        df = df.copy()
        
        if 'source_ip' not in df.columns:
            df['source_ip'] = 'unknown'
        df['source_ip'] = _str_col_with_unknown(df['source_ip'])
        
        # Extract IP class (synthetic TEST-NET ranges only - simulation, not real reputation)
        df['ip_class'] = df['source_ip'].apply(lambda x: 
            'normal' if x.startswith('192.0.2.') else
            'suspicious' if x.startswith('198.51.100.') else
            'high_risk' if x.startswith('203.0.113.') else 'unknown'
        )
        
        # Use pre-computed IP frequency; unseen IPs get the global mean frequency
        df['ip_event_count'] = df['source_ip'].map(self.ip_frequency).fillna(self._global_ip_freq)
        
        return df
    
    def transform_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform event features using stored baselines.
        
        FIX: Uses pre-computed event type frequency instead of computing from current DataFrame.
        Unseen event types receive the global mean training frequency.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with event features added
        """
        df = df.copy()
        
        if 'event_type' not in df.columns:
            df['event_type'] = 'unknown'
        df['event_type'] = _str_col_with_unknown(df['event_type'])
        
        # Use pre-computed event type frequency
        df['event_type_frequency'] = df['event_type'].map(self.event_type_frequency).fillna(self._global_event_freq)
        
        # Create event type categories
        sensitive_events = ['grade_download', 'student_export', 'course_export', 'bulk_download']
        df['is_sensitive_event'] = df['event_type'].isin(sensitive_events).astype(int)
        
        return df
    
    def is_fitted(self) -> bool:
        """Check if baselines have been fitted."""
        return self._is_fitted

    def fingerprint(self) -> str:
        """Deterministic digest of fitted feature-baseline state."""
        payload = {
            'artifact_version': self.ARTIFACT_VERSION,
            'is_fitted': self._is_fitted,
            'user_baselines': self.user_baselines,
            'user_totals': self.user_totals,
            'ip_frequency': self.ip_frequency,
            'event_type_frequency': self.event_type_frequency,
            'global_stats': self.global_stats,
            'global_ip_freq': self._global_ip_freq,
            'global_event_freq': self._global_event_freq,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:16]


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamps.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with additional time features
    """
    df = df.copy()
    
    # Convert timestamp to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.day_of_week
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_after_hours'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
    
    return df


def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract user-based aggregations as features.
    
    NOTE: This function computes aggregations from the CURRENT DataFrame.
    For production use with baseline_manager, use FeatureBaselineManager.transform_user_features() instead.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        DataFrame with user features
    """
    df = df.copy()
    
    # User activity aggregations
    user_agg = df.groupby('user_id').agg({
        'api_requests': ['sum', 'mean', 'max'],
        'data_mb': ['sum', 'mean', 'max'],
        'courses_accessed': ['sum', 'mean', 'max'],
        'timestamp': 'count'
    }).reset_index()
    
    # Flatten column names
    user_agg.columns = ['user_id', 'user_total_api', 'user_mean_api', 'user_max_api',
                        'user_total_data', 'user_mean_data', 'user_max_data',
                        'user_total_courses', 'user_mean_courses', 'user_max_courses',
                        'user_event_count']
    
    # Merge back to main dataframe
    df = df.merge(user_agg, on='user_id', how='left')
    
    return df


def extract_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract IP-based features.
    
    NOTE: This function computes IP frequency from the CURRENT DataFrame.
    For production use with baseline_manager, use FeatureBaselineManager.transform_ip_features() instead.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        DataFrame with IP features
    """
    df = df.copy()
    
    # Extract IP class (for documentation-only ranges)
    df['ip_class'] = df['source_ip'].apply(lambda x: 
        'normal' if x.startswith('192.0.2.') else
        'suspicious' if x.startswith('198.51.100.') else
        'high_risk' if x.startswith('203.0.113.') else 'unknown'
    )
    
    # IP frequency features
    ip_freq = df.groupby('source_ip').size().reset_index(name='ip_event_count')
    df = df.merge(ip_freq, on='source_ip', how='left')
    
    return df


def extract_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract event type features.
    
    NOTE: This function computes event frequency from the CURRENT DataFrame.
    For production use with baseline_manager, use FeatureBaselineManager.transform_event_features() instead.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        DataFrame with event features
    """
    df = df.copy()
    
    # Event type frequency
    event_counts = df['event_type'].value_counts()
    df['event_type_frequency'] = df['event_type'].map(event_counts)
    
    # Create event type categories
    sensitive_events = ['grade_download', 'student_export', 'course_export', 'bulk_download']
    df['is_sensitive_event'] = df['event_type'].isin(sensitive_events).astype(int)
    
    return df


def engineer_features(df: pd.DataFrame, baseline_manager: Optional[FeatureBaselineManager] = None,
                      leave_one_out: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply all feature engineering steps.

    FIX: When baseline_manager is provided, uses pre-computed baselines
    instead of computing aggregations from current DataFrame.

    Args:
        df: Raw security log DataFrame
        baseline_manager: Optional FeatureBaselineManager with pre-computed baselines
        leave_one_out: TRAINING-TIME ONLY flag forwarded to
            transform_user_features (honest leave-one-out z-scores).
            Must remain False at prediction time.

    Returns:
        Tuple of (engineered DataFrame, list of feature column names)
    """
    # Apply time features
    df = extract_time_features(df)

    # Apply user features with or without baseline manager
    if baseline_manager is not None and baseline_manager.is_fitted():
        df = baseline_manager.transform_user_features(df, leave_one_out=leave_one_out)
    else:
        # Fallback to original method if no baseline manager provided
        df = extract_user_features(df)
    
    # Apply IP features with or without baseline manager
    if baseline_manager is not None and baseline_manager.is_fitted():
        df = baseline_manager.transform_ip_features(df)
    else:
        # Fallback to original method
        df = extract_ip_features(df)
    
    # Apply event features with or without baseline manager
    if baseline_manager is not None and baseline_manager.is_fitted():
        df = baseline_manager.transform_event_features(df)
    else:
        # Fallback to original method
        df = extract_event_features(df)
    
    # Select numerical features for ML
    feature_columns = [
        'courses_accessed', 'accounts_accessed', 'api_requests',
        'data_mb', 'token_age_minutes', 'after_hours',
        'hour', 'day_of_week', 'is_weekend', 'is_after_hours',
        'user_total_api', 'user_mean_api', 'user_max_api',
        'user_total_data', 'user_mean_data', 'user_max_data',
        'user_total_courses', 'user_mean_courses', 'user_max_courses',
        'user_total_accounts', 'user_mean_accounts', 'user_max_accounts',
        'user_api_z', 'user_data_z', 'user_courses_z', 'user_accounts_z',
        'user_event_count', 'ip_event_count', 'event_type_frequency',
        'is_sensitive_event'
    ]
    
    # Keep only columns that exist
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    return df, feature_columns


def scale_features(df: pd.DataFrame, feature_columns: List[str], scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    FIX: Accepts an optional scaler to use for transform-only during inference.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        scaler: Optional fitted StandardScaler for inference
        
    Returns:
        Tuple of (DataFrame with scaled features, scaler)
    """
    df = df.copy()
    
    # Create or use provided scaler
    if scaler is None:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[feature_columns].fillna(0))
    else:
        # Use provided scaler (for inference)
        scaled_features = scaler.transform(df[feature_columns].fillna(0))
    
    # Create scaled feature columns
    scaled_columns = [f'{col}_scaled' for col in feature_columns]
    df_scaled = pd.DataFrame(scaled_features, columns=scaled_columns, index=df.index)
    
    return pd.concat([df, df_scaled], axis=1), scaler


def get_feature_matrix(df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    """
    Get feature matrix for ML models.
    
    Args:
        df: DataFrame with features
        feature_columns: List of feature column names
        
    Returns:
        2D numpy array of feature values
    """
    numeric = df[feature_columns].apply(pd.to_numeric, errors='coerce')
    return numeric.fillna(0).values
