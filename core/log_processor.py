"""
Security Log Processor for CAID

This module provides functionality for loading, processing, and analyzing
security logs from CSV files.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import os

from utils.validators import (
    validate_security_log,
    clean_security_log,
    REQUIRED_COLUMNS,
    get_validation_summary
)


def load_security_log(file_path: str) -> pd.DataFrame:
    """
    Load security log from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the security logs
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    if df.empty:
        raise ValueError(f"CSV file is empty or contains no records: {file_path}")
    
    return df


def process_security_log(file_path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load and process security log with validation.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (cleaned DataFrame, validation report dict)
    """
    # Load raw data
    df_raw = load_security_log(file_path)
    
    # Validate
    is_valid, errors = validate_security_log(df_raw)
    
    report = {
        "file_path": file_path,
        "total_records": len(df_raw),
        "is_valid": is_valid,
        "errors": errors,
        "validation_passed": is_valid
    }
    
    if not is_valid:
        report["validation_passed"] = False
        return df_raw, report
    
    # Clean data
    df_clean = clean_security_log(df_raw)
    
    report["cleaned_records"] = len(df_clean)
    report["dropped_records"] = len(df_raw) - len(df_clean)
    report["validation_passed"] = True
    
    return df_clean, report


def load_sample_dataset(dataset_type: str = "all") -> pd.DataFrame:
    """
    Load the built-in synthetic dataset.
    
    Args:
        dataset_type: Type of dataset to load
                      ("all", "normal", "suspicious", "high_risk", "evaluation")
        
    Returns:
        DataFrame with the requested dataset
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    if dataset_type == "all":
        file_path = os.path.join(data_dir, "raw", "security_logs.csv")
    elif dataset_type == "normal":
        file_path = os.path.join(data_dir, "raw", "normal_activity.csv")
    elif dataset_type == "suspicious":
        file_path = os.path.join(data_dir, "raw", "suspicious_activity.csv")
    elif dataset_type == "high_risk":
        file_path = os.path.join(data_dir, "raw", "high_risk_activity.csv")
    elif dataset_type == "evaluation":
        file_path = os.path.join(data_dir, "processed", "evaluation_data.csv")
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample dataset not found: {file_path}")
    
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    if df.empty:
        raise ValueError(f"Sample dataset is empty: {file_path}")
    
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sample dataset {file_path} is missing required columns: {missing}"
        )
    
    return df


def get_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate statistics for a security log dataset.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        "total_events": len(df),
        "unique_users": df["user_id"].nunique() if "user_id" in df.columns else 0,
        "event_types": df["event_type"].value_counts().to_dict() if "event_type" in df.columns else {},
        "roles": df["role"].value_counts().to_dict() if "role" in df.columns else {},
        "time_range": {},
        "api_requests": {
            "total": df["api_requests"].sum() if "api_requests" in df.columns else 0,
            "mean": df["api_requests"].mean() if "api_requests" in df.columns else 0,
            "max": df["api_requests"].max() if "api_requests" in df.columns else 0
        },
        "data_transferred_mb": df["data_mb"].sum() if "data_mb" in df.columns else 0,
        "labels": df["label"].value_counts().to_dict() if "label" in df.columns else {}
    }
    
    # Time range
    if "timestamp" in df.columns:
        stats["time_range"] = {
            "start": df["timestamp"].min(),
            "end": df["timestamp"].max()
        }
    
    return stats


def filter_events(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filter security events by criteria.
    
    Args:
        df: Security log DataFrame
        **kwargs: Filter criteria (user_id, role, event_type, risk_level, etc.)
        
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    if "user_id" in kwargs:
        df_filtered = df_filtered[df_filtered["user_id"] == kwargs["user_id"]]
    
    if "role" in kwargs:
        df_filtered = df_filtered[df_filtered["role"] == kwargs["role"]]
    
    if "event_type" in kwargs:
        df_filtered = df_filtered[df_filtered["event_type"] == kwargs["event_type"]]
    
    if "label" in kwargs:
        df_filtered = df_filtered[df_filtered["label"] == kwargs["label"]]
    
    if "after_hours" in kwargs:
        df_filtered = df_filtered[df_filtered["after_hours"] == kwargs["after_hours"]]
    
    if "risk_level" in kwargs:
        df_filtered = df_filtered[df_filtered["risk_level"] == kwargs["risk_level"]]
    
    if "is_anomaly" in kwargs:
        df_filtered = df_filtered[df_filtered["is_anomaly"] == kwargs["is_anomaly"]]
    
    return df_filtered


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect duplicate records in the security log.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        DataFrame with duplicate records
    """
    duplicates = df[df.duplicated(keep=False)]
    if "timestamp" in duplicates.columns:
        duplicates = duplicates.sort_values("timestamp")
    return duplicates


def search_events(df: pd.DataFrame, search_query: str) -> pd.DataFrame:
    """
    Search events across multiple columns using a global search query.
    
    Args:
        df: Security log DataFrame
        search_query: Search string to look for
        
    Returns:
        Filtered DataFrame with matching records
    """
    if not search_query:
        return df.copy()
    
    df_search = df.copy()
    
    # Convert search query to string
    search_lower = str(search_query).lower().strip()
    
    # Search across relevant columns
    mask = pd.Series([False] * len(df_search), index=df_search.index)
    
    # Search in user_id
    if "user_id" in df_search.columns:
        mask = mask | df_search["user_id"].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
    
    # Search in role
    if "role" in df_search.columns:
        mask = mask | df_search["role"].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
    
    # Search in source_ip
    if "source_ip" in df_search.columns:
        mask = mask | df_search["source_ip"].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
    
    # Search in event_type
    if "event_type" in df_search.columns:
        mask = mask | df_search["event_type"].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
    
    # Search in timestamp
    if "timestamp" in df_search.columns:
        mask = mask | df_search["timestamp"].astype(str).str.lower().str.contains(search_lower, na=False, regex=False)
    
    return df_search[mask]


def search_events_with_filters(df: pd.DataFrame, search_query: str = "", **kwargs) -> pd.DataFrame:
    """
    Search and filter events with multiple criteria.
    
    Args:
        df: Security log DataFrame
        search_query: Global search string
        **kwargs: Additional filter criteria
        
    Returns:
        Filtered DataFrame
    """
    # Apply text search first
    df_result = search_events(df, search_query)
    
    # Apply additional filters
    for key, value in kwargs.items():
        if value is not None and value != "" and key in df_result.columns:
            if isinstance(value, list):
                df_result = df_result[df_result[key].isin(value)]
            else:
                df_result = df_result[df_result[key] == value]
    
    return df_result


def get_user_activity(df: pd.DataFrame, user_id: str) -> Dict[str, Any]:
    """
    Get detailed activity statistics for a specific user.
    
    Args:
        df: Security log DataFrame
        user_id: User ID to investigate
        
    Returns:
        Dictionary with user activity statistics
    """
    if "user_id" not in df.columns:
        return {"error": "Column 'user_id' not found in DataFrame"}
    
    user_df = df[df["user_id"] == user_id]
    
    if user_df.empty:
        return {"error": f"User {user_id} not found"}
    
    # Basic user info
    user_info = user_df.iloc[0]
    
    # Calculate activity metrics
    activity = {
        "user_id": user_id,
        "role": str(user_info.get("role", "Unknown")),
        "total_events": len(user_df),
        "total_api_requests": int(user_df["api_requests"].sum()) if "api_requests" in user_df.columns else 0,
        "total_data_mb": float(user_df["data_mb"].sum()) if "data_mb" in user_df.columns else 0,
        "unique_source_ips": int(user_df["source_ip"].nunique()) if "source_ip" in user_df.columns else 0,
        "unique_courses_accessed": int(user_df["courses_accessed"].sum()) if "courses_accessed" in user_df.columns else 0,
        "after_hours_count": int(user_df["after_hours"].sum()) if "after_hours" in user_df.columns else 0,
        "anomaly_count": int(user_df["is_anomaly"].sum()) if "is_anomaly" in user_df.columns else 0,
        "risk_distribution": {},
        "event_types": {},
        "first_activity": str(user_df["timestamp"].min()) if "timestamp" in user_df.columns else "N/A",
        "last_activity": str(user_df["timestamp"].max()) if "timestamp" in user_df.columns else "N/A"
    }
    
    # Risk distribution
    if "risk_level" in user_df.columns:
        activity["risk_distribution"] = user_df["risk_level"].value_counts().to_dict()
    
    # Event types
    if "event_type" in user_df.columns:
        activity["event_types"] = user_df["event_type"].value_counts().to_dict()
    
    return activity


def get_ip_activity(df: pd.DataFrame, ip_address: str) -> Dict[str, Any]:
    """
    Get detailed activity statistics for a specific IP address.
    
    Args:
        df: Security log DataFrame
        ip_address: IP address to investigate
        
    Returns:
        Dictionary with IP activity statistics
    """
    if "source_ip" not in df.columns:
        return {"error": "Column 'source_ip' not found in DataFrame"}
    
    ip_df = df[df["source_ip"] == ip_address]
    
    if ip_df.empty:
        return {"error": f"IP {ip_address} not found"}
    
    # Basic IP info
    ip_info = {
        "ip_address": ip_address,
        "associated_users": list(ip_df["user_id"].unique()) if "user_id" in ip_df.columns else [],
        "event_count": len(ip_df),
        "first_observed": str(ip_df["timestamp"].min()) if "timestamp" in ip_df.columns else "N/A",
        "last_observed": str(ip_df["timestamp"].max()) if "timestamp" in ip_df.columns else "N/A",
        "event_types": {},
        "risk_classifications": {},
        "unusual_indicators": []
    }
    
    # Event types
    if "event_type" in ip_df.columns:
        ip_info["event_types"] = ip_df["event_type"].value_counts().to_dict()
    
    # Risk classifications
    if "risk_level" in ip_df.columns:
        ip_info["risk_classifications"] = ip_df["risk_level"].value_counts().to_dict()
    
    # Check for unusual indicators
    if "api_requests" in ip_df.columns:
        avg_api = ip_df["api_requests"].mean()
        if avg_api > 300:
            ip_info["unusual_indicators"].append(f"High average API requests: {avg_api:.0f}")
    
    if "data_mb" in ip_df.columns:
        avg_data = ip_df["data_mb"].mean()
        if avg_data > 10:
            ip_info["unusual_indicators"].append(f"High average data transfer: {avg_data:.2f} MB")
    
    if "is_anomaly" in ip_df.columns:
        anomaly_rate = ip_df["is_anomaly"].mean()
        if anomaly_rate > 0.5:
            ip_info["unusual_indicators"].append(f"High anomaly rate: {anomaly_rate * 100:.1f}%")
    
    return ip_info


def generate_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a chronological event timeline.
    
    Args:
        df: Security log DataFrame
        
    Returns:
        DataFrame with timeline events
    """
    if "timestamp" not in df.columns:
        return pd.DataFrame()
    
    # Create timeline dataframe with a proper datetime column
    timeline = df.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
    timeline = timeline.dropna(subset=["timestamp"])
    
    if timeline.empty:
        return pd.DataFrame()
    
    # Sort by timestamp
    timeline = timeline.sort_values("timestamp", ascending=False)
    
    # Add time-based columns for analysis
    timeline["date"] = timeline["timestamp"].dt.date
    timeline["hour"] = timeline["timestamp"].dt.hour
    
    return timeline


def get_timeline_events(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """
    Get recent timeline events.
    
    Args:
        df: Security log DataFrame
        n: Number of events to return
        
    Returns:
        DataFrame with timeline events
    """
    timeline = generate_timeline(df)
    
    if timeline.empty:
        return timeline
    
    return timeline.head(n).copy()
