"""
CSV Export Module for CAID

This module provides functionality for exporting security log data
in various formats for analysis and reporting.
"""

import pandas as pd
import io
from typing import Optional, Dict, Any


_FORMULA_PREFIXES = ('=', '+', '-', '@', '\t', '\r')


def _sanitize_csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralize spreadsheet formula injection in exported CSVs.

    Cells whose text begins with a formula trigger character are prefixed
    with a single quote so spreadsheet applications treat them as text.
    """
    df = df.copy()
    for col in df.columns:
        # pandas 3.x uses a dedicated 'str' dtype; sanitize every
        # non-numeric, non-datetime column regardless of dtype flavor.
        if (not pd.api.types.is_numeric_dtype(df[col])
                and not pd.api.types.is_datetime64_any_dtype(df[col])):
            df[col] = df[col].apply(
                lambda v: "'" + v
                if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES)
                else v
            )
    return df


def export_csv(df: pd.DataFrame, filename: str = "export.csv") -> bytes:
    """
    Export DataFrame to CSV format.
    
    Args:
        df: DataFrame to export
        filename: Output filename (used for content-disposition header)
        
    Returns:
        CSV content as bytes
    """
    return _sanitize_csv_frame(df).to_csv(index=False).encode('utf-8')


def get_download_button_csv(df: pd.DataFrame, filename: str = "export.csv", 
                            button_label: str = "Download CSV") -> tuple:
    """
    Get CSV export data for Streamlit download button.
    
    Args:
        df: DataFrame to export
        filename: Filename for download
        button_label: Label for the download button
        
    Returns:
        Tuple of (csv_data, filename, button_label)
    """
    csv = _sanitize_csv_frame(df).to_csv(index=False)
    return csv.encode('utf-8'), filename, button_label


def export_analysis_summary(df: pd.DataFrame, summary_stats: Dict[str, Any] = None) -> bytes:
    """
    Export analysis summary as CSV.
    
    Args:
        df: Analyzed DataFrame
        summary_stats: Additional statistics to include
        
    Returns:
        Analysis summary as CSV bytes
    """
    summary_data = []
    
    # Add overall statistics
    summary_data.append({
        'metric': 'Total Events',
        'value': len(df),
        'description': 'Total number of security events'
    })
    
    if 'user_id' in df.columns:
        summary_data.append({
            'metric': 'Unique Users',
            'value': df['user_id'].nunique(),
            'description': 'Number of unique users'
        })
    
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        for level, count in risk_counts.items():
            summary_data.append({
                'metric': f'{level} Events',
                'value': int(count),
                'description': f'Number of {level.lower()} risk events'
            })
    
    if 'is_anomaly' in df.columns:
        anomaly_count = int(df['is_anomaly'].sum())
        summary_data.append({
            'metric': 'ML Anomalies',
            'value': anomaly_count,
            'description': 'Events flagged by Isolation Forest'
        })
    
    # Add summary statistics if provided
    if summary_stats:
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)):
                summary_data.append({
                    'metric': key.replace('_', ' ').title(),
                    'value': value,
                    'description': ''
                })
    
    summary_df = pd.DataFrame(summary_data)
    return _sanitize_csv_frame(summary_df).to_csv(index=False).encode('utf-8')


def export_filtered_data(df: pd.DataFrame, filters: Dict[str, Any], 
                         search_query: str = "") -> bytes:
    """
    Export filtered data with filter information.
    
    Args:
        df: DataFrame to export
        filters: Applied filters
        search_query: Search query used
        
    Returns:
        Filtered data CSV bytes
    """
    export_df = df.copy()
    
    # Add filter metadata as comments
    metadata_lines = [
        "# Filtered Data Export",
        f"# Total Records: {len(export_df)}",
        f"# Search Query: {' '.join(str(search_query).split())}",
        "# Applied Filters:"
    ]
    
    for key, value in filters.items():
        if value is not None and value != '':
            # Collapse newlines so a crafted filter value cannot break out
            # of the '#' comment prefix and inject spreadsheet formulas.
            safe_value = ' '.join(str(value).split())
            metadata_lines.append(f"#   {key}: {safe_value}")
    
    # Convert to CSV and prepend metadata
    csv_content = _sanitize_csv_frame(export_df).to_csv(index=False)
    full_content = '\n'.join(metadata_lines) + '\n' + csv_content
    
    return full_content.encode('utf-8')


def export_ip_investigation(ip_data: Dict[str, Any], filename: str = "ip_investigation.csv") -> bytes:
    """
    Export IP investigation data to CSV.
    
    Args:
        ip_data: Dictionary with IP investigation results
        filename: Output filename
        
    Returns:
        IP investigation data as CSV bytes
    """
    data_rows = []
    
    # Add basic IP info
    data_rows.append({
        'field': 'IP Address',
        'value': ip_data.get('ip_address', ''),
        'type': 'basic'
    })
    
    data_rows.append({
        'field': 'Event Count',
        'value': ip_data.get('event_count', ''),
        'type': 'basic'
    })
    
    data_rows.append({
        'field': 'First Observed',
        'value': ip_data.get('first_observed', ''),
        'type': 'basic'
    })
    
    data_rows.append({
        'field': 'Last Observed',
        'value': ip_data.get('last_observed', ''),
        'type': 'basic'
    })
    
    # Add associated users
    users = ip_data.get('associated_users', [])
    data_rows.append({
        'field': 'Associated Users',
        'value': ', '.join(users) if users else 'None',
        'type': 'users'
    })
    
    # Add risk classifications
    risk_class = ip_data.get('risk_classifications', {})
    for level, count in risk_class.items():
        data_rows.append({
            'field': f'Risk: {level}',
            'value': count,
            'type': 'risk'
        })
    
    # Add event types
    event_types = ip_data.get('event_types', {})
    for event_type, count in event_types.items():
        data_rows.append({
            'field': f'Event: {event_type}',
            'value': count,
            'type': 'events'
        })
    
    # Add unusual indicators
    indicators = ip_data.get('unusual_indicators', [])
    for indicator in indicators:
        data_rows.append({
            'field': 'Unusual Indicator',
            'value': indicator,
            'type': 'alerts'
        })
    
    result_df = pd.DataFrame(data_rows)
    return _sanitize_csv_frame(result_df).to_csv(index=False).encode('utf-8')


def export_user_investigation(user_data: Dict[str, Any], filename: str = "user_investigation.csv") -> bytes:
    """
    Export user investigation data to CSV.
    
    Args:
        user_data: Dictionary with user investigation results
        filename: Output filename
        
    Returns:
        User investigation data as CSV bytes
    """
    data_rows = []
    
    # Add user info
    for key, value in user_data.items():
        if key != 'risk_distribution' and key != 'event_types':
            data_rows.append({
                'field': key.replace('_', ' ').title(),
                'value': value,
                'type': 'user_info'
            })
    
    # Add risk distribution
    risk_dist = user_data.get('risk_distribution', {})
    for level, count in risk_dist.items():
        data_rows.append({
            'field': f'Risk: {level}',
            'value': count,
            'type': 'risk'
        })
    
    # Add event types
    event_types = user_data.get('event_types', {})
    for event_type, count in event_types.items():
        data_rows.append({
            'field': f'Event: {event_type}',
            'value': count,
            'type': 'events'
        })
    
    result_df = pd.DataFrame(data_rows)
    return _sanitize_csv_frame(result_df).to_csv(index=False).encode('utf-8')


def export_timeline_events(timeline_df: pd.DataFrame, filename: str = "timeline_events.csv") -> bytes:
    """
    Export timeline events to CSV.
    
    Args:
        timeline_df: Timeline DataFrame
        filename: Output filename
        
    Returns:
        Timeline events as CSV bytes
    """
    if timeline_df.empty:
        return b''
    
    # Select relevant columns for export
    relevant_cols = ['timestamp', 'user_id', 'role', 'source_ip', 
                     'event_type', 'risk_level', 'is_anomaly', 'anomaly_score']
    
    # Filter to only existing columns
    export_cols = [col for col in relevant_cols if col in timeline_df.columns]
    
    export_df = timeline_df[export_cols].copy()
    
    # Format timestamp
    if 'timestamp' in export_df.columns:
        export_df['timestamp'] = export_df['timestamp'].astype(str)
    
    return _sanitize_csv_frame(export_df).to_csv(index=False).encode('utf-8')