"""
Data validators for CAID

This module provides validation functions for CSV security logs.
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime


REQUIRED_COLUMNS = [
    'timestamp', 'user_id', 'role', 'source_ip', 'event_type',
    'courses_accessed', 'accounts_accessed', 'api_requests',
    'data_mb', 'token_age_minutes', 'after_hours'
]

OPTIONAL_COLUMNS = ['label', 'session_id', 'auth_result']

# Valid event types for Canvas activity (synthetic research schema; expanded
# for independent v1.9/v2.0 generators). This is not a real Canvas taxonomy.
VALID_EVENT_TYPES = [
    'login', 'logout', 'course_view', 'discussion_post', 'discussion_view',
    'assignment_submit', 'assignment_view', 'file_view', 'file_upload',
    'quiz_submitted', 'quiz_view', 'grade_download', 'grade_view',
    'student_export', 'course_export', 'bulk_download', 'bulk_export',
    'content_view', 'sidebar_view', 'settings_view', 'user_view',
    'group_view', 'resource_view', 'enrollment_view', 'lti_launch',
    'mastery_check', 'wiki_review', 'wiki_edit', 'module_view',
    'page_view', 'announcement_view', 'submission_view', 'peer_review',
    'conference_join'
]

# IP address pattern (basic validation)
IP_PATTERN = re.compile(
    r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
    r'(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
)

# Synthetic IP ranges (for documentation)
SYNTHETIC_IP_RANGES = {
    'TEST_NET_1': '192.0.2.0/24',      # Documentation (RFC 5737)
    'TEST_NET_2': '198.51.100.0/24',   # Documentation (RFC 5737)
    'TEST_NET_3': '203.0.113.0/24'     # Documentation (RFC 5737)
}


def validate_csv_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that required columns are present in the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_required = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing_required.append(col)
    
    return len(missing_required) == 0, missing_required


def validate_numeric_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that numeric columns contain valid numeric values.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, problematic_columns)
    """
    numeric_cols = [
        'courses_accessed', 'accounts_accessed', 'api_requests',
        'data_mb', 'token_age_minutes', 'after_hours'
    ]
    
    problematic = []
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                problematic.append(col)
    
    return len(problematic) == 0, problematic


def validate_file_size(file_path: str, max_size_mb: float = 10.0) -> Tuple[bool, str]:
    """
    Validate file size is within acceptable limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import os
    
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed ({max_size_mb} MB)"
        return True, ""
    except OSError as e:
        return False, f"Error checking file size: {str(e)}"


def validate_file_extension(file_path: str, allowed_extensions: List[str] = ['.csv']) -> Tuple[bool, str]:
    """
    Validate file extension is in the allowed list.
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import os
    
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in [e.lower() for e in allowed_extensions]:
        return False, f"File extension '{ext}' is not allowed. Allowed: {', '.join(allowed_extensions)}"
    return True, ""


def validate_ip_format(ip: str) -> bool:
    """
    Validate that an IP address has a valid IPv4 format.
    
    Args:
        ip: IP address string to validate
        
    Returns:
        True if valid IPv4 format, False otherwise
    """
    if not isinstance(ip, str):
        return False
    
    ip = ip.strip()
    if not ip:
        return False
    
    return bool(IP_PATTERN.match(ip))


def validate_ip_range(ip: str) -> str:
    """
    Check if IP is in synthetic ranges (for documentation).
    
    Args:
        ip: IP address string
        
    Returns:
        Range type ('TEST_NET_1', 'TEST_NET_2', 'TEST_NET_3', or 'OTHER')
    """
    if not isinstance(ip, str):
        return 'OTHER'
    
    ip = ip.strip()
    
    if ip.startswith('192.0.2.'):
        return 'TEST_NET_1'
    elif ip.startswith('198.51.100.'):
        return 'TEST_NET_2'
    elif ip.startswith('203.0.113.'):
        return 'TEST_NET_3'
    else:
        return 'OTHER'


def validate_timestamp_format(timestamp: str) -> Tuple[bool, Optional[datetime]]:
    """
    Validate and parse timestamp string.
    
    Args:
        timestamp: Timestamp string to validate
        
    Returns:
        Tuple of (is_valid, parsed_datetime or None)
    """
    if not isinstance(timestamp, str):
        return False, None
    
    timestamp = timestamp.strip()
    if not timestamp:
        return False, None
    
    # Try common datetime formats
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d',
        '%d/%m/%Y %H:%M:%S',
        '%m/%d/%Y %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return True, datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    
    return False, None


def validate_numeric_value(value: Any, column_name: str, allow_zero: bool = True) -> Tuple[bool, Optional[float]]:
    """
    Validate that a value can be converted to a numeric type.
    
    Args:
        value: Value to validate
        column_name: Name of the column (for error messages)
        allow_zero: Whether zero values are allowed
        
    Returns:
        Tuple of (is_valid, numeric_value or None)
    """
    if pd.isna(value):
        return False, None
    
    try:
        numeric_val = float(value)
        if not allow_zero and numeric_val == 0:
            return False, None
        return True, numeric_val
    except (ValueError, TypeError):
        return False, None


def validate_event_type(event_type: str) -> Tuple[bool, str]:
    """
    Validate event type against known Canvas events.
    
    Args:
        event_type: Event type string to validate
        
    Returns:
        Tuple of (is_valid, normalized_event_type)
    """
    if not isinstance(event_type, str):
        return False, ""
    
    event_type = event_type.strip()
    if not event_type:
        return False, ""
    
    # Check against known valid event types (case-insensitive)
    event_type_lower = event_type.lower()
    for valid_type in VALID_EVENT_TYPES:
        if event_type_lower == valid_type.lower():
            return True, valid_type
    
    return False, event_type


def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all required columns are present in the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)
    
    return len(missing) == 0, missing


def validate_optional_columns(df: pd.DataFrame) -> List[str]:
    """
    Check for unexpected columns in the DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of unexpected column names
    """
    all_columns = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    unexpected = []
    
    for col in df.columns:
        if col not in all_columns:
            unexpected.append(col)
    
    return unexpected


def validate_data_types(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate data types for critical columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check string columns
    string_cols = ['user_id', 'role', 'source_ip', 'event_type']
    for col in string_cols:
        if col in df.columns:
            non_string = df[~df[col].apply(lambda x: isinstance(x, (str, type(None))) or pd.isna(x))]
            if len(non_string) > 0:
                issues.append({
                    'column': col,
                    'issue': 'Non-string values detected',
                    'count': len(non_string)
                })
    
    # Check numeric columns
    numeric_cols = ['courses_accessed', 'accounts_accessed', 'api_requests',
                   'data_mb', 'token_age_minutes', 'after_hours']
    for col in numeric_cols:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                issues.append({
                    'column': col,
                    'issue': 'Non-numeric values detected',
                    'count': df[col].apply(lambda x: pd.isna(pd.to_numeric(x, errors='coerce'))).sum()
                })
    
    return issues


def validate_missing_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Check for missing values in required columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of missing value issues
    """
    issues = []
    
    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            missing_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == '').sum()
            if missing_count > 0:
                issues.append({
                    'column': col,
                    'missing_count': int(missing_count),
                    'missing_percent': round(missing_count / len(df) * 100, 2)
                })
    
    return issues


def validate_duplicate_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect duplicate records based on all columns.
    
    Args:
        df: DataFrame to check
        
    Returns:
        List of duplicate issues (with count and percentage)
    """
    # Create a copy with hashable columns for duplicate detection
    # Convert unhashable types (lists, dicts, arrays) to strings.
    # hash() is checked first so pd.isna() only ever sees scalars —
    # array-like values would make pd.isna() return an array and
    # produce an ambiguous truth value.
    def _make_hashable(x):
        try:
            hash(x)
        except TypeError:
            return str(x)
        try:
            return '' if bool(pd.isna(x)) else x
        except (TypeError, ValueError):
            # Hashable but non-scalar (e.g. tuple): pd.isna() returns an
            # array — stringify so duplicated() can compare it.
            return str(x)

    df_check = df.copy()
    for col in df_check.columns:
        if df_check[col].dtype == 'object':
            df_check[col] = df_check[col].apply(_make_hashable)
    
    duplicate_mask = df_check.duplicated(keep=False)
    duplicate_count = duplicate_mask.sum()
    
    if duplicate_count > 0:
        # Count unique duplicate groups
        duplicate_groups = df_check[duplicate_mask].drop_duplicates()
        unique_duplicates = len(duplicate_groups)
        
        return [{
            'issue': 'Duplicate records found',
            'total_duplicate_records': int(duplicate_count),
            'unique_duplicate_groups': int(unique_duplicates),
            'duplicate_percent': round(duplicate_count / len(df) * 100, 2)
        }]
    
    return []


def validate_timestamps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate timestamp column format and convertibility.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of timestamp validation issues
    """
    issues = []
    
    if 'timestamp' not in df.columns:
        issues.append({
            'column': 'timestamp',
            'issue': 'Missing required column'
        })
        return issues
    
    # Try to parse timestamps
    parsed = pd.to_datetime(df['timestamp'], errors='coerce')
    invalid_count = parsed.isna().sum()
    
    if invalid_count > 0:
        issues.append({
            'column': 'timestamp',
            'issue': 'Invalid timestamp format',
            'invalid_count': int(invalid_count),
            'invalid_percent': round(invalid_count / len(df) * 100, 2)
        })
    
    return issues


def validate_ips(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate IP address format in source_ip column.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of IP validation issues
    """
    issues = []
    
    if 'source_ip' not in df.columns:
        issues.append({
            'column': 'source_ip',
            'issue': 'Missing required column'
        })
        return issues
    
    # Validate IP format
    invalid_ips = df['source_ip'].apply(
        lambda x: not validate_ip_format(str(x).strip()) if pd.notna(x) else True
    )
    invalid_count = invalid_ips.sum()
    
    if invalid_count > 0:
        issues.append({
            'column': 'source_ip',
            'issue': 'Invalid IP address format',
            'invalid_count': int(invalid_count),
            'invalid_percent': round(invalid_count / len(df) * 100, 2)
        })
    
    return issues


def validate_event_types(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate event_type values against known Canvas events.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of event type validation issues
    """
    issues = []
    
    if 'event_type' not in df.columns:
        issues.append({
            'column': 'event_type',
            'issue': 'Missing required column'
        })
        return issues
    
    # Validate event types
    invalid_events = df['event_type'].apply(
        lambda x: not validate_event_type(str(x).strip())[0] if pd.notna(x) else True
    )
    invalid_count = invalid_events.sum()
    
    if invalid_count > 0:
        issues.append({
            'column': 'event_type',
            'issue': 'Invalid or unknown event type',
            'invalid_count': int(invalid_count),
            'invalid_percent': round(invalid_count / len(df) * 100, 2)
        })
    
    return issues


def validate_user_ids(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Validate user_id column is not empty.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        List of user ID validation issues
    """
    issues = []
    
    if 'user_id' not in df.columns:
        issues.append({
            'column': 'user_id',
            'issue': 'Missing required column'
        })
        return issues
    
    # Check for empty user IDs
    empty_mask = df['user_id'].isna() | (df['user_id'].astype(str).str.strip() == '')
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        issues.append({
            'column': 'user_id',
            'issue': 'Empty or missing user IDs',
            'empty_count': int(empty_count),
            'empty_percent': round(empty_count / len(df) * 100, 2)
        })
    
    return issues


def validate_security_log(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Run all security log validations.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check for empty DataFrame
    if df.empty:
        errors.append("DataFrame is empty")
        return False, errors
    
    # Validate required columns
    is_valid, missing_cols = validate_required_columns(df)
    if not is_valid:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Validate optional columns
    unexpected_cols = validate_optional_columns(df)
    if unexpected_cols:
        errors.append(f"Unexpected columns found: {', '.join(unexpected_cols)}")
    
    # Validate timestamps
    timestamp_issues = validate_timestamps(df)
    for issue in timestamp_issues:
        errors.append(f"{issue['column']}: {issue['issue']} ({issue.get('invalid_count', 0)} records)")
    
    # Validate IP addresses
    ip_issues = validate_ips(df)
    for issue in ip_issues:
        errors.append(f"{issue['column']}: {issue['issue']} ({issue.get('invalid_count', 0)} records)")
    
    # Validate event types
    event_issues = validate_event_types(df)
    for issue in event_issues:
        errors.append(f"{issue['column']}: {issue['issue']} ({issue.get('invalid_count', 0)} records)")
    
    # Validate user IDs
    user_issues = validate_user_ids(df)
    for issue in user_issues:
        errors.append(f"{issue['column']}: {issue['issue']} ({issue.get('empty_count', 0)} records)")
    
    # Check for missing values
    missing_issues = validate_missing_values(df)
    for issue in missing_issues:
        errors.append(f"{issue['column']}: Missing values ({issue['missing_count']} records, {issue['missing_percent']}%)")
    
    # Check for duplicates
    duplicate_issues = validate_duplicate_records(df)
    for issue in duplicate_issues:
        errors.append(f"{issue['issue']} ({issue['total_duplicate_records']} records)")
    
    return len(errors) == 0, errors


def clean_security_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess security log data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Clean user_id
    df['user_id'] = df['user_id'].astype(str).str.strip()
    df = df[df['user_id'] != '']
    
    # Fill missing numeric values with 0
    numeric_cols = [
        'courses_accessed', 'accounts_accessed', 'api_requests',
        'data_mb', 'token_age_minutes', 'after_hours'
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def get_validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a comprehensive validation summary for a DataFrame.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary containing validation summary
    """
    summary = {
        'total_records': len(df),
        'required_columns_present': True,
        'missing_columns': [],
        'unexpected_columns': [],
        'has_invalid_timestamps': False,
        'invalid_timestamp_count': 0,
        'has_invalid_ips': False,
        'invalid_ip_count': 0,
        'has_invalid_event_types': False,
        'invalid_event_type_count': 0,
        'has_empty_user_ids': False,
        'empty_user_id_count': 0,
        'missing_value_columns': [],
        'duplicate_groups': 0,
        'total_duplicate_records': 0,
        'overall_valid': True
    }
    
    if df.empty:
        summary['overall_valid'] = False
        return summary
    
    # Check columns
    is_valid, missing_cols = validate_required_columns(df)
    summary['required_columns_present'] = is_valid
    summary['missing_columns'] = missing_cols
    
    summary['unexpected_columns'] = validate_optional_columns(df)
    
    # Check timestamps
    timestamp_issues = validate_timestamps(df)
    if timestamp_issues:
        summary['has_invalid_timestamps'] = True
        summary['invalid_timestamp_count'] = timestamp_issues[0].get('invalid_count', 0)
    
    # Check IPs
    ip_issues = validate_ips(df)
    if ip_issues:
        summary['has_invalid_ips'] = True
        summary['invalid_ip_count'] = ip_issues[0].get('invalid_count', 0)
    
    # Check event types
    event_issues = validate_event_types(df)
    if event_issues:
        summary['has_invalid_event_types'] = True
        summary['invalid_event_type_count'] = event_issues[0].get('invalid_count', 0)
    
    # Check user IDs
    user_issues = validate_user_ids(df)
    if user_issues:
        summary['has_empty_user_ids'] = True
        summary['empty_user_id_count'] = user_issues[0].get('empty_count', 0)
    
    # Check missing values
    missing_issues = validate_missing_values(df)
    for issue in missing_issues:
        summary['missing_value_columns'].append({
            'column': issue['column'],
            'count': issue['missing_count'],
            'percent': issue['missing_percent']
        })
    
    # Check duplicates
    duplicate_issues = validate_duplicate_records(df)
    if duplicate_issues:
        summary['duplicate_groups'] = duplicate_issues[0]['unique_duplicate_groups']
        summary['total_duplicate_records'] = duplicate_issues[0]['total_duplicate_records']
    
    # Determine overall validity
    if (not summary['required_columns_present'] or
        summary['has_invalid_timestamps'] or
        summary['has_invalid_ips'] or
        summary['has_invalid_event_types'] or
        summary['has_empty_user_ids'] or
        len(summary['missing_value_columns']) > 0):
        summary['overall_valid'] = False
    
    return summary

def clean_and_summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Strict cleaning stage for uploaded security logs (Milestone 1.7).

    Produces a row-level audit summary: retained / dropped / corrected counts
    plus per-issue breakdowns. Original input data is never mutated — the
    caller retains the raw frame for auditability.

    Corrections applied:
    - timestamp: unparseable -> row dropped
    - user_id: missing/blank -> row dropped
    - numeric fields: non-numeric/NaN -> coerced to 0 (counted as corrected)
    - after_hours: coerced to 0/1 (values >1 or non-numeric -> 0)

    Args:
        df: Raw uploaded DataFrame

    Returns:
        (cleaned DataFrame, summary dict)
    """
    summary = {
        'input_rows': int(len(df)),
        'retained_rows': 0,
        'dropped_rows': 0,
        'corrected_rows': 0,
        'dropped_invalid_timestamp': 0,
        'dropped_missing_user_id': 0,
        'corrected_numeric_fields': 0,
        'corrected_after_hours': 0,
        'columns_present': list(df.columns),
    }

    if df is None or len(df) == 0:
        summary['dropped_rows'] = summary['input_rows']
        return pd.DataFrame(), summary

    work = df.copy()
    required_present = all(c in work.columns for c in
                           ('timestamp', 'user_id'))
    if not required_present:
        # Cannot clean meaningfully without key columns; drop everything
        summary['dropped_rows'] = summary['input_rows']
        return pd.DataFrame(columns=df.columns), summary

    # --- timestamps ---
    parsed_ts = pd.to_datetime(work['timestamp'], errors='coerce')
    bad_ts = parsed_ts.isna()
    summary['dropped_invalid_timestamp'] = int(bad_ts.sum())
    work = work.loc[~bad_ts].copy()
    work['timestamp'] = parsed_ts.loc[work.index]

    # --- user_id ---
    uid = work['user_id'].astype(str).str.strip() if 'user_id' in work.columns else pd.Series('', index=work.index)
    bad_uid = uid.isna() | (uid == '')
    summary['dropped_missing_user_id'] = int(bad_uid.sum())
    work = work.loc[~bad_uid].copy()
    work['user_id'] = uid.loc[work.index]

    # --- numeric fields ---
    numeric_cols = ['courses_accessed', 'accounts_accessed', 'api_requests',
                    'data_mb', 'token_age_minutes']
    corrected_mask = pd.Series(False, index=work.index)
    for col in numeric_cols:
        if col in work.columns:
            coerced = pd.to_numeric(work[col], errors='coerce')
            bad = coerced.isna() | ~np.isfinite(coerced)
            if bad.any():
                summary['corrected_numeric_fields'] += int(bad.sum())
                corrected_mask |= bad
                work.loc[bad, col] = 0
            work[col] = coerced.fillna(0)

    # --- after_hours -> strict 0/1 ---
    if 'after_hours' in work.columns:
        ah = pd.to_numeric(work['after_hours'], errors='coerce')
        bad_ah = ah.isna() | ~ah.isin([0, 1])
        if bad_ah.any():
            summary['corrected_after_hours'] += int(bad_ah.sum())
            corrected_mask |= bad_ah
            ah = ah.mask(bad_ah, 0)
        work['after_hours'] = ah.fillna(0).astype(int)

    summary['retained_rows'] = int(len(work))
    summary['dropped_rows'] = summary['input_rows'] - summary['retained_rows']
    summary['corrected_rows'] = int(corrected_mask.sum())
    return work.reset_index(drop=True), summary
