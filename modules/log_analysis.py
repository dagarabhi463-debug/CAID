"""
Log Analysis Module for CAID

This module provides functionality for uploading, validating, searching,
filtering, and investigating security logs with a professional UI.
"""

import logging

import streamlit as st
import pandas as pd
import io
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

from core.log_processor import (
    get_dataset_statistics,
    search_events,
    filter_events,
    search_events_with_filters,
    get_user_activity,
    get_ip_activity,
    generate_timeline,
    detect_duplicates
)
from core.risk_engine import RiskEngine
from core.calibration import apply_to_engine as apply_risk_calibration
from utils.validators import (
    validate_security_log,
    get_validation_summary,
    REQUIRED_COLUMNS,
    clean_and_summarize,
)
from models.anomaly_detector import ensure_model_exists, AnomalyDetector
from modules.export import (
    get_download_button_csv,
    export_analysis_summary,
    export_filtered_data,
    export_ip_investigation,
    export_user_investigation,
    export_timeline_events
)


def display_upload_section():
    """Display professional file upload section with validation."""
    st.subheader("📁 Upload Security Log")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a security log CSV file with required columns",
            key="upload_section"
        )
    
    if uploaded_file:
        # Display file info
        file_size_kb = uploaded_file.size / 1024
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col_info = st.columns(3)
        with col_info[0]:
            st.info(f"**Filename:** {uploaded_file.name}")
        with col_info[1]:
            if file_size_mb < 1:
                st.info(f"**Size:** {file_size_kb:.1f} KB")
            else:
                st.info(f"**Size:** {file_size_mb:.2f} MB")
        with col_info[2]:
            st.info(f"**Lines:** ~{uploaded_file.size // 50:,} (estimated)")
        
        return uploaded_file
    
    return None


def validate_upload(file) -> Tuple[bool, str]:
    """
    Validate uploaded file with comprehensive checks.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Check file size (10MB limit)
    file_size_mb = file.size / (1024 * 1024)
    if file_size_mb > 10:
        return False, f"File size ({file_size_mb:.2f} MB) exceeds 10MB limit"
    
    # Check file extension
    if not file.name.lower().endswith('.csv'):
        return False, "Only CSV files are allowed"
    
    # Read and basic validation
    try:
        file_contents = file.read()
        file.seek(0)  # Reset file pointer
        
        # Try to parse as CSV
        df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')), nrows=5)
        
        if df.empty:
            return False, "CSV file appears to be empty"
        
        return True, "File valid"
    
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def process_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Process uploaded file and return DataFrame.
    
    Args:
        uploaded_file: Uploaded file object
        
    Returns:
        Processed DataFrame
    """
    # Read file
    file_contents = uploaded_file.read()
    
    # Create DataFrame
    df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
    
    return df


def display_data_quality(df: pd.DataFrame):
    """Display comprehensive data quality report."""
    st.subheader("📊 Data Quality Report")
    
    # Get validation summary
    summary = get_validation_summary(df)
    
    if summary['overall_valid']:
        st.success("✅ Data validation passed!")
    else:
        st.warning("⚠️ Data validation found issues")
    
    # Display summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", summary['total_records'])
    
    with col2:
        if summary['required_columns_present']:
            st.success("All Required Columns Present")
        else:
            st.error(f"Missing: {', '.join(summary['missing_columns'])}")
    
    with col3:
        if summary['duplicate_groups'] > 0:
            st.warning(f"{summary['total_duplicate_records']} Duplicate Records Found")
        else:
            st.success("No Duplicate Records")
    
    # Display missing values
    if summary['missing_value_columns']:
        st.subheader("Missing Values")
        missing_df = pd.DataFrame(summary['missing_value_columns'])
        st.dataframe(missing_df)
    
    # Display validation issues
    if not summary['overall_valid']:
        issues = []
        if summary['has_invalid_timestamps']:
            issues.append(f"Invalid timestamps: {summary['invalid_timestamp_count']} records")
        if summary['has_invalid_ips']:
            issues.append(f"Invalid IPs: {summary['invalid_ip_count']} records")
        if summary['has_invalid_event_types']:
            issues.append(f"Invalid event types: {summary['invalid_event_type_count']} records")
        if summary['has_empty_user_ids']:
            issues.append(f"Empty user IDs: {summary['empty_user_id_count']} records")
        
        if issues:
            st.subheader("Validation Issues")
            for issue in issues:
                st.warning(f"⚠️ {issue}")


def display_statistics(df: pd.DataFrame):
    """Display comprehensive dataset statistics."""
    st.subheader("📈 Dataset Statistics")
    
    # Get statistics
    stats = get_dataset_statistics(df)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", f"{stats['total_events']:,}")
    
    with col2:
        st.metric("Unique Users", f"{stats['unique_users']:,}")
    
    with col3:
        if stats['time_range']:
            st.metric("Time Range Start", str(stats['time_range']['start'])[:10])
        else:
            st.metric("Time Range Start", "N/A")
    
    with col4:
        if stats['time_range']:
            st.metric("Time Range End", str(stats['time_range']['end'])[:10])
        else:
            st.metric("Time Range End", "N/A")
    
    # Display API and data stats
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("API Requests")
        st.write(f"**Total:** {stats['api_requests']['total']:,.0f}")
        st.write(f"**Mean:** {stats['api_requests']['mean']:,.1f}")
        st.write(f"**Max:** {stats['api_requests']['max']:,.0f}")
    
    with col6:
        st.subheader("Data Transfer")
        st.write(f"**Total (MB):** {stats['data_transferred_mb']:,.2f}")
    
    # Display event types distribution
    if stats['event_types']:
        st.subheader("Event Types")
        event_df = pd.DataFrame({
            'Event Type': list(stats['event_types'].keys()),
            'Count': list(stats['event_types'].values())
        })
        event_df = event_df.sort_values('Count', ascending=False)
        st.dataframe(event_df, width='stretch')


def display_risk_analysis(df: pd.DataFrame):
    """Display risk analysis results."""
    st.subheader("🛡️ Risk Analysis")
    
    risk_engine = RiskEngine()
    # Use the same persisted calibrated thresholds as the main pipeline
    # when the production detector artifact is available (Milestone 1.9).
    try:
        apply_risk_calibration(risk_engine, detector=ensure_model_exists())
    except Exception:
        pass  # explicit uncalibrated_dataset_relative fallback inside engine
    df_with_risk = risk_engine.process_events(df)
    
    # Display risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        risk_counts = df_with_risk['risk_level'].value_counts()
        risk_df = risk_counts.reset_index()
        risk_df.columns = ['Risk Level', 'Count']
        st.dataframe(risk_df, width='stretch')
    
    with col2:
        # ML Anomaly distribution
        if 'is_anomaly' in df_with_risk.columns:
            st.subheader("ML Anomaly Distribution")
            anomaly_counts = df_with_risk['is_anomaly'].value_counts()
            anomaly_df = pd.DataFrame({
                'Category': ['Normal', 'Anomaly'],
                'Count': [anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)]
            })
            st.dataframe(anomaly_df, width='stretch')
    
    # Display top anomalies
    st.subheader("Top 10 Highest Risk Events")
    top_events = df_with_risk.nlargest(10, 'risk_score')
    
    display_cols = ['user_id', 'event_type', 'risk_score', 'risk_level', 'api_requests', 'data_mb']
    display_cols = [col for col in display_cols if col in top_events.columns]
    
    st.dataframe(top_events[display_cols], width='stretch')
    
    return df_with_risk


def display_validation_summary(df: pd.DataFrame):
    """Display comprehensive validation summary."""
    st.subheader("🔍 Validation Summary")
    
    summary = get_validation_summary(df)
    
    # Create validation status
    status_container = st.container()
    
    if summary['overall_valid']:
        status_container.success("✅ Data validation passed - ready for analysis")
    else:
        status_container.warning("⚠️ Data has validation issues - review before processing")
    
    # Display summary table
    summary_data = []
    
    summary_data.append({
        'Metric': 'Total Records',
        'Value': str(summary['total_records'])
    })
    
    summary_data.append({
        'Metric': 'Required Columns',
        'Value': 'All Present' if summary['required_columns_present'] else 'Missing'
    })
    
    summary_data.append({
        'Metric': 'Invalid Timestamps',
        'Value': str(summary['invalid_timestamp_count'])
    })
    
    summary_data.append({
        'Metric': 'Invalid IP Addresses',
        'Value': str(summary['invalid_ip_count'])
    })
    
    summary_data.append({
        'Metric': 'Duplicate Records',
        'Value': str(summary['total_duplicate_records'])
    })
    
    summary_data.append({
        'Metric': 'Missing Value Columns',
        'Value': str(len(summary['missing_value_columns']))
    })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, width='stretch')


def display_search_and_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display search and filter controls.
    
    Args:
        df: DataFrame to filter
        
    Returns:
        Filtered DataFrame
    """
    st.subheader("🔍 Search and Filter")
    
    # Create expandable filter section
    with st.expander("Advanced Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Global search
            search_query = st.text_input(
                "Global Search",
                placeholder="Search user_id, role, IP, event_type...",
                key="global_search"
            )
        
        with col2:
            # Role filter
            available_roles = ['All'] + list(df['role'].unique()) if 'role' in df.columns else ['All']
            selected_role = st.selectbox(
                "Role",
                available_roles,
                key="role_filter"
            )
        
        with col3:
            # Risk level filter
            available_risk = ['All'] + list(df['risk_level'].unique()) if 'risk_level' in df.columns else ['All']
            selected_risk = st.selectbox(
                "Risk Level",
                available_risk,
                key="risk_filter"
            )
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Event type filter
            if 'event_type' in df.columns:
                available_events = ['All'] + list(df['event_type'].unique())
                selected_event = st.selectbox(
                    "Event Type",
                    available_events,
                    key="event_filter"
                )
            else:
                selected_event = 'All'
        
        with col5:
            # After hours filter
            if 'after_hours' in df.columns:
                after_hours_options = ['All', 'Business Hours', 'After Hours']
                selected_after_hours = st.selectbox(
                    "Time of Day",
                    after_hours_options,
                    key="after_hours_filter"
                )
            else:
                selected_after_hours = 'All'
        
        with col6:
            # Anomaly filter
            if 'is_anomaly' in df.columns:
                anomaly_options = ['All', 'Normal', 'Anomaly']
                selected_anomaly = st.selectbox(
                    "ML Anomaly",
                    anomaly_options,
                    key="anomaly_filter"
                )
            else:
                selected_anomaly = 'All'
    
    # Apply filters
    df_filtered = df.copy()
    
    # Apply global search
    if search_query:
        df_filtered = search_events(df_filtered, search_query)
    
    # Apply role filter
    if selected_role != 'All':
        df_filtered = filter_events(df_filtered, role=selected_role)
    
    # Apply risk level filter
    if selected_risk != 'All':
        df_filtered = filter_events(df_filtered, risk_level=selected_risk)
    
    # Apply event type filter
    if selected_event != 'All':
        df_filtered = filter_events(df_filtered, event_type=selected_event)
    
    # Apply after hours filter
    if selected_after_hours == 'Business Hours':
        df_filtered = filter_events(df_filtered, after_hours=0)
    elif selected_after_hours == 'After Hours':
        df_filtered = filter_events(df_filtered, after_hours=1)
    
    # Apply anomaly filter
    if selected_anomaly == 'Normal':
        df_filtered = filter_events(df_filtered, is_anomaly=False)
    elif selected_anomaly == 'Anomaly':
        df_filtered = filter_events(df_filtered, is_anomaly=True)
    
    return df_filtered


def display_security_table(df: pd.DataFrame):
    """
    Display security events table.
    
    Args:
        df: DataFrame to display
    """
    st.subheader("📜 Security Events Table")
    
    if df.empty:
        st.warning("No events match the current filters")
        return
    
    # Display total count
    st.write(f"Showing {len(df)} events")
    
    # Create display dataframe
    display_cols = ['timestamp', 'user_id', 'role', 'source_ip', 'event_type', 
                    'risk_level', 'api_requests', 'data_mb']
    display_cols = [col for col in display_cols if col in df.columns]
    
    df_display = df[display_cols].copy()
    
    # Format timestamp
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = df_display['timestamp'].astype(str).str[:19]
    
    # Format risk level with color indicator
    if 'risk_level' in df_display.columns:
        df_display['Risk Indicator'] = df_display['risk_level'].apply(
            lambda x: '🟢' if x == 'Low' else ('🟡' if x == 'Medium' else ('🔴' if x in ['High', 'Critical'] else '⚪'))
        )
    
    st.dataframe(df_display, width='stretch')


def display_user_investigation(df: pd.DataFrame):
    """
    Display user activity investigation panel.
    
    Args:
        df: DataFrame with user data
    """
    st.subheader("👤 User Activity Investigation")
    
    if 'user_id' not in df.columns:
        st.warning("No user_id column available")
        return
    
    # User selector
    users = ['Select a user...'] + list(df['user_id'].unique())
    selected_user = st.selectbox(
        "Select User to Investigate",
        users,
        key="user_selector"
    )
    
    if selected_user and selected_user != 'Select a user...':
        # Get user activity
        user_activity = get_user_activity(df, selected_user)
        
        if 'error' in user_activity:
            st.error(user_activity['error'])
            return
        
        # Display user info cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### User Information")
            st.write(f"**User ID:** {user_activity['user_id']}")
            st.write(f"**Role:** {user_activity['role']}")
            st.write(f"**First Activity:** {user_activity['first_activity'][:10]}")
        
        with col2:
            st.markdown("### Activity Summary")
            st.write(f"**Total Events:** {user_activity['total_events']}")
            st.write(f"**API Requests:** {user_activity['total_api_requests']:,}")
            st.write(f"**Data Accessed:** {user_activity['total_data_mb']:.2f} MB")
        
        with col3:
            st.markdown("### Security Indicators")
            st.write(f"**Unique IPs:** {user_activity['unique_source_ips']}")
            st.write(f"**After Hours Events:** {user_activity['after_hours_count']}")
            st.write(f"**ML Anomalies:** {user_activity['anomaly_count']}")
        
        # Risk distribution
        if user_activity['risk_distribution']:
            st.subheader("Risk Distribution")
            risk_df = pd.DataFrame({
                'Risk Level': list(user_activity['risk_distribution'].keys()),
                'Count': list(user_activity['risk_distribution'].values())
            })
            st.dataframe(risk_df, width='stretch')
        
        # Event types
        if user_activity['event_types']:
            st.subheader("Event Type Distribution")
            event_df = pd.DataFrame({
                'Event Type': list(user_activity['event_types'].keys()),
                'Count': list(user_activity['event_types'].values())
            })
            st.dataframe(event_df, width='stretch')
        
        # Download user investigation
        csv_data, filename, _ = get_download_button_csv(
            export_user_investigation(user_activity),
            f"user_{selected_user}_investigation.csv",
            "Download User Investigation"
        )
        st.download_button(
            label="📥 Download User Investigation",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_user_{selected_user}"
        )


def display_ip_investigation(df: pd.DataFrame):
    """
    Display IP address investigation panel.
    
    Args:
        df: DataFrame with IP data
    """
    st.subheader("🌐 IP Address Investigation")
    
    if 'source_ip' not in df.columns:
        st.warning("No source_ip column available")
        return
    
    # IP selector
    ips = ['Select an IP...'] + list(df['source_ip'].unique())
    selected_ip = st.selectbox(
        "Select IP to Investigate",
        ips,
        key="ip_selector"
    )
    
    if selected_ip and selected_ip != 'Select an IP...':
        # Get IP activity
        ip_activity = get_ip_activity(df, selected_ip)
        
        if 'error' in ip_activity:
            st.error(ip_activity['error'])
            return
        
        # Display IP info cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### IP Information")
            st.write(f"**IP Address:** {ip_activity['ip_address']}")
            st.write(f"**Event Count:** {ip_activity['event_count']}")
            st.write(f"**First Observed:** {ip_activity['first_observed'][:10]}")
            st.write(f"**Last Observed:** {ip_activity['last_observed'][:10]}")
        
        with col2:
            st.markdown("### Associated Accounts")
            users = ip_activity.get('associated_users', [])
            st.write(f"**Accounts:** {', '.join(users)}" if users else "**Accounts:** None")
        
        # Risk classifications
        if ip_activity['risk_classifications']:
            st.subheader("Risk Classifications")
            risk_df = pd.DataFrame({
                'Risk Level': list(ip_activity['risk_classifications'].keys()),
                'Count': list(ip_activity['risk_classifications'].values())
            })
            st.dataframe(risk_df, width='stretch')
        
        # Event types
        if ip_activity['event_types']:
            st.subheader("Event Types")
            event_df = pd.DataFrame({
                'Event Type': list(ip_activity['event_types'].keys()),
                'Count': list(ip_activity['event_types'].values())
            })
            st.dataframe(event_df, width='stretch')
        
        # Unusual indicators
        if ip_activity['unusual_indicators']:
            st.subheader("⚠️ Unusual Activity Indicators")
            for indicator in ip_activity['unusual_indicators']:
                st.warning(f"• {indicator}")
        
        # Download IP investigation
        csv_data, filename, _ = get_download_button_csv(
            export_ip_investigation(ip_activity),
            f"ip_{selected_ip}_investigation.csv",
            "Download IP Investigation"
        )
        st.download_button(
            label="📥 Download IP Investigation",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"download_ip_{selected_ip}"
        )


def display_timeline_visualization(df: pd.DataFrame):
    """
    Display interactive event timeline.
    
    Args:
        df: DataFrame with events
    """
    st.subheader("⏰ Event Timeline")
    
    timeline = generate_timeline(df)
    
    if timeline.empty:
        st.warning("No timeline data available")
        return
    
    # Display recent events
    st.write("### Recent Events (Sorted by Timestamp)")
    
    # Create timeline display
    timeline_cols = ['timestamp', 'user_id', 'source_ip', 'event_type', 
                     'risk_level', 'is_anomaly']
    timeline_cols = [col for col in timeline_cols if col in timeline.columns]
    
    df_timeline = timeline[timeline_cols].head(100).copy()
    
    if 'timestamp' in df_timeline.columns:
        df_timeline['timestamp'] = df_timeline['timestamp'].astype(str)
    
    st.dataframe(df_timeline, width='stretch')
    
    # Show count
    st.write(f"Showing up to 100 most recent events")
    
    # Download timeline (export_timeline_events already returns CSV bytes)
    csv_data = export_timeline_events(timeline)
    st.download_button(
        label="📥 Download Timeline",
        data=csv_data,
        file_name="timeline_events.csv",
        mime="text/csv",
        key="download_timeline"
    )


def display_csv_exports(df: pd.DataFrame, original_df: pd.DataFrame = None):
    """
    Display CSV export functionality.
    
    Args:
        df: Processed DataFrame
        original_df: Original uploaded DataFrame (if available)
    """
    st.subheader("💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    # Export original data
    with col1:
        if original_df is not None:
            csv_data, filename, _ = get_download_button_csv(
                original_df,
                "original_upload.csv",
                "Download Original"
            )
            st.download_button(
                label="Download Original CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                key="download_original"
            )
    
    # Export cleaned data
    with col2:
        csv_data, filename, _ = get_download_button_csv(
            df,
            "cleaned_data.csv",
            "Download Cleaned CSV"
        )
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key="download_cleaned"
        )
    
    # Export analysis summary
    with col3:
        summary_stats = get_dataset_statistics(df)
        summary_data = export_analysis_summary(df, summary_stats)
        st.download_button(
            label="Download Analysis Summary",
            data=summary_data,
            file_name="analysis_summary.csv",
            mime="text/csv",
            key="download_summary"
        )


def display_log_analysis():
    """
    Main function to display the complete log analysis page.
    """
    st.subheader("Security Log Analysis")
    
    # Initialize session state
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
        st.session_state.df_processed = None
    
    # File upload
    uploaded_file = display_upload_section()
    
    if uploaded_file:
        # Validate file
        is_valid, message = validate_upload(uploaded_file)
        
        if not is_valid:
            st.error(f"Upload failed: {message}")
            return
        
        # Process file
        with st.spinner("Processing uploaded file..."):
            try:
                # Read uploaded file (preserve original)
                df_original = process_uploaded_file(uploaded_file)
                
                # Get validation summary
                validation_summary = get_validation_summary(df_original)
                
                # Show validation summary before processing
                display_validation_summary(df_original)
                
                # Ask user if they want to proceed
                if not validation_summary['overall_valid']:
                    st.warning("⚠️ Data has validation issues. Do you want to proceed?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Proceed with Analysis", key="proceed_button"):
                            proceed = True
                        else:
                            proceed = False
                    
                    with col2:
                        if st.button("Cancel", key="cancel_button"):
                            proceed = False
                
                else:
                    proceed = True
                    st.success("✅ Data validation passed!")
                
                if proceed:
                    # Analyze the data
                    df_processed = analyze_uploaded_data(df_original)
                    
                    # Store in session state
                    st.session_state.df_original = df_original.copy()
                    st.session_state.df_processed = df_processed.copy()
                    
                    st.success(f"Processed {len(df_processed)} records successfully!")
                    
                    # Display data quality
                    display_data_quality(df_processed)
                    
                    # Display statistics
                    display_statistics(df_processed)
                    
                    # Display risk analysis
                    df_with_risk = display_risk_analysis(df_processed)
                    
                    # Add search and filters
                    df_filtered = display_search_and_filters(df_with_risk)
                    
                    # Display filtered data
                    display_security_table(df_filtered)
                    
                    # Display investigation panels
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        display_user_investigation(df_with_risk)
                    
                    with col2:
                        display_ip_investigation(df_with_risk)
                    
                    # Display timeline
                    display_timeline_visualization(df_with_risk)
                    
                    # Display export options
                    display_csv_exports(df_processed, df_original)

                    # F-2 bridge: feed the uploaded dataset into the
                    # shared analysis state so Correlation and
                    # Investigation pages can consume it.
                    if st.button("Use uploaded data for Correlation & "
                                 "Investigation", key="f2_bridge"):
                        try:
                            from modules.correlation_dashboard import (
                                analyze_correlation)
                            from modules.analysis_state import ANALYSIS_KEY
                            with st.spinner("Running correlation pipeline "
                                            "on uploaded data…"):
                                result = analyze_correlation(df_processed)
                            result['scenario_key'] = 'uploaded'
                            result['scenario_label'] = \
                                f"Uploaded file ({uploaded_file.name})"
                            from modules.analysis_state import get_manager
                            get_manager().build_incidents(
                                result.get('correlations'))
                            st.session_state[ANALYSIS_KEY] = result
                            st.success("Uploaded data is now the active "
                                       "analysis — open Correlation or "
                                       "Investigation.")
                            st.rerun()
                        except Exception as exc:
                            from utils.errors import safe_error_message
                            st.error(safe_error_message(
                                exc, "Correlation analysis failed"))

                    # Re-run to show dashboard with new data
                    st.rerun()
                    
            except Exception as e:
                logger.exception("Log analysis upload/processing failed")
                st.error(f"Error processing file: {str(e)}")
                st.caption("Full details have been logged server-side; "
                           "check the application logs for the traceback.")


def analyze_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze uploaded data with feature engineering and ML.
    
    Args:
        df: Raw uploaded DataFrame
        
    Returns:
        Analyzed DataFrame
    """
    # Strict cleaning stage (Milestone 1.7): row-level validation/cleaning
    # with retained/dropped/corrected audit summary. The raw df_original is
    # preserved in session state for auditability.
    df, clean_summary = clean_and_summarize(df)
    if len(df) == 0:
        raise ValueError(
            "All uploaded rows failed validation/cleaning "
            f"({clean_summary['dropped_rows']} dropped). "
            "Check timestamps and user_id fields."
        )
    st.session_state.upload_clean_summary = clean_summary

    # AUD-003: uploaded data now runs through the SAME v2_0 detection
    # pipeline as scenario analysis — ML + rules + behavioral + session +
    # sequence channels with v2 calibration lookup. Output is a superset
    # of the legacy profile (adds session_*/sequence_*/operating_mode/
    # missing_channels columns); risk_level/combined_score semantics now
    # match the scenario pipeline exactly.
    from modules.session_intelligence import analyze_session_intelligence
    df_analyzed = analyze_session_intelligence(df)

    # Surface calibration/artifact status for the UI (Milestones 1.9/2.4).
    if 'risk_calibration_status' in df_analyzed.columns:
        st.session_state.risk_calibration_status = \
            df_analyzed['risk_calibration_status'].mode().iloc[0]
    if 'detection_calibration_status' in df_analyzed.columns:
        st.session_state.detection_calibration_status = \
            df_analyzed['detection_calibration_status'].mode().iloc[0]
    if 'threshold_version' in df_analyzed.columns and \
            df_analyzed['threshold_version'].notna().any():
        st.session_state.risk_threshold_version = \
            df_analyzed['threshold_version'].dropna().iloc[0]

    return df_analyzed