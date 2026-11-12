"""
Advanced Threat Detection Dashboard Module

This module provides the Streamlit interface for the Advanced Threat Detection page
(Milestone 3), displaying combined ML, rule-based, and behavioral analysis results.

ARCHITECTURE:
- Dashboard Functions: Display threat detection metrics and findings
- Visualization: Charts, tables, and summaries
- User Interaction: Filtering, investigation, and export options
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go


def display_threat_detection_header():
    """Display the main header for threat detection page."""
    st.header("🛡️ Advanced Threat Detection")
    st.markdown("Real-time threat detection combining ML, behavioral analysis, and security rules.")


def display_detection_metrics(df: pd.DataFrame):
    """Display key threat detection metrics."""
    if df is None or len(df) == 0:
        st.warning("No data available. Please load security logs.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.metric("Total Events Analyzed", f"{total:,}")
    
    with col2:
        ml_anomalies = int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0
        st.metric("ML-Detected Anomalies", f"{ml_anomalies:,}")
    
    with col3:
        rule_score_col = 'rule_detection_score' if 'rule_detection_score' in df.columns else (
            'rule_score' if 'rule_score' in df.columns else None
        )
        rule_events = (
            df[df[rule_score_col] > 0.3].shape[0]
            if rule_score_col else 0
        )
        st.metric("Rule-Triggered Events", f"{rule_events:,}")
    
    with col4:
        suspicious_events = df[df['combined_score'] >= 0.5].shape[0] if 'combined_score' in df.columns else 0
        st.metric("Combined Suspicious Events", f"{suspicious_events:,}")


def display_risk_distribution(df: pd.DataFrame):
    """Display risk level distribution."""
    if df is None or len(df) == 0:
        return
    
    st.subheader("Risk Distribution")
    
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            marker_colors=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Risk classification not available. Run detection engine first.")


def display_top_suspicious_users(df: pd.DataFrame, n: int = 10):
    """Display top suspicious users."""
    if df is None or len(df) == 0:
        return
    
    st.subheader("Top Suspicious Users")
    
    if 'combined_score' in df.columns:
        # Get users with highest combined scores
        top_users = df.groupby('user_id').agg({
            'combined_score': 'mean',
            'detection_id': 'count'
        }).reset_index()
        
        top_users.columns = ['user_id', 'avg_risk_score', 'event_count']
        top_users = top_users.sort_values('avg_risk_score', ascending=False).head(n)
        
        # Display as table
        st.dataframe(
            top_users.style.format({'avg_risk_score': '{:.3f}'})
            .map(lambda x: 'background-color: #ffe0b2' if x >= 0.7 else '', subset=['avg_risk_score'])
        )
        
        if len(top_users) > 0:
            st.caption("Hover over user IDs to investigate activity.")
    else:
        st.info("Risk scores not available. Run detection engine first.")


def display_detection_timeline(df: pd.DataFrame):
    """Display detection events over time."""
    if df is None or len(df) == 0:
        return
    
    st.subheader("Detection Timeline")
    
    if 'timestamp' in df.columns:
        # Check if required columns exist
        has_detection_id = 'detection_id' in df.columns
        has_combined_score = 'combined_score' in df.columns
        
        if not has_detection_id and not has_combined_score:
            st.info("Detection data not available. Run detection engine first.")
            return
        
        # Create time-based aggregation
        df_timeline = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_timeline['timestamp']):
            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'], errors='coerce')
        
        # Build aggregation dictionary
        agg_dict = {}
        if has_detection_id:
            agg_dict['detection_id'] = 'count'
        if has_combined_score:
            agg_dict['combined_score'] = 'mean'
        
        # Daily aggregation
        daily_counts = df_timeline.groupby(pd.Grouper(key='timestamp', freq='D')).agg(agg_dict).reset_index()
        
        # Rename columns appropriately
        if has_detection_id:
            daily_counts.columns = ['date', 'detection_count'] + list(daily_counts.columns[2:])
        else:
            daily_counts.columns = ['date'] + list(daily_counts.columns[1:])
        
        # Create bar chart
        fig = px.bar(
            daily_counts,
            x='date',
            y='detection_count' if has_detection_id else None,
            title='Daily Detection Events',
            labels={'detection_count': 'Events', 'date': 'Date'},
            color='combined_score' if has_combined_score else None,
            color_continuous_scale='RdYlGn_r'
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("Timestamp data not available.")


def display_triggered_rules(df: pd.DataFrame):
    """Display most frequently triggered detection rules."""
    if df is None or len(df) == 0:
        return
    
    st.subheader("Most Triggered Detection Rules")
    
    if 'triggered_rules' in df.columns:
        # Extract all triggered rules
        all_rules = []
        for _, row in df.iterrows():
            rules = row.get('triggered_rules', [])
            for rule in rules:
                all_rules.append(rule)
        
        if all_rules:
            rules_df = pd.DataFrame(all_rules)
            name_col = next(
                (c for c in ('rule_name', 'name', 'rule_id')
                 if c in rules_df.columns), None)
            if name_col is None:
                st.info("Triggered rules have no recognizable name field.")
                return
            rule_counts = rules_df[name_col].astype(str).value_counts()
            
            # Create horizontal bar chart
            fig = px.bar(
                x=rule_counts.values,
                y=rule_counts.index,
                orientation='h',
                title='Detection Rule Frequency',
                labels={'x': 'Count', 'y': 'Rule Name'}
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No detection rules have been triggered.")
    else:
        st.info("Rule detection results not available.")


def display_detection_details(df: pd.DataFrame, n: int = 100):
    """Display detailed detection findings."""
    if df is None or len(df) == 0:
        return
    
    st.subheader("Detection Findings")
    
    # Filter to suspicious events
    if 'combined_score' in df.columns:
        suspicious_df = df[df['combined_score'] >= 0.5].copy()
    else:
        suspicious_df = df.head(n)
    
    if len(suspicious_df) == 0:
        st.info("No suspicious events detected based on current thresholds.")
        return
    
    # Display in expandable format
    for idx, (i, row) in enumerate(suspicious_df.iterrows()):
        with st.expander(f"**{row.get('detection_id', f'DET-{i}')}: {row.get('user_id', 'Unknown')} - {row.get('risk_level', 'Unknown')}**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Timestamp:** {row.get('timestamp', 'N/A')}")
                st.write(f"**User ID:** {row.get('user_id', 'N/A')}")
                st.write(f"**Role:** {row.get('role', 'N/A')}")
                st.write(f"**Source IP:** {row.get('source_ip', 'N/A')}")
            
            with col2:
                st.write(f"**ML Anomaly:** {'Yes' if row.get('is_anomaly', False) else 'No'}")
                ml_display = row.get('anomaly_score_normalized', row.get('anomaly_score', 0))
                try:
                    ml_display = f"{float(ml_display):.3f}"
                except (TypeError, ValueError):
                    ml_display = 'N/A'
                st.write(f"**ML Score (normalized):** {ml_display}")
                rule_score = row.get('rule_detection_score', row.get('rule_score', 0))
                try:
                    rule_score = f"{float(rule_score):.3f}"
                except (TypeError, ValueError):
                    rule_score = 'N/A'
                st.write(f"**Rule Score:** {rule_score}")
                behavioral_score = row.get('behavioral_deviation_score')
                behavioral_missing = bool(row.get('behavioral_score_missing', False))
                try:
                    behavioral_score = (
                        'N/A' if behavioral_missing or pd.isna(behavioral_score)
                        else f"{float(behavioral_score):.3f}"
                    )
                except (TypeError, ValueError):
                    behavioral_score = 'N/A'
                st.write(f"**Behavioral Deviation:** {behavioral_score}")
                if behavioral_missing:
                    st.write("**Behavioral Status:** result unavailable (not scored as normal)")
                st.write(f"**Combined Score:** {row.get('combined_score', 0):.3f}")
            
            st.write(f"**Explanation:** {row.get('detection_explanation', 'N/A')}")
            
            if 'recommended_actions' in row.index:
                actions = row['recommended_actions']
                if actions:
                    st.write("**Recommended Actions:**")
                    for action in actions:
                        st.write(f"- {action}")


def display_filter_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Display filter controls for investigation."""
    if df is None or len(df) == 0:
        return df
    
    st.subheader("Filter Findings")
    
    # Create filters
    filtered_df = df.copy()
    
    # Risk level filter
    if 'risk_level' in filtered_df.columns:
        risk_levels = st.multiselect(
            "Risk Levels",
            options=filtered_df['risk_level'].unique().tolist(),
            default=filtered_df['risk_level'].unique().tolist()
        )
        if risk_levels:
            filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_levels)]
    
    # Score threshold filter
    if 'combined_score' in filtered_df.columns:
        score_threshold = st.slider(
            "Minimum Combined Score",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        filtered_df = filtered_df[filtered_df['combined_score'] >= score_threshold]
    
    # User filter
    if 'user_id' in filtered_df.columns:
        users = filtered_df['user_id'].unique().tolist()
        selected_users = st.multiselect(
            "Users",
            options=users,
            default=[]
        )
        if selected_users:
            filtered_df = filtered_df[filtered_df['user_id'].isin(selected_users)]
    
    return filtered_df


def display_detection_summary_stats(df: pd.DataFrame) -> Dict:
    """
    Display and return summary statistics for detection results.
    
    Args:
        df: DataFrame with detection results
        
    Returns:
        Dictionary with summary statistics
    """
    if df is None or len(df) == 0:
        return {}
    
    summary = {
        'total_events': len(df),
        'ml_anomalies': int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
        'rule_triggers': (
            int((df['rules_triggered'] > 0).sum())
            if 'rules_triggered' in df.columns
            else int((df['rule_score'] > 0).sum()) if 'rule_score' in df.columns
            else int((df['rule_detection_score'] > 0).sum()) if 'rule_detection_score' in df.columns
            else 0
        ),
        'suspicious_events': int((df['combined_score'] >= 0.5).sum()) if 'combined_score' in df.columns else 0,
    }
    
    # Display as cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", summary['total_events'])
    
    with col2:
        st.metric("ML Anomalies", summary['ml_anomalies'])
    
    with col3:
        st.metric("Rule Triggers", summary['rule_triggers'])
    
    with col4:
        st.metric("Suspicious Events", summary['suspicious_events'])
    
    return summary


def display_milestone_3_dashboard(df: pd.DataFrame):
    """
    Main function to display the complete Milestone 3 threat detection dashboard.
    
    Args:
        df: Security log DataFrame with all detection results
    """
    # Display header
    display_threat_detection_header()
    
    # Display summary metrics
    st.markdown("### 📊 Overview Metrics")
    display_detection_metrics(df)
    
    st.markdown("---")
    
    # Display risk distribution
    st.markdown("### 🎯 Risk Analysis")
    display_risk_distribution(df)
    
    # Display top suspicious users
    st.markdown("---")
    display_top_suspicious_users(df)
    
    # Display timeline
    st.markdown("---")
    display_detection_timeline(df)
    
    # Display triggered rules
    st.markdown("---")
    display_triggered_rules(df)
    
    # Display findings
    st.markdown("---")
    st.markdown("### 🔍 Detection Findings")
    
    # Apply filters
    filtered_df = display_filter_controls(df)
    
    # Display details
    if len(filtered_df) > 0:
        st.info(f"Showing {len(filtered_df)} filtered findings")
        display_detection_details(filtered_df)
    else:
        st.info("No findings match current filters.")


def display_detection_explanation(detection: Dict):
    """
    Display detailed explanation for a single detection.
    
    Args:
        detection: Detection dictionary with all relevant fields
    """
    st.subheader("Detection Explanation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detection ID:**", detection.get('detection_id', 'N/A'))
        st.write("**User ID:**", detection.get('user_id', 'N/A'))
        st.write("**Timestamp:**", detection.get('timestamp', 'N/A'))
        st.write("**Risk Level:**", detection.get('risk_level', 'N/A'))
    
    with col2:
        st.write("**ML Anomaly:**", detection.get('ml_anomaly', 'N/A'))
        st.write("**ML Score:**", f"{detection.get('ml_anomaly_score', 0):.3f}")
        st.write("**Combined Score:**", f"{detection.get('combined_score', 0):.3f}")
    
    st.write("**Explanation:**")
    st.write(detection.get('explanation', 'N/A'))
    
    if 'recommended_actions' in detection:
        st.write("**Recommended Actions:**")
        for action in detection['recommended_actions']:
            st.write(f"- {action}")


def get_detection_history() -> list:
    """
    Get detection history from session state.
    
    Returns:
        List of detection results
    """
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    return st.session_state.detection_history


def save_detection_result(detection_result: Dict):
    """
    Save detection result to session history.
    
    Args:
        detection_result: Detection results dictionary
    """
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    st.session_state.detection_history.append(detection_result)
