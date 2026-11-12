"""Event monitoring page — searchable, filterable view of scored events."""

import pandas as pd
import streamlit as st

from modules.analysis_state import analysis_or_prompt

SCORE_COLUMNS = [
    'anomaly_score_normalized', 'anomaly_score',
    'behavioral_deviation_score', 'sequence_deviation',
    'combined_score',
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    c1, c2 = st.columns(2)
    with c1:
        search = st.text_input(
            "Search", placeholder="user, session, IP, event type…",
            key='ev_search')
    with c2:
        min_score = st.slider("Minimum combined score", 0.0, 1.0, 0.0,
                              0.05, key='ev_min_score')

    if 'combined_score' in out.columns:
        out = out[_num(out['combined_score']) >= min_score]

    f1, f2, f3 = st.columns(3)
    with f1:
        levels = sorted(out['risk_level'].dropna().unique().tolist()) \
            if 'risk_level' in out.columns else []
        sel_levels = st.multiselect("Risk level", options=levels,
                                    default=levels, key='ev_risk')
        if sel_levels and 'risk_level' in out.columns:
            out = out[out['risk_level'].isin(sel_levels)]
    with f2:
        users = sorted(out['user_id'].dropna().unique().tolist()) \
            if 'user_id' in out.columns else []
        sel_users = st.multiselect("User", options=users, key='ev_user')
        if sel_users and 'user_id' in out.columns:
            out = out[out['user_id'].isin(sel_users)]
    with f3:
        etypes = sorted(out['event_type'].dropna().unique().tolist()) \
            if 'event_type' in out.columns else []
        sel_types = st.multiselect("Event type", options=etypes,
                                   key='ev_type')
        if sel_types and 'event_type' in out.columns:
            out = out[out['event_type'].isin(sel_types)]

    if 'timestamp' in out.columns:
        d1, d2 = st.columns(2)
        ts = pd.to_datetime(out['timestamp'], errors='coerce')
        valid = ts.dropna()
        if len(valid):
            with d1:
                start = st.date_input("From", value=valid.min().date(),
                                      key='ev_start')
            with d2:
                end = st.date_input("To", value=valid.max().date(),
                                    key='ev_end')
            mask = (ts.dt.date >= start) & (ts.dt.date <= end)
            out = out[mask.fillna(False)]

    if search:
        fields = [c for c in ['user_id', 'session_id', 'source_ip',
                              'event_type', 'detection_id']
                  if c in out.columns]
        mask = pd.Series(False, index=out.index)
        for col in fields:
            mask |= out[col].astype(str).str.contains(
                search, case=False, na=False, regex=False)
        out = out[mask]

    return out


def _event_detail(row: pd.Series) -> None:
    cols = [c for c in ['timestamp', 'user_id', 'session_id', 'source_ip',
                        'event_type', 'role'] if c in row.index]
    st.dataframe(pd.DataFrame([row[cols]]), hide_index=True,
                 width='stretch')
    score_cols = [c for c in SCORE_COLUMNS if c in row.index]
    if score_cols:
        st.markdown("**Scores**")
        st.dataframe(
            pd.DataFrame([{
                'metric': c,
                'value': (None if pd.isna(row[c]) else round(float(row[c]), 4))
            } for c in score_cols]),
            hide_index=True, width='stretch')
    for label, key in [('Detection explanation', 'detection_explanation'),
                       ('Operating mode', 'operating_mode'),
                       ('Missing channels', 'missing_channels'),
                       ('Triggered rules', 'triggered_rules'),
                       ('Recommended actions', 'recommended_actions')]:
        if key in row.index and row[key] not in (None, '', [], {}):
            st.markdown(f"**{label}:**")
            st.write(row[key])
    if 'structured_evidence' in row.index and row['structured_evidence']:
        st.markdown("**Structured evidence:**")
        try:
            st.json(row['structured_evidence'])
        except Exception:
            st.code(str(row['structured_evidence'])[:4000])


st.subheader("Event Monitoring")
st.caption("All analyzed events with ML, behavioral, sequence and "
           "combined risk scores. Scores are calibrated anomaly "
           "scores, not probabilities.")

result = analysis_or_prompt()
if result is not None:
    df = result.get('scored_events')
    if df is None or len(df) == 0:
        st.info("No scored events available.")
    else:
        filtered = _apply_filters(df)
        st.caption(f"Showing {len(filtered):,} of {len(df):,} events")

        show_cols = [c for c in [
            'timestamp', 'user_id', 'session_id', 'source_ip', 'event_type',
            'is_anomaly', 'anomaly_score_normalized',
            'behavioral_deviation_score', 'sequence_deviation',
            'combined_score', 'risk_level', 'operating_mode',
            'detection_id'] if c in filtered.columns]
        sort_col = 'combined_score' if 'combined_score' in filtered.columns \
            else show_cols[0]
        view = filtered.sort_values(sort_col, ascending=False,
                                    na_position='last')

        # ---- Pagination ---------------------------------------------------
        total = len(view)
        page_sizes = [100, 250, 500, 1000, 2500]
        pc1, pc2, pc3 = st.columns([1, 1, 2])
        with pc1:
            page_size = st.selectbox(
                "Rows per page", page_sizes, index=2, key='ev_page_size',
                disabled=total == 0)
        n_pages = max(1, (total + page_size - 1) // page_size)
        with pc2:
            page = st.number_input(
                "Page", min_value=1, max_value=n_pages, value=1,
                step=1, key='ev_page', disabled=total == 0)
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        with pc3:
            st.caption(f"Viewing rows {start + 1:,}–{end:,} of "
                       f"{total:,} (page {page}/{n_pages})")
        page_view = view[show_cols].iloc[start:end]
        st.dataframe(page_view, hide_index=True,
                     width='stretch', alt='Analyzed security events')

        from modules.export import export_csv
        st.download_button(
            "Download filtered events (CSV)",
            export_csv(filtered[show_cols]),
            "events.csv", "text/csv", width='stretch',
            key='ev_download')

        st.subheader("Event detail")
        if len(filtered) == 0:
            st.info("No events match the current filters.")
        else:
            detail_cols = [c for c in ['timestamp', 'user_id',
                                       'session_id', 'event_type',
                                       'combined_score', 'risk_level']
                           if c in filtered.columns]
            # Limit the selectbox to the highest-scoring 500 events, but
            # provide a text filter so any event can be located by id.
            options = filtered.sort_values(sort_col, ascending=False,
                                           na_position='last').head(500)
            labels = {
                idx: " | ".join(str(options.loc[idx, c])
                               for c in detail_cols[:4])
                for idx in options.index
            }
            pick = st.selectbox("Select event",
                                options=list(options.index),
                                format_func=lambda i: labels.get(i, str(i)),
                                key='ev_detail_pick')
            with st.expander("Event evidence", expanded=True):
                _event_detail(filtered.loc[pick])
