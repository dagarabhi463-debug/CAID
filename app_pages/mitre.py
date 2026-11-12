"""MITRE ATT&CK dashboard — framework data, coverage, and mappings."""

import pandas as pd
import streamlit as st

from core.mitre_attack import MitreAttackStore


@st.cache_resource
def get_store() -> MitreAttackStore:
    return MitreAttackStore()


store = get_store()
st.subheader("MITRE ATT&CK")
st.caption(
    "Mappings link CAID detection rules, correlation findings, and event "
    "evidence to ATT&CK Enterprise techniques. 'Confirmed' requires "
    "explicit evidence per CAID's curated mapping rules; 'potential' "
    "mappings need analyst review and are never treated as confirmed "
    "attacker behavior.")

# ---- Load / status -------------------------------------------------------
if not store.available:
    st.info("MITRE ATT&CK data is not loaded yet. It downloads the "
            "official STIX bundle once and caches it locally "
            "(data/mitre_attack/).")
    if st.button("Download MITRE ATT&CK data", key='mitre_load'):
        with st.spinner("Downloading ATT&CK STIX bundle…"):
            ok = store.ensure_data(allow_download=True)
        if ok:
            st.success("MITRE ATT&CK data loaded.")
            st.rerun()
        else:
            st.error(f"Could not load MITRE data: {store.load_error}")
    st.stop()

cov = store.coverage()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Techniques in framework", f"{cov['total_techniques']:,}")
c2.metric("Confirmed mappings", cov['confirmed_mappings'])
c3.metric("Potential mappings", cov['potential_mappings'])
c4.metric("Mapped CAID rules", len(cov['mapped_rules']))
c5.metric("Unmapped rules", len(cov['unmapped_rules']))
st.caption(f"Framework: {cov['framework_version']} · "
           f"retrieved {str(cov['retrieved_at'])[:10]}")

# ---- Coverage by tactic --------------------------------------------------
st.subheader("Technique coverage by tactic")
if cov['techniques_by_tactic']:
    tac_df = pd.DataFrame(
        [{'tactic': k, 'mapped_techniques': v}
         for k, v in sorted(cov['techniques_by_tactic'].items())])
    st.bar_chart(tac_df.set_index('tactic'), height=280,
                 alt='Mapped ATT&CK techniques per tactic')
else:
    st.info("No technique mappings recorded yet.")

# ---- Technique browser ---------------------------------------------------
st.subheader("Techniques")
tactics = sorted({t for tech in store.list_techniques() for t in tech.tactics})
sel_tactic = st.selectbox("Filter by tactic", options=['All'] + tactics,
                          key='mitre_tactic')
techs = store.list_techniques(None if sel_tactic == 'All' else sel_tactic)
labels = {t.technique_id: f"{t.technique_id} — {t.name}"
          + (" (sub-technique)" if t.is_subtechnique else "")
          for t in techs}
sel_tid = st.selectbox("Select technique", options=list(labels.keys()),
                       format_func=lambda k: labels[k], key='mitre_tech')
tech = store.get_technique(sel_tid)

if tech:
    st.markdown(f"### {tech.technique_id} — {tech.name}")
    st.caption("Tactics: " + ", ".join(tech.tactics)
               + (" · sub-technique" if tech.is_subtechnique else ""))
    if tech.description:
        with st.expander("Description"):
            st.write(tech.description)
    if st.button("Open on MITRE ATT&CK site", key='mitre_link'):
        import webbrowser
        webbrowser.open(store.technique_url(tech.technique_id))

    # CAID mappings referencing this technique
    from core.mitre_attack import CAID_MITRE_MAP
    refs = [r for r in CAID_MITRE_MAP if r['technique_id'] == tech.technique_id]
    st.markdown("**CAID detection mappings:**")
    if refs:
        st.dataframe(pd.DataFrame([{
            'caid_source': r['caid_source'],
            'confidence': r['state'],
            'rationale': r['rationale'],
            'mapping_source': r.get('mapping_source', ''),
        } for r in refs]), hide_index=True, width='stretch',
            alt='CAID mappings for technique')
    else:
        st.caption("No CAID rule or finding type maps to this technique.")

# ---- Live mappings from current analysis ---------------------------------
from modules.analysis_state import get_analysis  # noqa: E402

result = get_analysis()
if result is not None:
    st.subheader("Mappings in current analysis")
    corr_df = result.get('correlations')
    findings = corr_df.to_dict('records') \
        if corr_df is not None else []
    mapped_rows = []
    for f in findings:
        for m in store.map_finding(f):
            mapped_rows.append({
                'technique_id': m['technique_id'],
                'technique_name': m['technique_name'],
                'tactic': m['tactic'],
                'caid_source': m['caid_source'],
                'confidence': m['confidence'],
                'correlation_id': m['evidence'].get('correlation_id'),
                'entities': ', '.join(str(e) for e in
                                      (m['evidence'].get('entities') or [])[:3]),
            })
    if mapped_rows:
        mdf = pd.DataFrame(mapped_rows)
        st.dataframe(mdf, hide_index=True, width='stretch',
                     alt='ATT&CK mappings for current findings')
        csv = mdf.to_csv(index=False)
        st.download_button("Download mappings (CSV)", csv,
                           "mitre_mappings.csv", "text/csv",
                           key='mitre_csv')
    else:
        st.info("No ATT&CK mappings produced for the current analysis.")

    # Incidents
    from modules.analysis_state import get_manager  # noqa: E402
    inc_rows = []
    for inc in get_manager().list_incidents():
        for m in store.map_incident(inc):
            inc_rows.append({
                'incident_id': inc.get('incident_id'),
                'severity': inc.get('severity'),
                'status': inc.get('status'),
                'technique_id': m['technique_id'],
                'technique_name': m['technique_name'],
                'tactic': m['tactic'],
                'confidence': m['confidence'],
            })
    if inc_rows:
        st.markdown("**Incident → technique mappings:**")
        st.dataframe(pd.DataFrame(inc_rows), hide_index=True,
                     width='stretch', alt='Incident ATT&CK mappings')
else:
    st.info("Load and analyze a scenario to see live ATT&CK mappings.")

if cov['unmapped_rules']:
    with st.expander("Unmapped CAID rules (insufficient ATT&CK evidence)"):
        st.write(", ".join(cov['unmapped_rules']))
        st.caption("These rules have no curated ATT&CK mapping; they are "
                   "reported rather than guessed.")
