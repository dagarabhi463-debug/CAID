"""AI Security Analyst — Ollama-grounded explanations and hunting."""

import json
import os
import queue
import threading

import pandas as pd
import streamlit as st

from core.ai_analyst import (
    ask,
    explain_event,
    explain_finding,
    generate_investigation_summary,
    hunt,
    investigate_attack_chain,
    summarize_incident,
)
from core.ollama_client import OllamaClient
from modules.analysis_state import (
    analysis_or_prompt,
    get_manager,
)
from utils.errors import safe_error_message

CHAT_DIR = os.path.join('data', 'ai_chat')


def _chat_path(scenario_key: str) -> str:
    safe = ''.join(c if c.isalnum() or c in '-_' else '_'
                   for c in str(scenario_key or 'default'))
    os.makedirs(CHAT_DIR, exist_ok=True)
    return os.path.join(CHAT_DIR, f'{safe}.json')


def _load_chat(scenario_key: str) -> list:
    path = _chat_path(scenario_key)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_chat(scenario_key: str, messages: list) -> None:
    path = _chat_path(scenario_key)
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(messages, fh, indent=2)
    except Exception:
        pass


@st.cache_resource
def get_client() -> OllamaClient:
    return OllamaClient()


def _model() -> str:
    return st.session_state.get('ollama_model') or None


def _show_details(resp: dict) -> None:
    """Render provenance for an AI response (text shown separately)."""
    if resp.get('degraded') or resp.get('error'):
        hint = (' Try a smaller/faster model on the Settings page.'
                if 'timed out' in str(resp.get('error') or '').lower()
                else '')
        st.caption(f"AI degraded: {resp.get('error') or 'fallback summary'}"
                   f".{hint}")
    meta = []
    if resp.get('elapsed_ms') is not None:
        meta.append(f"{resp['elapsed_ms']} ms")
    usage = resp.get('usage') or {}
    if usage.get('total_tokens'):
        meta.append(f"{usage['total_tokens']} tokens")
    if meta:
        st.caption("Response: " + " · ".join(meta))
    if resp.get('structured_error'):
        st.caption(f"Structured output: {resp['structured_error']}")
    st.caption(f"Model: {resp.get('model', 'n/a')}")
    ev = resp.get('evidence_used') or []
    if ev:
        with st.expander("Evidence used for this answer"):
            for e in ev:
                st.markdown(f"- {e}")


def _render_structured(s: dict) -> None:
    """Render a schema-validated structured analyst response."""
    if s.get('assessment'):
        st.markdown(f"**Assessment:** {s['assessment']}")
    if s.get('observations'):
        st.markdown("**Observations**")
        for o in s['observations']:
            st.markdown(f"- {o}")
    if s.get('evidence_refs'):
        st.caption("Evidence refs: " + ", ".join(s['evidence_refs']))
    if s.get('hypotheses'):
        st.markdown("**Hypotheses (unproven)**")
        for h in s['hypotheses']:
            st.markdown(f"- {h}")
    if s.get('limitations'):
        st.markdown("**Limitations / uncertainty**")
        for l in s['limitations']:
            st.markdown(f"- {l}")
    st.caption(f"Confidence: {s.get('confidence', 'unknown')}")


def _run_analyst(fn, *args, structured: bool = False, **kwargs) -> dict:
    """Run an analyst function; structured mode is non-streaming."""
    if structured:
        with st.spinner("Generating structured analysis…"):
            return fn(*args, structured=True, **kwargs)
    return _ai_stream(fn, *args, **kwargs)


def _show_ai_result(resp: dict, structured: bool) -> None:
    """Show streamed/structured output plus provenance details."""
    s = resp.get('structured') if structured else None
    if s:
        _render_structured(s)
    elif structured and resp.get('text'):
        # Structured request failed validation — show raw model text.
        st.markdown(resp['text'])
    _show_details(resp)


def _ai_stream(fn, *args, **kwargs) -> dict:
    """Run a core analyst function, streaming text via ``st.write_stream``.

    The core function runs on a worker thread so generated chunks can be
    yielded to the UI as they arrive. Returns the final response dict.
    """
    q = queue.Queue()
    box = {}

    def worker():
        try:
            box['resp'] = fn(*args, on_chunk=q.put, **kwargs)
        except Exception as exc:  # surfaced below as a page error
            box['exc'] = exc
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    streamed = []

    def gen():
        while True:
            chunk = q.get()
            if chunk is None:
                break
            streamed.append(chunk)
            yield chunk

    st.write_stream(gen())
    resp = box.get('resp')
    if resp is None:
        raise box.get('exc') or RuntimeError('AI request failed')
    if not streamed and resp.get('text'):
        # Degraded/fallback path produced no chunks — show it statically.
        st.markdown(resp['text'])
    return resp


def _status_banner(client: OllamaClient) -> None:
    health = client.health()
    if not health['reachable']:
        st.warning("Ollama is not reachable — AI features are degraded. "
                   "CAID continues to work with deterministic summaries. "
                   "Configure Ollama on the Settings page.")
        return
    st.success(f"Ollama connected — {len(health['models'])} model(s) "
               f"installed, {health['latency_ms']} ms. "
               f"Active: `{health['resolved_model']}`")


st.subheader("AI Security Analyst")
st.caption("Explanations are generated by the local Ollama model from "
           "actual CAID detection evidence only. AI output is advisory "
           "and never modifies scores, thresholds, or incident status. "
           "Content inside log fields is treated as untrusted data.")

structured_mode = st.checkbox(
    "Structured JSON output", key='ai_structured_mode',
    help="Request a schema-validated JSON response with observations, "
         "hypotheses, and evidence references instead of free text. "
         "Applies to the explanation tabs; streaming is disabled in "
         "this mode.")

client = get_client()
_status_banner(client)

result = analysis_or_prompt()
if result is None:
    st.stop()
df = result.get('scored_events')
correlations = result.get('correlations')
incidents = get_manager().list_incidents()

if df is None or len(df) == 0:
    st.info("No analyzed events available.")
    st.stop()

tab_event, tab_finding, tab_incident, tab_chain, tab_hunt, tab_qa = st.tabs([
    "Event explanation", "Finding explanation", "Incident summary",
    "Attack chain", "Threat hunting", "Ask a question",
])

# ---- Event explanation ----------------------------------------------------
with tab_event:
    st.markdown("Select a scored event to explain why it was flagged.")
    candidates = df
    if 'combined_score' in df.columns:
        candidates = df.assign(_sort_score=pd.to_numeric(
            df['combined_score'], errors='coerce')).sort_values(
            '_sort_score', ascending=False, na_position='last') \
            .drop(columns='_sort_score').head(200)
    else:
        candidates = df.head(200)
    labels = {}
    for pos, (_, r) in enumerate(candidates.iterrows()):
        labels[pos] = (f"{r.get('user_id', '?')} · {r.get('event_type', '?')} · "
                       f"{str(r.get('timestamp', ''))[:19]} · "
                       f"score={r.get('combined_score', 'n/a')}")
    pick = st.selectbox("Event", options=list(labels.keys()),
                        format_func=lambda i: labels.get(i, str(i)),
                        key='ai_event_pick')
    if st.button("Explain with AI", key='ai_explain_event', type='primary'):
        try:
            resp = _run_analyst(explain_event, candidates.iloc[pick], df,
                                client=client, model=_model(),
                                structured=structured_mode)
        except Exception as exc:
            st.error(safe_error_message(exc, "AI request failed"))
        else:
            _show_ai_result(resp, structured_mode)

# ---- Finding explanation --------------------------------------------------
with tab_finding:
    if correlations is None or len(correlations) == 0:
        st.info("No correlation findings in the current analysis.")
    else:
        flabels = {}
        for _, f in correlations.iterrows():
            fid = f.get('correlation_id')
            flabels[fid] = (f"{f.get('correlation_type')} — "
                            f"{', '.join(str(e) for e in (f.get('entities') or [])[:3])} "
                            f"({float(f.get('correlation_score') or 0):.2f})")
        fchoice = st.selectbox("Finding", options=list(flabels.keys()),
                               format_func=lambda k: flabels.get(k, k),
                               key='ai_finding_pick')
        frow = correlations[correlations['correlation_id'] == fchoice].iloc[0]
        finding = frow.to_dict()
        finding['evidence'] = frow.get('evidence')
        if st.button("Explain finding", key='ai_explain_finding',
                     type='primary'):
            try:
                resp = _run_analyst(explain_finding, finding, client=client,
                                    model=_model(),
                                    structured=structured_mode)
            except Exception as exc:
                st.error(safe_error_message(exc, "AI request failed"))
            else:
                _show_ai_result(resp, structured_mode)

# ---- Incident summary -----------------------------------------------------
with tab_incident:
    if not incidents:
        st.info("No incidents recorded for the current store.")
    else:
        ilabels = {i['incident_id']:
                   f"{i['incident_id']} — {i.get('title', '')} "
                   f"[{i.get('severity')}/{i.get('status')}]"
                   for i in incidents}
        ichoice = st.selectbox("Incident", options=list(ilabels.keys()),
                               format_func=lambda k: ilabels.get(k, k),
                               key='ai_inc_pick')
        inc = get_manager().store.get(ichoice)
        if inc and st.button("Generate AI summary", key='ai_inc_sum',
                             type='primary'):
            try:
                resp = _run_analyst(summarize_incident, inc, client=client,
                                    model=_model(),
                                    structured=structured_mode)
            except Exception as exc:
                st.error(safe_error_message(exc, "AI request failed"))
            else:
                _show_ai_result(resp, structured_mode)
                note_text = resp.get('text', '')
                s = resp.get('structured')
                if s:
                    note_text = '\n'.join(
                        [s.get('assessment', '')] +
                        [f"- {o}" for o in s.get('observations', [])] +
                        [f"- HYPOTHESIS: {h}"
                         for h in s.get('hypotheses', [])])
                st.session_state['ai_inc_note'] = {
                    'incident_id': ichoice,
                    'text': note_text}
                st.caption("AI summaries are advisory only — incident "
                           "severity and status are unchanged.")
        pending = st.session_state.get('ai_inc_note')
        if pending and pending.get('incident_id') == ichoice \
                and pending.get('text'):
            if st.button("Attach AI summary to incident notes",
                         key='ai_inc_attach'):
                mgr = get_manager()
                mgr.add_note(ichoice,
                             "[AI analyst summary]\n"
                             + pending['text'][:4000],
                             author='ai-analyst')
                st.session_state.pop('ai_inc_note', None)
                st.success("AI summary attached to incident notes "
                           "(author: ai-analyst).")
                st.rerun()

# ---- Attack chain / automated investigation -------------------------------
with tab_chain:
    st.markdown("Reconstruct a retrospective attack chain and generate an "
                "automated investigation summary from grounded incident "
                "evidence only.")
    if not incidents:
        st.info("No incidents recorded for the current store.")
    else:
        chain_labels = {i['incident_id']:
                        f"{i['incident_id']} — {i.get('title', '')} "
                        f"[{i.get('severity')}/{i.get('status')}]"
                        for i in incidents}
        cchoice = st.selectbox("Incident", options=list(chain_labels.keys()),
                               format_func=lambda k: chain_labels.get(k, k),
                               key='ai_chain_pick')
        cinc = get_manager().store.get(cchoice)
        if cinc:
            c1, c2 = st.columns(2)
            if c1.button("Investigate attack chain", key='ai_chain_btn',
                         type='primary'):
                try:
                    resp = _run_analyst(investigate_attack_chain, cinc,
                                        scored_df=df, client=client,
                                        model=_model(),
                                        structured=structured_mode)
                except Exception as exc:
                    st.error(safe_error_message(exc, "AI request failed"))
                else:
                    _show_ai_result(resp, structured_mode)
                    st.caption(f"Matched event evidence: "
                               f"{resp.get('matched_event_count', 0)}; "
                               f"ATT&CK mappings: "
                               f"{resp.get('mitre_mapping_count', 0)}")
            if c2.button("Generate investigation summary",
                         key='ai_inv_summary_btn'):
                try:
                    resp = _run_analyst(generate_investigation_summary,
                                        cinc, scored_df=df, client=client,
                                        model=_model(),
                                        structured=structured_mode)
                except Exception as exc:
                    st.error(safe_error_message(exc, "AI request failed"))
                else:
                    _show_ai_result(resp, structured_mode)
                    st.caption("Automated summary is advisory only — "
                               "incident severity and status are "
                               "unchanged.")

# ---- Threat hunting -------------------------------------------------------
with tab_hunt:
    st.markdown("Ask a predefined hunting question about the analyzed "
                "dataset. Questions are matched to safe, read-only "
                "queries — no generated code is executed.")
    q = st.text_input("Question", key='ai_hunt_q',
                      placeholder="e.g. Show accounts with failed logins")
    if st.button("Run hunt", key='ai_hunt_btn', type='primary') and q:
        try:
            resp = _ai_stream(hunt, q, df, client=client, model=_model())
        except Exception as exc:
            st.error(safe_error_message(exc, "AI request failed"))
        else:
            if resp.get('query_label'):
                st.caption(f"Query: {resp['query_label']}")
            rows = resp.get('rows')
            if rows is not None and len(rows):
                st.dataframe(rows.head(100), hide_index=True,
                             width='stretch', alt='Threat hunt results')
                st.download_button("Download results (CSV)",
                                   rows.to_csv(index=False), "hunt.csv",
                                   "text/csv", key='ai_hunt_csv')
            _show_details(resp)

# ---- General Q&A (chat with memory) --------------------------------------
SUGGESTED_QUESTIONS = [
    "How many high-risk events are there?",
    "Which users had failed logins?",
    "Show sensitive operations by user",
    "What anomalies were detected?",
]
CHAT_KEY = 'ai_analyst_chat'

with tab_qa:
    st.markdown("Ask about the analyzed dataset in plain language. "
                "Answers are grounded in dataset statistics and a "
                "bounded sample of scored events — the AI can see the "
                "highest-risk and most recent events, not every row. "
                "Predefined hunting questions are answered with real "
                "query results.")
    scenario_key = result.get('scenario_key', 'default')
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = _load_chat(scenario_key)
    for msg in st.session_state[CHAT_KEY]:
        with st.chat_message(msg['role']):
            st.markdown(msg['text'])

    st.markdown("**Try asking:**")
    sugg_cols = st.columns(len(SUGGESTED_QUESTIONS))
    picked = None
    for i, s in enumerate(SUGGESTED_QUESTIONS):
        if sugg_cols[i].button(s, key=f'ai_suggest_{i}', width='stretch'):
            picked = s

    question = st.chat_input("Ask about the analyzed dataset…",
                             key='ai_chat_in')
    submitted = question or picked
    if submitted:
        history = list(st.session_state[CHAT_KEY])
        st.session_state[CHAT_KEY].append(
            {'role': 'user', 'text': submitted})
        with st.chat_message('user'):
            st.markdown(submitted)
        with st.chat_message('assistant'):
            try:
                resp = _ai_stream(ask, submitted, df, client=client,
                                  model=_model(), history=history)
            except Exception as exc:
                st.error(safe_error_message(exc, "AI request failed"))
                resp = {'text': f"_(AI request failed: {exc})_",
                        'degraded': True, 'error': str(exc),
                        'evidence_used': []}
            _show_details(resp)
            rows = resp.get('rows')
            if rows is not None and len(rows):
                st.dataframe(rows.head(100), hide_index=True,
                             width='stretch',
                             alt='Hunting query results')
        st.session_state[CHAT_KEY].append(
            {'role': 'assistant', 'text': resp.get('text', '')})
        _save_chat(scenario_key, st.session_state[CHAT_KEY])

    if st.session_state[CHAT_KEY]:
        if st.button("Clear chat history", key='ai_clear_chat'):
            st.session_state[CHAT_KEY] = []
            _save_chat(scenario_key, [])
            st.rerun()
