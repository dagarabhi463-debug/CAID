# AI Analyst Accuracy Benchmark

Addresses assessment items #29–31 and #38: *no quantitative AI accuracy
benchmark, hallucination rate not established, attack-chain stages can be
hypotheses, AI cannot independently verify intent.*

## What this is

`core/ai_evaluation.py` defines synthetic CAID scenarios with known
ground truth and scores analyst responses **deterministically** — no
LLM-as-judge. `scripts/evaluate_ai_analyst.py` runs them against a live
Ollama server (or `--mock` for plumbing checks) and writes
`reports/ai_evaluation.json`.

## Scenarios (9)

| id | path exercised |
|----|----------------|
| `event_bulk_download` | `explain_event` (structured) |
| `finding_shared_ip` | `explain_finding` — shared-IP-only caveat |
| `incident_summary` | `summarize_incident` |
| `incident_mitre` | `generate_investigation_summary` + ATT&CK maps |
| `attack_chain` | `investigate_attack_chain` + ATT&CK maps |
| `ask_event_count` | `ask` freeform (dataset stats) |
| `hunt_failed_auth` | `hunt` predefined query (rows verified deterministically) |
| `adversarial_injection` | `explain_event` with injection text in log fields |
| `empty_incident` | `summarize_incident` with zero findings |

## Metrics

All checks are heuristics over the response text, the structured JSON
payload, and the exact prompt sent to the model (captured via
`RecordingClient`):

- **term_recall** — fraction of expected ground-truth terms present
  (`_`/`-`/space treated as equivalent).
- **fabricated_entities** — identifier-like tokens in the response
  (IPs, `user_*`/`sess_*`/`inc_*`/`corr_*`/`det_*`/`evt_*` ids, ATT&CK
  `T####` ids, long hex) absent from the prompt = hallucinated
  references.
- **unsupported_numbers** — decimals and integers ≥100 in the response
  whose value does not appear in the prompt.
- **observations_unsupported** — structured `observations` whose
  numbers/identifiers are not traceable to the cited `E<n>` evidence
  (FACTS always count as citable).
- **forbidden_hits** — regex ground-truth violations (e.g. "confirmed
  compromise"), with a negation guard so disclaimers like *"not
  confirmed attacker behavior"* are not flagged. Match context is
  stored in `forbidden_snippets` for audit.
- **required_caveats** — scenarios that must hedge (shared-IP-only,
  potential ATT&CK, empty evidence) must contain caveat language.
- **deterministic_check / prompt_check** — hunt row correctness and
  prompt-injection sanitization (`[filtered]` + `<<UNTRUSTED_DATA>>`
  delimiters verified in the recorded prompt).
- **telemetry** — `elapsed_ms`, token usage, structured parse rate,
  dropped evidence refs.

## Known scorer limitations

- Identifier/number extraction is regex-based; it can miss paraphrased
  fabrications (e.g. "the attacker" with no concrete token) and can
  flag legitimate derivations.
- `term_recall` penalizes the model for not repeating raw IDs even when
  the prose is otherwise accurate — interpret as *citation
  specificity*, not correctness.
- The negation guard is a 48-char window heuristic; adversarial
  phrasing could evade or falsely trigger it.
- Scenarios are synthetic and small (9); results do not generalize to
  real log data.

## Results — llama3.1:8b @ localhost (2026-11-12)

See `reports/ai_evaluation.json` for the canonical run. Summary of the
latest run:

| metric | value |
|--------|-------|
| degraded | 0/9 |
| mean term_recall | 0.73–0.88 across runs |
| fabrication_rate | 0.0 |
| unsupported_claim_rate | 0.0 |
| forbidden_claim_scenarios | 0 |
| caveat_compliance | 1.0 |
| structured_parse_rate | 1.0 |
| mean dropped evidence refs | ~0.2–0.8 (varies per run) |
| latency mean / p95 | ~19.6–19.8s / ~35–42s |
| tokens mean | ~940–975 |

An earlier run caught a genuine fabrication: hunt narration cited
`T1071.001`, an ATT&CK technique absent from the evidence — exactly
the hallucination class this harness is meant to detect. LLM
nondeterminism means metrics vary between runs; treat single-run
numbers as indicative, not definitive. Re-run before drawing
conclusions:

```
python scripts/evaluate_ai_analyst.py
```

## What this does NOT establish

- Factual accuracy against *real* incidents (scenarios are synthetic).
- Robustness across other models — rerun with `--model` to compare.
- Resistance to sophisticated prompt injection (one adversarial
  scenario only).
- That a clean score proves correctness — it bounds known failure
  modes only.
