# Phase 5.1 Audit: Existing AI Evaluation

## Date
2026-11-12

## Files Audited

- `core/ai_evaluation.py` (651 lines)
- `scripts/evaluate_ai_analyst.py` (136 lines)
- `core/ai_analyst.py` (701 lines)
- `core/ai_context.py` (528 lines)
- `core/ollama_client.py` (310 lines)
- `tests/test_ai_evaluation.py` (304 lines)
- `tests/test_ai_reliability.py` (305 lines)
- `reports/ai_evaluation.json` (existing results)

## Current State Summary

### Strengths

1. **Deterministic scoring** - No LLM-as-judge; all checks are regex/string-based
2. **Prompt recording** - `RecordingClient` captures exact prompts for evidence validation
3. **Fabrication detection** - Identifier extraction (IPs, user/session/incident IDs, ATT&CK T####)
4. **Negation guard** - 48-char window prevents false positives on disclaimers
5. **Mock plumbing tests** - `MockClient` and `JsonClient` enable offline testing
6. **Telemetry** - Latency, token counts, structured parse rate, dropped refs
7. **Structured validation** - JSON schema validation with evidence ref filtering
8. **9 existing scenarios** covering main analyst paths

### Existing Scenarios (9)

| ID | Path | Category |
|----|------|----------|
| event_bulk_download | explain_event | Evidence grounding |
| finding_shared_ip | explain_finding | Evidence grounding (shared-IP caveat) |
| incident_summary | summarize_incident | Incident investigation |
| incident_mitre | generate_investigation_summary | MITRE ATT&CK |
| attack_chain | investigate_attack_chain | MITRE ATT&CK |
| ask_event_count | ask | Numerical reasoning |
| hunt_failed_auth | hunt | Numerical reasoning |
| adversarial_injection | explain_event | Prompt injection |
| empty_incident | summarize_incident | Reliability (empty evidence) |

### Existing Metrics

- `mean_term_recall` - fraction of expected terms present
- `fabrication_rate` - scenarios with fabricated entities
- `unsupported_claim_rate` - scenarios with unsupported numbers/observations
- `forbidden_claim_scenarios` - forbidden pattern hits
- `caveat_compliance` - scenarios requiring caveats that passed
- `structured_parse_rate` - structured JSON parse success
- `mean_dropped_refs` - average invalid evidence refs
- `latency_ms_mean/p95` - latency telemetry
- `tokens_mean` - token count telemetry

## Missing Requirements (Phase 5.1)

### 1. Repeatability Testing
- **Current**: Single run per scenario
- **Needed**: Configurable repetitions (default 10), individual run recording, pass rates, worst-case failures, failed response preservation

### 2. Evidence Grounding Expansion (6 scenarios → ~12)
Missing categories:
- Missing event evidence
- Contradictory evidence
- Incorrect event references
- Unsupported user attribution
- Unsupported numerical claims

### 3. Incident Investigation Expansion (2 scenarios → ~8)
Missing categories:
- Incomplete incident
- Multiple related findings
- Conflicting indicators
- Shared-device-only evidence
- Uncertain attack-chain reconstruction

### 4. MITRE ATT&CK Expansion (2 scenarios → ~6)
Missing categories:
- Unsupported technique mapping
- Incorrect technique identifier
- Tactic/technique mismatch
- Potential technique requiring caveat
- No applicable technique

### 5. Prompt Injection Expansion (1 scenario → ~6)
Missing categories:
- Instructions inside user-agent fields
- Malicious text in filenames
- Fake system instructions
- Attempts to obtain secrets
- Attempts to change detection conclusions
- Data exfiltration requests

### 6. Numerical Reasoning Expansion (2 scenarios → ~5)
Missing categories:
- Incorrect event counts
- Percentages derived from supplied evidence
- Missing numerical evidence
- Unsupported severity statistics

### 7. Reliability Scenarios (1 → ~8)
Missing categories:
- Empty model response
- Invalid JSON
- Missing required fields
- Truncated response
- Ollama timeout
- Model unavailable
- Fallback response

### 8. New Metrics Required

Missing from current aggregation:
- `heuristic_detected_fabrication_rate` (per-run fabrication detection)
- `evidence_reference_precision` - fraction of refs that are valid
- `evidence_reference_recall` - fraction of evidence items cited
- `unsupported_claim_rate` (exists but refine)
- `caveat_compliance_rate` (exists)
- `attack_mapping_accuracy` - ATT&CK validation
- `structured_schema_validity_rate` - schema conformance
- `scenario_pass_rate` - fraction of runs passing all checks
- `repeatability_rate` - per-scenario consistency across runs
- `timeout_rate` - failed due to timeout
- `fallback_rate` - degraded responses
- `mean_latency_ms` (exists)
- `p50_latency_ms` (missing)
- `p95_latency_ms` (exists)
- `mean_tokens` (exists)

### 9. Evidence Validator Improvements

Current `check_observation_support`:
- Checks numbers/identifiers against cited evidence pool
- Distinguishes observations but not explicitly labeled as observed vs hypothesized
- Does not validate attack-chain stage labels
- Does not require explicit insufficient-evidence response for empty evidence

Needed:
- Strict evidence reference validation (unknown refs must fail)
- Distinguish direct observations, derived observations, hypotheses
- Attack-chain stages labeled as observed/hypothesized
- Reject unsupported attacker intent claims
- Empty evidence → explicit insufficient-evidence response requirement

### 10. ATT&CK Validation

Current:
- Scenarios include MITRE mappings
- Term recall checks for technique IDs
- No dedicated technique validation

Needed:
- Validate ATT&CK IDs against supplied evidence
- Validate against approved mapping catalogue
- Flag unsupported mappings
- Check tactic/technique alignment

## False Positive/Negative Analysis

### Identifier Extraction (Regex)

**False positives possible:**
- Generic tokens like `e1`-`e20` are filtered
- But valid derivations could be flagged (e.g., legitimate new session IDs in hypotheses)
- Long hex strings might be used in legitimate contexts

**False negatives possible:**
- Paraphrased fabrications without concrete identifiers (e.g., "the attacker" with no ID)
- Fabricated entity not matching regex patterns
- Fabricated terms within allowed derivations

### Term Recall

**False positives:**
- Penalizes model for not repeating raw IDs when prose is accurate
- `term_recall` measures citation specificity, not factual correctness

**False negatives:**
- Synonyms not in expected_terms list
- Model correctly describes fact without exact term

### Negation Guard

**False positives:**
- 48-char window may be too narrow or too wide
- Complex negation structures may not be detected

**False negatives:**
- Adversarial phrasing that evades the negation check
- Negation word appears but is actually a false claim

### Mock Mode Interpretation

**Risk identified:**
- Mock client returns deterministic low-quality response
- If reported as model accuracy, would be misleading
- Current script prints "mode: mock (plumbing check only)" but reports are JSON-only
- Need to distinguish mock vs live in report metadata

## Determinism Analysis

### Deterministic Components
- Regex extraction: deterministic
- String normalization: deterministic
- Score aggregation: deterministic arithmetic
- Scenario selection: deterministic when specified

### Non-Deterministic Components
- Model responses: inherently non-deterministic (Ollama)
- Term recall varies by model phrasing
- Fabrication detection depends on model wording

### Mock Mode
- Fully deterministic by design
- Should never be presented as live model accuracy

## Traceability

### Every Expected Fact Has Traceable Evidence?
- Current scenarios: YES - evidence is fully defined in fixtures
- Check: `RecordingClient` captures prompt, used as evidence pool
- Validation: Numbers/identifiers checked against evidence pool

### Scoring Deterministic?
- For a given response text: YES
- Across model runs: NO (model varies)
- Recommendation: Record individual runs separately, aggregate after

## Mock Mode Issues

**Current handling:**
- `MockClient` returns minimal valid JSON
- Script prints "mode: mock (plumbing check only)"
- Report includes `model: 'mock-model'`

**Risk:**
- Reports don't explicitly label mock vs live in summary
- Future automation might misinterpret mock results

**Fix required:**
- Add `is_mock: true` to report metadata
- Add evaluation mode field to summary
- Clear warning in report text

## Scoring Ambiguity

### Scenarios with Ambiguous Expected Answers?
- Current 9 scenarios: NO - all have clear ground truth
- New scenarios must maintain this

### Ground Truth Completeness
- Evidence: complete in fixtures
- Expected terms: clearly specified
- Forbidden patterns: clear regex
- Required caveats: clear list
- Required evidence refs: tracked

## Implementation Plan

### Phase 1: Preserve Baseline
1. Copy `reports/ai_evaluation.json` → `reports/ai_evaluation_v1_baseline.json`
2. Copy `reports/ai_evaluation_run2.json` → `reports/ai_evaluation_v1_run2_baseline.json`

### Phase 2: Extend `core/ai_evaluation.py`
1. Add `run_scenario_repeatedly(scenario, client, model, runs=10)` function
2. Add `aggregate_repeated_runs(results_by_scenario)` function
3. Add `validate_attack_mapping(response, evidence)` function
4. Add `validate_evidence_references(response, evidence)` with stricter checks
5. Add `check_empty_evidence_response(response)` function
6. Extend `build_scenarios()` to return ~35 scenarios (append new ones)
7. Add new metrics to `aggregate()`: precision, recall, pass_rate, repeatability_rate, etc.
8. Add `mock_marker` field to report metadata

### Phase 3: Update `scripts/evaluate_ai_analyst.py`
1. Add `--runs` argument (default 10)
2. Add `--scenario` support for repeated runs
3. Add report versioning (v2 format)
4. Add baseline preservation logic
5. Generate separate files:
   - `reports/ai_evaluation.json` (main report)
   - `reports/ai_evaluation_summary.md` (human-readable)
   - `reports/ai_evaluation_scenarios.json` (scenario definitions)
6. Add metadata: model, version, config, date, repetitions, mode (mock/live)

### Phase 4: Add New Scenarios
Create in `core/ai_evaluation.py`:

**Evidence grounding (+6):**
- evidence_missing_event
- evidence_contradictory
- evidence_incorrect_ref
- evidence_unsupported_user
- evidence_unsupported_numerical
- evidence_partial

**Incident investigation (+6):**
- incident_incomplete
- incident_multiple_findings
- incident_conflicting_indicators
- incident_shared_device_only
- incident_uncertain_chain
- incident_edge_case

**MITRE ATT&CK (+4):**
- mitre_unsupported_mapping
- mitre_incorrect_id
- mitre_tactic_mismatch
- mitre_no_applicable

**Prompt injection (+5):**
- injection_user_agent
- injection_filename
- injection_fake_system
- injection_secret_request
- injection_detection_override

**Numerical reasoning (+3):**
- numeric_incorrect_count
- numeric_percentage_derived
- numeric_missing_evidence

**Reliability (+7):**
- reliability_empty_response
- reliability_invalid_json
- reliability_missing_fields
- reliability_truncated
- reliability_timeout
- reliability_unavailable
- reliability_fallback

Total: 9 existing + 31 new = 40 scenarios

### Phase 5: Add Tests
Create `tests/test_ai_evaluation_v2.py`:
- Test new metrics
- Test repeated-run aggregation
- Test evidence reference validation
- Test ATT&CK validation
- Test new scenarios (sample)
- Test empty evidence handling
- Test timeout/fallback handling
- Test mock mode metadata

### Phase 6: Run Verification
1. Run full test suite: `pytest tests/`
2. Run mock evaluation: `python scripts/evaluate_ai_analyst.py --mock --runs 10`
3. Run live evaluation (bounded): `python scripts/evaluate_ai_analyst.py --runs 3`
4. Verify reports generated correctly
5. Verify mock metadata present

### Phase 7: Documentation
Create `docs/research/phase_5_1_ai_reliability_benchmark.md`:
- Approach and methodology
- Scenario catalog
- Metric definitions
- Known limitations
- Baseline preservation notes
- Execution instructions

## Risk Mitigation

1. **Mock mode confusion** - Add explicit `is_mock` flag to report
2. **Ambiguous ground truth** - Review each new scenario for clarity
3. **False positives in fabrication** - Document limitation
4. **Determinism** - Keep scoring deterministic, document model nondeterminism
5. **Detection engine** - No changes to detection artifacts, scores, thresholds
6. **Regression** - Run full test suite before commit
7. **Baseline preservation** - Copy existing reports before overwriting

## Files to Modify

- `core/ai_evaluation.py` - extend with v2 logic
- `scripts/evaluate_ai_analyst.py` - add --runs, v2 report format
- `tests/test_ai_evaluation_v2.py` - new file
- `docs/research/phase_5_1_ai_reliability_benchmark.md` - new file
- `reports/ai_evaluation_v1_baseline.json` - new file (preserved copy)
- `reports/ai_evaluation_summary.md` - new file
- `reports/ai_evaluation_scenarios.json` - new file

## Files to Create

- `tests/test_ai_evaluation_v2.py`
- `docs/research/phase_5_1_ai_reliability_benchmark.md`
- `reports/ai_evaluation_v1_baseline.json` (copy)
- `reports/ai_evaluation_summary.md`
- `reports/ai_evaluation_scenarios.json`

## Acceptance Criteria

- [ ] Baseline reports preserved
- [ ] ~40 scenarios total with unambiguous ground truth
- [ ] Repeated runs recorded individually
- [ ] Failed responses preserved
- [ ] Evidence-reference validation strict
- [ ] Observed facts vs hypotheses distinguished
- [ ] Mock runs marked as mock in reports
- [ ] No detection-engine modifications
- [ ] All existing tests pass
- [ ] New tests cover new metrics
- [ ] Live Ollama evaluation runs successfully
- [ ] Mock evaluation runs successfully
- [ ] Documentation complete
- [ ] Git status shows clean staged changes
