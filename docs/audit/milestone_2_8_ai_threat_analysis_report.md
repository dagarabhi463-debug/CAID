# Milestone 2.8 — Advanced AI Threat Analysis Report

Date: 2026-11-12  
Scope: standalone defensive research prototype; synthetic data only.

## Implemented

- **Attack-chain investigation**
  - Added `attack_chain_context()` in `core/ai_context.py` to build a grounded evidence bundle from incident findings, matched scored events, and MITRE ATT&CK mappings.
  - Added `investigate_attack_chain()` in `core/ai_analyst.py`: asks Ollama to reconstruct observed/hypothesized/unsupported chain stages using only supplied evidence, with explicit uncertainty and no state mutation.

- **Automated investigation summaries**
  - Added `generate_investigation_summary()` in `core/ai_analyst.py`: combines incident context and attack-chain evidence into a read-only analyst summary with overview, entities, evidence, potential ATT&CK techniques, uncertainty, and next steps.

- **MITRE grounding**
  - Both new AI functions can load local ATT&CK mappings through `MitreAttackStore.ensure_data(allow_download=False)` when available; mappings remain labelled `potential`/`confirmed` by the existing mapper and are never treated as proof of attacker behavior.

- **Expanded safe threat hunting**
  - Added predefined read-only queries in `core/ai_context.py`:
    - after-hours sensitive operations,
    - authentication failure bursts,
    - shared IPs used by multiple users,
    - users with broad sensitive-operation activity.
  - No generated code, SQL, or shell execution was added.

- **Streamlit UI**
  - Added an **Attack chain** tab to `app_pages/ai_analyst.py` with incident selection, attack-chain investigation, and automated investigation summary actions.

## Safety contract preserved

- AI output is advisory only.
- No AI path modifies detection scores, thresholds, operating modes, incident severity/status, or correlation state.
- Log-derived content remains sanitized and wrapped as untrusted data.
- Ollama remains optional; deterministic evidence summaries are returned when unavailable.

## Verification

```powershell
.venv/Scripts/python.exe -m py_compile core/ai_context.py core/ai_analyst.py app_pages/ai_analyst.py
.venv/Scripts/python.exe -m pytest tests/test_ai_grounding.py -q
```

Result: `16 passed`.

New tests cover:

- attack-chain context uses real incident/event/MITRE evidence,
- deterministic fallback for attack-chain investigation,
- deterministic fallback for automated investigation summary,
- all predefined hunt queries still execute on scored data.

## Limitations

- Live Ollama generation was not exercised in this environment; deterministic fallback behavior is tested.
- Attack-chain stages are hypotheses grounded in retrospective correlation evidence, not confirmed intrusion timelines.
- ATT&CK coverage depends on the locally cached MITRE bundle and existing mapping rules.
