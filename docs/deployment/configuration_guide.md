# CAID Configuration Guide

All configuration is via environment variables (`.env`, gitignored) or the Settings page. No secrets are required for core detection.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama server endpoint. `https://` supported. |
| `OLLAMA_MODEL` | `llama3.1:8b` | Preferred model; falls back to first installed model. |
| `OLLAMA_TIMEOUT` | `60` | Request timeout in seconds (5–300 via UI). |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis Streams queue. `rediss://` supported; credentials may be embedded in URL. |

`.env.example` documents all of the above — copy to `.env`.

## Settings page (Data & Admin → Settings)

- Ollama base URL / timeout / model picker (session-scoped; "Test connection" probes `/api/tags`).
- MITRE dataset reload (`allow_download=True` fetches the official bundle once into `data/mitre_attack/`).
- Read-only views: model artifact status, calibration status, incident store path.

## Failure-mode behavior (verified by tests/manual checks)

| Condition | Behavior |
|---|---|
| Missing/invalid env vars | Defaults apply; `OLLAMA_TIMEOUT` non-numeric → 60s fallback |
| Ollama down / model missing | Typed `OllamaError` subclasses → deterministic fallback summaries; detection unaffected |
| Redis down | asyncio queue used; Redis tests skip; worker continues |
| Corrupt model/calibration artifact | `ensure_*` helpers refit-or-fail explicitly; fingerprint mismatch → uncalibrated status surfaced, never silent |
| Missing MITRE dataset | Page shows "not loaded"; optional one-time download |
| Corrupt incident store | `IncidentStore.load` → empty store + `load_error` warning (no crash) |
| Unwritable reports dir | Export writes fail with surfaced error; core pipeline unaffected |
| Interrupted streaming worker | Pending entries recovered via `claim_pending`/XAUTOCLAIM; checkpoint restores tracker + dedup state |

## Security notes

- Keep the app bound to localhost; there is no authentication (see `docs/security/vulnerability_register.md` SEC-005).
- `REDIS_URL`/`OLLAMA_BASE_URL` accept remote endpoints — use TLS (`rediss://`, `https://`) and URL-embedded credentials if ever remote.
