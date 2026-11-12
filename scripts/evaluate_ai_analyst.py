"""
AI Analyst accuracy benchmark (limitations #29-31, #38).

Runs the grounded-evaluation scenarios in ``core.ai_evaluation``
against a live Ollama server and scores factual recall, fabricated
entities, unsupported claims, caveat compliance, citation validity,
and latency/token usage. Writes ``reports/ai_evaluation.json``.

Phase 5.1 adds repeatability testing, expanded scenarios (~40),
and additional metrics (evidence validation, ATT&CK mapping,
pass rates, repeatability rates).

Usage:
    python scripts/evaluate_ai_analyst.py                # live Ollama (single run)
    python scripts/evaluate_ai_analyst.py --runs 10       # repeated runs (default 10)
    python scripts/evaluate_ai_analyst.py --mock         # plumbing check
    python scripts/evaluate_ai_analyst.py --scenario event_bulk_download
    python scripts/evaluate_ai_analyst.py --model llama3.1:8b

Environment: OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT.
Exit code 0 always — this is a measurement tool, not a gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.ai_evaluation import (
    aggregate_repeated_runs,
    build_scenarios,
    run_benchmark,
    run_scenario_repeatedly,
)  # noqa: E402
from core.ollama_client import OllamaClient  # noqa: E402

OUT_PATH = os.path.join(ROOT, 'reports', 'ai_evaluation.json')
SUMMARY_PATH = os.path.join(ROOT, 'reports', 'ai_evaluation_summary.md')
SCENARIOS_PATH = os.path.join(ROOT, 'reports', 'ai_evaluation_scenarios.json')


class MockClient:
    """Deterministic stand-in for plumbing tests (no Ollama needed).

    Returns a minimal valid structured response so the harness can be
    exercised end-to-end; scores will be low — that is expected.
    """

    model = 'mock-model'
    last_stats = {'total_tokens': 50, 'model': 'mock-model'}
    timeout = 30

    def resolve_model(self):
        return 'mock-model'

    def generate(self, prompt, model=None, system=None, stream=False):
        payload = json.dumps({
            'assessment': 'mock response for testing',
            'observations': ['mock observation based on evidence'],
            'evidence_refs': ['E1'],
            'hypotheses': [],
            'limitations': ['mock mode - no actual analysis'],
            'confidence': 'low'})
        if stream:
            return iter([payload])
        return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--mock', action='store_true',
                    help='use deterministic mock client (no Ollama)')
    ap.add_argument('--model', default=None,
                    help='Ollama model override')
    ap.add_argument('--scenario', action='append', default=None,
                    help='run only this scenario id (repeatable)')
    ap.add_argument('--runs', type=int, default=10,
                    help='number of repetitions per scenario (default 10)')
    ap.add_argument('--out', default=OUT_PATH)
    args = ap.parse_args()

    scenarios = build_scenarios()
    if args.scenario:
        wanted = set(args.scenario)
        scenarios = [s for s in scenarios if s['id'] in wanted]
        missing = wanted - {s['id'] for s in scenarios}
        if missing:
            print(f'unknown scenario(s): {sorted(missing)}',
                  file=sys.stderr)
            return 2

    is_mock = args.mock
    health = {'base_url': 'mock://localhost', 'latency_ms': 0}
    if is_mock:
        client, model = MockClient(), 'mock-model'
        print('mode: mock (plumbing check only)')
    else:
        client = OllamaClient(model=args.model) if args.model \
            else OllamaClient()
        health = client.health(timeout=10)
        if not health['reachable']:
            print('Ollama is not reachable at '
                  f'{health["base_url"]}: {health["error"]}',
                  file=sys.stderr)
            print('Start Ollama or rerun with --mock.', file=sys.stderr)
            return 1
        model = health['resolved_model']
        print(f'mode: live Ollama @ {health["base_url"]} '
              f'(model={model}, latency={health["latency_ms"]}ms)')

    # Phase 5.1: Repeated runs
    print(f'running {len(scenarios)} scenarios with {args.runs} repetitions each...')
    scenario_results = []
    for scenario in scenarios:
        print(f"  {scenario['id']}...", end=' ', flush=True)
        sr = run_scenario_repeatedly(scenario, client, model=model, runs=args.runs)
        scenario_results.append(sr)
        print(f"pass_rate={sr['summary']['scenario_pass_rate']:.2f}")

    report = aggregate_repeated_runs(scenario_results)

    # Add metadata
    generated_at = time.strftime('%Y-%m-%dT%H:%M:%S')
    report['metadata'] = {
        'version': '2.0',
        'generated_at': generated_at,
        'model': model,
        'is_mock': is_mock,
        'repetitions': args.runs,
        'scenario_count': len(scenarios),
        'total_runs': report['summary']['total_runs'],
        'ollama_base_url': health.get('base_url') if not is_mock else None,
        'ollama_config': {
            'model': model,
            'timeout': getattr(client, 'timeout', None) if not is_mock else None,
        } if not is_mock else None,
    }
    report['summary']['generated_at'] = generated_at

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, default=str)

    # Generate scenario catalog
    scenarios_catalog = [
        {
            'id': s['id'],
            'description': s['description'],
            'category': categorize_scenario(s['id']),
        }
        for s in scenarios
    ]
    with open(SCENARIOS_PATH, 'w', encoding='utf-8') as fh:
        json.dump(scenarios_catalog, fh, indent=2)

    # Generate summary markdown
    generate_summary_md(report, SUMMARY_PATH, scenarios_catalog)

    s = report['summary']
    print(f"\n--- Summary ---")
    print(f"scenarios={s['scenario_count']} total_runs={s['total_runs']} "
          f"degraded={s['degraded']}")
    print(f"overall_pass_rate={s['overall_pass_rate']} "
          f"repeatability_rate={s['repeatability_rate']}")
    print(f"mean_term_recall={s['mean_term_recall']} "
          f"fabrication_rate={s['fabrication_rate']} "
          f"unsupported_claim_rate={s['unsupported_claim_rate']}")
    print(f"caveat_compliance={s['caveat_compliance']} "
          f"structured_parse_rate={s['structured_parse_rate']}")
    print(f"latency_ms_mean={s['latency_ms_mean']} "
          f"p50={s['latency_ms_p50']} p95={s['latency_ms_p95']} "
          f"tokens_mean={s['tokens_mean']}")
    print(f"timeout_rate={s['timeout_rate']} "
          f"fallback_rate={s['fallback_rate']}")
    if s.get('attack_mapping_accuracy'):
        print(f"attack_mapping_accuracy={s['attack_mapping_accuracy']}")

    print(f'\nreports written to:')
    print(f'  {args.out}')
    print(f'  {SUMMARY_PATH}')
    print(f'  {SCENARIOS_PATH}')
    return 0


def categorize_scenario(scenario_id: str) -> str:
    """Map scenario ID to category for reporting."""
    if scenario_id.startswith('evidence_'):
        return 'Evidence grounding'
    if scenario_id.startswith('incident_'):
        return 'Incident investigation'
    if scenario_id.startswith('mitre_'):
        return 'MITRE ATT&CK'
    if scenario_id.startswith('injection_'):
        return 'Prompt injection'
    if scenario_id.startswith('numeric_'):
        return 'Numerical reasoning'
    if scenario_id.startswith('reliability_'):
        return 'Reliability'
    if scenario_id in ['event_bulk_download', 'finding_shared_ip']:
        return 'Evidence grounding'
    if scenario_id in ['incident_summary', 'incident_mitre', 'attack_chain']:
        return 'Incident investigation'
    if scenario_id in ['ask_event_count', 'hunt_failed_auth']:
        return 'Numerical reasoning'
    if scenario_id == 'adversarial_injection':
        return 'Prompt injection'
    if scenario_id == 'empty_incident':
        return 'Reliability'
    return 'Other'


def generate_summary_md(report: Dict, out_path: str, scenarios_catalog: List[Dict]):
    """Generate human-readable markdown summary."""
    lines = [
        '# AI Analyst Reliability Benchmark Summary',
        '',
        f'**Version:** {report["metadata"]["version"]}',
        f'**Generated:** {report["metadata"]["generated_at"]}',
        f'**Model:** {report["metadata"]["model"]}',
        f'**Mode:** {"Mock (plumbing check)" if report["metadata"]["is_mock"] else "Live Ollama"}',
        f'**Repetitions:** {report["metadata"]["repetitions"]}',
        f'**Scenarios:** {report["metadata"]["scenario_count"]}',
        f'**Total Runs:** {report["metadata"]["total_runs"]}',
        '',
        '## Overall Metrics',
        '',
    ]
    s = report['summary']
    lines.extend([
        f"- **Overall pass rate:** {s['overall_pass_rate']:.2%}",
        f"- **Repeatability rate:** {s['repeatability_rate']:.2%}",
        f"- **Mean term recall:** {s['mean_term_recall']:.3f}",
        f"- **Fabrication rate:** {s['fabrication_rate']:.2%}",
        f"- **Unsupported claim rate:** {s['unsupported_claim_rate']:.2%}",
        f"- **Caveat compliance:** {s['caveat_compliance']:.2%}" if s.get('caveat_compliance') else "- **Caveat compliance:** N/A",
        f"- **Structured parse rate:** {s['structured_parse_rate']:.2%}" if s.get('structured_parse_rate') else "- **Structured parse rate:** N/A",
        f"- **Latency (mean/p50/p95):** {s['latency_ms_mean']:.1f}ms / {s['latency_ms_p50']:.1f}ms / {s['latency_ms_p95']:.1f}ms" if s.get('latency_ms_mean') else "- **Latency:** N/A",
        f"- **Mean tokens:** {s['tokens_mean']:.1f}" if s.get('tokens_mean') else "- **Mean tokens:** N/A",
        f"- **Timeout rate:** {s['timeout_rate']:.2%}",
        f"- **Fallback rate:** {s['fallback_rate']:.2%}",
    ])
    if s.get('attack_mapping_accuracy'):
        lines.append(f"- **ATT&CK mapping accuracy:** {s['attack_mapping_accuracy']:.2%}")

    lines.extend([
        '',
        '## Scenario Results',
        '',
        '| Scenario | Category | Pass Rate | Worst Failures |',
        '|----------|----------|-----------|----------------|',
    ])

    for sr in report['scenario_results']:
        sid = sr['scenario_id']
        cat = next((c['category'] for c in scenarios_catalog if c['id'] == sid), 'Other')
        pass_rate = sr['summary']['scenario_pass_rate']
        worst = sr['summary'].get('worst_failures', [])
        worst_desc = f"{len(worst)} failure(s)" if worst else "None"
        lines.append(f"| {sid} | {cat} | {pass_rate:.2%} | {worst_desc} |")

    lines.extend([
        '',
        '## Known Limitations',
        '',
        '- Identifier/number extraction is regex-based; may miss paraphrased fabrications.',
        '- Term recall measures citation specificity, not factual correctness.',
        '- Negation guard is a 48-char window heuristic.',
        '- Scenarios are synthetic; results do not generalize to real log data.',
        '- Mock results are plumbing checks only, not model accuracy.',
        '- Clean scores do not prove correctness; they bound known failure modes.',
        '',
        '## What This Does NOT Establish',
        '',
        '- Factual accuracy against real incidents.',
        '- Robustness across other models.',
        '- Resistance to sophisticated prompt injection.',
        '- That a clean score proves correctness.',
        '',
        '---',
        '',
        f'*Report generated by Phase 5.1 benchmark (v{report["metadata"]["version"]}).*',
    ])

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))


if __name__ == '__main__':
    raise SystemExit(main())
