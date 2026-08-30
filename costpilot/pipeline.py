"""CostPilot pipeline: one command, full report.

Usage:
    python pipeline.py <repo_path> [calls_per_day] [usage_json]

Scans a repository, estimates costs (L1 static / L2 scenario / L3 observed),
and produces optimization recommendations with reject-optimization judgment.

This is the demo entry point: a judge runs one command and sees everything.
"""

from __future__ import annotations

import sys

from costpilot.models import OptimizationKind
from costpilot.optimizer import find_optimization_candidates
from costpilot.pricing import observed_cost_from_file, profile_cost, scenario_monthly_cost
from costpilot.scanner import discover_ai_calls


def render_report(repo: str, calls_per_day: int = 100, usage_json: str | None = None) -> str:
    sites = discover_ai_calls(repo)
    estimates = profile_cost(sites)
    scenario = scenario_monthly_cost(estimates, calls_per_day=calls_per_day)
    candidates = find_optimization_candidates(estimates)

    lines: list[str] = []
    lines.append(f"# CostPilot Report: {repo}")
    lines.append("")
    lines.append(f"- **Call sites detected:** {len(sites)}")
    lines.append(
        f"- **Priced sites:** {scenario['priced_sites']} / "
        f"**Unpriced:** {scenario['unpriced_sites']}"
    )
    lines.append(
        f"- **Monthly cost @ {calls_per_day} calls/day:** ${scenario['total_usd']} "
        f"_(scenario estimate, not a real bill)_"
    )
    lines.append("")

    if usage_json:
        obs = observed_cost_from_file(usage_json)
        lines.append("## Observed cost (L3: usage snapshot)")
        if obs["priced"]:
            lines.append(
                f"- **Actual spend (snapshot):** ${obs['total_usd']}: "
                f"{obs['calls']} calls / {obs['input_tokens']} input tokens / "
                f"{obs['output_tokens']} output tokens (model `{obs['model']}`)"
            )
        else:
            lines.append(f"- _{obs['reason']}_")
        lines.append("")

    if sites:
        lines.append("## Detected LLM calls")
        lines.append("")
        lines.append("| file:line | framework | model | source | conf | est. input chars |")
        lines.append("|---|---|---|---|---|---|")
        for s in sites:
            lines.append(
                f"| `{s.file}:{s.line}` | {s.framework.value} | "
                f"`{s.model or 'unknown'}` | {s.model_source} | "
                f"{s.confidence:.2f} | {s.estimated_input_chars} |"
            )
        lines.append("")

    lines.append("## Optimization recommendations")
    lines.append("")
    if not candidates:
        lines.append("_No actionable recommendations._")
    for c in candidates:
        icon = (
            "🔒 KEEP"
            if c.rejected
            else ("💰 DOWNGRADE" if c.kind == OptimizationKind.MODEL_DOWNGRADE else "✂️ COMPRESS")
        )
        saving = f" ~{c.expected_saving_ratio:.0%}" if c.expected_saving_ratio else ""
        lines.append(f"- **{icon}** `{c.label}`: {c.suggestion}{saving}")
        lines.append(f"  - {c.reason}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    repo = sys.argv[1]
    calls_per_day = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    usage_json = sys.argv[3] if len(sys.argv) > 3 else None
    print(render_report(repo, calls_per_day, usage_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
