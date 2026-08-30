"""P2b: the optimization loops: deterministic and agent-driven, both safely
executing through the same policy-gated closed loop.

Discover -> Decide -> Simulate -> Patch -> Validate -> Branch -> Commit
         -> PR Artifact -> Human Review.

Security contract (approved decision):
- The agent (and these loops) only ever calls the GitProvider tool API and the
  allowlisted validation runner. There is no arbitrary bash/git path.
- Validation failure aborts the loop: no commit, no change request.
- The agent-driven loop adds a second gate: a change is only executed when the
  agent approves it AND the rule-based allowlist accepts it.
"""

from __future__ import annotations

from pathlib import Path

from .agent_loop import DecisionRecord, build_contexts
from .git_provider import GitProvider, LocalGitProvider, PRArtifact
from .optimizer import find_optimization_candidates
from .patcher import build_changes
from .pricing import profile_cost
from .scanner import discover_ai_calls
from .validator import validate_patch


def _approve_ids(
    decisions: dict[str, DecisionRecord],
) -> set[str]:
    return {d.candidate_id for d in decisions.values() if d.is_approve}


def _execute_approved(
    repo: Path,
    provider: GitProvider,
    branch: str,
    triples: list,
    candidates: list,
    decision_note: str,
) -> dict:
    """Shared execution half: patch -> validate -> commit -> PR artifact."""
    if not triples:
        return {
            "changed": False,
            "artifact": None,
            "candidates": candidates,
            "reason": "no approved optimization; everything else kept or handed to human",
        }

    changes = [c for _, c, _ in triples]
    provider.create_branch(repo, branch)
    provider.apply_changes(repo, changes)
    changed_files = sorted({c.file for c in changes})

    # Validate: failure aborts (no commit, no PR); "no-tests" is neutral.
    # On failure the patched files are *actually* reverted (honesty contract).
    validation = validate_patch(repo, changed_files)
    if validation.get("syntax") != "ok" or validation.get("tests") == "failed":
        provider.revert_changes(repo, changed_files)
        return {
            "changed": False,
            "artifact": None,
            "candidates": candidates,
            "validation": validation,
            "reason": "validation failed; changes reverted, no commit",
        }

    sha = provider.commit(repo, f"costpilot: optimize {len(changes)} call site(s)", changed_files)
    before_total = sum(sim["before_per_1k"] for _, _, sim in triples)
    after_total = sum(sim["after_per_1k"] for _, _, sim in triples)
    reasoning = "; ".join(f"{c.site.file}:{c.site.line} {c.suggestion}" for c, _, _ in triples)
    risk = (
        "Savings are price-saving potential from public pricing (not realized cost). "
        "The downgrade satisfies the explicit safety criteria (short prompt, bounded "
        "output, literal model source, repository tests pass). Not evaluated: output "
        "quality, refusal rate, JSON schema stability, tool-calling behavior, latency, "
        "and token-usage variance of the target model. These are the human's checklist "
        "when reviewing and merging the diff."
    )
    artifact = PRArtifact(
        branch=branch,
        commit=sha,
        changed_files=changed_files,
        before_cost=round(before_total, 4),
        after_cost=round(after_total, 4),
        expected_saving_ratio=(
            round(1.0 - after_total / before_total, 4) if before_total > 0 else None
        ),
        validation=validation,
        reasoning=f"{decision_note}\n\n{reasoning}",
        risk=risk,
        decision="MODEL_DOWNGRADE",
    )
    artifact_path = provider.create_change_request(repo, artifact)
    return {
        "changed": True,
        "artifact": artifact,
        "artifact_path": artifact_path,
        "candidates": candidates,
        "validation": validation,
    }


def run_optimization_loop(
    repo: Path,
    provider: GitProvider | None = None,
    branch: str = "costpilot/optimize",
) -> dict:
    """Deterministic loop (rule-based decide, no LLM)."""
    provider = provider or LocalGitProvider()

    sites = discover_ai_calls(repo)
    estimates = profile_cost(sites)
    est_map = {(e.site.file, e.site.line): e for e in estimates}
    candidates = find_optimization_candidates(estimates)

    triples = build_changes(candidates, est_map)
    if not triples:
        return {
            "changed": False,
            "artifact": None,
            "candidates": candidates,
            "reason": "no provable optimization; everything else kept or handed to human",
        }
    return _execute_approved(
        repo,
        provider,
        branch,
        triples,
        candidates,
        decision_note="decision: deterministic rule-based review",
    )


def run_agentic_loop(
    repo: Path,
    decisions_factory,
    provider: GitProvider | None = None,
    branch: str = "costpilot/optimize",
) -> dict:
    """Agent-driven loop: an LLM reviews structured candidate facts and approves
    changes; execution still passes through the same policy-gated layer.

    Two gates: agent approval + rule-based allowlist (build_changes).
    """
    provider = provider or LocalGitProvider()

    sites = discover_ai_calls(repo)
    estimates = profile_cost(sites)
    est_map = {(e.site.file, e.site.line): e for e in estimates}
    candidates = find_optimization_candidates(estimates)

    contexts = build_contexts(candidates, est_map)
    if not contexts:
        return {
            "changed": False,
            "artifact": None,
            "candidates": candidates,
            "reason": "no candidates to review",
        }

    decisions = decisions_factory(contexts)
    approved = _approve_ids(decisions)

    # gate 1: agent approval; gate 2: rule-based allowlist
    triples = build_changes(candidates, est_map)
    triples = [t for t in triples if f"c{candidates.index(t[0])}" in approved]
    notes = "; ".join(f"{candidate_id}={d.action}" for candidate_id, d in sorted(decisions.items()))
    return _execute_approved(
        repo,
        provider,
        branch,
        triples,
        candidates,
        decision_note=f"decision: agent-reviewed ({notes or 'no decisions'})",
    )
