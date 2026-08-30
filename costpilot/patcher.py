"""P2b: conservative patch generation: simulate before proving, patch after proving.

Review requirement: "I changed this, tests passed, estimated cost dropped X%"
- never "I recommend this". A candidate only becomes a Change after simulation
proves the expected saving, and only MODEL_DOWNGRADE (exact model-string swap)
is executable in P2b. Everything else stays a recommendation.
"""

from __future__ import annotations

from .git_provider import Change
from .models import CostEstimate, OptimizationCandidate, OptimizationKind
from .pricing import get_price


def simulate_optimization(est: CostEstimate, new_model: str) -> dict:
    """Per-1K-call cost before/after switching to new_model.

    Provable via the public pricing table and the same token estimate used by L1.
    Returns priced=False when either model is unknown (no guessing).
    """
    price_new = get_price(new_model)
    if est.price_per_1m_in is None or price_new is None:
        return {"priced": False}
    before = (est.cost_per_invocation or 0.0) * 1000.0  # per 1K calls
    after = (
        est.input_tokens / 1_000_000 * price_new["input"]
        + est.output_tokens / 1_000_000 * price_new["output"]
    ) * 1000.0
    if before <= 0:
        return {"priced": False}
    return {
        "priced": True,
        "before_per_1k": round(before, 4),
        "after_per_1k": round(after, 4),
        "saving_ratio": round(1.0 - after / before, 4),
    }


def build_changes(
    candidates: list[OptimizationCandidate],
    estimates: dict[tuple[str, int], CostEstimate],
) -> list[tuple[OptimizationCandidate, Change, dict]]:
    """Turn provable DOWNGRADE candidates into concrete Changes.

    Policy: only non-rejected MODEL_DOWNGRADE with a known target model and a
    priced estimate become changes; simulation must confirm a real saving.
    Returns (candidate, change, simulation) triples.
    """
    changes: list[tuple[OptimizationCandidate, Change, dict]] = []
    for candidate in candidates:
        if candidate.rejected or candidate.kind != OptimizationKind.MODEL_DOWNGRADE:
            continue
        if candidate.site.model is None or candidate.new_model is None:
            continue
        if candidate.site.model_source != "literal":
            # constant/param_default/env_default resolve through shared definitions;
            # swapping them would affect every caller: too risky for an autonomous
            # patch. Hand to human instead.
            continue
        est = estimates.get((candidate.site.file, candidate.site.line))
        if est is None or not est.is_priced:
            continue
        sim = simulate_optimization(est, candidate.new_model)
        if not sim["priced"] or (sim.get("saving_ratio") or 0.0) <= 0.0:
            continue
        change = Change(
            file=candidate.site.file,
            line=candidate.site.line,
            old_model=candidate.site.model,
            new_model=candidate.new_model,
        )
        changes.append((candidate, change, sim))
    return changes
