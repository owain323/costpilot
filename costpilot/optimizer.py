"""P1-D5: find_optimization_candidates: optimization candidates with judgment.

Design principles (core review requirement):
- Cost optimization with judgment: seeing an expensive model does not mean blindly
  swapping to a cheaper one: judge whether "the task can tolerate a cheaper model";
  if not, KEEP (reject optimization, with a reason).
- Expected savings are only given when provable (based on public pricing
  comparison), never made up.
"""

from __future__ import annotations

from .models import CostEstimate, OptimizationCandidate, OptimizationKind
from .pricing import PRICING

# Models with a known cheaper alternative: model -> cheaper_model
# expected_saving_ratio uses a weighted unit price (input 0.7 / output 0.3),
# labeled as a pricing comparison rather than measured data.
_DOWNGRADE_MAP: dict[str, str] = {
    "gpt-4o": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "claude-sonnet-4-5": "claude-haiku-3-5",
    "claude-opus-4": "claude-sonnet-4-5",
}

# Conservative reject conditions: exceeding any of these means
# "cannot confirm the task tolerates a cheaper model" -> KEEP
_MAX_PROMPT_CHARS_FOR_DOWNGRADE = 1500  # long prompt = possibly a complex task
_MAX_OUTPUT_TOKENS_FOR_DOWNGRADE = 2048  # large output = generation-style task
_LONG_PROMPT_CHARS = 8000  # beyond this, prompt is considered compressible


def _weighted_price(model: str) -> float:
    p = PRICING[model]
    return 0.7 * p["input"] + 0.3 * p["output"]


def find_optimization_candidates(estimates: list[CostEstimate]) -> list[OptimizationCandidate]:
    """Generate optimization candidates for every call site (including reject-optimization);
    unpriced call sites are honestly flagged "hand to human"."""
    candidates: list[OptimizationCandidate] = []
    for est in estimates:
        site = est.site
        model = site.model or "unknown"

        if not est.is_priced:
            # Cannot price (unknown model / not in table): hand to human, do not force a suggestion
            candidates.append(
                OptimizationCandidate(
                    site=site,
                    kind=OptimizationKind.KEEP,
                    suggestion="No actionable optimization available (model is not priced)",
                    reason=(
                        f"model {model!r} is not in the pricing table; "
                        "cannot estimate cost: hand to human to confirm model and usage"
                    ),
                    expected_saving_ratio=None,
                    confidence=round(site.confidence * 0.8, 2),
                    rejected=True,
                )
            )
            continue

        # Judgment 1: can we downgrade to a cheaper model?
        cheaper = _DOWNGRADE_MAP.get(model)
        if cheaper and cheaper in PRICING:
            simple_enough = (
                site.estimated_input_chars <= _MAX_PROMPT_CHARS_FOR_DOWNGRADE
                and site.params.get("max_tokens", 0) <= _MAX_OUTPUT_TOKENS_FOR_DOWNGRADE
            )
            if simple_enough and site.model_source == "literal":
                saving = 1 - _weighted_price(cheaper) / _weighted_price(model)
                candidates.append(
                    OptimizationCandidate(
                        site=site,
                        kind=OptimizationKind.MODEL_DOWNGRADE,
                        suggestion=f"Change model from {model!r} to {cheaper!r}",
                        reason=(
                            f"short prompt ({site.estimated_input_chars} chars) + bounded output; "
                            "can tolerate a cheaper model; pricing comparison shows "
                            f"~{saving:.0%} expected saving"
                        ),
                        expected_saving_ratio=round(saving, 3),
                        confidence=round(site.confidence * 0.85, 2),
                        new_model=cheaper,
                    )
                )
            elif simple_enough:
                # Constant / config / env-derived model: patching it means
                # changing a shared definition that every caller sees. That is
                # a bigger decision than a one-line swap -> hand to a human,
                # never auto-approve (README: "Models that come from constants,
                # config, or env vars are left to a human").
                saving = 1 - _weighted_price(cheaper) / _weighted_price(model)
                candidates.append(
                    OptimizationCandidate(
                        site=site,
                        kind=OptimizationKind.KEEP,
                        suggestion="Keep the current model (hand to human)",
                        reason=(
                            f"model {model!r} is referenced via a shared "
                            f"{site.model_source} definition (not a literal here); "
                            "changing it affects every caller, so a human reviews "
                            f"it first (cheaper alternative {cheaper!r} exists, "
                            f"~{saving:.0%} price potential)"
                        ),
                        expected_saving_ratio=None,
                        confidence=round(site.confidence * 0.9, 2),
                        rejected=True,
                    )
                )
            else:
                candidates.append(
                    OptimizationCandidate(
                        site=site,
                        kind=OptimizationKind.KEEP,
                        suggestion="Keep the current model (no downgrade)",
                        reason=(
                            "cannot confirm the task tolerates a cheaper model: "
                            f"prompt {site.estimated_input_chars} chars / max_tokens "
                            f"{site.params.get('max_tokens', 'unset')}: conservatively "
                            "rejected, hand to human"
                        ),
                        expected_saving_ratio=None,
                        confidence=round(site.confidence * 0.9, 2),
                        rejected=True,
                    )
                )
        elif model and site.estimated_input_chars > _LONG_PROMPT_CHARS:
            # Judgment 2: long prompt -> compressible
            candidates.append(
                OptimizationCandidate(
                    site=site,
                    kind=OptimizationKind.PROMPT_COMPRESSION,
                    suggestion=(
                        f"Compress the prompt of the {model!r} call "
                        f"(currently ~{site.estimated_input_chars} chars)"
                    ),
                    reason=(
                        "prompt is long; compression reduces input-token cost; "
                        "human should confirm whether info is redundant"
                    ),
                    expected_saving_ratio=None,  # depends on content; no guessing
                    confidence=round(site.confidence * 0.7, 2),
                )
            )
        else:
            # No actionable optimization available: say so honestly, do not force one
            candidates.append(
                OptimizationCandidate(
                    site=site,
                    kind=OptimizationKind.KEEP,
                    suggestion="No actionable optimization available",
                    reason=(
                        f"model {model!r} has no known cheaper alternative, or task "
                        "characteristics are insufficient to judge; human may consider "
                        "caching or batched calls"
                    ),
                    expected_saving_ratio=None,
                    confidence=round(site.confidence * 0.8, 2),
                    rejected=True,
                )
            )

    candidates.sort(key=lambda c: (c.rejected, -(c.expected_saving_ratio or 0.0)))
    return candidates
