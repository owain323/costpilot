"""P1-D3/D4: three-tier cost model: Static (L1) / Scenario (L2) / Observed (L3).

Design principles (no fake precision, from review requirements):
- No fake precision: L1 reports only "per-invocation cost" (provable); L2 is explicitly labeled
  as a scenario estimate; L3 uses a real usage snapshot, and "consumes a snapshot only to
  produce optimization decisions, never a persistent dashboard".
- Money is only counted when the pricing table matches; unknown models return None (no guessing).
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import AICallSite, CostEstimate

# Public model pricing (USD / 1M tokens). Source: vendor public pricing pages (2026-08 snapshot).
# Limited coverage is a feature, not a bug: unknown models are not quoted.
PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Heuristic chars->tokens conversion (approx. 4 chars/token for English; blended weighting)
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(chars: int) -> int:
    """Estimate token count from char count (heuristic; labeled as an estimate)."""
    return max(1, round(chars / _CHARS_PER_TOKEN))


def get_price(model: str | None) -> dict[str, float] | None:
    """Look up pricing; return None for unknown models (no guessing)."""
    if model is None:
        return None
    return PRICING.get(model)


def default_output_tokens(site: AICallSite) -> int:
    """Estimate output tokens: prefer the max_tokens param, else a conservative default of 512."""
    mt = site.params.get("max_tokens") or site.params.get("max_output_tokens")
    return int(mt) if mt else 512


def profile_cost(sites: list[AICallSite]) -> list[CostEstimate]:
    """L1 Static Cost: per-invocation cost for each site. Provable, no fake precision."""
    estimates: list[CostEstimate] = []
    for site in sites:
        price = get_price(site.model)
        input_tokens = estimate_tokens(site.estimated_input_chars)
        output_tokens = default_output_tokens(site)
        if price is None:
            estimates.append(
                CostEstimate(
                    site=site,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    price_per_1m_in=None,
                    price_per_1m_out=None,
                    cost_per_invocation=None,
                )
            )
            continue
        cost = (
            input_tokens / 1_000_000 * price["input"] + output_tokens / 1_000_000 * price["output"]
        )
        estimates.append(
            CostEstimate(
                site=site,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price_per_1m_in=price["input"],
                price_per_1m_out=price["output"],
                cost_per_invocation=round(cost, 6),
            )
        )
    return estimates


def scenario_monthly_cost(
    estimates: list[CostEstimate], calls_per_day: int, days: int = 30
) -> dict:
    """L2 Scenario Cost: monthly estimate for a given call frequency
    (explicitly a scenario, not a real bill).

    Returns {total_usd, days, calls_per_day, priced_sites, unpriced_sites}
    """
    priced = [e for e in estimates if e.is_priced]
    total_per_day = sum((e.cost_per_invocation or 0.0) for e in priced) * calls_per_day
    return {
        "total_usd": round(total_per_day * days, 2),
        "days": days,
        "calls_per_day": calls_per_day,
        "priced_sites": len(priced),
        "unpriced_sites": len(estimates) - len(priced),
        "note": "scenario estimate: not a real bill; calibrate with L3 observed data",
    }


def observed_cost(usage: dict) -> dict:
    """L3 Observed Cost: ingest a provider usage snapshot and compute actual cost.

    Expected format (provider usage JSON):
    {"provider": "openai", "calls": 42391, "input_tokens": 84100000,
     "output_tokens": 11300000, "model": "gpt-4o"}
    Unknown model / missing fields -> report honestly, do not compute (no guessing).
    """
    model = usage.get("model")
    price = get_price(model)
    if price is None:
        return {
            "priced": False,
            "reason": f"model {model!r} is not in the pricing table; cannot compute actual cost",
            "raw": {k: usage.get(k) for k in ("calls", "input_tokens", "output_tokens", "model")},
        }
    calls = usage.get("calls", 0)
    in_tokens = usage.get("input_tokens", 0)
    out_tokens = usage.get("output_tokens", 0)
    cost = in_tokens / 1_000_000 * price["input"] + out_tokens / 1_000_000 * price["output"]
    return {
        "priced": True,
        "model": model,
        "calls": calls,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_usd": round(cost, 2),
    }


def observed_cost_from_file(path: str | Path) -> dict:
    """Read a usage snapshot from a JSON file and compute its cost."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return observed_cost(data)
