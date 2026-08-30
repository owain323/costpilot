"""Core data structures for CostPilot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Framework(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_RESPONSES = "openai_responses"
    LANGCHAIN = "langchain"
    STRANDS = "strands"


@dataclass(frozen=True)
class AICallSite:
    """A single LLM API call site in a codebase (structured facts for the agent)."""

    file: str  # Path relative to the repo root
    line: int  # 1-based line number
    framework: Framework  # Detected framework
    model: str | None  # Model name extracted from code (only if resolvable)
    confidence: float  # 0-1 detection confidence
    call_expr: str  # Truncated call expression (for diagnostics/reporting)
    params: dict = field(default_factory=dict)  # Extracted params (max_tokens/temperature etc.)
    estimated_input_chars: int = (
        0  # Approx chars of messages (length only, never content: prompt-injection defense)
    )
    model_source: str = "literal"  # literal/constant/param_default/env_default/unknown

    @property
    def sort_key(self) -> tuple:
        return (self.file, self.line)


@dataclass(frozen=True)
class CostEstimate:
    """Per-invocation cost estimate (L1 Static Cost): provable estimates only,
    no fake precision."""

    site: AICallSite
    input_tokens: int  # Estimated input token count
    output_tokens: int  # Estimated output token count (max_tokens or default)
    price_per_1m_in: float | None  # Non-None only when the pricing table has a match
    price_per_1m_out: float | None
    cost_per_invocation: float | None  # Set when priced; None = cannot price (no guessing)

    @property
    def is_priced(self) -> bool:
        return self.cost_per_invocation is not None


class OptimizationKind(str, Enum):
    MODEL_DOWNGRADE = "model_downgrade"  # Switch to a cheaper model
    PROMPT_COMPRESSION = "prompt_compression"  # Compress the prompt
    CACHE_SUGGESTION = "cache_suggestion"  # Suggest adding a cache
    KEEP = "keep"  # Reject optimization (with judgment)


@dataclass(frozen=True)
class OptimizationCandidate:
    """An optimization candidate, including reject-optimization judgments."""

    site: AICallSite
    kind: OptimizationKind
    suggestion: str  # Concrete, actionable change
    reason: str  # Why (including the reason for rejecting optimization)
    expected_saving_ratio: float | None  # Expected saving (only when provable)
    confidence: float  # Suggestion confidence
    rejected: bool = False  # True = keep as-is (reject optimization)
    new_model: str | None = None  # Target model for MODEL_DOWNGRADE (patcher input)

    @property
    def label(self) -> str:
        return f"{self.site.file}:{self.site.line}"
