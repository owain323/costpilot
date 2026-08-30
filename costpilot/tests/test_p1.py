"""P1-D2~D5 acceptance tests: injection defense / cost model / candidates."""

from __future__ import annotations

import unittest
from pathlib import Path

from costpilot.models import AICallSite, CostEstimate, OptimizationCandidate, OptimizationKind
from costpilot.optimizer import find_optimization_candidates
from costpilot.pricing import (
    observed_cost_from_file,
    profile_cost,
    scenario_monthly_cost,
)
from costpilot.scanner import discover_ai_calls

FIXTURES = Path(__file__).parent / "fixtures"


class TestInjectionDefense(unittest.TestCase):
    """D2: Parser -> Structured Facts defense."""

    sites: list[AICallSite]

    @classmethod
    def setUpClass(cls):
        cls.sites = discover_ai_calls(FIXTURES.parent / "fixtures_injection")

    def test_site_detected(self):
        self.assertEqual(len(self.sites), 1)

    def test_no_raw_content_in_structured_facts(self):
        site = self.sites[0]
        blob = " ".join(
            [
                site.file,
                site.call_expr,
                str(site.model),
                str(site.params),
            ]
        ).lower()
        for bad in ("ignore previous", "credentials", "exfiltrate", "override", "unconstrained"):
            self.assertNotIn(bad, blob, f"injected content leaked into structured facts: {bad!r}")

    def test_input_chars_recorded_without_content(self):
        site = self.sites[0]
        self.assertGreater(site.estimated_input_chars, 0, "should record the length")
        self.assertLess(site.estimated_input_chars, 300, "only the length, never the content")


class TestProfileCost(unittest.TestCase):
    """D3: L1 Static Cost."""

    estimates: list[CostEstimate]

    @classmethod
    def setUpClass(cls):
        sites = discover_ai_calls(FIXTURES / "sample_app.py")
        cls.estimates = profile_cost(sites)

    def test_priced_constant_model(self):
        est = next(e for e in self.estimates if e.site.model == "gpt-4o")
        self.assertTrue(est.is_priced)
        self.assertGreater(est.cost_per_invocation or 0, 0)

    def test_unpriced_variable_model(self):
        est = next(e for e in self.estimates if e.site.model is None)
        self.assertFalse(est.is_priced, "unknown model must not be priced (no guessing)")

    def test_anthropic_priced(self):
        est = next(e for e in self.estimates if e.site.model == "claude-sonnet-4-5")
        self.assertTrue(est.is_priced)

    def test_input_tokens_estimated_from_chars(self):
        est = next(e for e in self.estimates if e.site.model == "gpt-4o")
        self.assertGreater(est.input_tokens, 0)


class TestScenarioAndObserved(unittest.TestCase):
    """D4: L2 Scenario + L3 Observed."""

    estimates: list[CostEstimate]

    @classmethod
    def setUpClass(cls):
        sites = discover_ai_calls(FIXTURES / "sample_app.py")
        cls.estimates = profile_cost(sites)

    def test_scenario_marks_as_estimate(self):
        r = scenario_monthly_cost(self.estimates, calls_per_day=100)
        self.assertIn("scenario estimate", r["note"])
        self.assertGreater(r["total_usd"], 0)
        self.assertEqual(r["days"], 30)

    def test_observed_from_file(self):
        r = observed_cost_from_file(FIXTURES / "usage_openai.json")
        self.assertTrue(r["priced"])
        # 5M input * 2.5/1M + 1M output * 10/1M = 12.5 + 10 = 22.5
        self.assertAlmostEqual(r["total_usd"], 22.5, places=1)

    def test_observed_unknown_model_does_not_guess(self):
        r = observed_cost_from_file(FIXTURES / "usage_unknown.json")
        self.assertFalse(r["priced"])


class TestOptimizationCandidates(unittest.TestCase):
    """D5: optimization candidates + reject-optimization judgment."""

    candidates: list[OptimizationCandidate]

    @classmethod
    def setUpClass(cls):
        sites = discover_ai_calls(FIXTURES / "sample_app.py")
        cls.candidates = find_optimization_candidates(profile_cost(sites))

    def test_short_prompt_downgrade_suggested(self):
        c = next(c for c in self.candidates if c.site.model == "gpt-4o")
        self.assertEqual(c.kind, OptimizationKind.MODEL_DOWNGRADE)
        self.assertFalse(c.rejected)
        self.assertIsNotNone(c.expected_saving_ratio)
        assert c.expected_saving_ratio is not None
        self.assertGreater(c.expected_saving_ratio, 0.9, "gpt-4o->mini should be ~94%")

    def test_unknown_model_never_forced(self):
        unknown = [c for c in self.candidates if c.site.model is None]
        self.assertTrue(unknown, "unpriced call sites must produce a candidate (hand to human)")
        for c in unknown:
            self.assertEqual(c.kind, OptimizationKind.KEEP)
            self.assertTrue(c.rejected, "must not suggest optimizations for unknown models")

    def test_all_candidates_have_reason(self):
        for c in self.candidates:
            self.assertTrue(c.reason)
            self.assertTrue(c.suggestion)


if __name__ == "__main__":
    unittest.main()
