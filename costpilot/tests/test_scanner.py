"""D1 acceptance tests: the discover_ai_calls scanner."""

from __future__ import annotations

import unittest
from pathlib import Path

from costpilot.models import AICallSite, Framework
from costpilot.scanner import discover_ai_calls

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURES_REAL = Path(__file__).parent / "fixtures_real"


class TestDiscoverAICalls(unittest.TestCase):
    sites: list[AICallSite]

    @classmethod
    def setUpClass(cls):
        cls.sites = discover_ai_calls(FIXTURES / "sample_app.py")

    def test_finds_all_expected_calls(self):
        self.assertEqual(len(self.sites), 5, f"expected 5 call sites, got: {self.sites}")

    def test_line_anchors_on_model_kwarg_not_call_node(self):
        """Regression: scanner.line must point to the `model=` kwarg line.

        Before the fix, scanner reported the enclosing Call node's lineno
        (which is one line *above* the model= kwarg for multi-line calls),
        and the demo's refine regex matched `model="gpt-4o"` against
        `model="gpt-4o-mini"` by prefix, collapsing three distinct sites
        onto one row in the UI with contradictory verdicts.
        """
        # Direct OpenAI SDK call (multi-line): model= is on line 14, not 13.
        s = next(
            s
            for s in self.sites
            if s.framework == Framework.OPENAI
            and s.model == "gpt-4o"
            and s.model_source == "literal"
        )
        self.assertEqual(s.line, 14, f"literal gpt-4o should anchor line 14, got {s.line}")

        # Constant model (multi-line): model= is on line 24, not 23.
        s = next(
            s
            for s in self.sites
            if s.framework == Framework.OPENAI and s.model_source == "constant"
        )
        self.assertEqual(s.line, 24, f"constant-model site should anchor line 24, got {s.line}")

        # Anthropic (multi-line): model= on line 34.
        s = next(s for s in self.sites if s.framework == Framework.ANTHROPIC)
        self.assertEqual(s.line, 34, f"anthropic site should anchor line 34, got {s.line}")

        # LangChain ChatOpenAI (single-line): model= on line 43, same line as call.
        s = next(
            s
            for s in self.sites
            if s.framework == Framework.LANGCHAIN and s.model_source == "literal"
        )
        self.assertEqual(s.line, 43, f"langchain literal should anchor line 43, got {s.line}")

    def test_no_two_sites_share_a_line(self):
        """Each model= row must produce exactly one site on that line.

        The previous bug collapsed three distinct sites onto line 43
        (APPROVE/KEEP/NOT PRICED). Lines and sites are now 1:1.
        """
        from collections import Counter

        line_counts = Counter(s.line for s in self.sites)
        for ln, n in line_counts.items():
            self.assertEqual(n, 1, f"line {ln} has {n} sites (expected 1)")

    def test_openai_sdk_with_constant_model(self):
        s = next(s for s in self.sites if s.model == "gpt-4o")
        self.assertEqual(s.framework, Framework.OPENAI)
        self.assertGreaterEqual(s.confidence, 0.9)
        self.assertIn("max_tokens", s.params)
        self.assertEqual(s.params["max_tokens"], 100)

    def test_model_from_module_constant(self):
        # MODEL_NAME = "gpt-4o" at module level -> resolvable with provenance
        s = next(s for s in self.sites if s.model == "gpt-4o" and s.model_source == "constant")
        self.assertEqual(s.framework, Framework.OPENAI)

    def test_anthropic_sdk(self):
        s = next(s for s in self.sites if s.model == "claude-sonnet-4-5")
        self.assertEqual(s.framework, Framework.ANTHROPIC)
        self.assertGreaterEqual(s.confidence, 0.9)

    def test_langchain_constructor(self):
        s = next(s for s in self.sites if s.model == "gpt-4o-mini")
        self.assertEqual(s.framework, Framework.LANGCHAIN)
        self.assertEqual(s.model, "gpt-4o-mini")

    def test_langchain_invoke_tracked_variable(self):
        s = next(s for s in self.sites if "invoke" in s.call_expr)
        self.assertEqual(s.framework, Framework.LANGCHAIN)
        # model is unknown here -> source penalty (0.10) lowers confidence below 0.8
        self.assertGreaterEqual(s.confidence, 0.7)

    def test_no_false_positive_on_helper(self):
        for s in self.sites:
            self.assertNotIn("helper", s.file)

    def test_has_call_expr_for_reporting(self):
        for s in self.sites:
            self.assertTrue(s.call_expr, "every call site must have a call_expr for reporting")


class TestRealWorldPatterns(unittest.TestCase):
    """P1.5: patterns found in real open-source repos during the benchmark."""

    sites: list[AICallSite]

    @classmethod
    def setUpClass(cls):
        cls.sites = discover_ai_calls(FIXTURES_REAL)

    def test_finds_all_real_pattern_calls(self):
        self.assertEqual(len(self.sites), 8, f"expected 8 sites, got: {self.sites}")

    def test_split_invocation_attribute_form(self):
        # self.client.create(...) where self.client was assigned .chat.completions
        s = next(s for s in self.sites if "gpt-4o-mini" in (s.model or ""))
        self.assertEqual(s.framework, Framework.OPENAI)
        self.assertGreaterEqual(s.confidence, 0.85)

    def test_split_invocation_name_form(self):
        # fallback_client.create(...) where fallback_client was assigned .chat.completions
        s = next(s for s in self.sites if "fallback_client" in s.call_expr)
        self.assertEqual(s.framework, Framework.OPENAI)

    def test_split_invocation_parse(self):
        # self.client.parse(...) on the same stored object
        s = next(s for s in self.sites if "self.client.parse" in s.call_expr)
        self.assertEqual(s.framework, Framework.OPENAI)

    def test_beta_parse_pattern(self):
        # client.beta.chat.completions.parse(...)
        s = next(s for s in self.sites if "beta" in s.call_expr)
        self.assertEqual(s.framework, Framework.OPENAI)
        self.assertGreaterEqual(s.confidence, 0.9)

    def test_model_from_module_constant_real(self):
        s = next(s for s in self.sites if "DEFAULT_MODEL" in s.call_expr)
        self.assertEqual(s.model, "gpt-4o-mini")
        self.assertEqual(s.model_source, "constant")

    def test_model_from_env_default(self):
        s = next(s for s in self.sites if "getenv" in s.call_expr)
        self.assertEqual(s.model, "gpt-4o")
        self.assertEqual(s.model_source, "env_default")
        self.assertLess(s.confidence, 0.96, "env-derived model must have reduced confidence")

    def test_model_from_param_default(self):
        s = next(s for s in self.sites if s.model == "claude-haiku-3-5")
        self.assertEqual(s.framework, Framework.ANTHROPIC)
        self.assertEqual(s.model_source, "param_default")

    def test_unresolvable_model_stays_unknown(self):
        s = next(s for s in self.sites if "pick_model" in s.call_expr)
        self.assertIsNone(s.model, "unresolvable model must never be guessed")
        self.assertEqual(s.model_source, "unknown")


if __name__ == "__main__":
    unittest.main()
