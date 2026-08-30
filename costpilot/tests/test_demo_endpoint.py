"""End-to-end acceptance for the web demo's analysis entry point.

The web demo renders four facts from the analyze_code() output:
  - call_sites
  - priced
  - unknown_pricing
  - summary.saving_ratio
Plus a "no provable downgrade" / "unpriced left untouched" caveat line.
These tests pin the contract so that any UI that maps those values to a
visible label cannot drift from what the engine actually produced.
"""

from __future__ import annotations

import unittest

from demo.app import analyze_code


class DemoEndpointContractTest(unittest.TestCase):
    """Pin the JSON shape and honest reporting of the live scan demo."""

    def test_empty_input(self):
        r = analyze_code("")
        self.assertEqual(r["call_sites"], 0)
        self.assertEqual(r["priced"], 0)
        self.assertEqual(r["unknown_pricing"], 0)
        self.assertIsNone(r["summary"]["saving_ratio"])
        # No fake success: a demo page must not be able to claim "downgrades N"
        # when there are zero call sites.
        self.assertEqual(len(r["candidates"]), 0)

    def test_javascript_input_is_rejected_with_clear_error(self):
        r = analyze_code("const x = 1;")
        self.assertIn("error", r)
        self.assertIn("Python", r["error"])
        # Front-end uses this exact key to render the red error banner;
        # if the engine silently returns a scan, the UI will show a "0 calls"
        # result and hide the error -- so this must stay an error path.
        self.assertNotIn("candidates", r)

    def test_python_with_provable_downgrade(self):
        code = (
            "from openai import OpenAI\n"
            "c = OpenAI()\n"
            "c.chat.completions.create(model='gpt-4o', messages=[])\n"
        )
        r = analyze_code(code)
        self.assertNotIn("error", r)
        self.assertGreaterEqual(r["call_sites"], 1)
        self.assertGreaterEqual(r["priced"], 1)
        self.assertIsNotNone(r["summary"]["saving_ratio"])
        self.assertGreater(r["summary"]["saving_ratio"], 0.0)
        # Must include the downgraded site so the sidebar can render an APPROVE card.
        downs = [c for c in r["candidates"] if c["decision"] == "model_downgrade"]
        self.assertGreater(len(downs), 0)
        # Every downgrade must carry the cost evidence the summary uses.
        for d in downs:
            self.assertIn("before_per_1k", d)
            self.assertIn("after_per_1k", d)
            self.assertIn("expected_saving_ratio", d)
        # pricing_basis must list every model in this scan with public prices.
        pb = r["pricing_basis"]
        self.assertEqual(pb["snapshot"], "2026-08")
        self.assertIn("models", pb)
        models = {m["model"] for m in pb["models"]}
        self.assertIn("gpt-4o", models)
        priced = [m for m in pb["models"] if m["priced"]]
        self.assertGreater(len(priced), 0)
        for m in priced:
            self.assertIsNotNone(m["input_per_1m"])
            self.assertIsNotNone(m["output_per_1m"])

    def test_python_no_provable_downgrade(self):
        # gpt-4o-mini has no cheaper entry in the downgrade map.
        code = (
            "from openai import OpenAI\n"
            "c = OpenAI()\n"
            "c.chat.completions.create(model='gpt-4o-mini', messages=[])\n"
        )
        r = analyze_code(code)
        self.assertNotIn("error", r)
        self.assertGreaterEqual(r["priced"], 1)
        downs = [c for c in r["candidates"] if c["decision"] == "model_downgrade"]
        self.assertEqual(len(downs), 0)
        # The UI must be able to say "no provable downgrade" honestly.
        kept = [c for c in r["candidates"] if c["decision"] != "model_downgrade"]
        self.assertGreaterEqual(len(kept), 1)

    def test_shared_constant_model_never_auto_approved(self):
        """Regression: a model referenced via a shared constant must NOT show
        APPROVE — it surfaces as HUMAN/KEEP because patching the shared
        definition affects every caller (README promise: constants left to
        a human). Previously the optimizer emitted model_downgrade for any
        priced model regardless of source, so `model=MODEL_NAME` (constant
        "gpt-4o") was shown as a downgrade suggestion on line 24."""
        code = (
            "from openai import OpenAI\n"
            "MODEL_NAME = 'gpt-4o'\n"
            "c = OpenAI()\n"
            "c.chat.completions.create(model=MODEL_NAME, messages=[])\n"
        )
        r = analyze_code(code)
        self.assertNotIn("error", r)
        # The constant-derived site exists, is priced, but must not be an auto-downgrade.
        downs = [c for c in r["candidates"] if c["decision"] == "model_downgrade"]
        self.assertEqual(
            len(downs), 0, f"constant-derived model must not be APPROVED: {r['candidates']}"
        )
        # And the candidate must carry the human-review signal for the UI pill.
        site = next(c for c in r["candidates"] if c["source"] == "constant")
        self.assertEqual(site["decision"], "KEEP")
        self.assertIn("human", site["suggestion"].lower())
        # Saving must be None here: no literal site contributes.
        self.assertIsNone(r["summary"]["saving_ratio"])


if __name__ == "__main__":
    unittest.main()
