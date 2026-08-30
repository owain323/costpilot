"""Frontend-safe contract: malicious-looking model names or messages must reach
the JSON output as plain strings (the UI's esc() is responsible for the final
HTML encoding; the engine must not pre-mangle or interpret them).
"""

from __future__ import annotations

import unittest

from demo.app import analyze_code


class XSSContractTest(unittest.TestCase):
    """Pin that the engine treats user-provided strings as opaque data."""

    def test_malicious_model_name_passes_through_untouched(self):
        code = (
            "from openai import OpenAI\n"
            "c = OpenAI()\n"
            "c.chat.completions.create(model='<img src=x onerror=alert(1)>', messages=[])\n"
        )
        r = analyze_code(code)
        self.assertNotIn("error", r)
        # The unpriced model name should appear verbatim in the candidate row;
        # the UI is responsible for HTML-encoding it via esc() before rendering.
        rows = r["candidates"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "<img src=x onerror=alert(1)>")

    def test_malicious_suggestion_is_only_text(self):
        # A model that triggers the downgrade path. The suggestion string comes
        # from the optimizer (controlled source), but the rendered label is the
        # model's quoted value -- which is user-controlled and must be plain text.
        code = (
            "from openai import OpenAI\n"
            "c = OpenAI()\n"
            'c.chat.completions.create(model="gpt-4o", messages=[])\n'
        )
        r = analyze_code(code)
        self.assertNotIn("error", r)
        for row in r["candidates"]:
            for field in ("model", "suggestion", "decision"):
                self.assertIsInstance(row[field], str)

    def test_html_meta_in_malformed_code_does_not_infect_error(self):
        # A syntax-error with HTML-like content must surface as a plain error
        # string, not be interpreted by the engine.
        code = "def broken(</script><img src=x onerror=alert(1)>:"
        self.assertIn("error", analyze_code(code))


if __name__ == "__main__":
    unittest.main()
