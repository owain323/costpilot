"""P2b-agent live demo: full agentic loop with a real local LLM (Strands + Ollama).

Reproduces the closed loop with real model-driven decisions:
  discover -> profile -> rule candidates -> LLM review (structured facts only)
  -> approve/keep/ask_human -> policy-gated execute -> commit -> PR artifact.

Usage:
    python benchmark/demo_agentic.py [model_id]

Requires: Ollama running on localhost:11434 with the model pulled.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from costpilot.agent_loop import build_contexts, strands_decisions_factory
from costpilot.optimizer import find_optimization_candidates
from costpilot.pricing import profile_cost
from costpilot.runner import run_agentic_loop
from costpilot.scanner import discover_ai_calls

HERE = Path(__file__).parent
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"

# A tiny test suite shipped into the demo repo so validation shows real evidence:
# "Ran N tests, OK" instead of "NO TESTS RAN".
_TEST_SAMPLE_APP = '''\
"""Smoke tests for the demo repo."""

import sys
import unittest
from unittest import mock

# The demo repo has no external SDK packages installed; provide lightweight stubs
# so ``import sample_app`` succeeds without openai/anthropic/langchain.
for _name in ("openai", "anthropic", "langchain_openai"):
    _mod = type(sys)("fake")
    _mod.OpenAI = mock.Mock
    _mod.Anthropic = mock.Mock
    _mod.ChatOpenAI = mock.Mock
    sys.modules[_name] = _mod

import sample_app


class TestSampleApp(unittest.TestCase):
    def test_helper(self):
        self.assertEqual(sample_app.helper(3), 6)

    @mock.patch("sample_app.client.chat.completions.create")
    def test_classify_returns_string(self, mock_create):
        mock_create.return_value.choices[0].message.content = "classified"
        self.assertEqual(sample_app.classify("hi"), "classified")


if __name__ == "__main__":
    unittest.main()
'''


def _make_demo_repo() -> Path:
    demo = HERE / "demo_agentic_repo"
    if demo.exists():
        shutil.rmtree(demo)
    demo.mkdir(parents=True)
    src = HERE.parent / "costpilot" / "tests" / "fixtures" / "sample_app.py"
    shutil.copy(src, demo / "sample_app.py")
    (demo / "test_sample_app.py").write_text(_TEST_SAMPLE_APP, encoding="utf-8")

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=str(demo), capture_output=True, text=True, check=True
        )

    git("init", "-q")
    git("config", "user.email", "demo@costpilot.local")
    git("config", "user.name", "CostPilot Demo")
    git("add", "--", "sample_app.py", "test_sample_app.py")
    git("commit", "-m", "initial")
    return demo


def main() -> int:
    print(f"== P2b-agent live demo (model: {MODEL_ID}) ==")
    demo = _make_demo_repo()

    # Show the structured facts the agent will see (never raw source)
    sites = discover_ai_calls(demo)
    estimates = profile_cost(sites)
    est_map = {(e.site.file, e.site.line): e for e in estimates}
    candidates = find_optimization_candidates(estimates)
    contexts = build_contexts(candidates, est_map)
    print(f"\ncandidates: {len(contexts)}")
    for c in contexts:
        saving = f"{c.saving_ratio:.0%}" if c.saving_ratio is not None else "-"
        print(
            f"  {c.id} {c.file}:{c.line} {c.model} ({c.model_source}) saving={saving}"
        )

    t0 = time.time()
    result = run_agentic_loop(
        demo, decisions_factory=strands_decisions_factory(model_id=MODEL_ID)
    )
    print(f"\nagentic loop took {time.time() - t0:.0f}s")
    print(f"changed: {result['changed']} | reason: {result.get('reason')}")

    artifact = result.get("artifact")
    if artifact is not None:
        print(f"branch={artifact.branch} commit={artifact.commit[:8]}")
        print(
            f"files={artifact.changed_files} saving={artifact.expected_saving_ratio:.0%}"
        )
        print(f"validation={artifact.validation}")
        print(f"reasoning: {artifact.reasoning[:200]}")
    changed = (demo / "sample_app.py").read_text(encoding="utf-8")
    print(f"\nfile changed: gpt-4o-mini present = {'gpt-4o-mini' in changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
