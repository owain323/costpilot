"""P2b-agent acceptance tests: decision toolbox, context building,
and the agent-driven loop (deterministic factory, no LLM in CI)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from costpilot.agent_loop import (
    CandidateContext,
    DecisionToolbox,
    build_contexts,
    rule_based_decisions_factory,
)
from costpilot.optimizer import find_optimization_candidates
from costpilot.pricing import profile_cost
from costpilot.runner import run_agentic_loop
from costpilot.scanner import discover_ai_calls

FIXTURES = Path(__file__).parent / "fixtures"


def _make_repo(src: Path) -> Path:
    """Throwaway git repo with one file committed (same helper as test_p2b)."""
    tmp = Path(tempfile.mkdtemp(prefix="costpilot_agent_"))
    shutil.copy(src, tmp / src.name)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=str(tmp), capture_output=True, text=True, check=True)

    git("init", "-q")
    git("config", "user.email", "test@costpilot.local")
    git("config", "user.name", "CostPilot Test")
    git("add", "--", src.name)
    for _ in range(3):
        git("commit", "-m", "initial")
        try:
            git("rev-parse", "HEAD")
            return tmp
        except subprocess.CalledProcessError:
            time.sleep(0.3)
    raise RuntimeError("could not create initial commit")


def _sites_estimates_candidates(repo: Path):
    sites = discover_ai_calls(repo)
    estimates = profile_cost(sites)
    est_map = {(e.site.file, e.site.line): e for e in estimates}
    candidates = find_optimization_candidates(estimates)
    return est_map, candidates


class TestDecisionToolbox(unittest.TestCase):
    """The only surface the agent may touch."""

    def setUp(self):
        self.contexts = [
            CandidateContext(
                id="c0",
                file="a.py",
                line=13,
                framework="openai",
                model="gpt-4o",
                model_source="literal",
                confidence=0.96,
                suggestion="Change model from 'gpt-4o' to 'gpt-4o-mini'",
                reason="short prompt; can tolerate a cheaper model",
                before_per_1k=4.02,
                after_per_1k=0.86,
                saving_ratio=0.785,
            )
        ]
        self.box = DecisionToolbox(self.contexts)

    def test_list_and_get(self):
        listing = json.loads(self.box.list_candidates())
        self.assertEqual(len(listing), 1)
        self.assertEqual(listing[0]["id"], "c0")
        detail = json.loads(self.box.get_candidate("c0"))
        self.assertEqual(detail["model"], "gpt-4o")
        self.assertEqual(detail["saving_ratio"], 0.785)

    def test_decide_valid_actions(self):
        self.assertIn("ok", json.loads(self.box.decide("c0", "approve", "safe")))
        self.assertEqual(self.box.decisions["c0"].action, "approve")

    def test_decide_rejects_invalid_action(self):
        r = json.loads(self.box.decide("c0", "rm -rf"))
        self.assertIn("error", r)
        self.assertNotIn("c0", self.box.decisions)

    def test_decide_rejects_unknown_candidate(self):
        r = json.loads(self.box.decide("c999", "approve"))
        self.assertIn("error", r)

    def test_decide_once_only(self):
        self.box.decide("c0", "approve")
        r = json.loads(self.box.decide("c0", "keep"))
        self.assertIn("error", r)

    def test_finish_marks_complete(self):
        self.box.finish()
        self.assertTrue(self.box.finished)


class TestAgenticLoop(unittest.TestCase):
    """End-to-end agent-driven loop with a deterministic decider (no LLM)."""

    def test_agent_approves_and_changes_land(self):
        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            result = run_agentic_loop(repo, decisions_factory=rule_based_decisions_factory())
            self.assertTrue(result["changed"], result.get("reason"))
            artifact = result["artifact"]
            assert artifact is not None
            self.assertEqual(artifact.changed_files, ["sample_app.py"])
            self.assertIn("agent-reviewed", artifact.reasoning)
            self.assertGreater(artifact.expected_saving_ratio or 0.0, 0.5)
            changed = (repo / "sample_app.py").read_text(encoding="utf-8")
            self.assertNotIn('model="gpt-4o"', changed)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_agent_keeps_everything_changes_nothing(self):
        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            # decider that keeps everything -> nothing executed
            keep_all = rule_based_decisions_factory(rule=lambda ctx: "keep")
            result = run_agentic_loop(repo, decisions_factory=keep_all)
            self.assertFalse(result["changed"])
            self.assertIn("no approved optimization", result["reason"])
            changed = (repo / "sample_app.py").read_text(encoding="utf-8")
            self.assertIn('model="gpt-4o"', changed, "file must remain untouched")
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_constant_derived_model_blocked_by_rule_layer(self):
        """Re-audit P1-6: agent approval alone is not enough.

        sample_app.py has a gpt-4o call site that resolves through a module
        constant (MODEL_NAME). Approving everything must still leave that site
        untouched: the rule allowlist only permits literal model strings.
        """
        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            approve_all = rule_based_decisions_factory(rule=lambda ctx: "approve")
            result = run_agentic_loop(repo, decisions_factory=approve_all)
            self.assertTrue(result["changed"], result.get("reason"))
            changed = (repo / "sample_app.py").read_text(encoding="utf-8")
            # literal gpt-4o sites were downgraded...
            self.assertNotIn('model="gpt-4o"', changed)
            # ...but the constant-derived site still reads MODEL_NAME (untouched)
            self.assertIn("MODEL_NAME", changed)
            artifact = result["artifact"]
            assert artifact is not None
            self.assertEqual(artifact.changed_files, ["sample_app.py"])
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_build_contexts_has_sim_for_provable_only(self):
        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            est_map, candidates = _sites_estimates_candidates(repo)
            contexts = build_contexts(candidates, est_map)
            downgrade_ctx = next(c for c in contexts if "gpt-4o-mini" in c.suggestion)
            self.assertIsNotNone(downgrade_ctx.saving_ratio)
            self.assertGreater(downgrade_ctx.saving_ratio or 0.0, 0.9)
            # KEEP candidates carry no simulation numbers
            keep_ctx = next(
                c for c in contexts if c.suggestion == "No actionable optimization available"
            )
            self.assertIsNone(keep_ctx.saving_ratio)
        finally:
            shutil.rmtree(repo, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
