"""P2b acceptance tests: policy gate + local git closed loop + PR artifact."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from costpilot.git_provider import Change, LocalGitProvider, _assert_git_args
from costpilot.runner import run_optimization_loop

FIXTURES = Path(__file__).parent / "fixtures"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout.strip()


def _init_repo(files: dict[str, str]) -> Path:
    """Create a throwaway git repo with the given files committed."""
    tmp = Path(tempfile.mkdtemp(prefix="costpilot_p2b_"))
    for name, content in files.items():
        (tmp / name).write_text(content, encoding="utf-8")
    _git(tmp, "init", "-q")
    _git(tmp, "config", "user.email", "test@costpilot.local")
    _git(tmp, "config", "user.name", "CostPilot Test")
    _git(tmp, "add", "--", *files)
    # Windows filesystem timing can occasionally swallow a commit; verify + retry
    for _ in range(3):
        _git(tmp, "commit", "-m", "initial")
        try:
            _git(tmp, "rev-parse", "HEAD")
            return tmp
        except subprocess.CalledProcessError:
            time.sleep(0.3)
    raise RuntimeError("could not create initial commit")


def _make_repo(src: Path) -> Path:
    return _init_repo({src.name: src.read_text(encoding="utf-8")})


def _git_can_stage_symlink(repo: Path, link: Path) -> bool:
    """True when git on this platform can see and stage a symlink.

    Some setups (Windows with developer mode) allow os.symlink() while
    git itself cannot stage the link, so the symlink attack vector does
    not exist there and the rejection tests cannot run meaningfully.
    """
    probe = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", str(link)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return probe.returncode == 0 and bool(probe.stdout.strip())


class TestPolicyGate(unittest.TestCase):
    """The agent/system boundary: allowlist enforcement."""

    def test_branch_name_must_have_prefix(self):
        provider = LocalGitProvider()
        tmp = Path(tempfile.mkdtemp(prefix="costpilot_gate_"))
        with self.assertRaises(PermissionError):
            provider.create_branch(tmp, "feature/foo")

    def test_commit_message_must_have_prefix(self):
        provider = LocalGitProvider()
        tmp = Path(tempfile.mkdtemp(prefix="costpilot_gate_"))
        with self.assertRaises(PermissionError):
            provider.commit(tmp, "fix stuff", [])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_git_subcommands_allowlist(self):
        with self.assertRaises(PermissionError):
            _assert_git_args(("push",))
        with self.assertRaises(PermissionError):
            _assert_git_args(("reset", "--hard"))
        with self.assertRaises(PermissionError):
            _assert_git_args(("checkout", "-b", "x", "--force"))
        with self.assertRaises(PermissionError):
            _assert_git_args(("clean", "-fd"))
        _assert_git_args(("checkout", "-b", "costpilot/x"))  # allowed

    def test_apply_changes_refuses_wrong_line(self):
        provider = LocalGitProvider()
        tmp = Path(tempfile.mkdtemp(prefix="costpilot_gate_"))
        (tmp / "a.py").write_text("model = 'gpt-4o'\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            provider.apply_changes(
                tmp, [Change(file="a.py", line=5, old_model="gpt-4o", new_model="gpt-4o-mini")]
            )
        shutil.rmtree(tmp, ignore_errors=True)


class TestHonestyFixes(unittest.TestCase):
    """P0-6: real revert on validation failure + precise staging on commit."""

    def _git(self, repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
        ).stdout.strip()

    def test_revert_changes_restores_worktree(self):
        repo = _init_repo({"a.py": "x = 'gpt-4o'\n"})
        try:
            (repo / "a.py").write_text("x = 'gpt-4o-mini'\n", encoding="utf-8")
            LocalGitProvider().revert_changes(repo, ["a.py"])
            self.assertEqual((repo / "a.py").read_text(encoding="utf-8"), "x = 'gpt-4o'\n")
            self.assertEqual(self._git(repo, "status", "--porcelain"), "")
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_validation_failure_actually_reverts(self):
        from unittest import mock

        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            with mock.patch(
                "costpilot.runner.validate_patch",
                return_value={"syntax": "ok", "tests": "failed", "summary": "broken"},
            ):
                result = run_optimization_loop(repo)
            self.assertFalse(result["changed"])
            self.assertIn("reverted", result["reason"])
            # the working tree must be clean after the revert (not left dirty)
            self.assertEqual(self._git(repo, "status", "--porcelain"), "")
            changed = (repo / "sample_app.py").read_text(encoding="utf-8")
            self.assertIn('model="gpt-4o"', changed, "patch must have been undone")
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_commit_stages_only_patched_files(self):
        provider = LocalGitProvider()
        repo = _init_repo({"a.py": "x = 1\n"})
        try:
            (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # patched file
            (repo / "unrelated.py").write_text("y = 9\n", encoding="utf-8")  # user's own file
            provider.commit(repo, "costpilot: test precise staging", ["a.py"])
            # unrelated.py must NOT be in the commit (still untracked)
            self.assertEqual(self._git(repo, "status", "--porcelain"), "?? unrelated.py")
            tree_files = self._git(repo, "show", "--name-only", "--format=", "HEAD").strip()
            self.assertIn("a.py", tree_files)
            self.assertNotIn("unrelated.py", tree_files)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_commit_rejects_unsafe_paths(self):
        provider = LocalGitProvider()
        tmp = Path(tempfile.mkdtemp(prefix="costpilot_gate_"))
        try:
            # Use a real absolute path recognized on both POSIX and Windows.
            abs_path = str(tmp / "evil.py")
            self.assertTrue(Path(abs_path).is_absolute())
            with self.assertRaises(PermissionError):
                provider.commit(tmp, "costpilot: x", [abs_path])
            with self.assertRaises(PermissionError):
                provider.commit(tmp, "costpilot: x", ["../outside.py"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_commit_rejects_symlink_to_outside(self):
        provider = LocalGitProvider()
        repo = _init_repo({"real.py": 'model = "gpt-4o"\n'})
        try:
            outside = repo.parent / "outside.py"
            outside.write_text('OUTSIDE = "original"\n', encoding="utf-8")
            link = repo / "evil.py"
            try:
                os.symlink(outside, link)
            except OSError:
                self.skipTest("OS does not allow unprivileged symlinks")
            if not _git_can_stage_symlink(repo, link):
                self.skipTest("git on this platform cannot stage symlinks")
            with self.assertRaises(PermissionError):
                provider.commit(repo, "costpilot: x", ["evil.py"])
            self.assertEqual(outside.read_text(encoding="utf-8"), 'OUTSIDE = "original"\n')
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_apply_changes_rejects_symlink_to_outside(self):
        provider = LocalGitProvider()
        repo = _init_repo({"real.py": 'model = "gpt-4o"\n'})
        try:
            outside = repo.parent / "outside.py"
            outside.write_text('OUTSIDE = "original"\n', encoding="utf-8")
            link = repo / "evil.py"
            try:
                os.symlink(outside, link)
            except OSError:
                self.skipTest("OS does not allow unprivileged symlinks")
            if not _git_can_stage_symlink(repo, link):
                self.skipTest("git on this platform cannot stage symlinks")
            with self.assertRaises(PermissionError):
                provider.apply_changes(
                    repo,
                    [Change(file="evil.py", line=1, old_model="gpt-4o", new_model="gpt-4o-mini")],
                )
            self.assertEqual(outside.read_text(encoding="utf-8"), 'OUTSIDE = "original"\n')
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_create_branch_rejects_invalid_ref(self):
        provider = LocalGitProvider()
        repo = _init_repo({"a.py": "x = 1\n"})
        try:
            with self.assertRaises(PermissionError):
                provider.create_branch(repo, "costpilot/../../escaped")
            with self.assertRaises(PermissionError):
                provider.create_branch(repo, "costpilot/foo bar")
            with self.assertRaises(PermissionError):
                provider.create_branch(repo, "costpilot/@{-1}")
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_create_branch_rejects_existing(self):
        provider = LocalGitProvider()
        repo = _init_repo({"a.py": "x = 1\n"})
        try:
            provider.create_branch(repo, "costpilot/test")
            with self.assertRaises(RuntimeError):
                provider.create_branch(repo, "costpilot/test")
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_commit_rejects_dirty_working_tree(self):
        provider = LocalGitProvider()
        repo = _init_repo({"a.py": "x = 1\n", "b.py": "y = 1\n"})
        try:
            (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # CostPilot patch
            (repo / "b.py").write_text("y = 2\n", encoding="utf-8")  # user's pre-existing change
            with self.assertRaises(RuntimeError):
                provider.commit(repo, "costpilot: test", ["a.py"])
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_commit_allows_only_patched_files_dirty(self):
        provider = LocalGitProvider()
        repo = _init_repo({"a.py": "x = 1\n"})
        try:
            (repo / "a.py").write_text("x = 2\n", encoding="utf-8")  # only CostPilot patch
            sha = provider.commit(repo, "costpilot: test", ["a.py"])
            self.assertIsNotNone(sha)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    """End-to-end: discover -> decide -> patch -> validate -> commit -> PR artifact."""

    def test_full_loop_produces_artifact(self):
        repo = _make_repo(FIXTURES / "sample_app.py")
        try:
            result = run_optimization_loop(repo)
            self.assertTrue(result["changed"], result.get("reason"))
            artifact = result["artifact"]
            assert artifact is not None

            self.assertTrue(artifact.branch.startswith("costpilot/"))
            self.assertIsNotNone(artifact.commit)
            self.assertEqual(artifact.changed_files, ["sample_app.py"])
            self.assertGreater(artifact.expected_saving_ratio or 0.0, 0.5)
            self.assertEqual(artifact.decision, "MODEL_DOWNGRADE")
            self.assertEqual(artifact.validation["syntax"], "ok")
            # sample app has no tests -> "no-tests" (neutral) is correct
            self.assertIn(artifact.validation["tests"], ("passed", "no-tests"))

            # branch exists with a commit on top of the initial one
            branches = _git(repo, "branch")
            self.assertIn(artifact.branch, branches)
            commits = _git(repo, "log", "--oneline", "-2").splitlines()
            self.assertEqual(len(commits), 2)

            # the artifact file exists and is human-readable
            path = result["artifact_path"]
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("Expected saving", content)
            self.assertIn("The agent owns the work. The human owns the decision.", content)

            # the risk section names the explicit safety criteria and the
            # dimensions that are NOT evaluated (semantic equivalence is a
            # human decision at merge time, never claimed by the agent)
            self.assertIn("explicit safety criteria", artifact.risk)
            self.assertIn("Not evaluated", artifact.risk)
            for dim in (
                "output quality",
                "refusal rate",
                "JSON schema stability",
                "tool-calling behavior",
                "latency",
                "token-usage variance",
            ):
                self.assertIn(dim, artifact.risk)
            self.assertNotIn("semantically equivalent", artifact.risk)

            # file on disk actually changed the model string
            changed = (repo / "sample_app.py").read_text(encoding="utf-8")
            self.assertNotIn('model="gpt-4o"', changed)
            self.assertIn('model="gpt-4o-mini"', changed)
        finally:
            shutil.rmtree(repo, ignore_errors=True)

    def test_loop_on_clean_repo_does_nothing(self):
        # a repo without any LLM calls -> no change, honest reason
        tmp = _init_repo({"hello.py": "print('hi')\n"})
        try:
            result = run_optimization_loop(tmp)
            self.assertFalse(result["changed"])
            self.assertIn("no provable optimization", result["reason"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
