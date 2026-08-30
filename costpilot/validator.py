"""P2b: patch validation: syntax / scan re-check / tests, all via allowlisted commands.

Security model: the agent cannot run arbitrary commands. Validation runs either
pure-Python checks (AST parse, re-scan, cost re-profile) or a single allowlisted
test command (python -m unittest <discover>). No shell=True, no arbitrary flags.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from .pricing import profile_cost
from .scanner import discover_ai_calls


def _syntax_check(paths: list[Path]) -> dict:
    """Parse every changed file; fail on any syntax error."""
    errors: list[str] = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            errors.append(f"{path.name}:{exc.lineno}: {exc.msg}")
    return {"syntax": "ok" if not errors else f"failed: {errors}"}


def _run_allowlisted_tests(repo: Path) -> dict:
    """Run the repository's unit tests via a single allowlisted command.

    Uses sys.executable (the interpreter running CostPilot) so the tests see
    the same installed dependencies; a bare `python` would resolve to whatever
    is on PATH and may not have the deps installed.
    Exit code 5 / 'NO TESTS RAN' means the repo has no tests: recorded as
    "no-tests" (neutral, not a failure); any other nonzero exit is a failure.
    """
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    no_tests = result.returncode == 5 or "NO TESTS RAN" in result.stdout
    if result.returncode == 0:
        status = "passed"
    elif no_tests:
        status = "no-tests"
    else:
        status = "failed"
    summary = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("Ran ", "OK", "FAILED", "NO TESTS RAN"))
    ]
    return {
        "tests": status,
        "summary": " | ".join(summary) if summary else (result.stderr.strip() or "n/a"),
    }


def validate_patch(repo: Path, changed_files: list[str]) -> dict:
    """Validate applied changes: syntax -> tests -> scan re-check.

    Returns a structured validation record for the PR artifact.
    """
    paths = [repo / f for f in changed_files]
    checks: dict = {}
    checks.update(_syntax_check(paths))
    checks.update(_run_allowlisted_tests(repo))
    # re-scan: the call site must still be detected after the change
    sites = discover_ai_calls(repo)
    checks["call_sites_after"] = len(sites)
    priced_after = sum(1 for e in profile_cost(sites) if e.is_priced)
    checks["priced_sites_after"] = priced_after
    return checks
