#!/usr/bin/env python3
"""Cross-platform quality gates (Windows/macOS/Linux).

Equivalent to run_checks.sh but runs anywhere Python does — no bash, no
Git Bash, no WSL required. Judge opens the zip, runs this file, sees the
same five gates.

Usage:
    python run_checks.py            # uses the python on PATH
    python run_checks.py /path/to/python   # explicit interpreter
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def gate(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✅" if ok else "❌"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    results: list[bool] = []

    # 1/5 ruff format --check
    code, out = run([py, "-m", "ruff", "format", "--check", "costpilot", "demo"])
    results.append(gate("ruff format --check", code == 0, out.splitlines()[-1] if out else ""))

    # 2/5 ruff check
    code, out = run([py, "-m", "ruff", "check", "costpilot", "demo"])
    results.append(gate("ruff check (zero tolerance)", code == 0, out.splitlines()[-1] if out else ""))

    # 3/5 mypy
    code, out = run([py, "-m", "mypy", "costpilot", "--no-incremental"])
    results.append(gate("mypy", code == 0, out.splitlines()[-1] if out else ""))

    # 4/5 unit tests
    code, out = run([py, "-m", "unittest", "discover", "-s", "costpilot/tests"])
    match = re.search(r"Ran (\d+) tests", out)
    test_count = match.group(1) if match else "?"
    tail = out.splitlines()
    tail = "\n".join(tail[-3:]) if tail else ""
    results.append(gate(f"unit tests ({test_count})", code == 0, tail.replace("\n", " | ")))

    # 5/5 language gate: zero CJK in code
    cjk = re.compile(r"[\u4e00-\u9fff]")
    hits: list[str] = []
    for p in (ROOT / "costpilot", ROOT / "demo"):
        for f in p.rglob("*.py"):
            try:
                if cjk.search(f.read_text(encoding="utf-8")):
                    hits.append(str(f.relative_to(ROOT)))
            except (UnicodeDecodeError, OSError):
                pass
    results.append(gate("zero CJK in code", not hits, ", ".join(hits[:5]) if hits else ""))

    print()
    if all(results):
        print("🎉 ALL GATES PASSED — clean-environment reproduction OK")
        return 0
    print("✗ SOME GATES FAILED — see output above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
