"""Git gate: block CJK content and filenames from entering the repository.

Why: this repository is a public hackathon submission. Internal working
documents are written in Chinese and must never be committed, force-added,
or pushed to any public surface. This checker is the last line of defense
after .gitignore rules.

Modes:
    pre-commit            check everything staged for the next commit
    pre-push              check everything about to be pushed (reads the
                          push spec from stdin)

Behavior is fail-closed: if the checker itself cannot run (missing python,
git error, undecodable state), the git operation is BLOCKED, not allowed.

Exit code 0 = clean; 1 = violation found or internal error.
"""

from __future__ import annotations

import subprocess
import sys

# CJK coverage: unified ideographs (+ext A/B, compat), CJK punctuation,
# fullwidth forms. Deliberately broad — one stray character blocks.
CJK_RANGES = (
    (0x3040, 0x30FF),   # hiragana / katakana
    (0x3400, 0x4DBF),   # CJK ext A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0x20000, 0x2A6DF),  # CJK ext B
    (0x3000, 0x303F),   # CJK punctuation
    (0xFF01, 0xFF60),   # fullwidth forms
)

MAX_REPORT = 8  # max violations printed before truncating


def first_cjk(text: str) -> tuple[str, int] | None:
    """Return (char, line_number) of the first CJK char, or None."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            cp = ord(ch)
            for lo, hi in CJK_RANGES:
                if lo <= cp <= hi:
                    return ch, lineno
    return None


def git(*args: str, binary: bool = False) -> bytes | str:
    r = subprocess.run(["git", *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args[:3])} failed: {r.stderr.decode('utf-8', 'replace')[:200]}"
        )
    return r.stdout if binary else r.stdout.decode("utf-8", "replace")


def staged_paths() -> list[str]:
    out = git("diff", "--cached", "--name-only", "--diff-filter=d", "-z", binary=True)
    assert isinstance(out, bytes)
    return [f.decode("utf-8", "replace") for f in out.split(b"\0") if f]


def push_spec_paths(local_sha: str, remote_sha: str) -> list[str]:
    """Files changed in the push range that still exist at the local tip."""
    if remote_sha.strip("0") == "":
        out = git("ls-tree", "-r", "--name-only", "-z", local_sha, binary=True)
    else:
        out = git("diff", "--name-only", "-z", f"{remote_sha}..{local_sha}", binary=True)
    assert isinstance(out, bytes)
    changed = [f.decode("utf-8", "replace") for f in out.split(b"\0") if f]
    present = []
    for f in changed:
        r = subprocess.run(["git", "cat-file", "-e", f"{local_sha}:{f}"], capture_output=True)
        if r.returncode == 0:
            present.append(f)
    return present


def check_blob(spec: str, path: str, violations: list[str]) -> None:
    """Check one file: first its name, then its staged/pushed blob content."""
    bad = first_cjk(path)
    if bad:
        violations.append(f"{path}: CJK character {bad[0]!r} in FILE NAME")
        return
    r = subprocess.run(["git", "show", spec], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"cannot read blob {spec}")
    data = r.stdout
    if not data:
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        # try utf-16 (BOM) before treating as binary
        try:
            text = data.decode("utf-16")
        except UnicodeDecodeError:
            return  # binary file: filename check already done
    bad = first_cjk(text)
    if bad:
        ch, lineno = bad
        violations.append(f"{path}:{lineno}: CJK character {ch!r} in content")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre-commit"
    try:
        if mode == "pre-commit":
            files = staged_paths()
            violations: list[str] = []
            for f in files:
                check_blob(f":{f}", f, violations)
        elif mode == "pre-push":
            violations = []
            for line in sys.stdin.read().splitlines():
                parts = line.split()
                if len(parts) != 4:
                    continue
                _local_ref, local_sha, _remote_ref, remote_sha = parts
                for f in push_spec_paths(local_sha, remote_sha):
                    check_blob(f"{local_sha}:{f}", f, violations)
        else:
            print(f"cjk-gate: unknown mode {mode!r}", file=sys.stderr)
            return 1
    except Exception as exc:  # fail closed
        print("cjk-gate: BLOCKED (internal error, fail-closed):", exc, file=sys.stderr)
        return 1

    if violations:
        print(
            "cjk-gate: BLOCKED — CJK content must never enter this public repository.",
            file=sys.stderr,
        )
        print("  Offending staged/pushed files:", file=sys.stderr)
        for v in violations[:MAX_REPORT]:
            print(f"    - {v}", file=sys.stderr)
        if len(violations) > MAX_REPORT:
            print(f"    ... and {len(violations) - MAX_REPORT} more", file=sys.stderr)
        print(
            "  Fix: remove the file from the commit (git restore --staged <file>)\n"
            "  and add it to .gitignore. Internal docs stay local, never in git.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
