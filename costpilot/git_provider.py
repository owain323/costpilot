"""P2b: GitProvider abstraction + LocalGitProvider + PR artifact.

Design (approved decision):
- P2b ships a *local* git loop only: no GitHub. Local is the official isolation
  layer, not a stand-in demo.
- The agent never touches raw git/bash. It calls the GitProvider tool API;
  the provider enforces an operation allowlist (policy gate):
  "The agent decides WHAT to do, the system decides HOW it may be done."
- GitHubProvider implements the same interface in P2c.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Change:
    """A single conservative code change (model-name replacement only)."""

    file: str  # path relative to repo root
    line: int  # 1-based line of the call site
    old_model: str  # current model string (exact quoted value)
    new_model: str  # replacement model string


@dataclass
class PRArtifact:
    """Structured change-request record: the local equivalent of a PR."""

    branch: str
    commit: str | None
    changed_files: list[str] = field(default_factory=list)
    before_cost: float | None = None
    after_cost: float | None = None
    expected_saving_ratio: float | None = None
    validation: dict = field(default_factory=dict)
    reasoning: str = ""
    risk: str = ""
    decision: str = ""
    status: str = "ready_for_human_review"

    def render_markdown(self) -> str:
        """Render a complete, human-readable change request (the local 'PR')."""
        lines: list[str] = []
        lines.append(f"# CostPilot Change Request: `{self.branch}`")
        lines.append("")
        lines.append(f"- **Status:** {self.status}")
        lines.append(f"- **Decision:** {self.decision}")
        if self.commit:
            lines.append(f"- **Commit:** `{self.commit}`")
        lines.append(f"- **Changed files:** {', '.join(self.changed_files) or '(none)'}")
        lines.append("")
        lines.append("## Cost evidence")
        if self.before_cost is not None and self.after_cost is not None:
            lines.append(
                f"- **Before:** ${self.before_cost:.4f} / 1K calls "
                f"→ **After:** ${self.after_cost:.4f} / 1K calls"
            )
            if self.expected_saving_ratio is not None:
                lines.append(f"- **Expected saving:** {self.expected_saving_ratio:.1%}")
        else:
            lines.append("- _No cost change (KEEP / not priced)._")
        lines.append("")
        lines.append("## Validation")
        for key, value in self.validation.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
        lines.append("## Reasoning")
        lines.append(self.reasoning or "_n/a_")
        lines.append("")
        lines.append("## Risk")
        lines.append(self.risk or "_none stated_")
        lines.append("")
        lines.append("> The agent owns the work. The human owns the decision.")
        return "\n".join(lines)


class GitProvider(ABC):
    """Tool API exposed to the agent. Implementations enforce their own policy gates."""

    @abstractmethod
    def create_branch(self, repo: Path, branch: str) -> None:
        """Create a branch. Implementations restrict branch naming (e.g. costpilot/ prefix)."""

    @abstractmethod
    def apply_changes(self, repo: Path, changes: list[Change]) -> None:
        """Apply conservative changes. Implementations restrict what can be changed."""

    @abstractmethod
    def revert_changes(self, repo: Path, files: list[str]) -> None:
        """Revert uncommitted changes to the given files (validation-failure path)."""

    @abstractmethod
    def commit(self, repo: Path, message: str, files: list[str]) -> str:
        """Commit only the files CostPilot patched; return the commit sha."""

    @abstractmethod
    def create_change_request(self, repo: Path, artifact: PRArtifact) -> Path:
        """Produce the change request (local: PR artifact file; P2c: GitHub PR)."""


# --- Policy gate: operation allowlist ---------------------------------------

_BRANCH_PREFIX = "costpilot/"
_COMMIT_PREFIX = "costpilot:"
# how many lines below a call's start line the model kwarg may sit
_MODEL_KWARG_WINDOW = 6
# git subcommands the provider may run (everything else is denied)
_ALLOWED_GIT_SUBCOMMANDS = (
    "checkout",
    "add",
    "commit",
    "rev-parse",
    "status",
    "log",
    "diff",
    "write-tree",
    "commit-tree",
    "symbolic-ref",
    "restore",  # validation-failure revert only (--source=HEAD --worktree)
)
# any argument starting with "-" must be in this allowlist (no --force/--hard/etc.)
_ALLOWED_GIT_FLAGS = (
    "-b",
    "--short",
    "--oneline",
    "-1",
    "--stat",
    "-m",
    "--name-only",
    "--porcelain",  # status --porcelain (machine-parseable, stable v1 format)
    "--",  # end-of-options separator (used as `git add -- .`)
    "--verify",  # rev-parse --verify refs/... (branch existence check)
    "-p",  # commit-tree parent
    "--source",  # restore --source=HEAD
    "--worktree",  # restore --worktree (files only, never staged index changes)
)


def _assert_git_args(args: tuple[str, ...]) -> None:
    """Policy gate: reject any git invocation outside the allowlist.

    Supports --flag=value forms (e.g. --source=HEAD) by matching the flag part.
    """
    if not args or args[0] not in _ALLOWED_GIT_SUBCOMMANDS:
        raise PermissionError(f"git subcommand not allowed: {args[:1]}")
    for arg in args[1:]:
        if arg.startswith("-"):
            flag = arg.split("=", 1)[0]
            if flag not in _ALLOWED_GIT_FLAGS:
                raise PermissionError(f"git flag not allowed: {arg}")


def _assert_safe_relpath(path: str) -> None:
    """Reject absolute paths or paths that escape the repo root."""
    p = Path(path)
    if p.is_absolute():
        raise PermissionError(f"absolute path not allowed: {path}")
    if ".." in p.parts:
        raise PermissionError(f"path escapes repo root: {path}")


def _resolve_repo_path(repo: Path, relpath: str) -> Path:
    """Resolve a repo-relative path and enforce containment + symlink policy.

    CostPilot's safety promise is that the agent can only modify files inside
    the repo it was given. This helper rejects:
      - paths with ``..`` or absolute components (string-level)
      - direct symlinks (common escape vector)
      - any path whose realpath resolves outside the repo (catches indirect
        symlinks, junctions on Windows, and traversal attempts)
    """
    _assert_safe_relpath(relpath)
    target = repo / relpath

    # Direct symlink/junction on the target itself: deny outright.
    # We use lstat semantics via is_symlink(); resolve() would follow it.
    if target.is_symlink():
        raise PermissionError(f"symlink target not allowed: {relpath}")

    # Final containment check: resolve follows symlinks/junctions, so if any
    # parent component escapes the repo, the resolved path will land outside.
    repo_root = repo.resolve()
    resolved = target.resolve(strict=False)
    if resolved != repo_root and repo_root not in resolved.parents:
        raise PermissionError(f"path escapes repo root after resolve: {relpath}")

    return target


def _assert_branch_name(branch: str) -> None:
    """Enforce a safe branch name under the costpilot/ namespace.

    Branch names become ref paths under .git/refs/heads/. A simple prefix check
    is not enough: ``costpilot/../../escaped`` would write outside refs/heads.
    We therefore validate with git's own ref-format rules *before* any filesystem
    operation.
    """
    if not branch.startswith(_BRANCH_PREFIX):
        raise PermissionError(f"branch must start with {_BRANCH_PREFIX!r}: {branch}")
    # Whitelist the allowed character set first (fast, auditable fail).
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./")
    if not set(branch).issubset(allowed):
        raise PermissionError(f"branch contains illegal characters: {branch}")
    # Delegate final ref semantics to git itself.
    result = subprocess.run(
        ["git", "check-ref-format", "--allow-onelevel", branch],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise PermissionError(f"branch fails git ref-format check: {branch}")


def _assert_clean_except(repo: Path, allowed_files: set[str]) -> None:
    """Refuse to commit if tracked files outside our patch set are dirty.

    Precise staging (``git add -- <files>``) already avoids sweeping in
    untracked files. But if the user had pre-existing modifications in the
    *same* file we patched, ``git add`` would commit those too. This helper
    raises before we stage anything, forcing a clean working tree for every
    tracked file except the ones CostPilot itself is about to commit.

    Untracked files (``??``) are ignored: they are not at risk of being
    committed and should not block a normal run.
    """
    output = _run_git(repo, "status", "--porcelain", strip_output=False)
    for line in output.splitlines():
        if not line.strip():
            continue
        # porcelain v1 format: XY <path> or XY <path> -> <origpath>
        status_code = line[:2]
        file_path = line[3:].strip().split(" -> ")[-1]
        if status_code == "??":
            continue  # untracked files are not swept by git add -- file
        if file_path not in allowed_files:
            raise RuntimeError(
                f"working tree has unrelated changes: {file_path}; "
                "commit a clean state before running CostPilot"
            )


def _assert_commit_message(message: str) -> None:
    if not message.startswith(_COMMIT_PREFIX):
        raise PermissionError(f"commit message must start with {_COMMIT_PREFIX!r}")


def _run_git(repo: Path, *args: str, strip_output: bool = True) -> str:
    """Run an allowlisted git command; raise on any failure."""
    _assert_git_args(args)
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    out = result.stdout
    return out.strip() if strip_output else out


def _update_head_ref(repo: Path, sha: str) -> None:
    """Point HEAD at `sha` by writing the resolved ref file directly.

    Handles both symbolic HEAD (ref: refs/heads/X) and detached HEAD.
    """
    head_file = repo / ".git" / "HEAD"
    content = head_file.read_text(encoding="utf-8").strip()
    if content.startswith("ref: "):
        target = content[len("ref: ") :]
        ref_path = repo / ".git" / Path(target)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(f"{sha}\n", encoding="utf-8")
    else:
        head_file.write_text(f"{sha}\n", encoding="utf-8")


class LocalGitProvider(GitProvider):
    """Local git loop with a hard policy gate (no push/force/reset/rm/clean).

    Ref handling is owned by the provider: git creates objects (write-tree /
    commit-tree) while ref files are written directly. This sidesteps the
    Windows issue where git's own loose-ref writes are silently swallowed
    (observed on git 2.55.0.windows.3) and keeps every ref operation auditable.
    """

    def create_branch(self, repo: Path, branch: str) -> None:
        _assert_branch_name(branch)
        current = _run_git(repo, "rev-parse", "HEAD")  # base commit
        ref_path = repo / ".git" / "refs" / "heads" / branch
        if ref_path.exists():
            raise RuntimeError(f"branch {branch!r} already exists")
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(f"{current}\n", encoding="utf-8")
        _run_git(repo, "symbolic-ref", "HEAD", f"refs/heads/{branch}")
        # verify the switch really took effect
        got = _run_git(repo, "rev-parse", "HEAD")
        if got != current:
            raise RuntimeError(f"branch {branch!r} switch failed: HEAD={got} expected={current}")

    def apply_changes(self, repo: Path, changes: list[Change]) -> None:
        """Replace the exact quoted model string near the call site (conservative).

        The model kwarg can sit a few lines below the call's start line (the
        scanner reports the call's start line), so a small window is searched.
        Only the quoted literal is touched; anything else is left untouched.
        """
        for change in changes:
            path = _resolve_repo_path(repo, change.file)
            if not path.is_file():
                raise FileNotFoundError(f"change target not found: {change.file}")
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            if change.line < 1 or change.line > len(lines):
                raise RuntimeError(f"line {change.line} out of range in {change.file}")
            quoted_old = f'"{change.old_model}"'
            quoted_new = f'"{change.new_model}"'
            start = max(0, change.line - 1)
            window = lines[start : start + _MODEL_KWARG_WINDOW]
            for offset, line in enumerate(window):
                if quoted_old in line:
                    lines[start + offset] = line.replace(quoted_old, quoted_new)
                    break
            else:
                raise RuntimeError(f"{change.file}:{change.line} does not contain {quoted_old}")
            path.write_text("".join(lines), encoding="utf-8")

    def revert_changes(self, repo: Path, files: list[str]) -> None:
        """Revert uncommitted working-tree changes to the given files.

        Uses `git restore --source=HEAD --worktree`: only ever touches the
        listed files, never the index, and cannot delete anything.
        """
        for f in files:
            _resolve_repo_path(repo, f)
        _run_git(repo, "restore", "--source=HEAD", "--worktree", "--", *files)

    def commit(self, repo: Path, message: str, files: list[str]) -> str:
        """Create a commit object via git, then update the ref ourselves.

        Only the files CostPilot patched are staged (precise staging: a user's
        unrelated uncommitted work is never swept into our commit). Ref updates
        are written directly (reliable on Windows, fully auditable).
        """
        _assert_commit_message(message)
        allowed = set(files)
        for f in files:
            _resolve_repo_path(repo, f)
        _assert_clean_except(repo, allowed)
        _run_git(repo, "add", "--", *files)
        tree = _run_git(repo, "write-tree")
        try:
            parent = _run_git(repo, "rev-parse", "HEAD")
            sha = _run_git(repo, "commit-tree", tree, "-p", parent, "-m", message)
        except RuntimeError:
            # no parent (fresh repo): root commit
            sha = _run_git(repo, "commit-tree", tree, "-m", message)
        _update_head_ref(repo, sha)
        # verify the ref update actually landed
        got = _run_git(repo, "rev-parse", "HEAD")
        if got != sha:
            raise RuntimeError(f"commit ref update failed: HEAD={got} expected={sha}")
        return sha

    def create_change_request(self, repo: Path, artifact: PRArtifact) -> Path:
        """Write the PR artifact as markdown beside the repo (not committed)."""
        out_dir = repo.parent / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = artifact.branch.replace("/", "__")
        out_path = out_dir / f"{safe_name}.md"
        out_path.write_text(artifact.render_markdown(), encoding="utf-8")
        artifact.status = "ready_for_human_review"
        return out_path
