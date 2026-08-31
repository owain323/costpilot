"""GitHubProvider: publish the local artifact as a real GitHub pull request.

Scope (deliberately minimal, P2c):
- Branching, patching, validation, and commit stay in LocalGitProvider. The
  local loop remains the source of truth and the policy gate.
- The only new behavior is create_change_request. When a GITHUB_TOKEN is set
  and the repo's origin points at github.com, the artifact markdown becomes
  the pull request body via the REST API (POST /repos/{owner}/{repo}/pulls).
- Without a token, without a github.com origin, or on any API error, the
  provider degrades honestly: the artifact file records what happened and the
  local path is returned. Nothing fails silently, and nothing fails loudly
  enough to break the optimization loop.

Like LocalGitProvider, this is system code. The agent never touches raw git,
HTTP, or this layer; it only ever calls the GitProvider tool API.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .git_provider import Change, GitProvider, LocalGitProvider, PRArtifact

_GITHUB_ORIGIN = re.compile(
    r"^(?:https://github\.com/|git@github\.com:)(?P<owner>[^/]+)/(?P<repo>.+?)(?:\.git)?/?$"
)


def _parse_github_origin(url: str) -> tuple[str, str] | None:
    """Return (owner, repo) for a github.com origin URL, else None."""
    m = _GITHUB_ORIGIN.match(url.strip())
    if m is None:
        return None
    return m.group("owner"), m.group("repo")


def _origin_url(repo: Path) -> str | None:
    """Read the origin URL. Read-only git call made by the provider, never by the agent."""
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


class GitHubProvider(GitProvider):
    """LocalGitProvider plus exactly one thing: publishing the artifact as a PR."""

    def __init__(self) -> None:
        self._local = LocalGitProvider()

    def create_branch(self, repo: Path, branch: str) -> None:
        self._local.create_branch(repo, branch)

    def apply_changes(self, repo: Path, changes: list[Change]) -> None:
        self._local.apply_changes(repo, changes)

    def revert_changes(self, repo: Path, files: list[str]) -> None:
        self._local.revert_changes(repo, files)

    def commit(self, repo: Path, message: str, files: list[str]) -> str:
        return self._local.commit(repo, message, files)

    def create_change_request(self, repo: Path, artifact: PRArtifact) -> Path:
        """Write the local artifact, attempt the GitHub publish, record the outcome."""
        artifact_path = self._local.create_change_request(repo, artifact)
        record = self._publish(repo, artifact)
        _append_record(artifact_path, record)
        return artifact_path

    def _publish(self, repo: Path, artifact: PRArtifact) -> str:
        """Try to open a pull request; always return an honest one-line outcome."""
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            return "not published: GITHUB_TOKEN is not set"

        explicit = os.environ.get("COSTPILOT_GITHUB_REPO", "")
        if explicit:
            parts = explicit.split("/")
            origin = (parts[0], parts[1]) if len(parts) == 2 else None
        else:
            url = _origin_url(repo)
            origin = _parse_github_origin(url) if url else None
        if origin is None:
            return "not published: origin is not a github.com repository"

        owner, name = origin
        base = os.environ.get("COSTPILOT_GITHUB_BASE", "main")
        try:
            import httpx  # transitive dependency of the ollama client

            response = httpx.post(
                f"https://api.github.com/repos/{owner}/{name}/pulls",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                json={
                    "title": f"CostPilot change request: {artifact.branch}",
                    "body": artifact.render_markdown(),
                    "head": f"{owner}:{artifact.branch}",
                    "base": base,
                },
                timeout=30,
            )
        except Exception as exc:
            return f"not published: {type(exc).__name__}"
        if response.status_code != 201:
            return f"not published: GitHub API returned {response.status_code}"
        pr_url = response.json().get("html_url", "")
        if not pr_url:
            return "not published: GitHub API response had no html_url"
        return f"published: {pr_url}"


def _append_record(artifact_path: Path, record: str) -> None:
    """Append the publish outcome to the artifact so the record stays honest."""
    block = ["", "## GitHub pull request", "", f"- {record}", ""]
    with open(artifact_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(block))
