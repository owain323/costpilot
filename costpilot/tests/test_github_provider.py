"""GitHubProvider acceptance tests: honest publish and honest degrade, no network."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from costpilot.git_provider import PRArtifact
from costpilot.github_provider import GitHubProvider, _parse_github_origin


def _artifact() -> PRArtifact:
    return PRArtifact(
        branch="costpilot/optimize",
        commit="9e69cb5b",
        changed_files=["sample_app.py"],
        before_cost=1.01,
        after_cost=0.0606,
        expected_saving_ratio=0.94,
        validation={"syntax": "ok", "tests": "passed"},
        reasoning="decision: agent-reviewed (c0=approve; c1=keep)",
        risk="Savings are price-saving potential from public pricing (not realized cost).",
        decision="MODEL_DOWNGRADE",
    )


class TestOriginParsing(unittest.TestCase):
    def test_https_url(self):
        self.assertEqual(
            _parse_github_origin("https://github.com/owain323/costpilot.git"),
            ("owain323", "costpilot"),
        )

    def test_https_url_without_suffix(self):
        self.assertEqual(
            _parse_github_origin("https://github.com/owain323/costpilot"),
            ("owain323", "costpilot"),
        )

    def test_ssh_url(self):
        self.assertEqual(
            _parse_github_origin("git@github.com:owain323/costpilot.git"),
            ("owain323", "costpilot"),
        )

    def test_non_github_origin_is_rejected(self):
        self.assertIsNone(_parse_github_origin("https://gitlab.com/owain323/costpilot.git"))


class TestPublishBehavior(unittest.TestCase):
    """No network in tests: httpx.post is mocked, outcomes are read off the artifact."""

    def test_no_token_degrades_honestly(self):
        provider = GitHubProvider()
        with tempfile.TemporaryDirectory() as tmp:
            env = {"COSTPILOT_GITHUB_REPO": "owain323/costpilot"}
            with mock.patch.dict("os.environ", env, clear=True):
                path = provider.create_change_request(Path(tmp), _artifact())
            content = path.read_text(encoding="utf-8")
        self.assertIn("## GitHub pull request", content)
        self.assertIn("not published: GITHUB_TOKEN is not set", content)
        # the artifact body is intact: the record is appended, nothing else changes
        self.assertIn("Expected saving", content)
        self.assertIn("The agent owns the work. The human owns the decision.", content)

    def test_publish_posts_artifact_and_records_url(self):
        provider = GitHubProvider()
        response = mock.Mock(
            status_code=201,
            json=lambda: {"html_url": "https://github.com/owain323/costpilot/pull/7"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "GITHUB_TOKEN": "token-for-tests",
                "COSTPILOT_GITHUB_REPO": "owain323/costpilot",
                "COSTPILOT_GITHUB_BASE": "main",
            }
            with (
                mock.patch.dict("os.environ", env, clear=True),
                mock.patch("httpx.post", return_value=response) as post,
            ):
                path = provider.create_change_request(Path(tmp), _artifact())
            content = path.read_text(encoding="utf-8")
        self.assertIn("published: https://github.com/owain323/costpilot/pull/7", content)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["head"], "owain323:costpilot/optimize")
        self.assertEqual(payload["base"], "main")
        self.assertIn("Expected saving:** 94.0%", payload["body"])
        self.assertIn("The agent owns the work. The human owns the decision.", payload["body"])

    def test_api_failure_records_reason_without_raising(self):
        provider = GitHubProvider()
        response = mock.Mock(status_code=401, json=lambda: {"message": "Bad credentials"})
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "GITHUB_TOKEN": "token-for-tests",
                "COSTPILOT_GITHUB_REPO": "owain323/costpilot",
            }
            with (
                mock.patch.dict("os.environ", env, clear=True),
                mock.patch("httpx.post", return_value=response),
            ):
                path = provider.create_change_request(Path(tmp), _artifact())
            content = path.read_text(encoding="utf-8")
        self.assertIn("not published: GitHub API returned 401", content)
        self.assertNotIn("- published:", content)


if __name__ == "__main__":
    unittest.main()
