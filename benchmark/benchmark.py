"""P1.5 benchmark: measure the scanner against real repositories.

Purpose: establish a baseline (recall / precision / runtime) on real-world
code before adding any agentic layer. Measurement only: no scanner changes.

Outputs a JSON report plus a human-readable summary.
"""

from __future__ import annotations

import importlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from costpilot.models import Framework
from costpilot.scanner import ScanConfig, discover_ai_calls

# strands-agents is located dynamically from the installed package (no hardcoded path)
_STRANDS_PKG = Path(importlib.import_module("strands").__file__).resolve().parent

REPOS_ROOT = Path(__file__).resolve().parent / "repos"
_MANIFEST_PATH = Path(__file__).resolve().parent / "repos.json"


def _ensure_repos() -> dict[str, str]:
    """Self-healing benchmark inputs: fetch missing repos from pinned SHAs.

    Uses codeload tarballs (no git history, tiny download, SHA-locked content)
    so the reported recall numbers are reproducible by a third party.
    """
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    resolved: dict[str, str] = {}
    for name, spec in manifest.items():
        target = REPOS_ROOT / name
        if not target.exists():
            _download_tarball(name, spec["url"], target)
        resolved[name] = str(target)
    return resolved


def _download_tarball(name: str, url: str, target: Path) -> None:
    print(f"[benchmark] fetching {name} from codeload ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=180) as resp:
        data = resp.read()
    tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    top = tar.getnames()[0].split("/")[0]
    REPOS_ROOT.mkdir(parents=True, exist_ok=True)
    tar.extractall(REPOS_ROOT)
    extracted = REPOS_ROOT / top
    if not extracted.exists():
        raise RuntimeError(f"tarball for {name} did not extract {top}")
    extracted.rename(target)


REPOS: dict[str, str] = _ensure_repos()
REPOS["strands-agents"] = str(_STRANDS_PKG)

# Independent "hard pattern" references (file-level) used for a coarse recall estimate.
# These are the deterministic invocation shapes our rules claim to detect.
HARD_PATTERNS: list[tuple[str, str]] = [
    ("openai_sdk", r"chat\.completions\.create"),
    ("anthropic_sdk", r"messages\.create"),
    ("langchain_ctor", r"ChatOpenAI\("),
    ("langchain_invoke", r"\.invoke\("),
]

FRAMEWORK_TO_PATTERN: dict[str, str] = {
    Framework.OPENAI.value: "openai_sdk",
    Framework.ANTHROPIC.value: "anthropic_sdk",
    Framework.LANGCHAIN.value: "langchain_ctor",
}

SKIP_PARTS = (".git", "node_modules", "venv", ".venv", "dist", "build")


def count_py_files(root: Path) -> int:
    return sum(
        1 for p in root.rglob("*.py") if not any(part in SKIP_PARTS for part in p.parts)
    )


def grep_hit_files(root: Path, pattern: str) -> set[str]:
    rx = re.compile(pattern)
    hits: set[str] = set()
    for p in root.rglob("*.py"):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if rx.search(text):
            hits.add(str(p.relative_to(root)).replace("\\", "/"))
    return hits


def scan_one(name: str, path: str) -> dict:
    root = Path(path)
    config = ScanConfig()
    t0 = time.perf_counter()
    sites = discover_ai_calls(str(root), config=config)
    seconds = round(time.perf_counter() - t0, 3)

    by_framework: dict[str, int] = {}
    for s in sites:
        by_framework[s.framework.value] = by_framework.get(s.framework.value, 0) + 1
    unknown = sum(1 for s in sites if s.model is None)
    sites_plain = [
        {
            "file": s.file,
            "line": s.line,
            "fw": s.framework.value,
            "model": s.model,
            "conf": s.confidence,
        }
        for s in sites
    ]

    # File-level recall per hard pattern: scanner-hit files / grep-hit files
    pattern_recall: dict[str, dict] = {}
    for pat_name, rx in HARD_PATTERNS:
        grep_hits = grep_hit_files(root, rx)
        if not grep_hits:
            pattern_recall[pat_name] = {
                "grep_files": 0,
                "scanner_files": 0,
                "recall": None,
                "missed": [],
            }
            continue
        # map scanner sites to the matching pattern by framework where possible
        scanner_files = set()
        for s in sites:
            target = FRAMEWORK_TO_PATTERN.get(s.framework.value)
            if target == pat_name or (pat_name == "langchain_invoke"):
                # langchain ctor vs invoke are both langchain; count both for invoke check
                scanner_files.add(s.file)
        covered = grep_hits & scanner_files
        pattern_recall[pat_name] = {
            "grep_files": len(grep_hits),
            "scanner_files": len(scanner_files),
            "recall": round(len(covered) / len(grep_hits), 3),
            "missed": sorted(grep_hits - covered)[:8],
        }
    # langchain ctor / invoke with no grep hits: still record keys
    for pat_name, _ in HARD_PATTERNS:
        if pat_name not in pattern_recall:
            pattern_recall[pat_name] = {
                "grep_files": 0,
                "scanner_files": 0,
                "recall": None,
                "missed": [],
            }

    return {
        "name": name,
        "py_files": count_py_files(root),
        "sites": len(sites),
        "seconds": seconds,
        "sites_per_sec": round(len(sites) / seconds, 2) if seconds > 0 else None,
        "by_framework": by_framework,
        "unknown_model": unknown,
        "unknown_rate": round(unknown / len(sites), 3) if sites else None,
        "pattern_recall": pattern_recall,
        "all_sites": sites_plain,
    }


def main() -> int:
    report: dict = {"repos": []}
    for name, path in REPOS.items():
        print(f"--- scanning {name} ...", file=sys.stderr)
        r = scan_one(name, path)
        report["repos"].append(r)
        print(
            f"{name}: {r['py_files']} py files, {r['sites']} sites, "
            f"{r['seconds']}s, {r['by_framework']}, unknown {r['unknown_rate']}",
            file=sys.stderr,
        )

    out = Path("benchmark/benchmark-report.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to {out}")

    # Human summary
    print("\n===== SUMMARY =====")
    for r in report["repos"]:
        print(f"\n## {r['name']}")
        print(
            f"  py files: {r['py_files']} | sites: {r['sites']} | time: {r['seconds']}s"
        )
        print(f"  frameworks: {r['by_framework']} | unknown rate: {r['unknown_rate']}")
        for pat, v in r["pattern_recall"].items():
            rec = f"{v['recall']:.0%}" if v["recall"] is not None else "n/a"
            print(f"  recall[{pat}]: {rec} ({v['grep_files']} grep files)")
            if v["missed"]:
                print(f"    missed sample: {v['missed'][:4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
