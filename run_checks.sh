#!/usr/bin/env bash
# CostPilot quality gates: run all checks with one command.
# Any failure = do not proceed to the next step (one-step-one-gate discipline).
# Portable: uses $PYTHON if set, else python from PATH. No machine-specific paths.
set -euo pipefail

PY="${PYTHON:-python}"
cd "$(dirname "$0")"

echo "=== [1/5] ruff format --check ==="
"$PY" -m ruff format --check costpilot demo || { echo "✗ format: run 'python -m ruff format costpilot demo'"; exit 1; }

echo "=== [2/5] ruff check (zero tolerance) ==="
"$PY" -m ruff check costpilot demo || { echo "✗ lint: fix warnings"; exit 1; }

echo "=== [3/5] mypy ==="
# --no-incremental: avoid cache cleanup (sandbox blocks recycle-bin deletes on Windows)
"$PY" -m mypy costpilot --no-incremental || { echo "✗ types: fix type errors"; exit 1; }

echo "=== [4/5] unit tests ==="
"$PY" -m unittest discover -s costpilot/tests 2>&1 | tail -3

echo "=== [5/5] language gate: zero CJK in code ==="
if grep -rlP '[\x{4e00}-\x{9fff}]' --include='*.py' costpilot/ demo/ 2>/dev/null; then
  echo "✗ CJK characters found in code (language red line)"; exit 1
fi
echo "✓ no CJK in code"

echo ""
echo "✅ ALL GATES PASSED"
