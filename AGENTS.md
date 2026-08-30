# AGENTS.md

## What this is
CostPilot: an agent that finds expensive LLM calls in your code, prices them against public rate cards, and only proposes a change when it can prove the saving. Hackathon entry ("Agents for Humans" theme).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
./run_checks.sh                                     # all gates
python costpilot/pipeline.py costpilot/tests/fixtures/sample_app.py 100   # static path
uvicorn demo.app:app --port 8000                    # live scan-only demo (from repo root)
```

## Stack
Python 3.13, AST-based static analysis, Strands Agents SDK for the decision loop (Ollama default, Bedrock optional). Demo is a single-file HTML frontend (no build step) served by FastAPI.

## Layout and conventions
- `costpilot/` — scanner, pricing, optimizer, validator, git_provider (the pipeline)
- `demo/` — live demo: `demo/app.py` (FastAPI, /scan) + `demo/static/index.html` (all UI in one file)
- `benchmark/` — real-repo benchmark, SHA-locked, self-healing fetch
- `docs/` — architecture + governance docs
- Local working documents are git-ignored and never published
- Demo deploys to `costpilot.owain32380.cn` (server: /opt/costpilot-demo, uvicorn demo.app:app, static served from demo/static/)
- Rules: pure English in the UI, no em dash, no AI-sounding words, gates in run_checks.sh

## Current state
Demo is the active surface: scan → summary (saving%) → proposed-change cards → click a card jumps to the line. Badge legend modal explains APPROVE / KEEP / NOT PRICED / HUMAN. Deployed and live.
