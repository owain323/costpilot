"""CostPilot live demo: scan-only web preview.

Paste code (or pick a sample) and see LLM call sites with pricing evidence
and optimization suggestions. That is all: no LLM, no git, no writes to your
code, no persistence. The smallest possible surface for a public demo.

Run locally:
    pip install fastapi uvicorn
    uvicorn demo.app:app --port 8000
"""

from __future__ import annotations

import ast
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from costpilot.optimizer import find_optimization_candidates
from costpilot.patcher import simulate_optimization
from costpilot.pricing import profile_cost
from costpilot.scanner import discover_ai_calls

app = FastAPI(title="CostPilot demo", docs_url=None, redoc_url=None)

MAX_INPUT_CHARS = 200_000
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10  # requests per window per IP
_hits: dict[str, list[float]] = defaultdict(list)

SAMPLE_PATH = (
    Path(__file__).resolve().parent.parent / "costpilot" / "tests" / "fixtures" / "sample_app.py"
)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    window = _hits[ip]
    _hits[ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_hits[ip]) >= RATE_LIMIT_MAX:
        return True
    _hits[ip].append(now)
    return False


_PAGE_PATH = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(
        _PAGE_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/sample")
def sample() -> JSONResponse:
    return JSONResponse(
        {"code": SAMPLE_PATH.read_text(encoding="utf-8")},
        headers={"Cache-Control": "no-store"},
    )


@app.post("/scan")
async def scan(request: Request) -> JSONResponse:
    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        return JSONResponse({"error": "rate limit reached, try again in a minute"}, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
    code = body.get("code", "")
    if not isinstance(code, str):
        return JSONResponse({"error": 'body must be JSON: {"code": "..."}'}, status_code=400)
    if len(code) > MAX_INPUT_CHARS:
        return JSONResponse(
            {"error": f"input too large (max {MAX_INPUT_CHARS} chars)"}, status_code=413
        )

    try:
        result = analyze_code(code)
        if result.get("error"):
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": f"analysis failed: {type(exc).__name__}"}, status_code=400)


def analyze_code(code: str) -> dict:
    """Run the static pipeline on pasted code and return a structured report."""
    # fail fast and explain: non-Python input or a typo both land here
    try:
        ast.parse(code)
    except SyntaxError as exc:
        if re.search(r"^\s*(const|let|var|function|export|async)\b", code, re.M) or "=>" in code:
            return {
                "error": "that looks like JavaScript, not Python. CostPilot scans Python code only."
            }
        return {"error": "syntax error: " + exc.msg}
    with tempfile.TemporaryDirectory(prefix="costpilot_demo_") as tmp:
        src = Path(tmp) / "app.py"
        src.write_text(code, encoding="utf-8")
        sites = discover_ai_calls(src)
        estimates = profile_cost(sites)
        candidates = find_optimization_candidates(estimates)

    def _model_line(approx: int, model_value) -> int:
        """No-op: scanner now reports the model-kwarg line directly (regression
        test: costpilot/tests/test_scanner.py::TestScannerLineAnchorsOnModelKwarg).

        Previously this function tried to refine the scanner's Call-node line to
        the actual model= line with a regex, but the regex matched
        `model="gpt-4o"` against `model="gpt-4o-mini"` by prefix and matched
        any `model=` row for an unresolved site. Result: three sites from
        sample_app.py were collapsed onto line 43 with contradictory verdicts
        (APPROVE / KEEP / NOT PRICED) and the saving shown was wrong.
        """
        del model_value  # explicitly unused; kept for API compatibility
        return approx

    est_by_site = {(e.site.file, e.site.line): e for e in estimates}
    rows: list[dict] = []
    before_total = after_total = 0.0
    for cand in candidates:
        est = est_by_site.get((cand.site.file, cand.site.line))
        row = {
            "line": _model_line(cand.site.line, cand.site.model),
            "model": cand.site.model,
            "framework": cand.site.framework.value,
            "source": cand.site.model_source,
            "decision": "KEEP" if cand.rejected else cand.kind.value,
            "suggestion": cand.suggestion,
            "confidence": cand.confidence,
        }
        if est is not None and est.is_priced:
            row["cost_per_invocation_usd"] = est.cost_per_invocation
        if cand.expected_saving_ratio is not None:
            row["expected_saving_ratio"] = cand.expected_saving_ratio
        # provable downgrade that the rule layer would actually execute:
        # literal model strings only (shared definitions are left to a human)
        if (
            est is not None
            and est.is_priced
            and cand.new_model
            and not cand.rejected
            and cand.site.model_source == "literal"
        ):
            sim = simulate_optimization(est, cand.new_model)
            if sim.get("priced"):
                before_total += sim["before_per_1k"]
                after_total += sim["after_per_1k"]
                row["before_per_1k"] = sim["before_per_1k"]
                row["after_per_1k"] = sim["after_per_1k"]
        rows.append(row)

    # Collapse "constructor + call-site" duplicates (LangChain pattern):
    # `llm = ChatOpenAI(model='x', ...)` on line N followed by
    # `llm.invoke(...)` on line N+1 produces two sites — the call-site has no
    # model of its own and shows as NOT PRICED. That is honest but confusing
    # (one logical unit, two cards). When an unresolvable site sits within
    # ±2 lines of a site that DID resolve a model, drop the unresolvable one;
    # the constructor card already carries the model and cost.
    if rows:
        lines_with_model = {r["line"] for r in rows if r.get("model")}
        rows = [
            r
            for r in rows
            if r.get("model") is not None
            or not any(abs(r["line"] - other) <= 2 for other in lines_with_model)
        ]

    priced = sum(1 for e in estimates if e.is_priced)
    saving = round(1.0 - after_total / before_total, 4) if before_total > 0 else None

    # pricing_basis: only the models involved in *this* scan, with their public
    # unit prices. Unknown models are listed honestly with null prices so the UI
    # can show "not in price table" rather than hiding them.
    seen: dict[str, dict] = {}
    for est in estimates:
        model = est.site.model or "unknown"
        if model not in seen:
            seen[model] = {
                "model": model,
                "input_per_1m": est.price_per_1m_in,
                "output_per_1m": est.price_per_1m_out,
                "priced": est.is_priced,
            }

    return {
        "call_sites": len(sites),
        "priced": priced,
        "unknown_pricing": len(sites) - priced,
        "summary": {
            "before_per_1k": round(before_total, 4),
            "after_per_1k": round(after_total, 4),
            "saving_ratio": saving,
        },
        "candidates": rows,
        "pricing_basis": {
            "snapshot": "2026-08",
            "source": "vendor public pricing pages",
            "unit": "USD per 1M tokens",
            "models": list(seen.values()),
        },
        "note": "static estimates only; unknown pricing is never guessed",
    }
