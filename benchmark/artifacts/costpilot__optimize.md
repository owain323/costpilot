# CostPilot Change Request: `costpilot/optimize`

- **Status:** ready_for_human_review
- **Decision:** MODEL_DOWNGRADE
- **Commit:** `9e69cb5b4b20d32b7337e987b95bc91282101199`
- **Changed files:** sample_app.py

## Cost evidence
- **Before:** $1.0100 / 1K calls → **After:** $0.0606 / 1K calls
- **Expected saving:** 94.0%

## Validation
- **syntax:** ok
- **tests:** passed
- **summary:** ..
----------------------------------------------------------------------
Ran 2 tests in 0.001s

OK
- **call_sites_after:** 5
- **priced_sites_after:** 4

## Reasoning
decision: agent-reviewed (c0=approve; c1=keep; c2=keep; c3=keep; c4=keep)

sample_app.py:14 Change model from 'gpt-4o' to 'gpt-4o-mini'

## Risk
Savings are price-saving potential from public pricing (not realized cost). The downgrade satisfies the explicit safety criteria (short prompt, bounded output, literal model source, repository tests pass). Not evaluated: output quality, refusal rate, JSON schema stability, tool-calling behavior, latency, and token-usage variance of the target model. These are the human's checklist when reviewing and merging the diff.

> The agent owns the work. The human owns the decision.