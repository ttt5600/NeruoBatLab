"""mbot -- Metaculus FutureEval / MiniBench forecasting bot.

Pipeline: fetch open tournament questions -> route by type -> research (AskNews + free official-data tools)
-> N independent LLM samples -> median aggregation -> calibration guards -> validated payload
-> official REST submission (POST /api/questions/forecast/ + private comment), logged to a JSONL ledger
that also enforces dedupe and the monthly spend cap.

Entry point: `python -m mbot --help` (see README.md).
"""

__version__ = "0.1.0"
