"""Pre-registered go/kill rule for the pilot, applied mechanically (`python -m mbot evaluate`).

Written 2026-11-25, BEFORE any real forecast or paid call. Do not edit the thresholds after the first live run; a
change needs a new dated rule and applies only to rounds that start after it. It merges the two rules on record and
keeps the stricter side of each: notes/metaculus_bot_tourney.md "Plan if go" step 3 (success: coverage >= 95% AND
avg peer >= +5 over 2 rounds; kill: coverage < 90% OR avg < 0) and the build brief (continue if mean peer >= 0 over
>= 40 questions; kill if < -2 or spend > cap).

Window: the first two MiniBench rounds the bot runs in unattended (from the first live `run_start` in the ledger),
judged after every question in them has closed. If fewer than 40 forecast questions have resolved with a score by
then, extend round by round until 40 have. The coverage kill applies as soon as >= 40 questions are eligible and the
spend kill applies at once; both can end the pilot before the window closes.

  KILL      coverage < 90%  OR  mean spot peer score < 0  OR  any calendar month's logged spend > the cap
  GO        coverage >= 95% AND mean spot peer >= +5 AND n_scored >= 40
            -> enter the full Spring 2027 season, still under the same monthly cap
  HOLD      anything else with n_scored >= 40 (mean in [0, +5) or coverage in [90%, 95%))
            -> MiniBench only for 2 more rounds, same cap, no tuning on open questions, then re-judge with this rule;
               a second consecutive HOLD counts as KILL
  PENDING   n_scored < 40 and no kill condition met yet

Metrics. coverage = questions forecast / questions in the watched tournaments that opened after the first live run
and have closed. Score = Metaculus' `spot_peer_score` for our forecast (the tournament scoring; my_forecasts.score_data),
averaged over forecast questions that resolved (annulled/ambiguous excluded). The mean is reported with a
question-level bootstrap 95% CI; when the CI spans 0 the write-up must say "not distinguishable from 0" even if the
point estimate decides the zone.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from .backtest import bootstrap_ci
from .ledger import Ledger
from .questions import questions_from_post

RULE_DATE = "2026-11-25"
MIN_SCORED = 40
KILL_COVERAGE = 0.90
GO_COVERAGE = 0.95
KILL_MEAN = 0.0
GO_MEAN = 5.0


def monthly_spend(ledger: Ledger) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for r in ledger.rows():
        if r.get("kind") in ("llm_call", "llm_error"):
            out[str(r.get("ts", ""))[:7]] += float(r.get("cost_usd") or 0.0)
    return dict(out)


def decide(n_eligible: int, n_forecast: int, scores: list[float], month_spend: dict[str, float], cap: float,
           previous_verdict: str | None = None) -> dict:
    coverage = n_forecast / n_eligible if n_eligible else None
    mean, lo, hi = bootstrap_ci(scores) if scores else (None, None, None)
    over_cap = {m: round(v, 2) for m, v in month_spend.items() if v > cap + 1e-9}
    reasons = []
    if coverage is not None and n_eligible >= MIN_SCORED and coverage < KILL_COVERAGE:
        reasons.append(f"coverage {coverage:.1%} < {KILL_COVERAGE:.0%}")
    if over_cap:
        reasons.append(f"spend over the ${cap:.0f} cap in {over_cap}")
    if len(scores) >= MIN_SCORED and mean < KILL_MEAN:
        reasons.append(f"mean spot peer {mean:+.2f} < {KILL_MEAN:+.0f}")
    if reasons:
        verdict = "KILL"
    elif len(scores) < MIN_SCORED:
        verdict = "PENDING"
        reasons.append(f"only {len(scores)} scored questions (< {MIN_SCORED})")
    elif coverage is not None and coverage >= GO_COVERAGE and mean >= GO_MEAN:
        verdict = "GO"
        reasons.append(f"coverage {coverage:.1%} >= {GO_COVERAGE:.0%} and mean {mean:+.2f} >= {GO_MEAN:+.0f}")
    else:
        verdict = "KILL" if previous_verdict == "HOLD" else "HOLD"
        reasons.append(f"coverage {coverage:.1%}, mean {mean:+.2f}: between the kill and go bars"
                       + ("; second consecutive HOLD" if previous_verdict == "HOLD" else ""))
    ci_note = None
    if mean is not None:
        ci_note = ("not distinguishable from 0" if lo <= 0 <= hi else ("above 0" if lo > 0 else "below 0"))
    return {"rule_date": RULE_DATE, "verdict": verdict, "reasons": reasons, "n_eligible": n_eligible,
            "n_forecast": n_forecast, "coverage": coverage, "n_scored": len(scores),
            "mean_spot_peer": mean, "ci95": [lo, hi] if mean is not None else None, "ci_reading": ci_note,
            "total_spot_peer": sum(scores) if scores else 0.0, "month_spend": month_spend, "cap": cap}


def first_live_run(ledger: Ledger) -> datetime | None:
    for r in ledger.rows():
        if r.get("kind") == "run_start" and r.get("dry_run") is False:
            return datetime.fromisoformat(r["ts"])
    return None


def collect(api, ledger: Ledger, tournaments: list[str], since: datetime | None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    submitted = ledger.submitted_question_ids()
    eligible, forecast, scores, rows = 0, 0, [], []
    seen: set[int] = set()
    for t in tournaments:
        for post in api.list_posts(t, statuses=["closed", "resolved"], page_size=100, max_pages=20):
            for q in questions_from_post(post):
                if q.question_id in seen:
                    continue
                seen.add(q.question_id)
                if since and (q.open_time is None or q.open_time < since):
                    continue
                if q.scheduled_close_time and q.scheduled_close_time > now:
                    continue
                eligible += 1
                did = q.question_id in submitted or q.already_forecast
                forecast += did
                s = q.my_score_data.get("spot_peer_score")
                ok = did and q.resolution not in (None, "annulled", "ambiguous") and s is not None
                if ok:
                    scores.append(float(s))
                rows.append({"question_id": q.question_id, "tournament": t, "forecast": did,
                             "resolution": q.resolution, "spot_peer": s if ok else None})
    return {"eligible": eligible, "forecast": forecast, "scores": scores, "rows": rows}
