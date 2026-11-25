"""Unattended runner: poll open tournament questions, forecast every one not yet forecast, submit, log.

Never double-submits. Four independent guards, any one of which is enough:
  1. a `submitted` row for the question id in the ledger            (our own record, survives restarts)
  2. `my_forecasts.latest` set on the question in the listing         (Metaculus' record)
  3. the same check on a fresh GET of the post right before the POST (closes the race with a parallel run)
  4. an exclusive run lock (two scheduled runs never overlap)
and a forecast POST that fails ambiguously is never retried blindly: the post is re-read and the outcome recorded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Config
from .forecaster import Forecaster
from .ledger import Ledger, RunLock, SpendCapExceeded, SpendGuard
from .metaculus_api import MetaculusAPI, MetaculusError
from .questions import Question, questions_from_post, route


def load_posts_file(path: str | Path) -> list[dict]:
    body = json.loads(Path(path).read_text())
    if isinstance(body, dict):
        body = body.get("results") or body.get("posts") or []
    return list(body)


def _find_question(post: dict, question_id: int) -> Question | None:
    for q in questions_from_post(post):
        if q.question_id == question_id:
            return q
    return None


class Runner:
    def __init__(self, cfg: Config, forecaster: Forecaster, ledger: Ledger, api: MetaculusAPI | None,
                 dry_run: bool = True, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 printer: Callable[[str], None] = print):
        self.cfg, self.fc, self.ledger, self.api = cfg, forecaster, ledger, api
        self.dry_run = dry_run
        self.now_fn = now_fn
        self.print = printer
        if not dry_run:
            if api is None:
                raise ValueError("a live run needs the Metaculus API (METACULUS_TOKEN)")
            if any(l.provider == "mock" for l in forecaster.llms):
                raise ValueError("refusing to submit forecasts produced by the mock LLM; use --dry-run")

    # ------------------------------------------------------------------ discovery
    def candidates(self, tournaments: list[str], posts_override: list[dict] | None = None) -> tuple[list[Question], list[dict]]:
        submitted = self.ledger.submitted_question_ids()
        now = self.now_fn()
        seen: set[int] = set()
        picked: list[Question] = []
        skips: list[dict] = []
        sources: list[tuple[str, list[dict]]] = []
        if posts_override is not None:
            sources.append(("file", posts_override))
        else:
            for t in tournaments:
                try:
                    sources.append((t, self.api.list_open_posts(t)))
                except MetaculusError as e:
                    skips.append({"tournament": t, "reason": "list_error", "error": str(e)[:300]})
        for src, posts in sources:
            for post in posts:
                for q in questions_from_post(post):
                    if q.question_id in seen:
                        continue
                    seen.add(q.question_id)
                    reason = None
                    if (q.status or "open") != "open":
                        reason = f"status_{q.status}"
                    elif route(q) == "unsupported":
                        reason = f"unsupported_type_{q.type}"
                    elif q.question_id in submitted:
                        reason = "already_submitted_ledger"
                    elif q.already_forecast:
                        reason = "already_forecast_on_metaculus"
                    else:
                        ml = q.minutes_left(now)
                        if ml is not None and ml < self.cfg.min_minutes_left:
                            reason = f"closing_in_{ml:.1f}_min"
                    if reason:
                        skips.append({"question_id": q.question_id, "post_id": q.post_id, "source": src, "reason": reason})
                    else:
                        picked.append(q)
        far = datetime.max.replace(tzinfo=timezone.utc)
        picked.sort(key=lambda q: q.scheduled_close_time or far)     # most urgent first
        return picked[: self.cfg.max_questions_per_run], skips

    # ------------------------------------------------------------------ comments left unposted by a crash
    def retry_missing_comments(self) -> int:
        rows = list(self.ledger.rows())
        posted = {r["question_id"] for r in rows if r.get("kind") == "comment_posted"}
        n = 0
        for r in rows:
            if r.get("kind") == "submitted" and r["question_id"] not in posted and r.get("comment"):
                try:
                    self.api.post_comment(r["post_id"], r["comment"])
                    self.ledger.append({"kind": "comment_posted", "question_id": r["question_id"], "retry": True})
                    posted.add(r["question_id"])
                    n += 1
                except Exception as e:  # noqa: BLE001
                    self.ledger.append({"kind": "comment_error", "question_id": r["question_id"], "error": str(e)[:300]})
        return n

    # ------------------------------------------------------------------ submission
    def submit(self, fr, run_id: str) -> str:
        q = fr.question
        try:
            fresh = _find_question(self.api.get_post(q.post_id), q.question_id)
        except MetaculusError as e:
            self.ledger.append({"kind": "submit_error", "run_id": run_id, "question_id": q.question_id,
                                "stage": "recheck", "error": str(e)[:300]})
            return "recheck_error"
        if fresh is None or (fresh.status or "open") != "open":
            self.ledger.append({"kind": "skip", "run_id": run_id, "question_id": q.question_id, "reason": "closed_before_submit"})
            return "closed"
        if fresh.already_forecast:
            self.ledger.append({"kind": "skip", "run_id": run_id, "question_id": q.question_id, "reason": "already_forecast_recheck"})
            return "already"
        try:
            self.api.post_forecast(q.question_id, fr.payload)
        except Exception as e:  # noqa: BLE001 -- ambiguous: did it land? ask Metaculus, never blind-retry
            landed = False
            try:
                again = _find_question(self.api.get_post(q.post_id), q.question_id)
                landed = bool(again and again.already_forecast)
            except Exception:  # noqa: BLE001
                pass
            if not landed:
                self.ledger.append({"kind": "submit_error", "run_id": run_id, "question_id": q.question_id,
                                    "post_id": q.post_id, "stage": "post_forecast", "error": f"{type(e).__name__}: {str(e)[:300]}"})
                return "submit_error"
        self.ledger.append({"kind": "submitted", "run_id": run_id, "question_id": q.question_id, "post_id": q.post_id,
                            "type": q.type, "payload_kind": fr.family, "readable": fr.readable, "comment": fr.comment})
        try:
            self.api.post_comment(q.post_id, fr.comment)
            self.ledger.append({"kind": "comment_posted", "run_id": run_id, "question_id": q.question_id})
        except Exception as e:  # noqa: BLE001
            self.ledger.append({"kind": "comment_error", "run_id": run_id, "question_id": q.question_id, "error": str(e)[:300]})
        return "submitted"

    # ------------------------------------------------------------------ one pass
    def run(self, tournaments: list[str] | None = None, posts_override: list[dict] | None = None) -> dict:
        tournaments = tournaments or self.cfg.tournaments
        run_id = self.now_fn().strftime("%Y%m%dT%H%M%SZ")
        guard = self.fc.guard
        self.ledger.append({"kind": "run_start", "run_id": run_id, "dry_run": self.dry_run, "tournaments": tournaments,
                            "models": [l.name for l in self.fc.llms], "n_samples": self.cfg.n_samples,
                            "budget_note": self.fc.budget_note,
                            "cap_usd": self.cfg.monthly_cap_usd, "spent_usd": round(guard.spent, 4) if guard else None})
        summary = {"run_id": run_id, "dry_run": self.dry_run, "candidates": 0, "forecast_ok": 0, "forecast_failed": 0,
                   "submitted": 0, "skipped": 0, "spend_cap_hit": False, "comment_retries": 0, "outcomes": {}}
        if not self.dry_run:
            summary["comment_retries"] = self.retry_missing_comments()
        picked, skips = self.candidates(tournaments, posts_override)
        summary["candidates"], summary["skipped"] = len(picked), len(skips)
        for s in skips:
            self.ledger.append({"kind": "skip", "run_id": run_id, **s})
        for q in picked:
            try:
                fr = self.fc.forecast(q, run_id=run_id)
            except SpendCapExceeded as e:
                summary["spend_cap_hit"] = True
                self.ledger.append({"kind": "skip", "run_id": run_id, "question_id": q.question_id, "reason": "spend_cap",
                                    "error": str(e)})
                self.print(f"STOP spend cap: {e}")
                break
            self.ledger.append({"kind": "forecast", "run_id": run_id, "dry_run": self.dry_run, **fr.log()})
            if not fr.ok:
                summary["forecast_failed"] += 1
                self.print(f"FAIL q{q.question_id} {q.title[:70]!r}: {fr.error}")
                continue
            summary["forecast_ok"] += 1
            self.print(f"{'DRY ' if self.dry_run else ''}q{q.question_id} [{q.type}] {q.title[:80]!r}\n"
                       f"    -> {fr.readable}  (samples ok {sum(s.ok for s in fr.samples)}/{len(fr.samples)}, "
                       f"cost ${fr.cost_usd:.4f}){'  WARN ' + '; '.join(fr.warnings) if fr.warnings else ''}")
            if self.dry_run:
                continue
            outcome = self.submit(fr, run_id)
            summary["outcomes"][str(q.question_id)] = outcome
            summary["submitted"] += outcome == "submitted"
        summary["spent_usd_month"] = round(guard.spent, 4) if guard else None
        self.ledger.append({"kind": "run_end", **summary})
        return summary


def run_locked(cfg: Config, runner: Runner, **kw) -> dict:
    with RunLock(cfg.lock_path):
        return runner.run(**kw)
