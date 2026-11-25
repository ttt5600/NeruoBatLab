"""JSONL ledger: one row per event. It is the single source of truth for (a) dedupe -- a question with a
`submitted` row is never posted again -- and (b) the monthly spend cap, summed from `llm_call` rows.

Row kinds: run_start, question_seen, skip, llm_call, llm_error, forecast, dry_run, submitted, submit_error,
comment_error, run_end. Every row has ts (UTC ISO) and kind.
"""
from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ledger:
    def __init__(self, path: str | Path, now_fn: Callable[[], datetime] = utcnow):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.now_fn = now_fn

    def append(self, row: dict) -> dict:
        row = {"ts": self.now_fn().isoformat(timespec="seconds"), **row}
        line = json.dumps(row, default=str, sort_keys=False)
        with open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def rows(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except json.JSONDecodeError:
                    continue      # a torn last line after a crash must not break dedupe or the cap

    def submitted_question_ids(self) -> set[int]:
        return {int(r["question_id"]) for r in self.rows() if r.get("kind") == "submitted" and r.get("question_id") is not None}

    def spent_month(self, now: datetime | None = None) -> float:
        now = now or utcnow()
        tot = 0.0
        for r in self.rows():
            if r.get("kind") not in ("llm_call", "llm_error"):
                continue
            ts = r.get("ts", "")
            if ts[:7] != now.strftime("%Y-%m"):
                continue
            tot += float(r.get("cost_usd") or 0.0)
        return tot


class SpendCapExceeded(RuntimeError):
    pass


class SpendGuard:
    """Hard monthly cap. Before every LLM call: refuse if (spent this UTC month + the call's worst-case cost) > cap.
    After the call: log its actual cost. Spend is read from the ledger at start, so restarts never reset it."""

    def __init__(self, ledger: Ledger, cap_usd: float, now_fn: Callable[[], datetime] = utcnow):
        self.ledger = ledger
        self.cap = float(cap_usd)
        self.now_fn = now_fn
        self._month = now_fn().strftime("%Y-%m")
        self._spent = ledger.spent_month(now_fn())

    @property
    def spent(self) -> float:
        m = self.now_fn().strftime("%Y-%m")
        if m != self._month:            # month rolled over mid-run
            self._month, self._spent = m, self.ledger.spent_month(self.now_fn())
        return self._spent

    def remaining(self) -> float:
        return self.cap - self.spent

    def check(self, worst_case_usd: float) -> None:
        if self.spent + worst_case_usd > self.cap:
            raise SpendCapExceeded(f"spend cap: ${self.spent:.2f} spent this month + worst case ${worst_case_usd:.3f} "
                                   f"> cap ${self.cap:.2f}")

    def record(self, row: dict) -> None:
        self.ledger.append(row)
        self._spent += float(row.get("cost_usd") or 0.0)


class AlreadyRunning(RuntimeError):
    pass


class RunLock:
    """Exclusive non-blocking lock so two scheduled runs can never forecast the same question at once."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            self.fh.close()
            self.fh = None
            raise AlreadyRunning(f"another run holds {self.path}") from e
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{os.getpid()} {utcnow().isoformat(timespec='seconds')}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None
        return False
