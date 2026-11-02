"""Bet-row schema and the point-in-time checks (GOAL.md rule 4).

One row = one bet. Required columns:
    bet_id        unique per row
    cluster       event-day (or event-week for weekly markets) -- the bootstrap / concentration unit
    event_start   UTC time from which outcome information could exist (tip-off; local midnight of a weather day;
                  first public data release of a weekly series). A bet must be decided strictly before it.
    decided_at    UTC time the decision was made
    inputs_as_of  UTC time of the newest input the rule used
    price         price paid per $1 payout BEFORE fees (binary contract price; 1/decimal odds for a book)
    price_source  taker | maker_tape | book_quote | maker_fill (a real exchange fill of our resting order, forward
                  adapters only; PIT-checked like maker_tape: fill >= post and <= fill_deadline)
    price_time    UTC time of the quote (taker/book: must be <= decided_at; maker_tape: the modeled fill time,
                  which must be >= decided_at and < event_start... or <= settlement for markets that trade
                  through the event, see `maker_fill_until`)
    side          free text (yes/no/over/under/home/away)
    stake         $ of cost risked (>0)
    outcome       1 win, 0 loss, 0.5 push/void (stake returned)
    contracts     optional; if present the stake is contracts x all-in cost (per-contract staking, as the Kalshi
                  studies score); otherwise `stake` dollars of cost are risked per bet
    pnl           optional; if given it must match the evaluator's own recomputation (the rule cannot set costs)
    close_price   optional; same-side price at the close, same units as `price` (NaN when not observed)

Violations raise PITViolation with per-check counts; nothing is silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source",
            "price_time", "side", "stake", "outcome"]
OPTIONAL = ["pnl", "close_price", "fill_deadline", "contracts"]
PRICE_SOURCES = {"taker", "maker_tape", "book_quote", "maker_fill"}
MAKER_SOURCES = {"maker_tape", "maker_fill"}
TIME_COLS = ["event_start", "decided_at", "inputs_as_of", "price_time", "fill_deadline"]


class SchemaError(ValueError):
    pass


class PITViolation(ValueError):
    def __init__(self, counts: dict, examples: pd.DataFrame):
        self.counts = counts
        self.examples = examples
        super().__init__(f"point-in-time violations {counts} (of {counts.get('_rows')} rows); "
                         f"first offenders:\n{examples.head(5).to_string()}")


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


def normalize(rows: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in rows.columns]
    if missing:
        raise SchemaError(f"bet rows missing required columns {missing}")
    df = rows.copy()
    for c in TIME_COLS:
        if c in df.columns:
            df[c] = _utc(df[c])
    for c in ("price", "stake", "outcome"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "close_price" not in df.columns:
        df["close_price"] = np.nan
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    bad = {
        "bet_id_duplicate": int(df["bet_id"].duplicated().sum()),
        "price_source_unknown": int((~df["price_source"].isin(PRICE_SOURCES)).sum()),
        "price_outside_0_1": int((~df["price"].between(0, 1, inclusive="neither")).sum()),
        "stake_not_positive": int((~(df["stake"] > 0)).sum()),
        "outcome_not_0_half_1": int((~df["outcome"].isin([0, 0.5, 1])).sum()),
        "null_cluster": int(df["cluster"].isna().sum()),
    }
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise SchemaError(f"bet rows fail the schema: {bad}")
    return df


def pit_check(df: pd.DataFrame) -> dict:
    """Return per-check violation counts (0 everywhere = clean). Rows must already be normalized."""
    nat = {c: df[c].isna() for c in ("event_start", "decided_at", "inputs_as_of", "price_time")}
    maker = df["price_source"].isin(MAKER_SOURCES)
    deadline = df["fill_deadline"] if "fill_deadline" in df.columns else df["event_start"]
    deadline = deadline.fillna(df["event_start"])
    checks = {
        "missing_timestamp": nat["event_start"] | nat["decided_at"] | nat["inputs_as_of"] | nat["price_time"],
        "inputs_after_decision": df["inputs_as_of"] > df["decided_at"],
        "decided_after_event_start": df["decided_at"] >= df["event_start"],
        "quote_after_decision": (~maker) & (df["price_time"] > df["decided_at"]),
        "maker_fill_before_post": maker & (df["price_time"] < df["decided_at"]),
        "maker_fill_after_deadline": maker & (df["price_time"] > deadline),
    }
    return {k: int(v.fillna(True).sum()) if k == "missing_timestamp" else int(v.fillna(False).sum())
            for k, v in checks.items()}, checks


def enforce(rows: pd.DataFrame) -> pd.DataFrame:
    df = normalize(rows)
    counts, masks = pit_check(df)
    if any(counts.values()):
        any_bad = np.zeros(len(df), bool)
        for m in masks.values():
            any_bad |= m.fillna(True).to_numpy(bool)
        raise PITViolation({**{k: v for k, v in counts.items() if v}, "_rows": len(df)}, df[any_bad])
    return df
