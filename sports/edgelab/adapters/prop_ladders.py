"""Kalshi NBA prop ladders (kalshi_prop_ladders): forward bets from scripts/kalshi_prop_logger.py.

Sources (read-only): ~/.claude/quant-sports/data/kalshi_prop_log.jsonl (type 'rung' rows; picks have `pick` YES|NO)
and kalshi_prop_grades.jsonl (the logger's `grade`: ESPN box -> y / void, plus close_yes = last print before tip).
Selection = the logger's frozen flag (rows with a pick). Nothing is re-decided here.
  decided_at = price_time = inputs_as_of = the rung's ts (the tick snapshot; quotes fetched in that tick)
  price      = YES: ask; NO: 1 - bid (taker; fee from the spec's schedule; 1 contract)
  event_start= tip_utc; cluster = ET date of the tip
  outcome    = grades (hidden): y == 1 -> 'yes' wins, y == 0 -> 'no'; void (DNP / not in box) -> stake back (0.5;
               Kalshi actually settles inactive players at a fair price -- not modeled, counted in the report)
  close      = grades close_yes (YES) or 1 - close_yes (NO): the last public print before tip
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result", "y", "stat_value", "profit"]


def load(params: dict):
    lf, gf = qs_or_repo("kalshi_prop_log.jsonl"), qs_or_repo("kalshi_prop_grades.jsonl")
    R = pd.DataFrame([r for r in read_jsonl(lf) if r.get("type") == "rung"])
    G = pd.DataFrame([g for g in read_jsonl(gf) if not g.get("pending")])
    srcs = [str(lf), str(gf)]
    if R.empty:
        d = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        d.attrs["loader_report"] = {"source": str(lf), "exists": lf.exists(), "rungs_logged": 0}
        return d, srcs
    R = R.drop_duplicates("ticker", keep="first")
    if not G.empty:
        G = G.drop_duplicates("ticker", keep="last")
        R = R.merge(G.drop(columns=[c for c in ("event_id",) if c in G.columns]), on="ticker", how="left")
    for c in ("void", "y", "close_yes", "stat_value", "profit"):
        if c not in R:
            R[c] = np.nan
    R["bet_id"] = R["ticker"]
    R["date"] = pd.to_datetime(R["tip_utc"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    R["settled"] = R["void"].notna() | R["y"].notna()
    R["result"] = np.where(R["void"].fillna(False).astype(bool), "void",
                           np.where(R["y"] == 1, "yes", np.where(R["y"] == 0, "no", None)))
    R.loc[~R["settled"], "result"] = None
    R.attrs["loader_report"] = {"source": str(lf), "rungs_logged": int(len(R)), "picks": int(R["pick"].notna().sum())}
    return R, srcs


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "pick" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    pk = f["pick"].notna()
    ok = pk & f["settled"]
    g = f[ok]
    yes = (g["pick"] == "YES").to_numpy()
    price = np.where(yes, g["ask"].to_numpy(float), 1 - g["bid"].to_numpy(float))
    cy = pd.to_numeric(g["close_yes"], errors="coerce").to_numpy(float)
    ts = pd.to_datetime(g["ts"], utc=True)
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": pd.to_datetime(g["tip_utc"], utc=True).to_numpy(), "decided_at": ts.to_numpy(),
                      "inputs_as_of": ts.to_numpy(), "price": np.round(price, 4), "price_source": "taker",
                      "price_time": ts.to_numpy(), "side": np.where(yes, "yes", "no"), "contracts": 1.0,
                      "stake": np.round(price, 4), "close_price": np.where(yes, cy, 1 - cy)})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "picks": int(pk.sum()),
                "picks_unsettled_pending": int((pk & ~f["settled"]).sum()),
                "voids_scored_as_push": int((ok & (f["result"] == "void")).sum()), "bets": int(ok.sum()),
                "bet_dates": date_range(g["date"])})
    return with_report(b, rep)
