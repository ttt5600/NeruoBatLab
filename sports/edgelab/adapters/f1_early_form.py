"""F1 NBA early form (f1_nba_early_form): forward bets from the paper logger scripts/nba_early_form_logger.py.

Source (read-only): ~/.claude/quant-sports/data/nba_early_form_log.jsonl (launchd copy; repo copy as fallback).
Selection = the logger's own primary set (cmd_report): rows with fires == True and mins_to_tip >= 15. Nothing is
re-decided here.
  decided_at = price_time = inputs_as_of = the row's ts (the tick: ESPN form uses finals with ET date < game date;
               spread + Kalshi bid/ask fetched in the same tick, seconds after ts -- the logger keeps one stamp)
  price      = Kalshi ask of the fired side (taker; fee from the spec's schedule, 1 contract)
  event_start= ESPN start_utc (tip)
  outcome    = ESPN final (games_<season>.parquet, STATUS_FINAL), hidden; side = home|away
  close      = ESPN closing moneyline of the same side, power de-vigged (the logger's CLV). A fair-probability close vs
               a Kalshi ask: it nets out the vig but NOT Kalshi's spread, so CLV here is biased low by ~half a spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import QS_DATA, DATA
from ._common import date_range, empty_bets, imp_scripts, qs_or_repo, read_jsonl, with_report, first_existing

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result", "home_score", "away_score"]
MIN_FORCE = 15


def _season(d: pd.Timestamp) -> int:
    return d.year + 1 if d.month >= 8 else d.year


def load(params: dict):
    f = qs_or_repo("nba_early_form_log.jsonl")
    rows = read_jsonl(f)
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)}), [str(f)]
    d["date"] = pd.to_datetime(d["date"])
    d["bet_id"] = d["event_id"].astype(str)
    if d["bet_id"].duplicated().any():
        raise RuntimeError(f"{f}: duplicate event_id rows (the logger logs a game once)")
    srcs = [str(f)]
    res, hs, as_, close_h, close_a = {}, {}, {}, {}, {}
    for s in sorted({_season(x) for x in d["date"]}):
        gf = first_existing(QS_DATA / "espn" / "nba" / f"games_{s}.parquet", DATA / "espn" / "nba" / f"games_{s}.parquet")
        g = pd.read_parquet(gf)
        srcs.append(str(gf))
        g = g[g["status"] == "STATUS_FINAL"]
        for r in g.itertuples():
            k = str(r.event_id)
            res[k] = "home" if r.home_score > r.away_score else "away"
            hs[k], as_[k] = r.home_score, r.away_score
        lf = first_existing(QS_DATA / "espn" / "nba" / f"lines_{s}.parquet", DATA / "espn" / "nba" / f"lines_{s}.parquet")
        if lf.exists():
            ln = pd.read_parquet(lf)
            srcs.append(str(lf))
            for r in ln.itertuples():
                close_h[str(r.event_id)], close_a[str(r.event_id)] = r.home_ml_close, r.away_ml_close
    d["settled"] = d["bet_id"].map(lambda k: k in res)
    d["result"] = d["bet_id"].map(res)
    d["home_score"], d["away_score"] = d["bet_id"].map(hs), d["bet_id"].map(as_)
    d["home_ml_close"], d["away_ml_close"] = d["bet_id"].map(close_h), d["bet_id"].map(close_a)
    d.attrs["loader_report"] = {"source": str(f), "rows_logged": int(len(d)),
                                "fired": int(d["fires"].fillna(False).astype(bool).sum())}
    return d, srcs


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "fires" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    imp_scripts()
    import edge_math as em
    f = frame
    fired = f["fires"].fillna(False).astype(bool) & (f["mins_to_tip"] >= MIN_FORCE)
    ok = fired & f["settled"]
    g = f[ok]
    price, close = [], []
    for r in g.itertuples():
        k = r.kalshi or {}
        price.append(k.get(f"{r.side}_ask"))
        hc, ac = r.home_ml_close, r.away_ml_close
        if hc is not None and ac is not None and np.isfinite(hc) and np.isfinite(ac):
            ph = em.devig([hc, ac], "power")[0]
            close.append(ph if r.side == "home" else 1 - ph)
        else:
            close.append(np.nan)
    ts = pd.to_datetime(g["ts"], utc=True)
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": pd.to_datetime(g["start_utc"], utc=True).to_numpy(), "decided_at": ts.to_numpy(),
                      "inputs_as_of": ts.to_numpy(), "price": np.array(price, float), "price_source": "taker",
                      "price_time": ts.to_numpy(), "side": g["side"].to_numpy(), "contracts": 1.0,
                      "stake": np.array(price, float), "close_price": np.array(close, float)})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "fired_primary": int(fired.sum()),
                "fired_unsettled_pending": int((fired & ~f["settled"]).sum()),
                "fired_inside_15min_excluded": int((f["fires"].fillna(False).astype(bool) & (f["mins_to_tip"] < MIN_FORCE)).sum()),
                "bets": int(ok.sum()), "bet_dates": date_range(g["date"]), "close_missing": int(np.isnan(close).sum())})
    return with_report(b, rep)
