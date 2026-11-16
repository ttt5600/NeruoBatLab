"""CB1 crypto barrier taker forward paper (trial cb1_crypto_barrier_taker; logger scripts/crypto_barrier_forward.py,
prereg notes/crypto_barrier.md).

Source (read-only): ~/.claude/quant-sports/data/kx_forward/crypto_barrier.jsonl (launchd job crypto-barrier; repo data/
as fallback).
  `pick`   one per TRIAL market, the first UTC day the frozen rule fires: status bet (Kalshi GET started in
           [16:00, 16:30) UTC) | off_time. Only status == 'bet' is a bet.
  `settle` one per pick once the market settled: result yes|no.
Bet row: decided_at = end of the Kalshi GET; inputs_as_of = newest model input (DVOL candle end, Coinbase hour end,
  Coinbase ticker time; all < the Kalshi GET start); price = ask of the side bought (taker), price_time = GET start;
  event_start = scheduled END of the measurement window (rules_primary). IN-PLAY by construction: a one-touch window
  is already running at decision time (edgelab in-play convention: event_start = when the outcome is fixed at the latest).
  cluster = 'crypto_month|<ET month of the window end>' (all assets / MAX+MIN of a month share one path).
Outcome (hidden): `result` = the market's settled side. Unsettled bets are pending (counted, not scored).
"""
from __future__ import annotations

import pandas as pd

from ._common import AdapterError, date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    f = qs_or_repo("kx_forward/crypto_barrier.jsonl")
    rs = read_jsonl(f)
    picks = [r for r in rs if r.get("kind") == "pick"]
    if not picks:
        e = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        e.attrs["loader_report"] = {"source": str(f), "pick_rows": 0,
                                    "days": sum(1 for r in rs if r.get("kind") == "day")}
        return e, [str(f)]
    res = {}
    for r in rs:
        if r.get("kind") == "settle":
            if r["ticker"] in res and res[r["ticker"]] != r["result"]:
                raise AdapterError(f"{f}: {r['ticker']} settled twice with different results")
            res[r["ticker"]] = r["result"]
    d = pd.DataFrame(picks)
    if d["ticker"].duplicated().any():
        raise AdapterError(f"{f}: a market was picked twice: {d.loc[d['ticker'].duplicated(), 'ticker'].tolist()[:3]}")
    d["bet_id"] = d["ticker"].astype(str)
    d["date"] = pd.to_datetime(d["date"])                       # UTC decision date = the vault split column
    d["result"] = d["ticker"].map(res)
    d["settled"] = d["result"].isin(["yes", "no"])
    d.attrs["loader_report"] = {"source": str(f), "pick_rows": int(len(d)),
                                "days": sum(1 for r in rs if r.get("kind") == "day"),
                                "status": d["status"].value_counts().to_dict()}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "status" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    bet = frame["status"] == "bet"
    g = frame[bet & frame["settled"]]
    if not g["side"].isin(["yes", "no"]).all():
        raise AdapterError("bet rows with a side other than yes/no")
    rep.update({"rows_in_window": int(len(frame)), "dates": date_range(frame["date"]), "bets_logged": int(bet.sum()),
                "bets_unsettled_pending": int((bet & ~frame["settled"]).sum()), "bets": int(len(g)),
                "clusters": int(g["cluster"].nunique()) if len(g) else 0})
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["cluster"].to_numpy(),
                      "event_start": pd.to_datetime(g["window_end"], utc=True).to_numpy(),
                      "decided_at": pd.to_datetime(g["decided_at"], utc=True).to_numpy(),
                      "inputs_as_of": pd.to_datetime(g["inputs_as_of"], utc=True).to_numpy(),
                      "price": g["ask"].astype(float).to_numpy(), "price_source": "taker",
                      "price_time": pd.to_datetime(g["quote_time"], utc=True).to_numpy(),
                      "side": g["side"].to_numpy(), "stake": 1.0, "close_price": float("nan")})
    return with_report(b, rep)
