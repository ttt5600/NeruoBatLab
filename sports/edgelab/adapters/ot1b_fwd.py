"""OT1b monotone-implied crypto-taker forward paper (trial ot1b_crypto_taker_mono; logger scripts/ot1_forward.py
second rule column, prereg notes/ot1b_monotone.md). Identical to adapters/ot1_fwd.py except it reads `pick_b` rows.

Source (read-only): ~/.claude/quant-sports/data/kx_forward/ot1.jsonl (launchd job ot1; repo data/ as fallback).
  `pick_b` one per OT1b rule-passing market at the 11:00 ET decision: status bet | outside_frame | off_time. Only
           status == 'bet' (on time, inside the DEV amendment-2 frame) is a bet; the rest are counted in the report.
  `settle` one per event once final: results {ticker: 'yes'|'no'} for every rung.
Bet row (same staking as DEV edgelab/rules/opt_tails.py: $1 of cost per bet, exact taker fee from the spec):
  decided_at = end of the Kalshi /markets GET; inputs_as_of = newest Deribit trade used (< the Kalshi GET start);
  price = ask of the side bought at that GET (taker), price_time = start of that GET; event_start = close - 1 s;
  cluster = 'crypto|<ET close date>' (BTC + ETH on a day share one cluster, as in DEV). close_price: not logged (NaN).
Outcome (hidden): `result` = the winning side of the ticker. Unsettled bets are pending (counted, not scored).
"""
from __future__ import annotations

import pandas as pd

from ._common import AdapterError, date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    f = qs_or_repo("kx_forward/ot1.jsonl")
    rs = read_jsonl(f)
    picks = [r for r in rs if r.get("kind") == "pick_b"]
    if not picks:
        e = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        e.attrs["loader_report"] = {"source": str(f), "pick_rows": 0}
        return e, [str(f)]
    res = {}
    for r in rs:
        if r.get("kind") == "settle":
            for tk, v in r["results"].items():
                if tk in res and res[tk] != v:
                    raise AdapterError(f"{f}: {tk} settled twice with different results")
                res[tk] = v
    d = pd.DataFrame(picks)
    if d["ticker"].duplicated().any():
        raise AdapterError(f"{f}: a market was decided twice: {d.loc[d['ticker'].duplicated(), 'ticker'].tolist()[:3]}")
    d["bet_id"] = d["ticker"].astype(str)
    d["date"] = pd.to_datetime(d["date"])                     # ET close date = the vault split column
    d["result"] = d["ticker"].map(res)
    d["settled"] = d["result"].isin(["yes", "no"])
    days = [r for r in rs if r.get("kind") == "day"]
    d.attrs["loader_report"] = {"source": str(f), "pick_rows": int(len(d)), "days": len(days),
                                "day_status": pd.Series([x.get("status") for x in days]).value_counts().to_dict(),
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
                "bet_dates": date_range(g["date"])})
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(),
                      "cluster": ("crypto|" + g["date"].dt.strftime("%Y-%m-%d")).to_numpy(),
                      "event_start": pd.to_datetime(g["event_start"], utc=True).to_numpy(),
                      "decided_at": pd.to_datetime(g["decided_at"], utc=True).to_numpy(),
                      "inputs_as_of": pd.to_datetime(g["inputs_as_of"], utc=True).to_numpy(),
                      "price": g["ask"].astype(float).to_numpy(), "price_source": "taker",
                      "price_time": pd.to_datetime(g["quote_time"], utc=True).to_numpy(),
                      "side": g["side"].to_numpy(), "stake": 1.0, "close_price": float("nan")})
    return with_report(b, rep)
