"""CB2 (Deribit-smile barrier, cb2_crypto_barrier_smile_taker) / CB3 (MAX-rung NO, cb3_crypto_max_no_taker) forward paper.

Logger scripts/crypto_barrier_forward.py (same launchd job crypto-barrier, same daily snapshot as CB1); models
scripts/crypto_barrier_v2_model.py; prereg notes/crypto_barrier_v2.md. params: {rule: CB2 | CB3}.
Source (read-only): ~/.claude/quant-sports/data/kx_forward/crypto_barrier_v2.jsonl (repo data/ as fallback).
  `pick`   one per (rule, TRIAL market), the first UTC day that rule fires: status bet (Kalshi GET started in
           [16:00, 16:30) UTC) | off_time. Only status == 'bet' is a bet.
  `settle` one per picked market once settled: result yes|no (market-level, shared by both rules).
Bet row conventions = CB1 (edgelab/adapters/crypto_barrier_fwd.py): decided_at = end of the Kalshi GET; inputs_as_of =
  newest model input (DVOL, Deribit book summary, Coinbase hour end / ticker; all < Kalshi GET start); price = taker ask,
  price_time = GET start; event_start = scheduled window end (in-play by construction); cluster = ET month of window end.
Outcome (hidden): `result`. Unsettled bets are pending.
"""
from __future__ import annotations

import pandas as pd

from ._common import AdapterError, date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    rule = params.get("rule")
    if rule not in ("CB2", "CB3"):
        raise AdapterError(f"params.rule must be CB2 or CB3, got {rule!r}")
    f = qs_or_repo("kx_forward/crypto_barrier_v2.jsonl")
    rs = read_jsonl(f)
    picks = [r for r in rs if r.get("kind") == "pick" and r.get("rule") == rule]
    ndays = sum(1 for r in rs if r.get("kind") == "day")
    if not picks:
        e = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        e.attrs["loader_report"] = {"source": str(f), "rule": rule, "pick_rows": 0, "days": ndays}
        return e, [str(f)]
    res = {}
    for r in rs:
        if r.get("kind") == "settle":
            if r["ticker"] in res and res[r["ticker"]] != r["result"]:
                raise AdapterError(f"{f}: {r['ticker']} settled twice with different results")
            res[r["ticker"]] = r["result"]
    d = pd.DataFrame(picks)
    if d["ticker"].duplicated().any():
        raise AdapterError(f"{f}: {rule} picked a market twice: {d.loc[d['ticker'].duplicated(), 'ticker'].tolist()[:3]}")
    d["bet_id"] = rule + "|" + d["ticker"].astype(str)
    d["date"] = pd.to_datetime(d["date"])                       # UTC decision date = the vault split column
    d["result"] = d["ticker"].map(res)
    d["settled"] = d["result"].isin(["yes", "no"])
    d.attrs["loader_report"] = {"source": str(f), "rule": rule, "pick_rows": int(len(d)), "days": ndays,
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
    if params.get("rule") == "CB3" and not (g["side"] == "no").all():
        raise AdapterError("CB3 bet with a side other than no")
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
