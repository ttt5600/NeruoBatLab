"""E4 06:00-D add-on maker (e4t_0600_addon_fwd): forward bets from the paper logger scripts/e4_0600_paper.py.

Source (read-only): ~/.claude/quant-sports/data/kx_forward/e4_0600.jsonl (launchd job e4-0600; repo data/ as fallback).
  `pick`   one per (orig-7 E4 ticker, event day D) at 06:00-06:40 station-local on D: cohort = state of the LIVE 22:00
           maker order in the executor journal at decision time; status bid / skip_* / missed; q = hypothetical NO bid.
  `settle` one per bid after the market settled: tt_fill (a print strictly below q in (decided_at, close]), fill_ts,
           result (the exchange's own), close_no.
Bets = PRIMARY population only (prereg notes/ops_e4_0600.md): cohort == 'filled' AND status == 'bid' AND tt_fill.
  decided_at = inputs_as_of = the decision tick (book read then; forecasts are the frozen 22:00 D-1 pick)
  event_start = market close_time. This is an IN-PLAY decision (after local midnight of D, the market has seen the
                morning); following edgelab's in-play convention (rules/lock_in.py, cbb_ingame.py) event_start = the time
                the outcome is fixed. Disclosed: the decision is NOT pre-event.
  price = q, price_source maker_tape (modeled fill from the public tape), price_time = fill_ts, fill_deadline = close
  contracts 1, stake q; close_price = last NO-taker print in (decided_at, decided_at + 10 h] (E4 CLV close) or NaN.
Outcome (hidden): settle `result` ('no' -> our NO wins).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    f = qs_or_repo("kx_forward/e4_0600.jsonl")
    rs = read_jsonl(f)
    picks = [r for r in rs if r.get("kind") == "pick"]
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    if not picks:
        empty.attrs["loader_report"] = {"source": str(f), "pick_rows": 0}
        return empty, [str(f)]
    d = pd.DataFrame(picks)
    if d["ticker"].duplicated().any():
        raise RuntimeError(f"{f}: a ticker was decided twice: {d.loc[d.ticker.duplicated(), 'ticker'].tolist()[:3]}")
    st = [r for r in rs if r.get("kind") == "settle"]
    sd = {r["ticker"]: r for r in st}
    if len(sd) != len(st):
        raise RuntimeError(f"{f}: a ticker was settled twice")
    d["bet_id"] = d["ticker"]
    d["date"] = pd.to_datetime(d["event_day"])
    for k in ("tt_fill", "fill_ts", "result", "close_no"):
        d[k] = d["ticker"].map(lambda t, k=k: sd.get(t, {}).get(k))
    d["close_time"] = d["ticker"].map(lambda t: sd.get(t, {}).get("close_time")).fillna(d.get("close_time"))
    d["settled"] = d["result"].notna()
    d.attrs["loader_report"] = {"source": str(f), "pick_rows": int(len(d)), "settle_rows": len(st),
                                "cohort_status": {f"{a}|{b}": int(n) for (a, b), n in
                                                  d.groupby(["cohort", "status"]).size().items()}}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "status" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    prim = (f["cohort"] == "filled") & (f["status"] == "bid")
    filled = prim & (f["tt_fill"] == True)  # noqa: E712
    ok = filled & f["settled"]
    g = f[ok]
    dec = pd.to_datetime(g["decided_at"], utc=True)
    q = g["q"].to_numpy(float)
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["event_day"].to_numpy(),
                      "event_start": pd.to_datetime(g["close_time"], utc=True).to_numpy(),
                      "decided_at": dec.to_numpy(), "inputs_as_of": dec.to_numpy(), "price": q,
                      "price_source": "maker_tape", "price_time": pd.to_datetime(g["fill_ts"], utc=True).to_numpy(),
                      "side": "no", "contracts": 1.0, "stake": q,
                      "close_price": pd.to_numeric(g["close_no"], errors="coerce").to_numpy(float),
                      "fill_deadline": pd.to_datetime(g["close_time"], utc=True).to_numpy()})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "primary_bids": int(prim.sum()),
                "primary_tt_filled": int(filled.sum()), "primary_bids_unsettled": int((prim & ~f["settled"]).sum()),
                "bets": int(ok.sum()), "close_missing": int(np.isnan(b["close_price"].to_numpy(float)).sum())})
    return with_report(b, rep)
