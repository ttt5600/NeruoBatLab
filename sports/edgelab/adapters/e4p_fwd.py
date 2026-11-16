"""E4 probabilistic rule C / C-agree maker (e4p_c_maker_fwd, e4p_cagree_maker_fwd): forward bets from scripts/e4p_paper.py.

Source (read-only): ~/.claude/quant-sports/data/kx_forward/e4p.jsonl (launchd job e4p; repo data/ as fallback).
  `bracket` one per tradeable bracket decided at 22:00-22:50 station-local D-1: phit (frozen monthly EMOS fit), e4_own,
            e4_live_at_T, sel_c, sel_cagree, hypothetical maker q (E4 price rule), status (bid | skip_*).
  `settle`  one per sel_c bracket once settled: result, tt_fill (print strictly below q in (decided_at, close]), fill_ts,
            close_no, e4_live_final (ticker appeared in the live E4 log by settle time -> NOT a NEW pick, excluded).
params: {variant: C | C_agree}. Prereg sports/notes/ops_e4p_forward.md.
Bets = selected (sel_c / sel_cagree) & status bid & tt_fill & settled & not e4_live_final.
  decided_at = inputs_as_of = the decision tick (forecasts, NBS, quote and book read then)
  event_start = local midnight of D at the station (pre-event decision; = e4_live convention)
  price = q, price_source maker_tape, price_time = fill_ts, fill_deadline = close; contracts 1, stake q
  close_price = last NO-taker print in (decided_at, decided_at + 10 h] (E4 CLV close) or NaN.
Outcome (hidden): settle `result` ('no' -> our NO wins).
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._common import date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]
TZ = {"KXHIGHNY": "America/New_York", "KXHIGHCHI": "America/Chicago", "KXHIGHMIA": "America/New_York",
      "KXHIGHAUS": "America/Chicago", "KXHIGHDEN": "America/Denver", "KXHIGHLAX": "America/Los_Angeles",
      "KXHIGHPHIL": "America/New_York", "KXHIGHTATL": "America/New_York", "KXHIGHTBOS": "America/New_York",
      "KXHIGHTDAL": "America/Chicago", "KXHIGHTDC": "America/New_York", "KXHIGHTEWR": "America/New_York",
      "KXHIGHTHOU": "America/Chicago", "KXHIGHTLV": "America/Los_Angeles", "KXHIGHTMIN": "America/Chicago",
      "KXHIGHTNOLA": "America/Chicago", "KXHIGHTOKC": "America/Chicago", "KXHIGHTPHX": "America/Phoenix",
      "KXHIGHTSAN": "America/Los_Angeles", "KXHIGHTSATX": "America/Chicago", "KXHIGHTSDF": "America/Kentucky/Louisville",
      "KXHIGHTSEA": "America/Los_Angeles", "KXHIGHTSFO": "America/Los_Angeles", "KXHIGHTTTN": "America/New_York"}
SEL = {"C": "sel_c", "C_agree": "sel_cagree"}


def load(params: dict):
    f = qs_or_repo("kx_forward/e4p.jsonl")
    rs = read_jsonl(f)
    br = [r for r in rs if r.get("kind") == "bracket"]
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    if not br:
        empty.attrs["loader_report"] = {"source": str(f), "bracket_rows": 0}
        return empty, [str(f)]
    d = pd.DataFrame(br)
    if d["ticker"].duplicated().any():
        raise RuntimeError(f"{f}: a ticker was decided twice: {d.loc[d.ticker.duplicated(), 'ticker'].tolist()[:3]}")
    st = [r for r in rs if r.get("kind") == "settle"]
    sd = {r["ticker"]: r for r in st}
    if len(sd) != len(st):
        raise RuntimeError(f"{f}: a ticker was settled twice")
    d["bet_id"] = d["ticker"]
    d["date"] = pd.to_datetime(d["event_day"])
    for k in ("tt_fill", "fill_ts", "result", "close_no", "e4_live_final"):
        d[k] = d["ticker"].map(lambda t, k=k: sd.get(t, {}).get(k))
    d["close_time"] = d["ticker"].map(lambda t: sd.get(t, {}).get("close_time")).fillna(d.get("close_time"))
    d["settled"] = d["result"].notna()
    d.attrs["loader_report"] = {"source": str(f), "bracket_rows": int(len(d)), "settle_rows": len(st),
                                "sel_c": int(d["sel_c"].sum()), "sel_cagree": int(d["sel_cagree"].sum())}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    var = (params or {}).get("variant", "C")
    if var not in SEL:
        raise ValueError(f"variant must be one of {list(SEL)}")
    if frame.empty or "status" not in frame:
        return with_report(empty_bets(), {**rep, "variant": var, "rows_in_window": 0, "bets": 0})
    f = frame
    sel = f[SEL[var]].astype(bool) & (f["status"] == "bid")
    late = f["e4_live_final"] == True  # noqa: E712
    filled = sel & (f["tt_fill"] == True)  # noqa: E712
    ok = filled & f["settled"] & ~late
    g = f[ok]
    dec = pd.to_datetime(g["decided_at"], utc=True)
    mid = [datetime.combine(pd.Timestamp(d).date(), dtime(0), ZoneInfo(TZ[s])).astimezone(timezone.utc)
           for d, s in zip(g["event_day"], g["series"])]
    q = g["q"].to_numpy(float)
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["event_day"].to_numpy(),
                      "event_start": pd.to_datetime(pd.Series(mid, dtype=object), utc=True).to_numpy(),
                      "decided_at": dec.to_numpy(), "inputs_as_of": dec.to_numpy(), "price": q,
                      "price_source": "maker_tape", "price_time": pd.to_datetime(g["fill_ts"], utc=True).to_numpy(),
                      "side": "no", "contracts": 1.0, "stake": q,
                      "close_price": pd.to_numeric(g["close_no"], errors="coerce").to_numpy(float),
                      "fill_deadline": pd.to_datetime(g["close_time"], utc=True).to_numpy()})
    rep.update({"variant": var, "rows_in_window": int(len(f)), "dates": date_range(f["date"]), "selected_bids": int(sel.sum()),
                "tt_filled": int(filled.sum()), "excluded_e4_live_late": int((filled & late).sum()),
                "bids_unsettled": int((sel & ~f["settled"]).sum()), "bets": int(ok.sum()),
                "close_missing": int(np.isnan(b["close_price"].to_numpy(float)).sum())})
    return with_report(b, rep)
