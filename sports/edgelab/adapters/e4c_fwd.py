"""E4C cold-band maker paper (e4c_cold1_maker_fwd): forward bets from scripts/e4c_cold_paper.py.

Source (read-only): ~/.claude/quant-sports/data/kx_forward/e4c_cold.jsonl (repo data/ as fallback).
  `pick`   one per EXTRA bracket (top edge 1-3F below the coldest live member, not an E4 pick) selected from e4p's
           22:00-22:50 station-local D-1 decision rows: quote at decision, hypothetical maker q, status bid | skip_*.
  `settle` one per pick once settled: result, tt_fill (print strictly below q in (decided_at, close]), fill_ts, close_no,
           e4_live_final (ticker appeared in the live E4 log by settle time -> an E4 pick, excluded).
Bets = status bid & tt_fill & settled & not e4_live_final. Prereg sports/notes/e4_coldmargin.md.
  decided_at = inputs_as_of = e4p's decision tick; event_start = local midnight of D at the station
  price = q, price_source maker_tape, price_time = fill_ts, fill_deadline = close; contracts 1, stake q
Outcome (hidden): settle `result` ('no' -> our NO wins).
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ._common import date_range, empty_bets, qs_or_repo, read_jsonl, with_report
from .e4p_fwd import TZ

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    f = qs_or_repo("kx_forward/e4c_cold.jsonl")
    rs = read_jsonl(f)
    pk = [r for r in rs if r.get("kind") == "pick"]
    if not pk:
        empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
        empty.attrs["loader_report"] = {"source": str(f), "pick_rows": 0}
        return empty, [str(f)]
    d = pd.DataFrame(pk)
    if d["ticker"].duplicated().any():
        raise RuntimeError(f"{f}: a ticker was picked twice: {d.loc[d.ticker.duplicated(), 'ticker'].tolist()[:3]}")
    st = [r for r in rs if r.get("kind") == "settle"]
    sd = {r["ticker"]: r for r in st}
    if len(sd) != len(st):
        raise RuntimeError(f"{f}: a ticker was settled twice")
    d["bet_id"] = d["ticker"]
    d["date"] = pd.to_datetime(d["event_day"])
    for k in ("tt_fill", "fill_ts", "result", "close_no", "e4_live_final"):
        d[k] = d["ticker"].map(lambda t, k=k: sd.get(t, {}).get(k))
    d["close_time"] = d["ticker"].map(lambda t: sd.get(t, {}).get("close_time")).fillna(d["close_time"])
    d["settled"] = d["result"].notna()
    d.attrs["loader_report"] = {"source": str(f), "pick_rows": int(len(d)), "settle_rows": len(st)}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "status" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    sel = f["status"] == "bid"
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
                      "price_source": "maker_tape", "price_time": pd.to_datetime(g["fill_ts"], unit="s", utc=True).to_numpy(),
                      "side": "no", "contracts": 1.0, "stake": q,
                      "close_price": pd.to_numeric(g["close_no"], errors="coerce").to_numpy(float),
                      "fill_deadline": pd.to_datetime(g["close_time"], utc=True).to_numpy()})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "bids": int(sel.sum()),
                "tt_filled": int(filled.sum()), "excluded_e4_live_late": int((filled & late).sum()),
                "bids_unsettled": int((sel & ~f["settled"]).sum()), "bets": int(ok.sum()),
                "close_missing": int(np.isnan(b["close_price"].to_numpy(float)).sum())})
    return with_report(b, rep)
