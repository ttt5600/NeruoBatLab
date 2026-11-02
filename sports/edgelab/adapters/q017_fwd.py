"""Q017 B (q017_b_far_bracket): ERCOT daily-peak far-bracket bets from the forward paper logger scripts/q017_logger.py.

Sources (read-only): ~/.claude/quant-sports/data/q017/fwd/{quotes,vintages}.jsonl and q017/error_sample.npy.
Selection = the logger's own `report` path with the FROZEN forecast (F_dm1: last ERCOT vintage stamped <= 14:30 CT on
D-1), trials B06 + B10 via q017_ercot_load.decide / trial_bets (imported, not copied). One bet per (ticker, hour).
  decided_at = T = 06:00 / 10:00 CT on operating day D; price_time = the snapshot used (last in [T-45 min, T])
  inputs_as_of = the vintage stamp used (CT -> UTC)
  price      = the side's ask at that snapshot (taker; fee from the spec's schedule; 1 contract)
  event_start= 00:00 CT on D. CONSERVATIVE on purpose: the day's load (hence information about its peak) starts at
               midnight, so every B06/B10 bet (decided after midnight on D) is REJECTED by the PIT check
               (decided_after_event_start). Scoring Q017 needs the lead to pre-register what "before the outcome
               existed" means for an intraday-decided daily max (a NEW trial), not a tweak here.
  outcome    = public Kalshi market result (fetched only for tickers with a bet), hidden.
  close      = not available from this logger -> NaN (CLV coverage 0 -> rule 5e fails closed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import QS_DATA
from ._common import date_range, empty_bets, imp_scripts, market, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]
CT = "America/Chicago"


def load(params: dict):
    F = QS_DATA / "q017" / "fwd"
    vf, qf, ef = F / "vintages.jsonl", F / "quotes.jsonl", QS_DATA / "q017" / "error_sample.npy"
    srcs = [str(vf), str(qf), str(ef)]
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    if not (vf.exists() and qf.exists()):
        empty.attrs["loader_report"] = {"source": str(F), "exists": False}
        return empty, srcs
    imp_scripts()
    import q017_ercot_load as q
    v = pd.read_json(vf, lines=True)
    v["vintage"], v["day"] = pd.to_datetime(v.vintage), pd.to_datetime(v.day)
    qt = pd.read_json(qf, lines=True)
    qt["ts"], qt["opday"] = pd.to_datetime(qt.ts, utc=True), pd.to_datetime(qt.opday)
    r = np.load(ef)
    snaps, missing = [], 0
    for (tk, day), g in qt.groupby(["ticker", "opday"]):
        for h in q.TIMES:
            T_ct = day + pd.Timedelta(hours=h)
            T = T_ct.tz_localize(CT).tz_convert("UTC")
            s = g[(g.ts <= T) & (g.ts >= T - pd.Timedelta(minutes=45))]
            gv = v[(v.day == day) & (v.vintage <= day - pd.Timedelta(days=1) + pd.Timedelta(hours=14, minutes=30))]
            if s.empty or gv.empty:
                missing += 1
                continue
            s = s.sort_values("ts").iloc[-1]
            vi = gv.loc[gv.vintage.idxmax()]
            snaps.append({"ticker": tk, "opday": day, "hour": h, "K": s.K, "F": float(vi.peak), "yes_bid": s.yes_bid,
                          "yes_ask": s.yes_ask, "has_quote": True, "snap_ts": s.ts, "T": T,
                          "vintage_utc": pd.Timestamp(vi.vintage).tz_localize(CT).tz_convert("UTC")})
    rep = {"source": str(F), "quote_ticks": int(qt.ts.nunique()), "quote_span": [str(qt.ts.min()), str(qt.ts.max())],
           "op_days_quoted": int(qt.opday.nunique()), "decision_slots_without_snapshot_or_vintage": missing,
           "decision_snapshots": len(snaps)}
    if not snaps:
        empty.attrs["loader_report"] = rep
        return empty, srcs
    dd = pd.DataFrame(snaps)
    x = q.decide(dd, r, 1.0)
    out = []
    for t in ("B06", "B10"):
        b = q.trial_bets(x, t)
        if b.empty:
            continue
        h = 6 if t == "B06" else 10
        b = b.merge(x[x.hour == h][["ticker", "opday", "snap_ts", "T", "vintage_utc"]], on=["ticker", "opday"])
        b["hour"] = h
        out.append(b)
    if not out:
        empty.attrs["loader_report"] = {**rep, "signals": 0}
        return empty, srcs
    b = pd.concat(out, ignore_index=True)
    b["bet_id"] = b.ticker + "|" + b.hour.astype(str)
    b["date"] = b.opday
    res = {}
    for tk in b.ticker.unique():
        m = market(tk)
        res[tk] = m.get("result") if m.get("result") in ("yes", "no") else None
    b["result"] = b.ticker.map(res)
    b["settled"] = b.result.notna()
    b.attrs["loader_report"] = {**rep, "signals": int(len(b))}
    return b, srcs


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "side" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    g = f[f["settled"]]
    es = pd.to_datetime(g["opday"]).dt.tz_localize(CT).dt.tz_convert("UTC")
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": es.to_numpy(), "decided_at": pd.to_datetime(g["T"], utc=True).to_numpy(),
                      "inputs_as_of": pd.to_datetime(g["vintage_utc"], utc=True).to_numpy(),
                      "price": g["ask"].to_numpy(float), "price_source": "taker",
                      "price_time": pd.to_datetime(g["snap_ts"], utc=True).to_numpy(), "side": g["side"].to_numpy(),
                      "contracts": 1.0, "stake": g["ask"].to_numpy(float), "close_price": np.nan})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]),
                "signals_unsettled_pending": int((~f["settled"]).sum()), "bets": int(len(g))})
    return with_report(b, rep)
