"""Kalshi-execution forward paper trials (L32b-K1, K-NFL, K-CFB): bets from scripts/kx_forward.py.

Source (read-only): ~/.claude/quant-sports/data/kx_forward/<trial>.jsonl (launchd copy; repo data/ as fallback), written
by the launchd jobs kx-k1 / kx-knfl / kx-kcfb (notes/ops_forward_wiring.md). params: {trial: k1|knfl|kcfb}.
  `pick`   one per decision: status (bet / skip / no quote ...), S, ticker, side (yes|no), ask, fee, quote_end,
           decided_at, event_start. Only status == 'bet' rows are bets; the rest are counted in the report.
  `settle` one per bet once final: result = winning side on the ticker ('yes'/'no'/'void'), graded on the final vs S
           exactly as the pre-registered dev scripts; kalshi_result = the exchange's own result (hidden too; disagreements counted in load()).
Bet row:
  decided_at = inputs_as_of = the decision tick (K1: L32b tick; K-NFL: open_lines_logger tick; K-CFB: first-tight hour)
  price      = ask (taker; 1 - yes_bid for NO), price_source taker, price_time = quote candle end (<= decided_at)
  contracts  = 100 (the specs' exact-fee unit); stake = 100 x ask; fee from the spec's schedule (evaluator)
  event_start= tip / kickoff; cluster = event date (UTC date of event_start)
  close      = close_mid: same-side MID of the last candle ending <= event start (1-min for K1/K-NFL, hourly for K-CFB).
               A mid, not an ask: price-CLV vs our ask is biased LOW by ~half the spread. The pre-registered judge uses
               the points CLV (clv_S / clv) in the log, which edgelab does not score.
Outcome (hidden): `result`. Unsettled bets are pending (counted, not scored).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import AdapterError, date_range, empty_bets, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result", "kalshi_result"]
KEY = {"k1": "key", "knfl": "bet_key", "kcfb": "event", "k50nfl": "bet_key"}
CONTRACTS = 100.0


def load(params: dict):
    tr = params["trial"]
    if tr not in KEY:
        raise AdapterError(f"unknown kx trial {tr!r}")
    f = qs_or_repo(f"kx_forward/{tr}.jsonl")
    rs = read_jsonl(f)
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    picks = [r for r in rs if r.get("kind") == "pick"]
    if not picks:
        empty.attrs["loader_report"] = {"source": str(f), "pick_rows": 0}
        return empty, [str(f)]
    k = KEY[tr]
    d = pd.DataFrame(picks)
    if d[k].duplicated().any():
        raise AdapterError(f"{f}: a decision was logged twice: {d.loc[d[k].duplicated(), k].tolist()[:3]}")
    st = [r for r in rs if r.get("kind") == "settle"]
    sd = {r[k]: r for r in st}
    if len(sd) != len(st):
        raise AdapterError(f"{f}: a bet was settled twice")
    d["bet_id"] = d[k].astype(str)
    d["event_start"] = pd.to_datetime(d["event_start"], utc=True)
    d["date"] = d["event_start"].dt.tz_localize(None).dt.normalize()
    d["result"] = d[k].map(lambda x: sd.get(x, {}).get("result"))
    d["kalshi_result"] = d[k].map(lambda x: sd.get(x, {}).get("kalshi_result"))
    d["close_mid"] = d[k].map(lambda x: sd.get(x, {}).get("close_mid"))
    d["settled"] = d["result"].notna()
    sb = d[(d["status"] == "bet") & d["settled"]]
    kr = sb["kalshi_result"]
    d.attrs["loader_report"] = {"source": str(f), "pick_rows": int(len(d)),
                                "status": d["status"].value_counts().to_dict(), "settle_rows": len(st),
                                "kalshi_result_missing": int(kr.isna().sum()),
                                "kalshi_result_disagrees": int((kr.notna() & (kr != sb["result"])).sum())}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "status" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    bet = f["status"] == "bet"
    ok = bet & f["settled"]
    g = f[ok]
    if g["side"].isin(["yes", "no"]).sum() != len(g):
        raise AdapterError("bet rows with a side other than yes/no")
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "decisions": int(len(f)),
                "bets_logged": int(bet.sum()), "bets_unsettled_pending": int((bet & ~f["settled"]).sum()),
                "bets": int(ok.sum()), "bet_dates": date_range(g["date"]),
                "close_missing": int(pd.to_numeric(g["close_mid"], errors="coerce").isna().sum())})
    dec = pd.to_datetime(g["decided_at"], utc=True)
    price = g["ask"].to_numpy(float)
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                      "event_start": g["event_start"].to_numpy(), "decided_at": dec.to_numpy(),
                      "inputs_as_of": dec.to_numpy(), "price": price, "price_source": "taker",
                      "price_time": pd.to_datetime(g["quote_end"], utc=True).to_numpy(), "side": g["side"].to_numpy(),
                      "contracts": CONTRACTS, "stake": CONTRACTS * price,
                      "close_price": pd.to_numeric(g["close_mid"], errors="coerce").to_numpy(float)})
    return with_report(b, rep)
