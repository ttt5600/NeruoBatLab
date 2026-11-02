"""E1X-T T2 maker (tsa_t2_maker): forward bets from the paper logger scripts/e1x_tsa_logger.py.

Source (read-only): ~/.claude/quant-sports/data/e1x/tsa_paper.jsonl -- `post` rows (paper maker order at q), then
`fill` rows (first crossing public trade after the post: fill_ts) and `settle` rows (result), appended by `fills`.
  decided_at  = post ts; inputs_as_of = post ts - 1 h (daily TSA through the previous day, as rules/tsa.py)
  price       = q (paper maker); price_source maker_tape (a modeled fill from the public tape, not a real fill)
  price_time  = fill_ts; fill_deadline = the market's close_time (public market record, fetched for filled tickers)
  event_start = Tuesday 13:00Z of the week (first in-week TSA release; rules/tsa.py)
  outcome     = settle result (hidden); 1 contract
  close       = not logged -> NaN (CLV coverage 0 -> rule 5e fails closed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._common import date_range, empty_bets, market, qs_or_repo, read_jsonl, with_report

OUTCOME = {"kind": "result_side", "key": "bet_id"}
HIDDEN = ["result"]


def load(params: dict):
    f = qs_or_repo("e1x/tsa_paper.jsonl")
    rs = read_jsonl(f)
    posts = [r for r in rs if r.get("kind") == "post"]
    empty = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "bet_id": pd.Series(dtype=str)})
    if not posts:
        empty.attrs["loader_report"] = {"source": str(f), "posts": 0}
        return empty, [str(f)]
    fills = {r["ticker"]: r for r in rs if r.get("kind") == "fill"}
    res = {r["ticker"]: r["result"] for r in rs if r.get("kind") == "settle"}
    d = pd.DataFrame(posts)
    if d["ticker"].duplicated().any():
        raise RuntimeError(f"{f}: a ticker was posted twice")
    d["bet_id"] = d["ticker"]
    d["date"] = pd.to_datetime(d["event"].str.split("-").str[1], format="%y%b%d")
    d["fill_ts"] = d["ticker"].map(lambda t: fills.get(t, {}).get("fill_ts"))
    d["result"] = d["ticker"].map(res)
    d["settled"] = d["result"].notna()
    close = {}
    for t in d.loc[d.fill_ts.notna(), "ticker"]:
        close[t] = market(t).get("close_time")
    d["close_time"] = d["ticker"].map(close)
    d.attrs["loader_report"] = {"source": str(f), "posts": int(len(d)), "fills": int(d.fill_ts.notna().sum()),
                                "settles": int(d.settled.sum())}
    return d, [str(f)]


def rows(frame: pd.DataFrame, params: dict) -> pd.DataFrame:
    rep = dict(frame.attrs.get("loader_report") or {})
    if frame.empty or "fill_ts" not in frame:
        return with_report(empty_bets(), {**rep, "rows_in_window": 0, "bets": 0})
    f = frame
    filled = f["fill_ts"].notna()
    ok = filled & f["settled"]
    g = f[ok]
    ts = pd.to_datetime(g["ts"], utc=True)
    wend = pd.to_datetime(g["date"]).dt.tz_localize("UTC")
    b = pd.DataFrame({"bet_id": g["bet_id"].to_numpy(), "cluster": g["event"].to_numpy(),
                      "event_start": (wend - pd.Timedelta(days=5) + pd.Timedelta(hours=13)).to_numpy(),
                      "decided_at": ts.to_numpy(), "inputs_as_of": (ts - pd.Timedelta(hours=1)).to_numpy(),
                      "price": g["q"].to_numpy(float), "price_source": "maker_tape",
                      "price_time": pd.to_datetime(g["fill_ts"], utc=True).to_numpy(), "side": g["side"].to_numpy(),
                      "contracts": 1.0, "stake": g["q"].to_numpy(float), "close_price": np.nan,
                      "fill_deadline": pd.to_datetime(g["close_time"], utc=True).to_numpy()})
    rep.update({"rows_in_window": int(len(f)), "dates": date_range(f["date"]), "posted": int(len(f)),
                "filled": int(filled.sum()), "filled_unsettled_pending": int((filled & ~f["settled"]).sum()),
                "bets": int(ok.sum())})
    return with_report(b, rep)
