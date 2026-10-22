"""Per-name earnings stats from Yahoo (yfinance, free): EPS beat history, realized post-earnings moves, and the current
options-implied move (ATM straddle mid of the first expiry on/after the report).

    python scripts/earnings_moves.py BE PG LMT ...   -> prints a table, writes data/earnings/moves.json

Definitions
  beat      : Yahoo 'Reported EPS' > 'EPS Estimate' (strict, like the Polymarket contracts; Yahoo's consensus is
              non-GAAP-style "street" EPS -- NOT the same strike as a GAAP Polymarket market).
  timing    : report hour >= 16 ET -> AMC: move = close(report day) -> close(next day);
              hour < 12 -> BMO: move = close(previous day) -> close(report day).
  implied   : (call mid + put mid) at the strike nearest spot, first expiry >= report date, / spot. Includes the normal
              (non-earnings) variance of every trading day up to that expiry, so it slightly OVERSTATES the earnings move.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parents[1] / "data" / "earnings"


def one(tk: str, n_hist: int = 12) -> dict:
    t = yf.Ticker(tk)
    ed = t.get_earnings_dates(limit=n_hist + 8)
    if ed is None or ed.empty:
        return {"ticker": tk, "err": "no earnings dates"}
    ed = ed.sort_index()
    now = pd.Timestamp.now(tz=ed.index.tz)
    fut = ed[ed["Reported EPS"].isna() & (ed.index >= now - pd.Timedelta(days=1))]
    past = ed[ed["Reported EPS"].notna()].tail(n_hist)
    px = t.history(period="5y", auto_adjust=True)["Close"]
    px.index = px.index.tz_convert(None).normalize()
    moves, beats = [], []
    for ts, r in past.iterrows():
        d = ts.tz_convert("America/New_York")
        day = pd.Timestamp(d.date())
        if r["EPS Estimate"] == r["EPS Estimate"]:
            beats.append(int(r["Reported EPS"] > r["EPS Estimate"] + 1e-9))
        try:
            if d.hour >= 16:
                i = px.index.get_indexer([day], method="pad")[0]
                a, b = px.iloc[i], px.iloc[i + 1]
            else:
                i = px.index.get_indexer([day], method="bfill")[0]
                a, b = px.iloc[i - 1], px.iloc[i]
            moves.append(float(b / a - 1))
        except Exception:
            pass
    out = {"ticker": tk, "n_q": len(beats), "beat_rate": round(float(np.mean(beats)), 3) if beats else None,
           "beats_last12": "".join("B" if b else "m" for b in beats),
           "streak": next((i for i, b in enumerate(reversed(beats)) if not b), len(beats)),
           "moves": [round(m, 4) for m in moves],
           "mean_abs_move": round(float(np.mean(np.abs(moves))), 4) if moves else None,
           "median_abs_move": round(float(np.median(np.abs(moves))), 4) if moves else None,
           "up_rate": round(float(np.mean(np.array(moves) > 0)), 3) if moves else None}
    if not fut.empty:
        nx = fut.index[0].tz_convert("America/New_York")
        out["next"] = str(nx)
        out["eps_est"] = None if fut.iloc[0]["EPS Estimate"] != fut.iloc[0]["EPS Estimate"] else float(fut.iloc[0]["EPS Estimate"])
        out["timing"] = "AMC" if nx.hour >= 16 else ("BMO" if nx.hour < 12 else "?")
        try:
            spot = float(t.history(period="1d")["Close"].iloc[-1])
            exps = [e for e in t.options if pd.Timestamp(e) >= pd.Timestamp(nx.date())]
            # AMC report: the move lands the next trading day -> need expiry strictly after the report day
            if out["timing"] == "AMC":
                exps = [e for e in exps if pd.Timestamp(e) > pd.Timestamp(nx.date())]
            exp = exps[0]
            ch = t.option_chain(exp)
            k = ch.calls.loc[(ch.calls.strike - spot).abs().idxmin(), "strike"]
            c = ch.calls[ch.calls.strike == k].iloc[0]
            p = ch.puts[ch.puts.strike == k].iloc[0]
            strad = (c.bid + c.ask) / 2 + (p.bid + p.ask) / 2
            out.update({"spot": round(spot, 2), "expiry": exp, "strike": float(k), "straddle": round(float(strad), 3),
                        "implied_move": round(float(strad / spot), 4),
                        "straddle_spread": round(float((c.ask - c.bid) + (p.ask - p.bid)), 3)})
            if out["mean_abs_move"]:
                out["implied_over_hist"] = round(out["implied_move"] / out["mean_abs_move"], 2)
        except Exception as e:  # noqa: BLE001
            out["opt_err"] = str(e)[:80]
    return out


if __name__ == "__main__":
    res = []
    for tk in sys.argv[1:]:
        try:
            r = one(tk)
        except Exception as e:  # noqa: BLE001
            r = {"ticker": tk, "err": str(e)[:100]}
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "moves"}))
    fn = D / "moves.json"
    old = {r["ticker"]: r for r in json.load(open(fn))} if fn.exists() else {}
    old.update({r["ticker"]: r for r in res})
    json.dump(list(old.values()), open(fn, "w"), indent=1)
