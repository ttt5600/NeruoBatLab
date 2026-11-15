"""P1m: L39 earlier-in-month CLIMATOLOGY version (7-20 days left). DEV only. Pre-registration: notes/l39_expansion.md.

    python scripts/l39_p1m.py build    # dev ladders x decision days 7..20: ACIS state + climatology Q995 + hourly quotes
    python scripts/l39_p1m.py dev      # picks, maker fills, score (pass rule from the prereg)

Kalshi GETs go through scripts/kx_throttle.slot() (shared cross-process throttle) and the sports.http disk cache.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import kalshi_wx_winter as W
import kx_throttle
from sports import http as H

OUT = ROOT / "results" / "l39_expansion"
FRAME = OUT / "p1m_frame.parquet"
SPLIT = {"rain": pd.Timestamp("2026-07-01"), "snow": pd.Timestamp("2026-03-01")}
DL = range(7, 21)
QS = (0.99, 0.995, 0.999)
BAR = 0.05 / 11

_raw_fetch = W.fetch


def tfetch(url, params=None, **kw):
    """Throttle only Kalshi requests that are not already in the disk cache."""
    if "kalshi" in url:
        path = H.CACHE_DIR / (H._key(url, params) + ".json")
        if not (kw.get("cache", True) and path.exists()):
            kx_throttle.slot()
    return _raw_fetch(url, params, **kw)


W.fetch = tfetch


def climo(sid: str, var: str) -> pd.Series:
    df = W.acis(sid, "1991-01-01", "2026-06-30")
    return df.set_index("date")[var] if len(df) else pd.Series(dtype=float)


def quantiles(daily: pd.Series, d: pd.Timestamp, L: int, year: int) -> tuple[list[float], int]:
    s = daily[daily.index < pd.Timestamp(year, 1, 1)]
    if s.empty:
        return [np.nan] * len(QS), 0
    full = s.reindex(pd.date_range(s.index.min(), s.index.max()))
    roll = full.rolling(L, min_periods=L).sum().shift(-(L - 1))  # value at t = sum over t..t+L-1 (NaN if any missing)
    tots = []
    for y in range(1991, year):
        base = pd.Timestamp(y, d.month, min(d.day, 28))
        for k in range(-10, 11):
            t = base + pd.Timedelta(days=k)
            v = roll.get(t, np.nan)
            if pd.notna(v):
                tots.append(v)
    if len(tots) < 200:
        return [np.nan] * len(QS), len(tots)
    a = np.array(tots)
    return [float(np.quantile(a, q, method="higher")) for q in QS], len(tots)


def cmd_build() -> int:
    base = pd.read_parquet(ROOT / "data" / "kalshi_wx_winter.parquet")
    base = base[base.month < base.kind.map(SPLIT)]  # DEV only; sealed rows are never touched below
    mk = base.drop_duplicates("ticker")[["series", "event_ticker", "ticker", "kind", "sid", "tz", "month", "end", "X",
                                          "result", "close_time"]]
    print(f"dev ladders {mk.event_ticker.nunique()} city-months, {len(mk)} markets", flush=True)
    cl = {}
    rows = []
    for (series, ev), g in mk.groupby(["series", "event_ticker"]):
        sid, kind, tz, month, end = g.sid.iloc[0], g.kind.iloc[0], g.tz.iloc[0], g.month.iloc[0], g.end.iloc[0]
        var = "pcpn" if kind == "rain" else "snow"
        if (sid, var) not in cl:
            cl[(sid, var)] = climo(sid, var)
        daily = cl[(sid, var)]
        ob = W.acis(sid, str(month.date()), str(end.date()))
        n_missing = int(ob[var].isna().sum()) if len(ob) else 99
        t0 = W.decision_ts(end - pd.Timedelta(days=max(DL)), tz) - 3 * 3600
        t1 = W.decision_ts(end - pd.Timedelta(days=min(DL)), tz)
        cs = W.candles(series, ev, g.ticker.tolist(), t0, t1)
        for dl in DL:
            dd = end - pd.Timedelta(days=dl)
            if dd < month:
                continue
            T = W.decision_ts(dd, tz)
            obs_d = float(ob.loc[ob.date < dd, var].sum()) if n_missing == 0 else np.nan
            qv, nwin = quantiles(daily, dd, dl + 1, month.year)
            for _, m in g.iterrows():
                c = [x for x in cs.get(m.ticker, []) if x["end_period_ts"] <= T]
                last = c[-1] if c else None
                rows.append({"series": series, "event_ticker": ev, "ticker": m.ticker, "kind": kind, "sid": sid,
                             "tz": tz, "month": month, "end": end, "d": dd, "days_left": dl, "T": T,
                             "price_ts": last["end_period_ts"] if last else np.nan, "X": m.X, "result": m.result,
                             "close_time": m.close_time, "obs_d": obs_d, "acis_missing": n_missing, "n_win": nwin,
                             **{f"q{int(q * 1000)}": v for q, v in zip(QS, qv)},
                             "yes_bid": W._px(last, "yes_bid") if last else np.nan,
                             "yes_ask": W._px(last, "yes_ask") if last else np.nan,
                             "vol_window": float(sum(float(x.get("volume_fp", x.get("volume", 0)) or 0)
                                                     for x in cs.get(m.ticker, [])))})
        print(f"  {ev:24s} {sid} {kind} mk {len(g)} miss {n_missing} nwin {nwin}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(FRAME, index=False)
    print(f"-> {FRAME} {len(df)} rows")
    return 0


def picks(df: pd.DataFrame, qcol: str) -> pd.DataFrame:
    d = df[(df.acis_missing == 0) & df[qcol].notna()].copy()
    d = d[(d.yes_bid > 0) & (d.yes_ask < 1) & ((d.yes_ask - d.yes_bid) <= 0.10 + 1e-9) & (d.yes_ask > 0.01)]
    d = d[d["T"] < W.close_s(d.close_time)]
    s = d[(d.X - d.obs_d) > d[qcol]].copy()
    s = s.sort_values("d").drop_duplicates("ticker", keep="first")
    s["side"] = "no"
    s["won"] = (s.result == "no").astype(int)
    a = 1 - s.yes_bid
    s["taker_cost"] = a + W.FEE * a * (1 - a)
    return s


def cmd_dev() -> int:
    df = pd.read_parquet(FRAME)
    assert (df.month < df.kind.map(SPLIT)).all(), "sealed rows in frame"
    res = {}
    for qcol in ("q995", "q990", "q999"):
        s = picks(df, qcol)
        if not len(s):
            res[qcol] = {"bets": 0}
            continue
        s = W.maker(s)
        f = s[s.filled]
        r = {"taker": W.score(s, "taker_cost"), "maker": W.score(f, "maker_cost") if len(f) else {"bets": 0},
             "fill_rate": float(s.filled.mean()), "picks": int(len(s))}
        r["taker"]["null_p"] = W.null_p(s, "taker_cost")
        if len(f):
            r["maker"]["null_p"] = W.null_p(f, "maker_cost")
        res[qcol] = r
        s.to_parquet(OUT / f"p1m_picks_{qcol}.parquet", index=False)
        print(f"\n== {qcol} {'(DECISION)' if qcol == 'q995' else '(diagnostic)'}: picks {len(s)} fill {s.filled.mean():.2f}")
        for k in ("maker", "taker"):
            x = r[k]
            if x.get("bets"):
                print(f"  {k}: n {x['bets']} cm {x['city_months']} L {x['losses']} ROI {x['roi']:+.4f} units {x['units']:+.2f}"
                      f" CI [{x['ci_lo']:+.2f},{x['ci_hi']:+.2f}] p {x['p']:.4f} null_p {x['null_p'][0]:.4f}/{x['null_p'][1]:.4f}"
                      f" halves {x['half1']} {x['half2']} avg_cost {x['avg_cost']:.3f}")
    # tape for edgelab (input-true tickers of the decision quantile; fetched from the first qualifying T, cached)
    tp = []
    s = picks(df, "q995")
    for tk, T in zip(s.ticker, s["T"]):
        t = W.tape(tk, int(T))
        t["ticker"] = tk
        tp.append(t)
    (pd.concat(tp, ignore_index=True) if tp else pd.DataFrame(columns=["ts", "no_price", "count", "ticker"])
     ).to_parquet(OUT / "p1m_tape.parquet", index=False)
    m = res["q995"].get("maker", {})
    ok = bool(m.get("bets", 0) >= 60 and m["p"] < BAR and m["null_p"][0] < BAR and (m["half1"][0] or 0) > 0
              and (m["half2"][0] or 0) > 0)
    res["verdict"] = "pass" if ok else "fail"
    res["bar"] = BAR
    print(f"\nP1m verdict (q995 maker, bar {BAR:.5f}): {res['verdict']}")
    json.dump(res, open(OUT / "p1m_dev.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit({"build": cmd_build, "dev": cmd_dev}[sys.argv[1]]())
