"""DL-PW: Kalshi weekly chokepoint transits (PortWatch) far-bracket maker. Prereg: notes/data_ladders_far.md.
Stages: audit -> build -> candles -> trades   (scoring only via `python -m edgelab evaluate pw_t1_maker`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "dl"
SER = {"KXHORMUZWEEKLY": "chokepoint6", "KXPANAMAWEEKLY": "chokepoint2", "KXSUEZWEEKLY": "chokepoint1",
       "KXBABELMANDEBWEEKLY": "chokepoint4"}
M = 0.05
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def daily() -> dict[str, pd.Series]:
    df = pd.read_parquet(D / "portwatch_daily.parquet")
    d = df["date"]
    if pd.api.types.is_numeric_dtype(d):
        traps.check_epoch_seconds(d // 1000, "portwatch date ms")
        d = pd.to_datetime(d, unit="ms", utc=True)
    else:
        d = pd.to_datetime(d, utc=True)
    df["day"] = d.dt.tz_localize(None).dt.normalize()
    out = {}
    for pid, g in df.groupby("portid"):
        s = g.drop_duplicates("day").set_index("day")["n_total"].astype(float).sort_index()
        out[pid] = s
    return out


def wsum(s: pd.Series, wend: pd.Timestamp, sunsat: bool = False) -> float:
    end = wend - pd.Timedelta(days=1) if sunsat else wend
    days = pd.date_range(end - pd.Timedelta(days=6), end)
    if days.min() < s.index.min() or days.max() > s.index.max():
        return np.nan
    v = s.reindex(days)
    return float(v.sum()) if v.notna().all() else np.nan


def markets() -> pd.DataFrame:
    xs = []
    for ser, pid in SER.items():
        m = pd.read_parquet(D / f"{ser}_markets.parquet")
        m["series"], m["portid"] = ser, pid
        xs.append(m)
    m = pd.concat(xs, ignore_index=True)
    m["wend"] = pd.to_datetime(m.event_ticker.str.split("-").str[1], format="%y%b%d")
    m["K"] = m.floor_strike.astype(float)
    return m[m.K.notna() & (m.strike_type == "greater")].copy()


def cmd_audit() -> int:
    s = daily()
    m = markets()
    m = m[m.result.isin(["yes", "no"])]
    for sunsat in (False, True):
        ok = []
        for ev, g in m.groupby("event_ticker"):
            A = wsum(s[g.portid.iloc[0]], g.wend.iloc[0], sunsat)
            if np.isnan(A):
                continue
            ok.append(bool(((A > g.K) == (g.result == "yes")).all()))
        print("sunsat" if sunsat else "mon-sun", "events", len(ok), "consistent", sum(ok),
              f"{np.mean(ok):.3f}" if ok else "")
    return 0


def pub_times(m: pd.DataFrame) -> dict:
    """week end -> earliest Kalshi settlement_ts over the 4 series = proxy for when PortWatch published that week
    (publication is usually Tuesday ~14:13Z but was a day late on 10-13 and 10-27)."""
    st = pd.to_datetime(m["settlement_ts"], utc=True, format="ISO8601")
    return st.groupby(m["wend"]).min().dropna().to_dict()


def cmd_build() -> int:
    s = daily()
    m = markets()
    pub = pub_times(m)
    rows = []
    for ev, g in m.groupby("event_ticker"):
        wend = g.wend.iloc[0]
        pid = g.portid.iloc[0]
        x = s[pid]
        T = pd.Timestamp(wend - pd.Timedelta(days=4), tz="UTC")      # Wednesday 00:00Z of week W
        F = wsum(x, wend - pd.Timedelta(days=7))
        # residual weeks V: week end <= T - 9 days (published by the Tuesday before last)
        last_v = (T - pd.Timedelta(days=9)).tz_localize(None).normalize()
        ends = pd.date_range(x.index.min() + pd.Timedelta(days=13), last_v, freq="W-SUN")
        A = np.array([wsum(x, e) for e in ends])
        A0 = np.array([wsum(x, e - pd.Timedelta(days=7)) for e in ends])
        ok = ~np.isnan(A) & ~np.isnan(A0)
        r = np.log((A[ok] + 1) / (A0[ok] + 1))
        rmin, rmax = (r.min(), r.max()) if len(r) else (np.nan, np.nan)
        for _, k in g.iterrows():
            rows.append({"ticker": k.ticker, "event_ticker": ev, "series": k.series, "portid": pid, "wend": wend,
                         "K": k.K, "T": T, "open_time": k.open_time, "close_time": k.close_time,
                         "F": F, "pub_prev": pub.get(wend - pd.Timedelta(days=7), pd.NaT), "r_min": rmin, "r_max": rmax, "n_resid": int(ok.sum()),
                         "hi": (F + 1) * np.exp(rmax + M) - 1, "lo": (F + 1) * np.exp(rmin - M) - 1,
                         "_src": k._src, "result": k.result})
    b = pd.DataFrame(rows)
    b.to_parquet(D / "pw_build.parquet")
    print("rungs", len(b), "events", b.event_ticker.nunique())
    return 0


def rungs() -> pd.DataFrame:
    x = pd.read_parquet(D / "pw_build.parquet")
    x = x[x.F.notna() & (x.n_resid >= 52)]
    # point-in-time: week W-1 must have been published (settled on Kalshi) before T
    x = x[pd.to_datetime(x.pub_prev, utc=True) < pd.to_datetime(x["T"], utc=True)]
    x = x[pd.to_datetime(x.open_time, utc=True) < pd.to_datetime(x["T"], utc=True)]
    x = x[pd.to_datetime(x.close_time, utc=True) > pd.to_datetime(x["T"], utc=True)]
    x["side"] = np.where(x.K >= x.hi, "no", np.where(x.K <= x.lo, "yes", ""))
    return x[x.side != ""].copy()


def _ts(t) -> int:
    return int((pd.Timestamp(t) - EPOCH) // pd.Timedelta(seconds=1)) if pd.Timestamp(t).tzinfo else \
        int((pd.Timestamp(t, tz="UTC") - EPOCH) // pd.Timedelta(seconds=1))


def cmd_candles() -> int:
    x = rungs()
    cache = D / "pw_candles.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ts = _ts(m["T"])
        p = {"start_ts": ts - 12 * 3600, "end_ts": ts + 6 * 86400, "period_interval": 60}
        if m["_src"] == "/historical/markets":
            r = get(f"/historical/markets/{m.ticker}/candlesticks", p)
        else:
            r = get(f"/series/{m.series}/markets/{m.ticker}/candlesticks", p)
        out[m.ticker] = r.get("candlesticks", [])
    cache.write_text(json.dumps(out))
    print("rejected rungs", len(x), "events", x.event_ticker.nunique(), "with candles", sum(bool(v) for v in out.values()))
    return 0


def cmd_trades() -> int:
    x = rungs()
    cache = D / "pw_trades.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for _, m in x.iterrows():
        if m.ticker in out:
            continue
        ep = "/historical/trades" if m["_src"] == "/historical/markets" else "/markets/trades"
        tr, cur = [], None
        while True:
            p = {"ticker": m.ticker, "min_ts": _ts(m["T"]), "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            tr += r.get("trades", [])
            cur = r.get("cursor")
            if not cur or not r.get("trades"):
                break
        out[m.ticker] = tr
    cache.write_text(json.dumps(out))
    print("tickers", len(out), "trades", sum(len(v) for v in out.values()))
    return 0


if __name__ == "__main__":
    sys.exit({"audit": cmd_audit, "build": cmd_build, "candles": cmd_candles, "trades": cmd_trades}[sys.argv[1]]())
