"""G2 frame (prereg notes/gas_daily_v2.md): DEV events only (D <= 2026-08-31). Read-only Kalshi GETs.
-> data/gas_daily_v2/gd_frame.parquet, gd_candles.json, aaa_inputs.parquet"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from kx import get
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "gas_daily_v2"; OUT.mkdir(parents=True, exist_ok=True)
E = pd.Timestamp("1970-01-01", tz="UTC")
DEV_END = pd.Timestamp("2026-08-31")
DEC4_FROM = pd.Timestamp("2026-08-05")
sec = lambda x: int((pd.Timestamp(x) - E) // pd.Timedelta(seconds=1))

aaa = pd.read_parquet("aaa_kalshi.parquet")[["day", "v"]].set_index("day").v.sort_index()
rb = pd.read_parquet("rb_daily.parquet").Close
rb.index = pd.to_datetime(rb.index).tz_localize(None).normalize()
rb = rb.sort_index()

def rb_feats(d):
    r = rb[rb.index <= d - pd.Timedelta(days=2)]
    if len(r) < 11: return np.nan, np.nan
    return (r.iloc[-1] - r.iloc[-6]) * 100, (r.iloc[-6] - r.iloc[-11]) * 100

def model_probs(D, strikes):
    """All inputs strictly before D: AAA days <= D-1, RB settles <= D-2."""
    a = aaa[aaa.index < D]
    if len(a) == 0 or a.index[-1] != D - pd.Timedelta(days=1):
        return None
    c = (a.diff() * 100)
    days = a.index
    rows = []
    for d in days:
        c0 = c.get(d, np.nan); c1 = c.get(d - pd.Timedelta(days=1), np.nan); c2 = c.get(d - pd.Timedelta(days=2), np.nan)
        f5, f10 = rb_feats(d)
        rows.append((d, c0, c1, c2, f5, f10))
    tr = pd.DataFrame(rows, columns=["d", "c", "c1", "c2", "f5", "f10"]).dropna()
    h = 0.0005 if D < DEC4_FROM else 0.00005
    last = a.iloc[-1]
    out = {"n_train": len(tr)}
    sc = np.sqrt(2.0)
    # model M
    cD1, cD2 = c.get(D - pd.Timedelta(days=1), np.nan), c.get(D - pd.Timedelta(days=2), np.nan)
    f5, f10 = rb_feats(D)
    xD = np.array([cD1, cD2, f5, f10])
    if len(tr) >= 30 and np.isfinite(xD).all():
        A = np.c_[np.ones(len(tr)), tr[["c1", "c2", "f5", "f10"]].to_numpy()]
        b, *_ = np.linalg.lstsq(A, tr.c.to_numpy(), rcond=None)
        res = tr.c.to_numpy() - A @ b
        s = res[-60:].std(ddof=1)
        mu = last + (b[0] + xD @ b[1:]) / 100
        out["mu_m"], out["s_m"] = mu, s
        out["p_m"] = [1 - stats.t.cdf((X + h - mu) / (s / 100 / sc), 4) for X in strikes]
    # placebo RW
    cc = c.dropna()
    if len(cc) >= 30:
        s = cc.iloc[-60:].std(ddof=1)
        out["p_rw"] = [1 - stats.t.cdf((X + h - last) / (s / 100 / sc), 4) for X in strikes]
        pe = (cc - cc.shift(1)).dropna()
        if len(pe) >= 30 and np.isfinite(cD1):
            sm = np.sqrt(np.mean(pe.iloc[-60:] ** 2))
            mu = last + cD1 / 100
            out["p_mom"] = [1 - stats.t.cdf((X + h - mu) / (sm / 100 / sc), 4) for X in strikes]
    return out

mk = pd.read_parquet("markets.parquet")
mk = mk[mk.event_ticker.str.startswith("KXAAAGASD") & (mk.status == "finalized")].copy()
mk["date"] = pd.to_datetime(mk.event_ticker.str.split("-").str[1], format="%y%b%d")
mk = mk[mk.date <= DEV_END].copy()
print("dev markets", len(mk), "events", mk.event_ticker.nunique(), flush=True)
rows, cand = [], {}
for e, g in mk.groupby("event_ticker"):
    D = g.date.iloc[0]
    T = pd.Timestamp(D.year, D.month, D.day, 10, tz="America/New_York") - pd.Timedelta(days=1)
    T = T.tz_convert("UTC")
    strikes = g.floor_strike.tolist()
    mp = model_probs(D, strikes) or {}
    for i, r in enumerate(g.itertuples(index=False)):
        rows.append({"ticker": r.ticker, "event_ticker": e, "date": D, "X": r.floor_strike, "T": T,
                     "open": pd.Timestamp(r.open_time), "close": pd.Timestamp(r.close_time),
                     "p_m": mp["p_m"][i] if "p_m" in mp else np.nan, "mu_m": mp.get("mu_m", np.nan), "s_m": mp.get("s_m", np.nan),
                     "p_rw": mp["p_rw"][i] if "p_rw" in mp else np.nan, "p_mom": mp["p_mom"][i] if "p_mom" in mp else np.nan,
                     "n_train": mp.get("n_train", 0), "result": r.result, "expiration_value": r.expiration_value})
        d = get(f"/historical/markets/{r.ticker}/candlesticks",
                {"start_ts": sec(r.open_time) - 3600, "end_ts": sec(r.close_time) + 3600, "period_interval": 60})
        cs = d.get("candlesticks")
        if not cs:
            d = get(f"/series/KXAAAGASD/markets/{r.ticker}/candlesticks",
                    {"start_ts": sec(r.open_time) - 3600, "end_ts": sec(r.close_time) + 3600, "period_interval": 60})
            cs = d.get("candlesticks", [])
        cand[r.ticker] = cs
    print(e, len(g), "p_m" in mp, flush=True)
fr = pd.DataFrame(rows)
fr.to_parquet(OUT / "gd_frame.parquet", index=False)
(OUT / "gd_candles.json").write_text(json.dumps(cand))
aaa.reset_index().to_parquet(OUT / "aaa_inputs.parquet", index=False)
print("rows", len(fr), "with p_m", int(fr.p_m.notna().sum()), "events with p_m", fr[fr.p_m.notna()].event_ticker.nunique(),
      "markets with candles", sum(1 for v in cand.values() if v))
