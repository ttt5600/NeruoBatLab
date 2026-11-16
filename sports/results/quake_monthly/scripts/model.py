"""Frozen model (prereg in notes/quake_monthly.md): P(month max >= K | info at decision time t).

p(K,t) = 1 if observed max since month start (USGS, as of t) >= K
       else W(K, T) = empirical share of windows of length T = (month end - t) days whose max M >= K,
            windows starting at every 00:00 UTC day in [2000-01-01, decision date - 30 d) (strictly before t),
            catalog = USGS ComCat M>=6 events (current values, eventtype earthquake).
Empirical windows carry aftershock clustering automatically (a window gets one 'hit' however many events it has).
ETAS-lite guard: if an M >= 7.5 occurred in the 30 d before t, the trial SKIPS the event (rate elevated, model not
conditioned on it).
Also reported (descriptive): Poisson on declustered rate (one event per 3-day/any-location cluster).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parents[1]
cat = pd.read_parquet(OUT / "usgs_m6.parquet")[["time", "mag"]].sort_values("time")
RUNGS = [6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2]


def p_model(t: pd.Timestamp, month_start: pd.Timestamp, month_end: pd.Timestamp, catalog=cat, start="2000-01-01"):
    T_days = (month_end - t) / pd.Timedelta(days=1)          # traps: Timedelta division, no int64 // 1e9
    obs = catalog[(catalog.time >= month_start) & (catalog.time <= t)].mag.max()
    obs = -np.inf if pd.isna(obs) else obs
    hist = catalog[(catalog.time >= start) & (catalog.time < t - pd.Timedelta(days=30))]
    t0 = pd.Timestamp(start, tz="UTC")
    starts = pd.date_range(t0, t - pd.Timedelta(days=30) - pd.Timedelta(days=T_days), freq="D")
    tt = hist.time.values.astype("datetime64[ns]").astype("int64")
    mm = hist.mag.values
    s_ns = starts.values.astype("datetime64[ns]").astype("int64")
    e_ns = s_ns + int(T_days * 86400e9)
    out = {}
    for K in RUNGS:
        if obs >= K:
            out[K] = 1.0
            continue
        sel = tt[mm >= K - 1e-9]
        i0 = np.searchsorted(sel, s_ns, "left"); i1 = np.searchsorted(sel, e_ns, "left")
        out[K] = float(np.mean(i1 > i0))
    big = catalog[(catalog.time >= t - pd.Timedelta(days=30)) & (catalog.time <= t) & (catalog.mag >= 7.5)]
    return {"T_days": T_days, "obs_max": obs, "p": out, "guard_skip": bool(len(big)), "n_windows": len(starts)}


if __name__ == "__main__":
    t = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now(tz="UTC").floor("min"))
    if t.tzinfo is None: t = t.tz_localize("UTC")
    r = p_model(t, pd.Timestamp("2026-11-01", tz="UTC"), pd.Timestamp("2026-12-01", tz="UTC"))
    print(json.dumps(r, indent=1, default=float))
    # stability check: 5 blocks of history
    for a, b in [("1990", "1997"), ("1997", "2004"), ("2004", "2011"), ("2011", "2018"), ("2018", "2026")]:
        c = cat[(cat.time >= a) & (cat.time < b)]
        rr = p_model(pd.Timestamp(f"{b}-01-01", tz="UTC") - pd.Timedelta(days=0), pd.Timestamp(f"{b}-01-01", tz="UTC"),
                     pd.Timestamp(f"{b}-01-01", tz="UTC") + pd.Timedelta(days=r["T_days"]), catalog=c, start=f"{a}-01-01")
        print(a, b, {k: round(v, 3) for k, v in rr["p"].items()})
